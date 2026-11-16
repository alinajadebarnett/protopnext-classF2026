"""
ProtoPool: Interpretable Image Classification with Differentiable Prototypes Assignment
(Rymarczyk et al., ECCV 2022, https://arxiv.org/abs/2112.02902).

Ported from https://github.com/gmum/ProtoPool (`PrototypeChooser` in model.py, training loop in main.py).
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..activations import L2Activation
from ..embedding import EmbeddedBackbone
from ..prediction_heads import ProtoPoolPredictionHead
from ..prototype_layers import ProtoPoolPrototypeLayer
from ..prototypical_part_model import ProtoPNet
from ..train.metrics import TrainingMetrics
from ..train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    TrainLayersUsingProtoPNetNames,
)
from ..train.scheduling.types import (
    IterativePhase,
    Phase,
    StepContext,
    TrainingSchedule,
)

log = logging.getLogger(__name__)


class ProtoPool(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation: L2Activation,
        num_classes: int,
        num_prototypes: int,
        num_descriptive: int,
        tau: float = 0.5,
        warn_on_errors: bool = False,
        k_for_topk: int = 1,
    ):
        prototype_layer = ProtoPoolPrototypeLayer(
            num_prototypes=num_prototypes,
            num_classes=num_classes,
            activation_function=activation,
            latent_channels=add_on_layers.proto_channels,
            k_for_topk=k_for_topk,
        )

        prediction_head = ProtoPoolPredictionHead(
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            num_descriptive=num_descriptive,
            tau=tau,
            epsilon_val=activation.epsilon_val,
        )

        super(ProtoPool, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            warn_on_errors=warn_on_errors,
            k_for_topk=k_for_topk,
        )

        self.reinit_parameters()

    @property
    def num_classes(self) -> int:
        return self.prototype_prediction_head.num_classes

    @property
    def num_descriptive(self) -> int:
        return self.prototype_prediction_head.num_descriptive

    def reinit_parameters(self):
        """
        Add-on layer initialization from the original ProtoPool implementation.
        """
        with torch.no_grad():
            for m in self.add_on_layers.modules():
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)

    def slot_prototype_indices(self) -> torch.Tensor:
        return self.prototype_prediction_head.slot_prototype_indices()

    def effective_class_connections(self) -> torch.Tensor:
        return self.prototype_prediction_head.effective_class_connections()

    def prototype_class_usage(self) -> torch.Tensor:
        """
        Returns: (num_prototypes, num_classes) count of the slots of each class that use each prototype.
        """
        slot_indices = self.slot_prototype_indices()
        usage = torch.zeros(
            self.num_prototypes,
            self.num_classes,
            device=slot_indices.device,
        )
        class_indices = torch.arange(
            self.num_classes, device=slot_indices.device
        ).repeat_interleave(self.num_descriptive)
        usage.index_put_(
            (slot_indices.flatten(), class_indices),
            torch.ones_like(class_indices, dtype=usage.dtype),
            accumulate=True,
        )
        return usage

    def project(
        self, dataloader: torch.utils.data.DataLoader, transform: callable = None
    ):
        head = self.prototype_prediction_head
        if not head.hard_assignment:
            head.set_hard_assignment()
        self.prototype_layer.set_prototype_allowed_classes(
            self.prototype_class_usage() > 0
        )
        return super(ProtoPool, self).project(dataloader, transform=transform)

    def prune_duplicate_prototypes(self, decimal_precision=8) -> None:
        """
        Merges prototypes with identical tensors: slots pointing at a duplicate are redirected
        to the first copy, then the duplicates are removed from the layer and head.
        """
        head = self.prototype_prediction_head
        if not head.hard_assignment:
            log.warning(
                "Skipping ProtoPool duplicate pruning: the slot assignment is not hard yet (prototypes are not projected)."
            )
            return

        prototype_tensors = self.prototype_tensors
        num_prototypes = prototype_tensors.shape[0]
        canonical_index = torch.arange(num_prototypes, device=prototype_tensors.device)
        duplicate_indices = []
        for proto_ind in range(1, num_prototypes):
            equiv = (
                torch.isclose(
                    prototype_tensors[:proto_ind], prototype_tensors[proto_ind]
                )
                .flatten(start_dim=1)
                .all(dim=1)
            )
            if equiv.any():
                canonical_index[proto_ind] = canonical_index[
                    torch.argmax(equiv.int()).item()
                ]
                duplicate_indices.append(proto_ind)

        logger_msg = f"Pruning from {num_prototypes} prototypes to {num_prototypes - len(duplicate_indices)}"
        log.info(logger_msg)

        if len(duplicate_indices) == 0:
            return

        head.slot_prototype_index = canonical_index.to(
            head.slot_prototype_index.device
        )[head.slot_prototype_index]
        self.prune_prototypes_by_index(duplicate_indices)
        self.prototype_layer.set_prototype_allowed_classes(
            self.prototype_class_usage() > 0
        )

    def get_prototype_class_identity(self, label) -> torch.Tensor:
        return (self.prototype_class_usage() > 0).float()[:, label]

    def class_major_prototype_activations(
        self, prototype_activations: torch.Tensor
    ) -> torch.Tensor:
        slot_indices = self.slot_prototype_indices().flatten()
        return prototype_activations[:, slot_indices.to(prototype_activations.device)]

    def metric_prototypes_per_class(self) -> int:
        return self.num_descriptive

    def prototype_primary_classes(self) -> Tuple[torch.Tensor, int]:
        """
        Returns: the primary class of each prototype (the class with most slots using it, ties and unused
            prototypes broken by the soft assignment mass), and the number of classes.
        """
        soft_mass = (
            torch.softmax(self.prototype_prediction_head.proto_presence.detach(), dim=1)
            .sum(dim=-1)
            .T
        )  # [P, C]
        scores = self.prototype_class_usage() + 1e-3 * soft_mass.to(
            self.prototype_tensors.device
        )
        return torch.argmax(scores, dim=1), self.num_classes

    def describe_prototypes(self):
        ret_str = ""
        usage = self.prototype_class_usage()
        effective_connections = self.effective_class_connections()
        for proto_index, proto_info in enumerate(self.prototype_layer.prototype_meta):
            classes = torch.nonzero(usage[proto_index]).flatten().tolist()
            sample_id = getattr(proto_info, "sample_id", None)
            ret_str += (
                f"\nPrototype {proto_index} comes from sample {sample_id}."
                f"\n\tIt is used by the slots of classes {classes}, with an effective class connection vector of:"
                f"\n\t\t{effective_connections[:, proto_index]}"
            )
        return ret_str


@dataclass(frozen=True)
class GumbelScaleSchedule:
    """
    Gumbel-softmax scale annealing from ProtoPool: start * sqrt(alpha * epoch), capped at end,
    with alpha chosen so the cap is reached after `gumbel_epochs` epochs.

    The epoch is 1-indexed (global_step + 1) because the original never trains at epoch 0,
    where the scale would be 0.
    """

    gumbel_epochs: int = 30
    start: float = 1.3
    end: float = 1000.0
    enabled: bool = True

    def __call__(self, global_step: int) -> float:
        if not self.enabled:
            return 0.0
        epoch = global_step + 1
        if epoch >= self.gumbel_epochs:
            return self.end
        alpha = (self.end / self.start) ** 2 / self.gumbel_epochs
        return self.start * math.sqrt(alpha * epoch)


def mixup_batch(
    batch_data_dict: Dict[str, Any], alpha: float, generator: np.random.Generator
) -> Dict[str, Any]:
    """
    Mixup (Zhang et al. 2018) as used by ProtoPool: mixes images within the batch and adds
    `mixup_target_b` and `mixup_lambda` for MixupCrossEntropyCost.
    """
    img = batch_data_dict["img"]
    target = batch_data_dict["target"]
    lam = float(generator.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(img.shape[0], device=img.device)

    batch_data_dict["img"] = lam * img + (1 - lam) * img[index]
    batch_data_dict["mixup_target_b"] = target[index]
    batch_data_dict["mixup_lambda"] = lam
    return batch_data_dict


class ProtoPoolBackpropPhase(ClassificationBackpropPhase):
    """
    Backprop phase that sets the head's Gumbel scale at the start of each epoch and optionally
    applies mixup to every training batch.
    """

    def __init__(
        self,
        *args,
        gumbel_schedule: Optional[GumbelScaleSchedule] = None,
        mixup_alpha: Optional[float] = None,
        mixup_seed: Optional[int] = None,
        **kwargs,
    ):
        super(ProtoPoolBackpropPhase, self).__init__(*args, **kwargs)
        self._gumbel_schedule = gumbel_schedule
        self._mixup_alpha = mixup_alpha
        self._mixup_generator = np.random.default_rng(mixup_seed)

    @property
    def gumbel_schedule(self) -> Optional[GumbelScaleSchedule]:
        return self._gumbel_schedule

    @property
    def mixup_alpha(self) -> Optional[float]:
        return self._mixup_alpha

    def _pre_dataloader_step_init(self, model: ProtoPool, step_context: StepContext):
        super(ProtoPoolBackpropPhase, self)._pre_dataloader_step_init(
            model=model, step_context=step_context
        )
        head = model.prototype_prediction_head
        if self.gumbel_schedule is not None and not head.hard_assignment:
            head.gumbel_scale = self.gumbel_schedule(step_context.global_step)
            log.debug("Gumbel scale set to %s", head.gumbel_scale)

    def run_classification(
        self,
        batch_idx: int,
        model: ProtoPool,
        batch_data_dict: Dict[str, Any],
        epoch_metrics_dict: Dict[str, Any],
    ):
        if self.mixup_alpha:
            batch_data_dict = mixup_batch(
                batch_data_dict, self.mixup_alpha, self._mixup_generator
            )
        return super(ProtoPoolBackpropPhase, self).run_classification(
            batch_idx=batch_idx,
            model=model,
            batch_data_dict=batch_data_dict,
            epoch_metrics_dict=epoch_metrics_dict,
        )


@dataclass(frozen=True)
class TrainProtoPoolLayers(TrainLayersUsingProtoPNetNames):
    """
    Adds control over the slot assignment (`proto_presence`), which lives in the prediction head
    next to the class connection layer but is trained in different phases.
    """

    train_slot_assignment: bool = False

    def overwrite_grad_epoch_settings(self, name, param, setting_attr, should_train):
        if name.endswith("proto_presence"):
            param.requires_grad = self.train_slot_assignment


class ProtoPoolTrainingSchedule(TrainingSchedule):
    def __init__(
        self,
        *,
        model: ProtoPool,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        project_loader: torch.utils.data.DataLoader,
        loss: torch.nn.Module,
        last_only_loss: Optional[torch.nn.Module] = None,
        num_accumulation_batches: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        num_warm_epochs: int = 10,
        num_joint_epochs: int = 20,
        joint_lr_step_size: int = 5,
        joint_lr_step_gamma: float = 0.1,
        num_last_only_epochs: int = 25,
        last_only_lr_step_size: int = 5,
        last_only_lr_step_gamma: float = 0.95,
        last_only_weight_decay: float = 1e-3,
        gumbel_schedule: Optional[GumbelScaleSchedule] = GumbelScaleSchedule(),
        mixup_alpha: Optional[float] = 0.5,
        iterative_push_cycles: int = 0,
        iterative_joint_epochs: int = 5,
        iterative_joint_lr_multiplier: float = 0.1,
        phase_config_kwargs: Dict[str, Any] = {},
        extra_training_metrics: Optional[TrainingMetrics] = None,
    ):
        """
        Learning rates follow the original ProtoPool code, relative to `lr`:
            - add-on layers, prototypes and slot assignment: 3 * lr
            - backbone: lr / 10
            - last layer: lr / 10

        Args:
            - num_warm_epochs: epochs training the add-on layers, prototypes and slot assignment with a frozen backbone
            - num_joint_epochs: epochs also training the backbone, before the (first) projection
            - joint_lr_step_size, joint_lr_step_gamma: StepLR for the joint phase
            - num_last_only_epochs: epochs fine-tuning only the last layer after the final projection
            - last_only_lr_step_size, last_only_lr_step_gamma: StepLR for the last-only phase
            - gumbel_schedule: Gumbel scale annealing for the slot assignment. None or disabled uses a plain softmax.
            - mixup_alpha: mixup Beta(alpha, alpha) parameter for training batches. None or 0 disables mixup.
            - iterative_push_cycles: if > 0, repeat (last_only, joint, project) this many times after the first projection.
                The post-projection joint phase keeps the slot assignment hard and frozen, and has its own optimizer
                with a constant learning rate of `iterative_joint_lr_multiplier` times the joint learning rates,
                so decay does not compound across cycles.
            - iterative_joint_epochs: joint epochs in each iterative cycle
        """
        if last_only_loss is None:
            last_only_loss = loss

        def add_on_group(group_lr):
            return {
                "params": list(model.add_on_layers.parameters()),
                "lr": group_lr,
                "weight_decay": weight_decay,
            }

        def prototype_group(group_lr):
            return {"params": [model.prototype_layer.prototype_tensors], "lr": group_lr}

        def slot_group(group_lr):
            return {
                "params": [model.prototype_prediction_head.proto_presence],
                "lr": group_lr,
            }

        def backbone_group(group_lr):
            return {
                "params": list(model.backbone.parameters()),
                "lr": group_lr,
                "weight_decay": weight_decay,
            }

        eval = ClassificationInferencePhase(
            name="eval",
            dataloader=val_loader,
            loss=loss,
            extra_training_metrics=extra_training_metrics,
            **phase_config_kwargs,
        )

        backprop_shared_config = {
            "dataloader": train_loader,
            "num_accumulation_batches": num_accumulation_batches,
            "loss": loss,
            "extra_training_metrics": extra_training_metrics,
            "mixup_alpha": mixup_alpha,
            **phase_config_kwargs,
        }

        warm = ProtoPoolBackpropPhase(
            name="warm",
            optimizer=torch.optim.Adam(
                [add_on_group(3 * lr), slot_group(3 * lr), prototype_group(3 * lr)]
            ),
            set_training_layers_fn=TrainProtoPoolLayers(
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
                train_slot_assignment=True,
            ),
            gumbel_schedule=gumbel_schedule,
            **backprop_shared_config,
        )

        joint_optimizer = torch.optim.Adam(
            [
                backbone_group(lr / 10),
                add_on_group(3 * lr),
                slot_group(3 * lr),
                prototype_group(3 * lr),
            ]
        )
        joint = ProtoPoolBackpropPhase(
            name="joint",
            optimizer=joint_optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size,
                gamma=joint_lr_step_gamma,
            ),
            set_training_layers_fn=TrainProtoPoolLayers(
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
                train_slot_assignment=True,
            ),
            gumbel_schedule=gumbel_schedule,
            **backprop_shared_config,
        )

        project = ProjectPhase(dataloader=project_loader)

        last_only_optimizer = torch.optim.Adam(
            [
                {
                    "params": list(
                        model.prototype_prediction_head.class_connection_layer.parameters()
                    ),
                    "lr": lr / 10,
                    "weight_decay": last_only_weight_decay,
                }
            ]
        )
        last_only_shared_config = backprop_shared_config.copy()
        last_only_shared_config["loss"] = last_only_loss
        last_only = ProtoPoolBackpropPhase(
            name="last_only",
            optimizer=last_only_optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(
                last_only_optimizer,
                step_size=last_only_lr_step_size,
                gamma=last_only_lr_step_gamma,
            ),
            set_training_layers_fn=TrainProtoPoolLayers(
                train_backbone=False,
                train_add_on_layers=False,
                train_prototype_layer=False,
                train_prototype_prediction_head=True,
                train_slot_assignment=False,
            ),
            **last_only_shared_config,
        )

        maybe_iterative_phase = []
        if iterative_push_cycles > 0:
            post_project_lr = iterative_joint_lr_multiplier * lr
            joint_post_project = ProtoPoolBackpropPhase(
                name="joint",
                optimizer=torch.optim.Adam(
                    [
                        backbone_group(post_project_lr / 10),
                        add_on_group(3 * post_project_lr),
                        prototype_group(3 * post_project_lr),
                    ]
                ),
                set_training_layers_fn=TrainProtoPoolLayers(
                    train_backbone=True,
                    train_add_on_layers=True,
                    train_prototype_layer=True,
                    train_prototype_prediction_head=False,
                    train_slot_assignment=False,
                ),
                **backprop_shared_config,
            )
            maybe_iterative_phase = [
                IterativePhase(
                    phases=[
                        Phase(train=last_only, duration=num_last_only_epochs),
                        Phase(
                            train=joint_post_project, duration=iterative_joint_epochs
                        ),
                        Phase(train=project),
                    ],
                    iterations=iterative_push_cycles,
                )
            ]

        super(ProtoPoolTrainingSchedule, self).__init__(
            default_eval_phase=eval,
            phases=[
                Phase(train=warm, duration=num_warm_epochs),
                Phase(train=joint, duration=num_joint_epochs),
                Phase(train=project),
                *maybe_iterative_phase,
                Phase(train=last_only, duration=num_last_only_epochs),
            ],
        )
