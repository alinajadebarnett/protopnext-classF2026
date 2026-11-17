import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from protopnet.train.metrics import TrainingMetrics

from ..embedding import EmbeddedBackbone
from ..prediction_heads import LinearClassPrototypePredictionHead
from ..prototype_layers import ClassAwarePrototypeLayer
from ..prototypical_part_model import ProtoPNet
from ..train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    TrainLayersByComponent,
)
from ..train.scheduling.types import IterativePhase, Phase, TrainingSchedule


class VanillaProtoPNet(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,
        num_classes: int,
        num_prototypes_per_class: int,
        k_for_topk: int = 1,
        **kwargs,
    ):
        num_prototypes = num_classes * num_prototypes_per_class

        # TODO: SHOULD BE CALLED FROM SAME INFO AS SELF.prototype_meta
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)

        for j in range(num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1

        latent_channels = add_on_layers.proto_channels

        prototype_layer = ClassAwarePrototypeLayer(
            activation_function=activation,
            latent_channels=latent_channels,
            class_specific_project=True,
            prototype_class_identity=prototype_class_identity,
            k_for_topk=k_for_topk,
        )

        prediction_head = LinearClassPrototypePredictionHead(
            prototype_class_identity=prototype_class_identity,
            k_for_topk=k_for_topk,
        )

        super(VanillaProtoPNet, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
            **kwargs,
        )


#: Components each backprop stage trains by default (the original ProtoPNet schedule).
DEFAULT_TRAINABLE_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "warm": ("add_on_layers", "prototype_layer"),
    "joint": ("add_on_layers", "backbone", "prototype_layer"),
    "last_only": ("prototype_prediction_head",),
}

#: Order of parameter groups in each stage's optimizer. Matches the original
#: hand-written optimizers so that default runs are unchanged.
_OPTIMIZER_GROUP_ORDER = (
    "add_on_layers",
    "backbone",
    "prototype_layer",
    "prototype_prediction_head",
)


class ProtoPNetTrainingSchedule(TrainingSchedule):
    def __init__(
        self,
        *,
        model: ProtoPNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        project_loader: torch.utils.data.DataLoader,
        loss: torch.nn.Module,
        last_only_loss: Optional[torch.nn.Module] = None,
        num_accumulation_batches: int = 1,
        add_on_lr: float = 0.003,
        num_warm_epochs: int = 10,
        num_preproject_joint_epochs: int = 10,
        backbone_lr: float = 1e-4,
        prototype_lr: float = 0.003,
        pred_head_lr: float = 1e-4,
        weight_decay: float = 1e-3,
        joint_lr_step_size: int = 5,
        joint_lr_step_gamma: float = 0.1,
        last_only_epochs_per_project: int = 20,
        joint_epochs_per_phase: int = 10,
        post_project_phases: int = 12,
        final_last_only_epochs: Optional[int] = None,
        trainable_components: Optional[Mapping[str, Iterable[str]]] = None,
        freeze_batchnorm: bool = False,
        stage_learning_rates: Optional[Mapping[str, Mapping[str, float]]] = None,
        stage_order: Optional[Sequence[Mapping[str, Any]]] = None,
        phase_config_kwargs: Dict[str, Any] = {},
        extra_training_metrics: Optional[TrainingMetrics] = None,
    ):
        """
        Args:
            - model: model class object
            - train_loader: dataloader of the training set
            - ...
            - loss: loss terms
            - last_only_loss: a loss term that applies only the the last layers (prediction head)
            - num_accumulation_batches: how many batches in gradient accumulation
            - add_on_lr: learning rate of the addon layers
            - num_warm_epochs: warm epochs in original protopnet freeze the backbone+predictionhead, train only the prototype layers
            - num_preproject_joint_epochs: how many joint (end-to-end) training epochs before the first project phase
            - ...
            - joint_lr_step_size: number of epochs for each learning rate step down
            - last_only_epochs_per_project: epochs after project where most features are frozen, only the prediction head trains
            - joint_epochs_per_phase: number of joint (end-to-end) training epochs before each projection step after warmups
            - post_project_phases: how many times to loop (last-only epochs -> joint (end-to-end) epochs -> project epochs) after the first project
            - final_last_only_epochs: last-layer epochs in the very last stage,
                after the loop. None = same as last_only_epochs_per_project (original behaviour).
              Any stage with 0 epochs is left out of the schedule (e.g. num_warm_epochs=0 skips warm-up).
            - trainable_components: which components each stage trains, keyed by
                stage name ("warm", "joint", "last_only"). Stages you leave out keep their
                defaults (DEFAULT_TRAINABLE_COMPONENTS). Example:
                {"warm": ["prototype_layer"], "last_only": ["prototype_prediction_head", "add_on_layers"]}
            - freeze_batchnorm: if True, BatchNorm layers in frozen components keep
                their running statistics fixed during a stage
            - stage_learning_rates: learning rate for a component in one stage,
                keyed by stage then component. Anything not given uses the component's
                normal rate (add_on_lr, backbone_lr, prototype_lr, pred_head_lr). Example:
                {"joint": {"add_on_layers": 0.001}, "last_only": {"prototype_prediction_head": 0.0005}}
                A rate can only be set for a component that trains in that stage.
            - stage_order: the order of stages as a list, replacing the standard
                ProtoPNet order. Each item is either ``{"stage": name, "epochs": n}``
                (name is "warm", "joint", "last_only" or "project"; project takes no
                epochs) or ``{"repeat": n, "stages": [...]}`` with a list of such stages.
                Stages with 0 epochs and repeats of 0 are skipped. When given, the epoch
                arguments above (num_warm_epochs, ..., final_last_only_epochs) are not used.
                Example:
                [{"stage": "warm", "epochs": 5}, {"stage": "project"},
                 {"repeat": 3, "stages": [{"stage": "last_only", "epochs": 10},
                                          {"stage": "joint", "epochs": 5},
                                          {"stage": "project"}]}]
        """
        if last_only_loss is None:
            last_only_loss = loss

        # Merge user overrides with the defaults and check for typos
        stage_trainable = resolve_trainable_components(trainable_components)

        # The normal learning rate of each component
        default_learning_rates = {
            "add_on_layers": add_on_lr,
            "backbone": backbone_lr,
            "prototype_layer": prototype_lr,
            "prototype_prediction_head": pred_head_lr,
        }
        # Final learning rate of every trainable component in every stage
        stage_lrs = resolve_stage_learning_rates(
            stage_learning_rates, stage_trainable, default_learning_rates
        )

        # Every epoch count must be a whole number >= 0
        validate_epoch_counts(
            num_warm_epochs=num_warm_epochs,
            num_preproject_joint_epochs=num_preproject_joint_epochs,
            last_only_epochs_per_project=last_only_epochs_per_project,
            joint_epochs_per_phase=joint_epochs_per_phase,
            post_project_phases=post_project_phases,
            final_last_only_epochs=final_last_only_epochs,
        )
        # Not given -> same as before (final stage = in-loop last-only length)
        if final_last_only_epochs is None:
            final_last_only_epochs = last_only_epochs_per_project

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
            **phase_config_kwargs,
        }

        # Optimizer parameter group (params + learning rate) for one component in one stage
        def param_group(component: str, stage: str) -> Dict[str, Any]:
            lr = stage_lrs[stage][component]
            if component == "add_on_layers":
                return {
                    "params": list(model.add_on_layers.parameters()),
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            if component == "backbone":
                return {
                    "params": list(model.backbone.parameters()),
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            if component == "prototype_layer":
                return {
                    "params": [model.prototype_layer.prototype_tensors],
                    "lr": lr,
                }
            if component == "prototype_prediction_head":
                return {
                    # FIXME - shouldn't this be model.prototype_prediction_head.parameters()?
                    "params": list(
                        model.prototype_prediction_head.class_connection_layer.parameters()
                    ),
                    "lr": lr,
                }
            raise ValueError(f"No optimizer settings for component {component!r}")

        # A stage's optimizer contains exactly the components it trains,
        # so a component can never be "unfrozen but not updated"
        def optimizer_for(stage: str) -> torch.optim.Optimizer:
            trainable = stage_trainable[stage]
            return torch.optim.Adam(
                [
                    param_group(c, stage)
                    for c in _OPTIMIZER_GROUP_ORDER
                    if c in trainable
                ]
            )

        # Freezing rule for a stage
        def layers_for(stage: str) -> TrainLayersByComponent:
            return TrainLayersByComponent(
                trainable=stage_trainable[stage], freeze_batchnorm=freeze_batchnorm
            )

        warm = ClassificationBackpropPhase(
            name="warm",
            optimizer=optimizer_for("warm"),
            set_training_layers_fn=layers_for("warm"),
            **backprop_shared_config,
        )

        # One joint optimizer (and LR scheduler) is shared by every joint stage, as before
        joint_optimizer = optimizer_for("joint")

        joint = ClassificationBackpropPhase(
            name="joint",
            optimizer=joint_optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size,
                gamma=joint_lr_step_gamma,
            ),
            set_training_layers_fn=layers_for("joint"),
            **backprop_shared_config,
        )

        last_only_shared_config = backprop_shared_config.copy()
        last_only_shared_config["loss"] = last_only_loss
        last_only = ClassificationBackpropPhase(
            name="last_only",
            optimizer=optimizer_for("last_only"),
            set_training_layers_fn=layers_for("last_only"),
            **last_only_shared_config,
        )

        project = ProjectPhase(
            dataloader=project_loader,
        )

        # Add a stage only if it has at least 1 epoch
        def stage(train, epochs: int = 1) -> List[Phase]:
            return [Phase(train=train, duration=epochs)] if epochs > 0 else []

        # The repeated loop: last-only -> joint -> project
        cycle = [
            *stage(last_only, last_only_epochs_per_project),
            *stage(joint, joint_epochs_per_phase),
            *stage(project),
        ]

        maybe_iterative_phase = (
            [IterativePhase(phases=cycle, iterations=post_project_phases)]
            if post_project_phases > 0
            else []
        )

        if stage_order is None:
            phases = [
                *stage(warm, num_warm_epochs),  # skipped if 0
                *stage(joint, num_preproject_joint_epochs),  # skipped if 0
                *stage(project),
                *maybe_iterative_phase,
                *stage(last_only, final_last_only_epochs),  # own epoch count
            ]
        else:
            # A user-defined order of stages (e.g. from a schedule config file)
            phases = build_phases_from_stage_order(
                stage_order,
                {
                    "warm": warm,
                    "joint": joint,
                    "last_only": last_only,
                    "project": project,
                },
            )

        super().__init__(default_eval_phase=eval, phases=phases)

        # Keep the resolved settings so they can be printed or checked
        self._stage_trainable = stage_trainable
        self._stage_lrs = stage_lrs

    @property
    def stage_trainable_components(self) -> Dict[str, Tuple[str, ...]]:
        """Components trained in each backprop stage."""
        return dict(self._stage_trainable)

    @property
    def stage_learning_rates(self) -> Dict[str, Dict[str, float]]:
        """Learning rate of each trainable component in each backprop stage."""
        return {stage: dict(lrs) for stage, lrs in self._stage_lrs.items()}


def resolve_trainable_components(
    overrides: Optional[Mapping[str, Iterable[str]]],
) -> Dict[str, Tuple[str, ...]]:
    """
    Merge per-stage trainable-component overrides onto DEFAULT_TRAINABLE_COMPONENTS.

    Each value may be a list of component names or a comma-separated string.
    Raises ValueError for unknown stages, unknown components or empty stages.
    """
    resolved = dict(DEFAULT_TRAINABLE_COMPONENTS)
    if not overrides:
        return resolved  # nothing given: original behaviour

    # stage names must be warm / joint / last_only
    unknown_stages = set(overrides) - set(DEFAULT_TRAINABLE_COMPONENTS)
    if unknown_stages:
        raise ValueError(
            f"Unknown stage(s) {sorted(unknown_stages)} in trainable_components. "
            f"Valid stages: {sorted(DEFAULT_TRAINABLE_COMPONENTS)}"
        )

    for stage_name, components in overrides.items():
        # allow "a,b,c" strings (handy for the command line)
        if isinstance(components, str):
            components = [c for c in components.split(",") if c.strip()]
        components = tuple(c.strip() for c in components)

        # component names must be real ProtoPNet components
        unknown = set(components) - set(_OPTIMIZER_GROUP_ORDER)
        if unknown:
            raise ValueError(
                f"Unknown component(s) {sorted(unknown)} for stage {stage_name!r}. "
                f"Valid components: {list(_OPTIMIZER_GROUP_ORDER)}"
            )
        # a stage that trains nothing is almost certainly a mistake
        if not components:
            raise ValueError(f"Stage {stage_name!r} has no trainable components.")

        resolved[stage_name] = tuple(
            c for c in _OPTIMIZER_GROUP_ORDER if c in components
        )
    return resolved


def validate_epoch_counts(**counts: Optional[int]) -> None:
    """Raise ValueError unless every count is None (use default) or a non-negative int."""
    for name, value in counts.items():
        if value is None:
            continue  # None means "use the default"
        # bool is a subclass of int in Python, so reject it explicitly
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def resolve_stage_learning_rates(
    overrides: Optional[Mapping[str, Any]],
    stage_trainable: Mapping[str, Tuple[str, ...]],
    default_learning_rates: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Learning rate of every trainable component in every stage.

    Starts from each component's default rate and applies per-stage overrides.
    Each override may be a ``{component: rate}`` mapping or a
    ``"component=rate,component=rate"`` string. Raises ValueError for unknown
    stages or components, for a rate given to a component that is frozen in
    that stage, and for rates that are not positive finite numbers.
    """
    # start from the normal rates: this is exactly the original behaviour
    resolved = {
        stage: {c: default_learning_rates[c] for c in components}
        for stage, components in stage_trainable.items()
    }
    if not overrides:
        return resolved

    # stage names must be warm / joint / last_only
    unknown_stages = set(overrides) - set(stage_trainable)
    if unknown_stages:
        raise ValueError(
            f"Unknown stage(s) {sorted(unknown_stages)} in stage_learning_rates. "
            f"Valid stages: {sorted(stage_trainable)}"
        )

    for stage_name, rates in overrides.items():
        # allow "component=rate,component=rate" strings (handy for the command line)
        if isinstance(rates, str):
            rates = parse_learning_rate_string(rates)

        for component, lr in rates.items():
            if component not in default_learning_rates:
                raise ValueError(
                    f"Unknown component {component!r} in learning rates for stage {stage_name!r}. "
                    f"Valid components: {list(default_learning_rates)}"
                )
            # a rate for a frozen component would silently do nothing, so stop instead
            if component not in stage_trainable[stage_name]:
                raise ValueError(
                    f"{component!r} is frozen in stage {stage_name!r}, so it cannot have a "
                    f"learning rate there. Trainable in {stage_name!r}: "
                    f"{list(stage_trainable[stage_name])}"
                )
            # rate must be a positive, finite number (bool is excluded on purpose)
            if (
                isinstance(lr, bool)
                or not isinstance(lr, (int, float))
                or not math.isfinite(lr)
                or lr <= 0
            ):
                raise ValueError(
                    f"Learning rate for {component!r} in stage {stage_name!r} must be a "
                    f"positive number, got {lr!r}"
                )
            resolved[stage_name][component] = float(lr)
    return resolved


def parse_learning_rate_string(text: str) -> Dict[str, float]:
    """Parse ``"add_on_layers=0.001,prototype_layer=0.003"`` into a dict of rates."""
    rates = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Expected 'component=rate', got {item!r} "
                "(example: add_on_layers=0.001,prototype_layer=0.003)"
            )
        component, value = (part.strip() for part in item.split("=", 1))
        try:
            rates[component] = float(value)
        except ValueError:
            raise ValueError(
                f"Learning rate for {component!r} is not a number: {value!r}"
            )
    return rates


#: Stage names that can appear in a ``stage_order``.
STAGE_NAMES = ("warm", "joint", "last_only", "project")


def build_phases_from_stage_order(
    stage_order: Sequence[Mapping[str, Any]],
    stage_phases: Mapping[str, Any],
) -> List[Union[Phase, IterativePhase]]:
    """
    Turn a list of stages and repeat blocks into schedule phases.

    Each item is ``{"stage": name, "epochs": n}`` or ``{"repeat": n, "stages": [...]}``.
    ``project`` takes no epochs (it is always one step). Stages with 0 epochs and
    repeat blocks with 0 repeats are skipped. Repeat blocks cannot be nested.
    Raises ValueError with the position of the problem for any invalid item.
    """
    if isinstance(stage_order, (str, bytes)) or not isinstance(stage_order, Sequence):
        raise ValueError(
            f"The stage order must be a list, got {type(stage_order).__name__}"
        )
    if not stage_order:
        raise ValueError("The stage order is empty; it needs at least one stage.")

    phases: List[Union[Phase, IterativePhase]] = []
    for position, item in enumerate(stage_order, start=1):
        where = f"item {position}"
        if isinstance(item, Mapping) and "repeat" in item:
            phases.extend(_repeat_block(item, stage_phases, where))
        else:
            phases.extend(_single_stage(item, stage_phases, where))

    if not phases:
        raise ValueError(
            "Every stage in the stage order has 0 epochs or 0 repeats, "
            "so there is nothing to train."
        )
    return phases


def _single_stage(
    item: Any, stage_phases: Mapping[str, Any], where: str
) -> List[Phase]:
    """One ``{"stage": name, "epochs": n}`` item as a list of 0 or 1 phases."""
    if not isinstance(item, Mapping) or "stage" not in item:
        raise ValueError(
            f"{where}: expected {{stage: <name>, epochs: <n>}} or "
            f"{{repeat: <n>, stages: [...]}}, got {item!r}"
        )
    unknown_keys = set(item) - {"stage", "epochs"}
    if unknown_keys:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown_keys)}; "
            "a stage only takes 'stage' and 'epochs'"
        )

    name = item["stage"]
    if name not in STAGE_NAMES:
        raise ValueError(
            f"{where}: unknown stage {name!r}. Valid stages: {list(STAGE_NAMES)}"
        )

    if name == "project":
        if "epochs" in item:
            raise ValueError(
                f"{where}: 'project' takes no epochs (it is always a single step)"
            )
        return [Phase(train=stage_phases[name])]

    if "epochs" not in item:
        raise ValueError(f"{where}: stage {name!r} needs an 'epochs' value")
    epochs = item["epochs"]
    validate_epoch_counts(**{f"{where} ({name}) epochs": epochs})
    return [Phase(train=stage_phases[name], duration=epochs)] if epochs > 0 else []


def _repeat_block(
    item: Mapping[str, Any], stage_phases: Mapping[str, Any], where: str
) -> List[IterativePhase]:
    """One ``{"repeat": n, "stages": [...]}`` item as a list of 0 or 1 phases."""
    unknown_keys = set(item) - {"repeat", "stages"}
    if unknown_keys:
        raise ValueError(
            f"{where}: unknown key(s) {sorted(unknown_keys)}; "
            "a repeat block only takes 'repeat' and 'stages'"
        )
    validate_epoch_counts(**{f"{where} repeat": item["repeat"]})

    inner = item.get("stages")
    if isinstance(inner, (str, bytes)) or not isinstance(inner, Sequence) or not inner:
        raise ValueError(f"{where}: a repeat block needs a non-empty 'stages' list")

    cycle: List[Phase] = []
    for inner_position, inner_item in enumerate(inner, start=1):
        inner_where = f"{where}, stage {inner_position}"
        if isinstance(inner_item, Mapping) and "repeat" in inner_item:
            raise ValueError(f"{inner_where}: repeat blocks cannot be nested")
        cycle.extend(_single_stage(inner_item, stage_phases, inner_where))

    if item["repeat"] == 0 or not cycle:
        return []
    return [IterativePhase(phases=cycle, iterations=item["repeat"])]
