import json
import logging

import click
import torch
from click.core import ParameterSource

from protopnet.train.metrics import InterpretableTrainingMetrics

from . import datasets
from .activations import L2Activation
from .backbones import construct_backbone, default_batch_sizes_for_backbone
from .cli import common_training_options
from .embedding import AddonLayers
from .model_losses import loss_for_coefficients
from .models.protopool import GumbelScaleSchedule, ProtoPool, ProtoPoolTrainingSchedule
from .train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from .train.scheduling import Phase, PrunePrototypesPhase
from .train.scheduling.early_stopping import ProjectPatienceEarlyStopping
from .train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


def _activation_function_set_by_user() -> bool:
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    return ctx.get_parameter_source("activation_function") not in (
        None,
        ParameterSource.DEFAULT,
    )


@click.command("train-protopool")
@common_training_options
@click.option(
    "--num-prototypes", type=int, default=202, help="Size of the shared prototype pool"
)
@click.option(
    "--num-descriptive",
    type=int,
    default=10,
    help="Number of prototype slots per class",
)
@click.option(
    "--warm-epochs", type=int, default=10, help="Epochs with a frozen backbone"
)
@click.option(
    "--joint-epochs", type=int, default=20, help="Joint epochs before the projection"
)
@click.option(
    "--gumbel-epochs",
    type=int,
    default=30,
    help="Epochs over which the Gumbel scale is annealed",
)
@click.option(
    "--gumbel/--no-gumbel",
    default=True,
    help="Use Gumbel-softmax for the slot assignment",
)
@click.option(
    "--last-only-epochs",
    type=int,
    default=25,
    help="Epochs fine-tuning only the last layer",
)
@click.option(
    "--iterative-push-cycles",
    type=int,
    default=0,
    help="Number of (last-only, joint, project) cycles after the first projection. 0 is the original schedule.",
)
@click.option(
    "--iterative-joint-epochs",
    type=int,
    default=5,
    help="Joint epochs in each iterative cycle",
)
@click.option(
    "--mixup/--no-mixup", default=True, help="Apply mixup to training batches"
)
@click.option(
    "--lr", type=float, default=1e-3, help="Base learning rate (the original's --lr)"
)
@click.option(
    "--joint-lr-gamma", type=float, default=0.1, help="StepLR decay for the joint phase"
)
@click.option(
    "--last-only-lr-gamma",
    type=float,
    default=0.95,
    help="StepLR decay for the last-only phase",
)
@click.option(
    "--last-only-weight-decay",
    type=float,
    default=1e-3,
    help="Weight decay of the last layer",
)
@click.option("--cluster-coef", type=float, default=0.8)
@click.option("--separation-coef", type=float, default=-0.08)
@click.option("--slot-orthogonality-coef", type=float, default=1.0)
@click.option("--l1-coef", type=float, default=1e-4)
def run(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    dataset="cub200",
    verify=False,
    dry_run=False,
    run_id=None,
    activation_function="l2",
    save_on_proto_updates=False,
    phase_multiplier=1,
    proto_channels=None,
    fa_func="serial",
    fa_coef=None,
    num_prototypes=202,
    num_descriptive=10,
    warm_epochs=10,
    joint_epochs=20,
    gumbel_epochs=30,
    gumbel=True,
    last_only_epochs=25,
    iterative_push_cycles=0,
    iterative_joint_epochs=5,
    mixup=True,
    lr=1e-3,
    joint_lr_gamma=0.1,
    last_only_lr_gamma=0.95,
    last_only_weight_decay=1e-3,
    cluster_coef=0.8,
    separation_coef=-0.08,
    slot_orthogonality_coef=1.0,
    l1_coef=1e-4,
):
    """
    Train a ProtoPool (https://arxiv.org/abs/2112.02902).

    Defaults follow the original implementation (https://github.com/gmum/ProtoPool):
    learning rates are relative to --lr (add-on, prototypes and slot assignment 3*lr, backbone and last layer lr/10),
    the joint phase decays by --joint-lr-gamma every 5 epochs, and the last layer is fine-tuned for --last-only-epochs.

    --phase-multiplier scales every epoch count: phase lengths, Gumbel annealing and the StepLR step sizes.
    """
    if fa_coef is not None:
        raise click.BadParameter(
            "Fine annotation is not supported for ProtoPool: it assumes class-specific prototypes.",
            param_hint="--fa-coef",
        )

    if activation_function != "l2" and _activation_function_set_by_user():
        log.warning(
            "ProtoPool uses the L2 (log) activation; ignoring --activation-function=%s",
            activation_function,
        )
    activation_function = "l2"

    if proto_channels is None:
        proto_channels = 256

    if verify:
        log.info("Running in verification mode to check training will run all phases")
        warm_epochs = 1
        joint_epochs = 1
        gumbel_epochs = 2
        last_only_epochs = 1
        iterative_push_cycles = 0
        phase_multiplier = 1
        num_descriptive = 2
        num_prototypes = None
        proto_channels = 32
        save_on_proto_updates = True

    log.info("Training config %s", json.dumps(locals()))
    log.log(
        logging.INFO if dry_run else logging.DEBUG,
        f'Running protopool training as a {"dry_run" if dry_run else "full run"}',
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)
    # the original trains with batches of 80
    batch_sizes = {**batch_sizes, "train": 80 // num_accumulation_batches}

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)
    num_classes = split_dataloaders.num_classes

    if num_prototypes is None:
        num_prototypes = 2 * num_classes

    backbone_module = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes,
        input_channels=backbone_module.latent_dimension[0],
        proto_channels=proto_channels,
        num_addon_layers=1,
    )

    model = ProtoPool(
        backbone=backbone_module,
        add_on_layers=add_on_layers,
        activation=L2Activation(),
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        num_descriptive=num_descriptive,
    )

    train_logger = WeightsAndBiasesTrainLogger(
        device=device,
        calculate_best_for=["accuracy"],
        wandb_plots=(["pr", "roc", "conf_mat"] if num_classes <= 10 else None),
    )

    loss = loss_for_coefficients(
        {
            "mixup_cross_entropy": 1,
            "protopool_cluster": cluster_coef,
            "protopool_separation": separation_coef,
            "slot_orthogonality": slot_orthogonality_coef,
            "l1": l1_coef,
        }
    ).to(device)

    last_only_loss = loss_for_coefficients(
        {"mixup_cross_entropy": 1, "l1": l1_coef}
    ).to(device)

    training_schedule = ProtoPoolTrainingSchedule(
        model=model,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=loss,
        last_only_loss=last_only_loss,
        num_accumulation_batches=num_accumulation_batches,
        lr=lr,
        num_warm_epochs=warm_epochs * phase_multiplier,
        num_joint_epochs=joint_epochs * phase_multiplier,
        joint_lr_step_size=5 * phase_multiplier,
        joint_lr_step_gamma=joint_lr_gamma,
        num_last_only_epochs=last_only_epochs * phase_multiplier,
        last_only_lr_step_size=5 * phase_multiplier,
        last_only_lr_step_gamma=last_only_lr_gamma,
        last_only_weight_decay=last_only_weight_decay,
        gumbel_schedule=GumbelScaleSchedule(
            gumbel_epochs=gumbel_epochs * phase_multiplier, enabled=gumbel
        ),
        mixup_alpha=0.5 if mixup else None,
        iterative_push_cycles=iterative_push_cycles,
        iterative_joint_epochs=iterative_joint_epochs * phase_multiplier,
        phase_config_kwargs={"device": device},
        extra_training_metrics=InterpretableTrainingMetrics(
            num_classes=num_classes,
            proto_per_class=num_descriptive,
            acc_only=True,
            part_num=None,
            img_size=None,
            half_size=None,
            protopnet=model,
            device=device,
        ),
    )

    if verify:
        training_schedule.phases.append(
            Phase(
                train=PrunePrototypesPhase(),
                duration=1,
                eval=training_schedule.default_eval_phase,
            )
        )

    checkpoint_phases = {"project"} if save_on_proto_updates else set()

    trainer = MultiPhaseProtoPNetTrainer(
        device=device,
        metric_logger=train_logger,
        checkpoint_phase_starts=checkpoint_phases,
        checkpoint_phase_ends=checkpoint_phases,
        num_classes=num_classes,
    )

    if dry_run:
        log.info("Skipping training due to dry run: %s", training_schedule)
    else:
        trainer.train(
            run_id=run_id,
            model=model,
            training_schedule=training_schedule,
            target_metric_name="accuracy",
            early_stopping=ProjectPatienceEarlyStopping(project_patience=2),
        )
