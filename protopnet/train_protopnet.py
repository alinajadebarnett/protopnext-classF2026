import json
import logging

import click
import torch

from protopnet.train.metrics import InterpretableTrainingMetrics

from . import datasets
from .activations import lookup_activation
from .backbones import construct_backbone, default_batch_sizes_for_backbone
from .cli import common_training_options
from .embedding import AddonLayers
from .losses import ClassAwareExtraCalculations
from .model_losses import loss_for_coefficients
from .models.vanilla_protopnet import ProtoPNetTrainingSchedule, VanillaProtoPNet
from .pixel_space import BilinearUpsampleActivations
from .schedule_config import load_schedule_config, scale_stage_order
from .train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from .train.scheduling import Phase, PrunePrototypesPhase, RSampleInitPhase
from .train.scheduling.early_stopping import ProjectPatienceEarlyStopping
from .train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


@click.command("train-protopnet")
@common_training_options
@click.option(
    "--num-prototypes-per-class",
    type=int,
    default=10,
)
# Number of epochs for each stage. Leave a flag out to keep the
# original value. All counts are multiplied by --phase-multiplier.
@click.option(
    "--warm-epochs",
    type=click.IntRange(min=0),
    default=None,
    help="Warm-up epochs. 0 skips warm-up. Default: 5.",
)
@click.option(
    "--preproject-joint-epochs",
    type=click.IntRange(min=0),
    default=None,
    help="Joint epochs before the first project step. 0 skips them. Default: 5.",
)
@click.option(
    "--joint-epochs-per-cycle",
    type=click.IntRange(min=0),
    default=None,
    help="Joint epochs inside each repeated cycle. Default: 5.",
)
@click.option(
    "--last-only-epochs-per-cycle",
    type=click.IntRange(min=0),
    default=None,
    help="Last-layer epochs inside each repeated cycle. Default: 10.",
)
@click.option(
    "--num-cycles",
    type=click.IntRange(min=0),
    default=None,
    help="How many times to repeat (last-only -> joint -> project). Default: 20.",
)
@click.option(
    "--final-last-only-epochs",
    type=click.IntRange(min=0),
    default=None,
    help="Last-layer epochs in the final stage. Default: same as --last-only-epochs-per-cycle.",
)
# Choose which components each stage trains (comma-separated names).
# Leave a flag out to keep that stage's original setting.
@click.option(
    "--warm-trainable",
    type=str,
    default=None,
    help="Components trained during warm-up, e.g. 'add_on_layers,prototype_layer'.",
)
@click.option(
    "--joint-trainable",
    type=str,
    default=None,
    help="Components trained during joint stages, e.g. 'backbone,add_on_layers,prototype_layer'.",
)
@click.option(
    "--last-only-trainable",
    type=str,
    default=None,
    help="Components trained during last-layer stages, e.g. 'prototype_prediction_head'.",
)
# Learning rate of a component in one stage, as "component=rate" pairs.
# Leave a flag out to keep the normal rates (these values are NOT scaled by --lr-multiplier).
@click.option(
    "--warm-lr",
    type=str,
    default=None,
    help="Learning rates during warm-up, e.g. 'add_on_layers=0.003,prototype_layer=0.003'.",
)
@click.option(
    "--joint-lr",
    type=str,
    default=None,
    help="Learning rates during joint stages, e.g. 'backbone=0.0001,add_on_layers=0.001'.",
)
@click.option(
    "--last-only-lr",
    type=str,
    default=None,
    help="Learning rates during last-layer stages, e.g. 'prototype_prediction_head=0.0001'.",
)
# A YAML file that defines the whole schedule (instead of the per-stage options)
@click.option(
    "--schedule-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="YAML file defining the training schedule: stage order, epochs, trainable "
    "components and learning rates. Cannot be combined with the per-stage options.",
)
# Keep BatchNorm statistics fixed in frozen components
@click.option(
    "--freeze-batchnorm",
    is_flag=True,
    default=False,
    help="Keep BatchNorm running statistics fixed in frozen components.",
)
def run(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    pre_project_phase_len=5,
    phase_multiplier=1,
    latent_dim_multiplier_exp=None,
    joint_lr_step_size=5,
    post_project_phases=20,
    joint_epochs_per_phase=5,
    last_only_epochs_per_phase=10,
    cluster_coef=-0.8,
    separation_coef=0.08,
    l1_coef=0.0001,
    num_addon_layers=2,
    fa_func="serial",
    fa_coef=None,
    num_prototypes_per_class=10,
    lr_multiplier=1,
    lr_step_gamma=0.1,
    class_specific=True,
    dry_run=False,
    verify=False,
    dataset="CUB200",
    proto_channels=None,
    save_on_proto_updates=False,
    rsample_init=False,
    run_id=None,
    activation_function="l2",
    k_for_topk=1,
    warm_trainable=None,
    joint_trainable=None,
    last_only_trainable=None,
    freeze_batchnorm=False,
    warm_epochs=None,
    preproject_joint_epochs=None,
    joint_epochs_per_cycle=None,
    last_only_epochs_per_cycle=None,
    num_cycles=None,
    final_last_only_epochs=None,
    warm_lr=None,
    joint_lr=None,
    last_only_lr=None,
    schedule_config=None,
):
    """
    Train a Vanilla ProtoPNet.

    Args:
    - backbone: str - See backbones.py
    - pre_project_phase_len: int - number of epochs in each pre-project phase (warm-up, joint). Total preproject epochs is 2*pre_project_phase_len*phase_multiplier.
    - phase_multiplier: int - for each phase, multiply the number of epochs in that phase by this number
    - latent_dim_multiplier_exp: int - expotential of 2 for the latent dimension of the prototype layer
    - joint_lr_step_size: int - number of epochs between each step in the joint learning rate scheduler. Multiplied by phase_multiplier.
    - last_only_epochs_per_phase: int - coefficient for clustering loss
    - post_project_phases: int - number of times to iterate between last-only, joint, project after the initial pre-project phases
    - cluster_coef: float - coefficient for clustering loss term
    - separation_coef: float - coefficient for separation loss term
    - l1_coef: float - coefficient for clustering loss
    - fa_func: str - one of "serial", "l2", or "square" to indicate which type of fine annotation loss to use. if None, fine annotation is deactivated.
    - fa_coef: float - coefficient for fine annotation loss term
    - num_prototypes_per_class: int - number of prototypes to create for each class
    - lr_multiplier: float - multiplier for learning rates. The base values are from protopnet's training.
    - class_specific: boolean - whether to bind prototypes to individual classes or allow them to be distributed unevenly based on training
    - dry_run: bool - Configure the training run, but do not execute it
    - preflight: bool - Configure a training run for a single epoch of all phases
    - verify: bool - Configure a training run for a single epoch of all phases. Causes things to fail quicker if they will fail.
    - interpretable_metrics: bool - Whether to calculate interpretable metrics
    - dataset: str - name of the dataset to use
    - save_on_proto_updates: bool - whether or not to save intermediate models to run pacmap on
    - rsample_init: bool - whether to initialize prototypes as random parts of the training data images (True), instead of as random vectors of the correct length (False)
    - warm_trainable / joint_trainable / last_only_trainable: str - comma-separated components each stage trains
      (backbone, add_on_layers, prototype_layer, prototype_prediction_head). None keeps the original setting.
    - freeze_batchnorm: bool - keep BatchNorm running statistics fixed in frozen components
    - warm_epochs / preproject_joint_epochs: int - epochs for warm-up / joint-before-first-project.
      None = pre_project_phase_len. 0 skips the stage.
    - joint_epochs_per_cycle / last_only_epochs_per_cycle / num_cycles: int - override
      joint_epochs_per_phase / last_only_epochs_per_phase / post_project_phases.
    - final_last_only_epochs: int - last-layer epochs in the final stage. None = last-only-per-cycle.
      All epoch counts are multiplied by phase_multiplier.
    - warm_lr / joint_lr / last_only_lr: str - per-stage learning rates as
      "component=rate,component=rate". Components not listed keep their normal rate.
      Only components that train in that stage can be given a rate. Not scaled by lr_multiplier.
    - schedule_config: str - path to a YAML schedule config file (see README, "Schedule config files").
      Replaces the per-stage options above, which cannot be used together with it. Epochs in the file
      are multiplied by phase_multiplier; in verify mode every stage runs 1 epoch and each repeat runs once.
    """

    if schedule_config is not None:
        # A config file replaces the per-stage options; mixing them would be ambiguous
        per_stage_options = {
            "--warm-epochs": warm_epochs,
            "--preproject-joint-epochs": preproject_joint_epochs,
            "--joint-epochs-per-cycle": joint_epochs_per_cycle,
            "--last-only-epochs-per-cycle": last_only_epochs_per_cycle,
            "--num-cycles": num_cycles,
            "--final-last-only-epochs": final_last_only_epochs,
            "--warm-trainable": warm_trainable,
            "--joint-trainable": joint_trainable,
            "--last-only-trainable": last_only_trainable,
            "--freeze-batchnorm": freeze_batchnorm or None,
            "--warm-lr": warm_lr,
            "--joint-lr": joint_lr,
            "--last-only-lr": last_only_lr,
        }
        used = [name for name, value in per_stage_options.items() if value is not None]
        if used:
            raise click.UsageError(
                f"--schedule-config cannot be combined with {', '.join(used)}; "
                "put these settings in the config file instead."
            )
        config_kwargs = load_schedule_config(schedule_config)
        log.info("Using schedule config %s: %s", schedule_config, config_kwargs)
    else:
        config_kwargs = None

    # Per-stage flags override the shared defaults when given
    if joint_epochs_per_cycle is not None:
        joint_epochs_per_phase = joint_epochs_per_cycle
    if last_only_epochs_per_cycle is not None:
        last_only_epochs_per_phase = last_only_epochs_per_cycle
    if num_cycles is not None:
        post_project_phases = num_cycles

    if verify:
        log.info("Running in verification mode to check training will run all phases")
        pre_project_phase_len = 1
        post_project_phases = 0
        joint_epochs_per_phase = 1
        last_only_epochs_per_phase = 1
        phase_multiplier = 1
        # verify mode ignores per-stage epoch flags (always 1 epoch per stage)
        warm_epochs = preproject_joint_epochs = final_last_only_epochs = None
        latent_dim_multiplier_exp = -2
        proto_channels = None
        num_prototypes_per_class = 5
        num_addon_layers = 1
        save_on_proto_updates = True
        rsample_init = False
        activation_function = "cosine"

    log.info("Training config %s", json.dumps(locals()))
    log.log(
        logging.INFO if dry_run else logging.DEBUG,
        f'Running vanilla training as a {"dry_run" if dry_run else "full run"}',
    )

    setup = {
        "num_prototypes_per_class": num_prototypes_per_class,
        "coefs": {
            "cluster": cluster_coef,
            "separation": separation_coef,
            "l1": l1_coef,
            "cross_entropy": 1,
            "fa": fa_coef,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    post_forward_calculations = [ClassAwareExtraCalculations()]

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    if fa_coef is not None:
        post_forward_calculations.append(
            BilinearUpsampleActivations(split_dataloaders.image_size)
        )

    activation = lookup_activation(activation_function)

    backbone = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=setup["num_prototypes_per_class"]
        * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=(
            2**latent_dim_multiplier_exp
            if latent_dim_multiplier_exp is not None
            else None
        ),
        proto_channels=proto_channels,
        num_addon_layers=num_addon_layers,
    )

    vppn = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=split_dataloaders.num_classes,
        num_prototypes_per_class=setup["num_prototypes_per_class"],
        k_for_topk=k_for_topk,
    )

    train_logger = WeightsAndBiasesTrainLogger(
        device=setup["device"],
        calculate_best_for=["accuracy"],
        wandb_plots=(
            ["pr", "roc", "conf_mat"] if split_dataloaders.num_classes <= 10 else None
        ),
    )

    protopnet_loss = loss_for_coefficients(
        setup["coefs"],
        fa_func=fa_func,
        class_specific_cluster=class_specific,
    ).to(setup["device"])

    if config_kwargs is None:
        # Pass only the stages the user actually set; the rest keep their defaults
        trainable_components = {
            stage: components
            for stage, components in (
                ("warm", warm_trainable),
                ("joint", joint_trainable),
                ("last_only", last_only_trainable),
            )
            if components is not None
        }
        stage_learning_rates = {
            stage: rates
            for stage, rates in (
                ("warm", warm_lr),
                ("joint", joint_lr),
                ("last_only", last_only_lr),
            )
            if rates is not None
        }
        stage_order = None
    else:
        trainable_components = config_kwargs.get("trainable_components")
        stage_learning_rates = config_kwargs.get("stage_learning_rates")
        freeze_batchnorm = config_kwargs.get("freeze_batchnorm", False)
        stage_order = config_kwargs.get("stage_order")
        if stage_order is not None:
            stage_order = scale_stage_order(
                stage_order, multiplier=phase_multiplier, verify=verify
            )

    # TODO: add L1 term only for last_only
    training_schedule = ProtoPNetTrainingSchedule(
        model=vppn,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=protopnet_loss,
        last_only_loss=protopnet_loss,
        num_accumulation_batches=num_accumulation_batches,
        # Each can be set on its own; falls back to pre_project_phase_len
        num_warm_epochs=(pre_project_phase_len if warm_epochs is None else warm_epochs)
        * phase_multiplier,
        num_preproject_joint_epochs=(
            pre_project_phase_len
            if preproject_joint_epochs is None
            else preproject_joint_epochs
        )
        * phase_multiplier,
        backbone_lr=1e-4 * lr_multiplier,
        add_on_lr=0.003 * lr_multiplier,
        prototype_lr=0.003 * lr_multiplier,
        pred_head_lr=1e-4 * lr_multiplier,
        weight_decay=1e-3,
        joint_lr_step_size=joint_lr_step_size * phase_multiplier,
        joint_lr_step_gamma=lr_step_gamma,
        last_only_epochs_per_project=last_only_epochs_per_phase * phase_multiplier,
        joint_epochs_per_phase=joint_epochs_per_phase * phase_multiplier,
        post_project_phases=post_project_phases,
        # Own length for the final last-only stage
        final_last_only_epochs=(
            None
            if final_last_only_epochs is None
            else final_last_only_epochs * phase_multiplier
        ),
        trainable_components=trainable_components,
        freeze_batchnorm=freeze_batchnorm,
        stage_learning_rates=stage_learning_rates,
        stage_order=stage_order,
        phase_config_kwargs={
            "device": setup["device"],
            "post_forward_calculations": post_forward_calculations,
        },
        extra_training_metrics=InterpretableTrainingMetrics(
            num_classes=split_dataloaders.num_classes,
            proto_per_class=vppn.prototype_layer.num_prototypes_per_class,
            acc_only=True,
            part_num=None,
            img_size=None,
            half_size=None,
            protopnet=vppn,
            device=setup["device"],
        ),
    )

    # Print what trains in every stage, and at which learning rate
    log.info(
        "Trainable components per stage: %s",
        training_schedule.stage_trainable_components,
    )
    log.info("Learning rates per stage: %s", training_schedule.stage_learning_rates)

    if verify:
        # add a prune epoch
        training_schedule.phases.append(
            Phase(
                train=PrunePrototypesPhase(),
                duration=1,
                eval=training_schedule.default_eval_phase,
            )
        )

    if rsample_init:
        training_schedule.phases.insert(
            0,
            Phase(
                train=RSampleInitPhase(dataloader=split_dataloaders.project_loader),
                duration=1,
                eval=training_schedule.default_eval_phase,
            ),
        )

    checkpoint_phases = {"project"} if save_on_proto_updates else set()

    ppn_trainer = MultiPhaseProtoPNetTrainer(
        device=setup["device"],
        metric_logger=train_logger,
        checkpoint_phase_starts=checkpoint_phases,
        checkpoint_phase_ends=checkpoint_phases,
        num_classes=split_dataloaders.num_classes,
    )

    if dry_run:
        log.info("Skipping training due to dry run: %s", training_schedule)
    else:
        ppn_trainer.train(
            run_id=run_id,
            model=vppn,
            training_schedule=training_schedule,
            target_metric_name="accuracy",
            early_stopping=ProjectPatienceEarlyStopping(project_patience=2),
        )
