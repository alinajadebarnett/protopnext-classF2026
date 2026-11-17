import numpy as np
import pytest
import torch

from protopnet.models.vanilla_protopnet import ProtoPNetTrainingSchedule
from protopnet.train.scheduling.lr_schedules import (
    CosineDecay,
    PhaseRestartingLR,
    StepDecay,
    joint_lr_scheduler,
)
from protopnet.train.scheduling.types import PrePhaseSummary


def make_optimizer(*lrs):
    return torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr} for lr in lrs]
    )


def lrs_over_phase(scheduler, phase_length):
    """Starts a phase and returns the first group's learning rate in each epoch."""
    scheduler.start_phase(phase_length)
    lrs = []
    for _ in range(phase_length):
        lrs.append(scheduler.optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def test_step_decay_factor():
    decay = StepDecay(step_size=2, gamma=0.1)
    factors = [decay(epoch, phase_length=5) for epoch in range(5)]
    assert factors == pytest.approx([1, 1, 0.1, 0.1, 0.01])


def test_cosine_decay_factor():
    decay = CosineDecay()
    factors = [decay(epoch, phase_length=4) for epoch in range(5)]
    assert factors == pytest.approx([1, 0.8535534, 0.5, 0.1464466, 0])


def test_step_reset_restarts_in_every_phase():
    scheduler = joint_lr_scheduler(
        "step-reset", make_optimizer(1.0), step_size=2, gamma=0.1
    )
    assert lrs_over_phase(scheduler, 3) == pytest.approx([1, 1, 0.1])
    assert lrs_over_phase(scheduler, 3) == pytest.approx([1, 1, 0.1])


def test_cosine_restarts_in_every_phase_and_adapts_to_its_length():
    scheduler = joint_lr_scheduler(
        "cosine", make_optimizer(1.0), step_size=5, gamma=0.1
    )
    assert lrs_over_phase(scheduler, 2) == pytest.approx([1, 0.5])
    assert lrs_over_phase(scheduler, 4) == pytest.approx([1, 0.8535534, 0.5, 0.1464466])


def test_each_parameter_group_keeps_its_own_base_lr():
    scheduler = PhaseRestartingLR(make_optimizer(1e-4, 3e-3), StepDecay(1, 0.5))
    scheduler.start_phase(2)
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([5e-5, 1.5e-3])
    scheduler.start_phase(2)
    assert scheduler.get_last_lr() == pytest.approx([1e-4, 3e-3])


def test_step_is_the_default_and_never_restarts():
    scheduler = joint_lr_scheduler("step", make_optimizer(1.0), step_size=2, gamma=0.1)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert (scheduler.step_size, scheduler.gamma) == (2, 0.1)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="step-reset"):
        joint_lr_scheduler("linear", make_optimizer(1.0), step_size=5, gamma=0.1)


def test_step_before_start_phase_is_an_error():
    scheduler = PhaseRestartingLR(make_optimizer(1.0), CosineDecay())
    with pytest.raises(RuntimeError, match="start_phase"):
        scheduler.step()


def joint_lrs_per_epoch(schedule):
    """
    Drives the joint phases of a schedule the way the trainer does (before_training,
    then one scheduler step per epoch) and returns the backbone learning rate used in
    each joint epoch, grouped by joint phase.
    """
    lrs_by_phase = []
    first_step = 0
    for phase in schedule:
        if phase.name == "joint":
            phase.train.before_training(
                model=None,
                metric_logger=None,
                checkpointer=None,
                pre_phase_summary=PrePhaseSummary(
                    name=phase.name,
                    first_step=first_step,
                    initial_target_metric=("accuracy", None),
                    expected_last_step=first_step + phase.duration - 1,
                ),
            )
            lrs = []
            for _ in range(phase.duration):
                # parameter group 1 holds the backbone
                lrs.append(phase.train.optimizer.param_groups[1]["lr"])
                phase.train.scheduler.step()
            lrs_by_phase.append(lrs)
        first_step += phase.duration
    return lrs_by_phase


@pytest.fixture
def make_schedule(mock_protopnet, mock_dataloaders, mock_loss):
    def make(**schedule_kwargs):
        return ProtoPNetTrainingSchedule(
            model=mock_protopnet,
            train_loader=mock_dataloaders["train"],
            val_loader=mock_dataloaders["val"],
            project_loader=mock_dataloaders["project"],
            loss=mock_loss,
            backbone_lr=1e-4,
            num_preproject_joint_epochs=2,
            joint_epochs_per_phase=2,
            joint_lr_step_size=2,
            post_project_phases=2,
            phase_config_kwargs={"device": torch.device("cpu")},
            **schedule_kwargs,
        )

    return make


def test_default_joint_schedule_decays_across_joint_phases(make_schedule):
    np.testing.assert_allclose(
        joint_lrs_per_epoch(make_schedule()),
        [[1e-4, 1e-4], [1e-5, 1e-5], [1e-6, 1e-6]],
    )


def test_step_reset_joint_schedule_starts_every_joint_phase_at_the_base_lr(
    make_schedule,
):
    schedule = make_schedule(joint_lr_schedule="step-reset")
    np.testing.assert_allclose(joint_lrs_per_epoch(schedule), [[1e-4, 1e-4]] * 3)


def test_cosine_joint_schedule_anneals_within_every_joint_phase(make_schedule):
    schedule = make_schedule(joint_lr_schedule="cosine")
    np.testing.assert_allclose(joint_lrs_per_epoch(schedule), [[1e-4, 5e-5]] * 3)
