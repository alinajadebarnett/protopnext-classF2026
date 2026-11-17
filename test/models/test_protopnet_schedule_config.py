"""
Tests for configurable freezing (change 1), per-stage epoch counts (change 3)
and per-stage learning rates (change 4) in ProtoPNetTrainingSchedule.
"""

import pytest
import torch

from protopnet.models.vanilla_protopnet import (
    DEFAULT_TRAINABLE_COMPONENTS,
    ProtoPNetTrainingSchedule,
    parse_learning_rate_string,
)
from protopnet.train.scheduling import TrainLayersByComponent


@pytest.fixture
def make_schedule(mock_protopnet, mock_dataloaders, mock_loss):
    def _make(**kwargs):
        return ProtoPNetTrainingSchedule(
            model=mock_protopnet,
            train_loader=mock_dataloaders["train"],
            val_loader=mock_dataloaders["val"],
            project_loader=mock_dataloaders["project"],
            loss=mock_loss,
            phase_config_kwargs={"device": torch.device("cpu")},
            **kwargs,
        )

    return _make


def first_phase(schedule, name):
    return next(p for p in schedule if p.name == name)


def optimizer_param_ids(optimizer):
    return [sorted(id(p) for p in group["params"]) for group in optimizer.param_groups]


def component_param_ids(model, component):
    params = {
        "add_on_layers": model.add_on_layers.parameters(),
        "backbone": model.backbone.parameters(),
        "prototype_layer": [model.prototype_layer.prototype_tensors],
        "prototype_prediction_head": model.prototype_prediction_head.class_connection_layer.parameters(),
    }[component]
    return sorted(id(p) for p in params)


# --------------------------------------------------------------------------
# Backwards compatibility: defaults reproduce the original schedule
# --------------------------------------------------------------------------


def test_default_phase_sequence_and_durations_unchanged(make_schedule):
    schedule = make_schedule(
        num_warm_epochs=3,
        num_preproject_joint_epochs=4,
        joint_epochs_per_phase=5,
        last_only_epochs_per_project=6,
        post_project_phases=2,
    )
    assert [(p.name, p.duration) for p in schedule] == [
        ("warm", 3),
        ("joint", 4),
        ("project", 1),
        ("last_only", 6),
        ("joint", 5),
        ("project", 1),
        ("last_only", 6),
        ("joint", 5),
        ("project", 1),
        ("last_only", 6),  # final stage defaults to last_only_epochs_per_project
    ]


def test_default_optimizers_match_original_param_groups(make_schedule, mock_protopnet):
    """Same components, same order, same learning rates as the hand-written original."""
    schedule = make_schedule(
        add_on_lr=0.3, backbone_lr=0.1, prototype_lr=0.4, pred_head_lr=0.2
    )
    expected = {
        "warm": [("add_on_layers", 0.3), ("prototype_layer", 0.4)],
        "joint": [("add_on_layers", 0.3), ("backbone", 0.1), ("prototype_layer", 0.4)],
        "last_only": [("prototype_prediction_head", 0.2)],
    }
    for stage, groups in expected.items():
        optimizer = first_phase(schedule, stage).train.optimizer
        assert optimizer_param_ids(optimizer) == [
            component_param_ids(mock_protopnet, c) for c, _ in groups
        ]
        assert [g["lr"] for g in optimizer.param_groups] == [lr for _, lr in groups]


def test_default_trainable_components(make_schedule):
    schedule = make_schedule()
    assert schedule.stage_trainable_components == DEFAULT_TRAINABLE_COMPONENTS
    for stage, components in DEFAULT_TRAINABLE_COMPONENTS.items():
        layers = first_phase(schedule, stage).train.set_training_layers_fn
        assert isinstance(layers, TrainLayersByComponent)
        assert set(layers.trainable) == set(components)
        assert layers.freeze_batchnorm is False


def test_joint_optimizer_state_is_shared_across_repeated_joint_stages(make_schedule):
    """The original reuses one joint optimizer (and LR scheduler) for every joint stage."""
    schedule = make_schedule(post_project_phases=3)
    joint_phases = [p for p in schedule if p.name == "joint"]
    assert len(joint_phases) == 4
    assert len({id(p.train.optimizer) for p in joint_phases}) == 1


# --------------------------------------------------------------------------
# Change 1: choose which components each stage trains
# --------------------------------------------------------------------------


def test_override_trainable_components_changes_freezing_and_optimizer(
    make_schedule, mock_protopnet
):
    schedule = make_schedule(
        trainable_components={
            "warm": ["prototype_layer"],
            "last_only": ["prototype_prediction_head", "add_on_layers"],
        }
    )

    warm = first_phase(schedule, "warm").train
    assert warm.set_training_layers_fn.trainable == ("prototype_layer",)
    assert optimizer_param_ids(warm.optimizer) == [
        component_param_ids(mock_protopnet, "prototype_layer")
    ]

    last_only = first_phase(schedule, "last_only").train
    assert set(last_only.set_training_layers_fn.trainable) == {
        "prototype_prediction_head",
        "add_on_layers",
    }
    assert optimizer_param_ids(last_only.optimizer) == [
        component_param_ids(mock_protopnet, "add_on_layers"),
        component_param_ids(mock_protopnet, "prototype_prediction_head"),
    ]

    # joint was not overridden, so it keeps its default
    joint = first_phase(schedule, "joint").train
    assert set(joint.set_training_layers_fn.trainable) == set(
        DEFAULT_TRAINABLE_COMPONENTS["joint"]
    )


def test_trainable_components_accepts_comma_separated_string(make_schedule):
    schedule = make_schedule(
        trainable_components={"warm": "add_on_layers, prototype_layer, backbone"}
    )
    assert set(schedule.stage_trainable_components["warm"]) == {
        "add_on_layers",
        "prototype_layer",
        "backbone",
    }


def test_freeze_batchnorm_is_passed_to_every_stage(make_schedule):
    schedule = make_schedule(freeze_batchnorm=True)
    for stage in ("warm", "joint", "last_only"):
        assert first_phase(
            schedule, stage
        ).train.set_training_layers_fn.freeze_batchnorm


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"warmup": ["backbone"]}, "Unknown stage"),
        ({"warm": ["backbon"]}, "Unknown component"),
        ({"warm": []}, "no trainable components"),
    ],
)
def test_invalid_trainable_components_raise(make_schedule, overrides, match):
    with pytest.raises(ValueError, match=match):
        make_schedule(trainable_components=overrides)


# --------------------------------------------------------------------------
# Change 3: independent epoch counts per stage
# --------------------------------------------------------------------------


def test_final_last_only_epochs_is_independent(make_schedule):
    schedule = make_schedule(
        last_only_epochs_per_project=6, final_last_only_epochs=15, post_project_phases=1
    )
    durations = [(p.name, p.duration) for p in schedule]
    assert durations[-1] == ("last_only", 15)
    assert ("last_only", 6) in durations[:-1]  # in-cycle stage unchanged


def test_zero_epoch_stages_are_skipped(make_schedule):
    schedule = make_schedule(
        num_warm_epochs=0,
        num_preproject_joint_epochs=2,
        last_only_epochs_per_project=0,
        joint_epochs_per_phase=3,
        post_project_phases=2,
        final_last_only_epochs=4,
    )
    assert [(p.name, p.duration) for p in schedule] == [
        ("joint", 2),
        ("project", 1),
        ("joint", 3),
        ("project", 1),
        ("joint", 3),
        ("project", 1),
        ("last_only", 4),
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_warm_epochs": -1},
        {"joint_epochs_per_phase": 2.5},
        {"final_last_only_epochs": -3},
        {"post_project_phases": True},
    ],
)
def test_invalid_epoch_counts_raise(make_schedule, kwargs):
    with pytest.raises(ValueError, match="non-negative integer"):
        make_schedule(**kwargs)


def test_schedule_repr_shows_per_stage_durations(make_schedule):
    text = repr(
        make_schedule(
            num_warm_epochs=2,
            num_preproject_joint_epochs=3,
            post_project_phases=1,
            final_last_only_epochs=9,
        )
    )
    assert "Phase(warm" in text and "duration=2" in text
    assert "duration=9" in text


# --------------------------------------------------------------------------
# Change 4: learning rate per component per stage
# --------------------------------------------------------------------------

DEFAULT_LRS = dict(add_on_lr=0.3, backbone_lr=0.1, prototype_lr=0.4, pred_head_lr=0.2)


def lrs_by_component(schedule, model, stage):
    """{component: lr} for the optimizer of one stage."""
    optimizer = first_phase(schedule, stage).train.optimizer
    by_ids = {
        tuple(component_param_ids(model, c)): c
        for c in (
            "add_on_layers",
            "backbone",
            "prototype_layer",
            "prototype_prediction_head",
        )
    }
    return {
        by_ids[tuple(sorted(id(p) for p in g["params"]))]: g["lr"]
        for g in optimizer.param_groups
    }


def test_default_learning_rates_are_shared_across_stages(make_schedule):
    """No overrides: every stage uses the component's normal rate (original behaviour)."""
    schedule = make_schedule(**DEFAULT_LRS)
    assert schedule.stage_learning_rates == {
        "warm": {"add_on_layers": 0.3, "prototype_layer": 0.4},
        "joint": {"add_on_layers": 0.3, "backbone": 0.1, "prototype_layer": 0.4},
        "last_only": {"prototype_prediction_head": 0.2},
    }


def test_stage_learning_rate_override_changes_only_that_stage(
    make_schedule, mock_protopnet
):
    schedule = make_schedule(
        **DEFAULT_LRS,
        stage_learning_rates={"joint": {"add_on_layers": 0.05}},
    )
    # joint uses the new rate for add_on_layers, keeps the others
    assert lrs_by_component(schedule, mock_protopnet, "joint") == {
        "add_on_layers": 0.05,
        "backbone": 0.1,
        "prototype_layer": 0.4,
    }
    # warm still uses the normal add_on rate
    assert lrs_by_component(schedule, mock_protopnet, "warm") == {
        "add_on_layers": 0.3,
        "prototype_layer": 0.4,
    }


def test_override_applies_to_every_repeated_joint_stage(make_schedule):
    """All joint stages share one optimizer, so the override reaches every one of them."""
    schedule = make_schedule(
        post_project_phases=2, stage_learning_rates={"joint": {"backbone": 0.007}}
    )
    joint_phases = [p for p in schedule if p.name == "joint"]
    assert len(joint_phases) == 3
    for phase in joint_phases:
        lrs = [g["lr"] for g in phase.train.optimizer.param_groups]
        assert 0.007 in lrs


def test_learning_rate_string_format(make_schedule, mock_protopnet):
    schedule = make_schedule(
        **DEFAULT_LRS,
        stage_learning_rates={
            "warm": "add_on_layers=0.01, prototype_layer=0.02",
            "last_only": "prototype_prediction_head=5e-4",
        },
    )
    assert lrs_by_component(schedule, mock_protopnet, "warm") == {
        "add_on_layers": 0.01,
        "prototype_layer": 0.02,
    }
    assert lrs_by_component(schedule, mock_protopnet, "last_only") == {
        "prototype_prediction_head": 5e-4
    }


def test_rate_for_component_made_trainable_by_task_1(make_schedule, mock_protopnet):
    """Task 1 and task 4 together: unfreeze add_on_layers in last_only and give it its own rate."""
    schedule = make_schedule(
        **DEFAULT_LRS,
        trainable_components={
            "last_only": ["prototype_prediction_head", "add_on_layers"]
        },
        stage_learning_rates={"last_only": {"add_on_layers": 0.001}},
    )
    assert lrs_by_component(schedule, mock_protopnet, "last_only") == {
        "add_on_layers": 0.001,
        "prototype_prediction_head": 0.2,
    }


def test_joint_lr_scheduler_starts_from_overridden_rate(make_schedule):
    schedule = make_schedule(stage_learning_rates={"joint": {"backbone": 0.002}})
    scheduler = first_phase(schedule, "joint").train.scheduler
    assert 0.002 in scheduler.base_lrs


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"warmup": {"add_on_layers": 0.1}}, "Unknown stage"),
        ({"warm": {"add_on_layer": 0.1}}, "Unknown component"),
        ({"warm": {"backbone": 0.1}}, "frozen in stage 'warm'"),
        ({"warm": {"add_on_layers": 0}}, "positive number"),
        ({"warm": {"add_on_layers": -0.1}}, "positive number"),
        ({"warm": {"add_on_layers": float("nan")}}, "positive number"),
        ({"warm": {"add_on_layers": True}}, "positive number"),
        ({"warm": "add_on_layers"}, "component=rate"),
        ({"warm": "add_on_layers=fast"}, "not a number"),
    ],
)
def test_invalid_stage_learning_rates_raise(make_schedule, overrides, match):
    with pytest.raises(ValueError, match=match):
        make_schedule(stage_learning_rates=overrides)


def test_parse_learning_rate_string():
    assert parse_learning_rate_string("a=0.1, b=2e-3,") == {"a": 0.1, "b": 0.002}
