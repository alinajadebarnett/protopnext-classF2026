"""Tests for schedule config files (change 2) and the --schedule-config CLI option."""

from pathlib import Path
from unittest import mock

import pytest
import torch
from click.testing import CliRunner

import protopnet.train_protopnet as train_protopnet
from protopnet.models.vanilla_protopnet import ProtoPNetTrainingSchedule
from protopnet.schedule_config import (
    load_schedule_config,
    scale_stage_order,
    schedule_kwargs_from_config,
)
from protopnet.train.scheduling.types import IterativePhase

SCHEDULES_DIR = Path(__file__).resolve().parents[1] / "configs" / "schedules"


def write(tmp_path, text, name="schedule.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


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


def flat(schedule):
    """[(name, duration), ...] for every phase the schedule will run."""
    return [(p.name, p.duration) for p in schedule]


# --------------------------------------------------------------------------
# Reading config files
# --------------------------------------------------------------------------


def test_full_config(tmp_path):
    path = write(
        tmp_path,
        """
stages:
  warm:
    trainable: [prototype_layer]
    lr: {prototype_layer: 0.01}
  last_only:
    lr: {prototype_prediction_head: 0.0005}
freeze_batchnorm: true
schedule:
  - {stage: warm, epochs: 2}
  - {stage: project}
""",
    )
    assert load_schedule_config(path) == {
        "trainable_components": {"warm": ["prototype_layer"]},
        "stage_learning_rates": {
            "warm": {"prototype_layer": 0.01},
            "last_only": {"prototype_prediction_head": 0.0005},
        },
        "freeze_batchnorm": True,
        "stage_order": [{"stage": "warm", "epochs": 2}, {"stage": "project"}],
    }


def test_empty_file_means_all_defaults(tmp_path):
    assert load_schedule_config(write(tmp_path, "")) == {}


def test_stage_with_no_settings_is_allowed(tmp_path):
    assert load_schedule_config(write(tmp_path, "stages:\n  warm:\n")) == {}


@pytest.mark.parametrize(
    "text,match",
    [
        ("- just\n- a list\n", "top level must be a mapping"),
        ("epochs: 3\n", "unknown section"),
        ("stages: [warm, joint]\n", "'stages' must map"),
        ("stages:\n  warm: [backbone]\n", "settings for stage 'warm'"),
        ("stages:\n  warm:\n    epochs: 3\n", "unknown setting"),
        ("stages:\n  warm:\n    lr: 0.1\n", "'lr' for stage 'warm'"),
        ("freeze_batchnorm: yes please\n", "must be true or false"),
        ("schedule: warm\n", "'schedule' must be a list"),
        ("stages: {warm: [\n", "not valid YAML"),
    ],
)
def test_invalid_config_files(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_schedule_config(write(tmp_path, text))


def test_errors_name_the_file(tmp_path):
    path = write(tmp_path, "epochs: 3\n", name="my_schedule.yaml")
    with pytest.raises(ValueError, match="my_schedule.yaml"):
        load_schedule_config(path)


def test_schedule_kwargs_from_parsed_config():
    assert schedule_kwargs_from_config(None) == {}
    assert schedule_kwargs_from_config({"freeze_batchnorm": False}) == {
        "freeze_batchnorm": False
    }


# --------------------------------------------------------------------------
# Scaling epochs (phase multiplier and verify mode)
# --------------------------------------------------------------------------

ORDER = [
    {"stage": "warm", "epochs": 5},
    {"stage": "joint", "epochs": 0},
    {"stage": "project"},
    {
        "repeat": 3,
        "stages": [{"stage": "last_only", "epochs": 4}, {"stage": "project"}],
    },
]


def test_scale_stage_order_multiplies_epochs():
    assert scale_stage_order(ORDER, multiplier=2) == [
        {"stage": "warm", "epochs": 10},
        {"stage": "joint", "epochs": 0},
        {"stage": "project"},
        {
            "repeat": 3,
            "stages": [{"stage": "last_only", "epochs": 8}, {"stage": "project"}],
        },
    ]


def test_scale_stage_order_verify_runs_each_stage_once():
    assert scale_stage_order(ORDER, multiplier=7, verify=True) == [
        {"stage": "warm", "epochs": 1},
        {"stage": "joint", "epochs": 0},  # 0 still means skipped
        {"stage": "project"},
        {
            "repeat": 1,
            "stages": [{"stage": "last_only", "epochs": 1}, {"stage": "project"}],
        },
    ]


def test_scale_stage_order_does_not_change_its_input():
    before = [dict(item) for item in ORDER]
    scale_stage_order(ORDER, multiplier=3)
    assert [dict(item) for item in ORDER] == before


def test_scale_stage_order_leaves_malformed_items_for_validation():
    weird = [{"stage": "warm", "epochs": "five"}, "joint", {"repeat": 2}]
    assert scale_stage_order(weird, multiplier=2) == weird


# --------------------------------------------------------------------------
# Building the schedule from a stage order
# --------------------------------------------------------------------------


def test_stage_order_builds_the_given_sequence(make_schedule):
    schedule = make_schedule(
        stage_order=[
            {"stage": "joint", "epochs": 2},
            {"stage": "project"},
            {"stage": "warm", "epochs": 1},
            {
                "repeat": 2,
                "stages": [
                    {"stage": "last_only", "epochs": 3},
                    {"stage": "project"},
                ],
            },
        ]
    )
    assert flat(schedule) == [
        ("joint", 2),
        ("project", 1),
        ("warm", 1),
        ("last_only", 3),
        ("project", 1),
        ("last_only", 3),
        ("project", 1),
    ]
    assert isinstance(schedule.phases[-1], IterativePhase)


def test_stage_order_skips_zero_epochs_and_zero_repeats(make_schedule):
    schedule = make_schedule(
        stage_order=[
            {"stage": "warm", "epochs": 0},
            {"stage": "joint", "epochs": 1},
            {"repeat": 0, "stages": [{"stage": "last_only", "epochs": 3}]},
            {"repeat": 4, "stages": [{"stage": "last_only", "epochs": 0}]},
        ]
    )
    assert flat(schedule) == [("joint", 1)]


def test_stage_order_reuses_the_shared_joint_optimizer(make_schedule):
    schedule = make_schedule(
        stage_order=[
            {"stage": "joint", "epochs": 1},
            {"repeat": 2, "stages": [{"stage": "joint", "epochs": 1}]},
        ]
    )
    assert len({id(p.train.optimizer) for p in schedule}) == 1


def test_stage_order_keeps_trainable_and_learning_rate_settings(make_schedule):
    schedule = make_schedule(
        trainable_components={"joint": ["add_on_layers"]},
        stage_learning_rates={"joint": {"add_on_layers": 0.02}},
        stage_order=[{"stage": "joint", "epochs": 1}],
    )
    joint = next(iter(schedule)).train
    assert joint.set_training_layers_fn.trainable == ("add_on_layers",)
    assert [g["lr"] for g in joint.optimizer.param_groups] == [0.02]


@pytest.mark.parametrize(
    "stage_order,match",
    [
        ("warm", "must be a list"),
        ([], "empty"),
        ([{"stage": "warmup", "epochs": 1}], "item 1: unknown stage 'warmup'"),
        ([{"stage": "warm"}], "needs an 'epochs' value"),
        ([{"stage": "warm", "epochs": -1}], "non-negative integer"),
        ([{"stage": "warm", "epochs": 1.5}], "non-negative integer"),
        ([{"stage": "project", "epochs": 2}], "'project' takes no epochs"),
        ([{"stage": "warm", "epochs": 1, "lr": 0.1}], "unknown key"),
        (["warm"], "item 1: expected"),
        ([{"repeat": 2}], "non-empty 'stages' list"),
        ([{"repeat": -1, "stages": [{"stage": "project"}]}], "non-negative"),
        ([{"repeat": 2, "stages": [], "x": 1}], "unknown key"),
        (
            [
                {
                    "repeat": 2,
                    "stages": [{"repeat": 2, "stages": [{"stage": "project"}]}],
                }
            ],
            "item 1, stage 1: repeat blocks cannot be nested",
        ),
        (
            [{"stage": "joint", "epochs": 1}, {"stage": "joint", "epochs": "x"}],
            "item 2",
        ),
        ([{"stage": "warm", "epochs": 0}], "nothing to train"),
    ],
)
def test_invalid_stage_orders(make_schedule, stage_order, match):
    with pytest.raises(ValueError, match=match):
        make_schedule(stage_order=stage_order)


# --------------------------------------------------------------------------
# Example config files in configs/schedules
# --------------------------------------------------------------------------

# The defaults that `train-protopnet` passes to ProtoPNetTrainingSchedule
CLI_DEFAULT_EPOCHS = dict(
    num_warm_epochs=5,
    num_preproject_joint_epochs=5,
    last_only_epochs_per_project=10,
    joint_epochs_per_phase=5,
    post_project_phases=20,
)


def test_default_config_file_reproduces_the_default_schedule(make_schedule):
    from_file = make_schedule(
        **load_schedule_config(SCHEDULES_DIR / "protopnet_default.yaml")
    )
    default = make_schedule(**CLI_DEFAULT_EPOCHS)

    assert repr(from_file) == repr(default)
    assert flat(from_file) == flat(default)
    assert from_file.stage_trainable_components == default.stage_trainable_components
    assert from_file.stage_learning_rates == default.stage_learning_rates


@pytest.mark.parametrize("path", sorted(SCHEDULES_DIR.glob("*.yaml")))
def test_every_example_config_builds_a_schedule(make_schedule, path):
    schedule = make_schedule(**load_schedule_config(path))
    assert len(flat(schedule)) > 0


# --------------------------------------------------------------------------
# --schedule-config on the command line
# --------------------------------------------------------------------------


class _Stop(Exception):
    pass


def run_cli(args):
    """Run `train-protopnet` up to building the schedule; return its keyword args."""
    captured = {}

    def fake_schedule(**kwargs):
        captured.update(kwargs)
        raise _Stop()

    with mock.patch.object(
        train_protopnet, "ProtoPNetTrainingSchedule", fake_schedule
    ), mock.patch.object(
        train_protopnet.datasets,
        "training_dataloaders",
        return_value=mock.MagicMock(num_classes=3),
    ), mock.patch.object(
        train_protopnet, "construct_backbone"
    ), mock.patch.object(
        train_protopnet, "VanillaProtoPNet"
    ), mock.patch.object(
        train_protopnet, "WeightsAndBiasesTrainLogger"
    ), mock.patch.object(
        train_protopnet, "AddonLayers"
    ), mock.patch.object(
        train_protopnet, "loss_for_coefficients"
    ):
        result = CliRunner().invoke(train_protopnet.run, args)
    return captured, result


CONFIG_TEXT = """
stages:
  last_only:
    trainable: [prototype_prediction_head, add_on_layers]
    lr: {add_on_layers: 0.0005}
freeze_batchnorm: true
schedule:
  - {stage: warm, epochs: 2}
  - {stage: project}
  - repeat: 3
    stages:
      - {stage: last_only, epochs: 4}
"""


def test_cli_passes_config_file_settings_to_the_schedule(tmp_path):
    path = write(tmp_path, CONFIG_TEXT)
    kwargs, _ = run_cli(["--schedule-config", str(path), "--phase-multiplier", "2"])
    assert kwargs["trainable_components"] == {
        "last_only": ["prototype_prediction_head", "add_on_layers"]
    }
    assert kwargs["stage_learning_rates"] == {"last_only": {"add_on_layers": 0.0005}}
    assert kwargs["freeze_batchnorm"] is True
    # epochs are multiplied by --phase-multiplier
    assert kwargs["stage_order"] == [
        {"stage": "warm", "epochs": 4},
        {"stage": "project"},
        {"repeat": 3, "stages": [{"stage": "last_only", "epochs": 8}]},
    ]


def test_cli_verify_runs_config_stages_once(tmp_path):
    path = write(tmp_path, CONFIG_TEXT)
    kwargs, _ = run_cli(["--schedule-config", str(path), "--verify"])
    assert kwargs["stage_order"] == [
        {"stage": "warm", "epochs": 1},
        {"stage": "project"},
        {"repeat": 1, "stages": [{"stage": "last_only", "epochs": 1}]},
    ]


def test_cli_without_config_passes_no_stage_order():
    kwargs, _ = run_cli(["--joint-lr", "add_on_layers=0.001"])
    assert kwargs["stage_order"] is None
    assert kwargs["stage_learning_rates"] == {"joint": "add_on_layers=0.001"}


@pytest.mark.parametrize(
    "extra",
    [
        ["--warm-epochs", "3"],
        ["--joint-trainable", "backbone"],
        ["--last-only-lr", "prototype_prediction_head=0.001"],
        ["--freeze-batchnorm"],
    ],
)
def test_cli_config_cannot_be_combined_with_per_stage_options(tmp_path, extra):
    path = write(tmp_path, CONFIG_TEXT)
    kwargs, result = run_cli(["--schedule-config", str(path), *extra])
    assert kwargs == {}  # stopped before building anything
    assert result.exit_code != 0
    assert "cannot be combined with" in result.output
    assert extra[0] in result.output


def test_cli_reports_a_missing_config_file(tmp_path):
    _, result = run_cli(["--schedule-config", str(tmp_path / "missing.yaml")])
    assert result.exit_code != 0
    assert "does not exist" in result.output
