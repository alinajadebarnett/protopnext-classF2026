import math
import pickle

import numpy as np
import pytest
import torch

from protopnet.activations import CosPrototypeActivation, L2Activation
from protopnet.embedding import AddonLayers, EmbeddedBackbone
from protopnet.model_losses import loss_for_coefficients
from protopnet.models.protopool import (
    GumbelScaleSchedule,
    ProtoPool,
    ProtoPoolBackpropPhase,
    ProtoPoolTrainingSchedule,
    TrainProtoPoolLayers,
    mixup_batch,
)
from protopnet.pacmap import prototype_labels_and_num_classes
from protopnet.prototype_layers import ProtoPoolPrototypeLayer
from protopnet.train.metrics import InterpretableTrainingMetrics
from protopnet.train.scheduling.scheduling import ProjectPhase
from protopnet.train.scheduling.types import IterativePhase, StepContext

NUM_CLASSES, NUM_PROTOTYPES, NUM_DESCRIPTIVE = 3, 5, 2


def make_model(seed=0):
    torch.manual_seed(seed)
    backbone = EmbeddedBackbone(
        torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, stride=4), torch.nn.ReLU()),
        input_channels=(3, 32, 32),
    )
    add_on_layers = AddonLayers(
        NUM_PROTOTYPES, input_channels=8, proto_channels=16, num_addon_layers=1
    )
    return ProtoPool(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=L2Activation(),
        num_classes=NUM_CLASSES,
        num_prototypes=NUM_PROTOTYPES,
        num_descriptive=NUM_DESCRIPTIVE,
    )


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def dataloader():
    torch.manual_seed(1)
    images = torch.randn(12, 3, 32, 32)
    targets = torch.tensor([0, 1, 2] * 4)
    dataset = [
        {"img": images[i], "target": targets[i], "sample_id": str(i)}
        for i in range(len(targets))
    ]
    return torch.utils.data.DataLoader(dataset, batch_size=4)


@pytest.fixture
def loss():
    return loss_for_coefficients(
        {
            "mixup_cross_entropy": 1,
            "protopool_cluster": 0.8,
            "protopool_separation": -0.08,
            "slot_orthogonality": 1,
            "l1": 1e-4,
        }
    )


def make_schedule(model, dataloader, loss, **kwargs):
    return ProtoPoolTrainingSchedule(
        model=model,
        train_loader=dataloader,
        val_loader=dataloader,
        project_loader=dataloader,
        loss=loss,
        phase_config_kwargs={"device": "cpu"},
        **kwargs,
    )


def test_protopool_forward(model):
    output = model(
        torch.randn(4, 3, 32, 32),
        return_prototype_layer_output_dict=True,
        return_min_distances=True,
        return_proto_presence=True,
    )
    assert output["logits"].shape == (4, NUM_CLASSES)
    assert output["prototype_activations"].shape[:2] == (4, NUM_PROTOTYPES)
    assert output["prototype_distances"].shape == output["prototype_activations"].shape
    assert output["min_distances"].shape == (4, NUM_PROTOTYPES)
    assert output["proto_presence"].shape == (
        NUM_CLASSES,
        NUM_PROTOTYPES,
        NUM_DESCRIPTIVE,
    )


def test_protopool_layer_requires_l2_activation():
    with pytest.raises(AssertionError):
        ProtoPoolPrototypeLayer(
            num_prototypes=2,
            num_classes=2,
            activation_function=CosPrototypeActivation(),
        )


def test_protopool_project_is_class_restricted(model, dataloader):
    # prototypes 3 and 4 are not used by any class
    slots = torch.tensor([[0, 1], [1, 2], [2, 2]])
    model.prototype_prediction_head.set_hard_assignment(slots)

    model.project(dataloader)

    allowed = model.prototype_layer.prototype_allowed_classes
    assert allowed.tolist() == [
        [True, False, False],
        [True, True, False],
        [False, True, True],
        [False, False, False],
        [False, False, False],
    ]
    metas = model.prototype_layer.prototype_meta
    assert all(meta.source == "sample" for meta in metas)
    assert metas[0].target == 0
    assert metas[1].target in {0, 1}
    assert metas[2].target in {1, 2}
    assert torch.equal(model.slot_prototype_indices(), slots)
    assert model.prototypes_embedded()


def test_protopool_project_sets_hard_assignment(model, dataloader):
    expected = torch.argmax(model.prototype_prediction_head.proto_presence, dim=1)
    model.project(dataloader)
    assert model.prototype_prediction_head.hard_assignment
    assert torch.equal(model.slot_prototype_indices(), expected)


def test_protopool_logits_match_effective_connections_after_project(model, dataloader):
    model.project(dataloader)
    model.eval()
    output = model(
        torch.randn(4, 3, 32, 32), return_similarity_score_to_each_prototype=True
    )
    assert torch.allclose(
        output["logits"],
        output["similarity_score_to_each_prototype"]
        @ model.effective_class_connections().T,
        atol=1e-5,
    )


def test_protopool_prune_duplicate_prototypes(model, dataloader):
    model.prototype_prediction_head.set_hard_assignment(
        torch.tensor([[0, 1], [1, 2], [2, 4]])
    )
    model.project(dataloader)
    with torch.no_grad():
        model.prototype_layer.prototype_tensors[
            4
        ] = model.prototype_layer.prototype_tensors[0]
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    before = model(x)["logits"]

    model.prune_duplicate_prototypes()

    assert model.num_prototypes == NUM_PROTOTYPES - 1
    assert model.prototype_prediction_head.proto_presence.shape[1] == NUM_PROTOTYPES - 1
    assert model.slot_prototype_indices().tolist() == [[0, 1], [1, 2], [2, 0]]
    assert model.prototype_layer.prototype_allowed_classes.shape == (
        NUM_PROTOTYPES - 1,
        NUM_CLASSES,
    )
    assert torch.allclose(before, model(x)["logits"], atol=1e-5)


def test_protopool_prune_skipped_before_projection(model):
    model.prune_duplicate_prototypes()
    assert model.num_prototypes == NUM_PROTOTYPES


def test_protopool_metric_hooks(model, dataloader):
    model.prototype_prediction_head.set_hard_assignment(
        torch.tensor([[0, 1], [1, 2], [2, 2]])
    )
    acts = torch.arange(NUM_PROTOTYPES).float().view(1, -1, 1, 1).expand(2, -1, 3, 3)
    class_major = model.class_major_prototype_activations(acts)
    assert class_major.shape == (2, NUM_CLASSES * NUM_DESCRIPTIVE, 3, 3)
    assert class_major[0, :, 0, 0].tolist() == [0, 1, 1, 2, 2, 2]
    assert model.metric_prototypes_per_class() == NUM_DESCRIPTIVE

    labels, num_classes = prototype_labels_and_num_classes(model)
    assert num_classes == NUM_CLASSES
    assert labels.shape == (NUM_PROTOTYPES,)
    assert labels[0].item() == 0
    assert labels[2].item() == 2


def test_protopool_interpretable_metrics_use_class_major_activations(model):
    model.prototype_prediction_head.set_hard_assignment(
        torch.tensor([[0, 1], [1, 2], [2, 2]])
    )
    metrics = InterpretableTrainingMetrics(
        protopnet=model,
        num_classes=NUM_CLASSES,
        proto_per_class=model.metric_prototypes_per_class(),
        part_num=2,
        img_size=32,
        half_size=4,
    )
    acts = torch.rand(2, NUM_PROTOTYPES, 3, 3)
    assert metrics._class_major_activations(acts).shape == (
        2,
        NUM_CLASSES * NUM_DESCRIPTIVE,
        3,
        3,
    )


def test_protopool_state_dict_and_pickle_round_trip(model, dataloader):
    model.project(dataloader)
    model.prototype_prediction_head.gumbel_scale = 12.0
    model.eval()
    x = torch.randn(3, 3, 32, 32)
    expected = model(x)["logits"]

    restored = make_model(seed=5)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    assert restored.prototype_prediction_head.hard_assignment
    assert restored.prototype_prediction_head.gumbel_scale == 12.0
    assert torch.equal(
        restored.prototype_layer.prototype_allowed_classes,
        model.prototype_layer.prototype_allowed_classes,
    )
    assert torch.allclose(restored(x)["logits"], expected)

    unpickled = pickle.loads(pickle.dumps(model))
    assert torch.allclose(unpickled(x)["logits"], expected)


def test_gumbel_scale_schedule():
    schedule = GumbelScaleSchedule(gumbel_epochs=30)
    assert schedule(0) == pytest.approx(1000 * math.sqrt(1 / 30))
    assert schedule(0) > 0
    assert schedule(9) == pytest.approx(1000 * math.sqrt(10 / 30))
    assert schedule(29) == 1000
    assert schedule(100) == 1000
    assert all(schedule(i) < schedule(i + 1) for i in range(28))
    assert GumbelScaleSchedule(enabled=False)(5) == 0.0

    pickle.loads(pickle.dumps(schedule))


def test_mixup_batch():
    batch = {"img": torch.randn(6, 3, 4, 4), "target": torch.arange(6)}
    original_img = batch["img"].clone()
    mixed = mixup_batch(batch, alpha=0.5, generator=np.random.default_rng(0))
    assert mixed["img"].shape == original_img.shape
    assert mixed["mixup_target_b"].shape == (6,)
    assert 0 <= mixed["mixup_lambda"] <= 1
    assert sorted(mixed["mixup_target_b"].tolist()) == list(range(6))


def phase_names(schedule):
    return [phase.name for phase in schedule]


def test_protopool_schedule_default_phases(model, dataloader, loss):
    schedule = make_schedule(model, dataloader, loss)
    assert phase_names(schedule) == ["warm", "joint", "project", "last_only"]
    assert [phase.duration for phase in schedule.phases] == [10, 20, 1, 25]
    assert isinstance(schedule.phases[2].train, ProjectPhase)


def test_protopool_schedule_iterative_phases(model, dataloader, loss):
    schedule = make_schedule(
        model, dataloader, loss, iterative_push_cycles=2, iterative_joint_epochs=3
    )
    assert phase_names(schedule) == [
        "warm",
        "joint",
        "project",
        "last_only",
        "joint",
        "project",
        "last_only",
        "joint",
        "project",
        "last_only",
    ]
    iterative = schedule.phases[3]
    assert isinstance(iterative, IterativePhase)
    pre_project_joint = schedule.phases[1].train
    post_project_joint = iterative.phases[1].train
    assert post_project_joint is not pre_project_joint
    assert post_project_joint.optimizer is not pre_project_joint.optimizer
    assert post_project_joint.scheduler is None
    assert iterative.phases[1].duration == 3
    post_lrs = [group["lr"] for group in post_project_joint.optimizer.param_groups]
    assert post_lrs == pytest.approx([1e-5, 3e-4, 3e-4])
    post_params = {
        id(p) for g in post_project_joint.optimizer.param_groups for p in g["params"]
    }
    assert id(model.prototype_prediction_head.proto_presence) not in post_params


def test_protopool_schedule_learning_rates(model, dataloader, loss):
    schedule = make_schedule(model, dataloader, loss, lr=1e-3)
    warm, joint, _, last_only = [phase.train for phase in schedule.phases]

    warm_groups = warm.optimizer.param_groups
    assert [g["lr"] for g in warm_groups] == pytest.approx([3e-3, 3e-3, 3e-3])
    assert warm_groups[0]["weight_decay"] == 1e-3
    assert warm_groups[1]["params"][0] is model.prototype_prediction_head.proto_presence
    assert warm_groups[2]["params"][0] is model.prototype_layer.prototype_tensors

    joint_groups = joint.optimizer.param_groups
    assert [g["lr"] for g in joint_groups] == pytest.approx([1e-4, 3e-3, 3e-3, 3e-3])
    assert joint_groups[0]["weight_decay"] == 1e-3
    assert joint.scheduler.step_size == 5
    assert joint.scheduler.gamma == 0.1

    last_groups = last_only.optimizer.param_groups
    assert [g["lr"] for g in last_groups] == pytest.approx([1e-4])
    assert last_groups[0]["weight_decay"] == 1e-3
    assert (
        last_groups[0]["params"][0]
        is model.prototype_prediction_head.class_connection_layer.weight
    )
    assert last_only.scheduler.step_size == 5
    assert last_only.scheduler.gamma == 0.95

    # optimizers do not share param group dicts, so the joint scheduler cannot change warm learning rates
    joint.scheduler.step()
    for _ in range(5):
        joint.scheduler.step()
    assert [g["lr"] for g in warm_groups] == pytest.approx([3e-3, 3e-3, 3e-3])


def _requires_grad(model):
    head = model.prototype_prediction_head
    return {
        "backbone": all(p.requires_grad for p in model.backbone.parameters()),
        "add_on_layers": all(p.requires_grad for p in model.add_on_layers.parameters()),
        "prototypes": model.prototype_layer.prototype_tensors.requires_grad,
        "slot_assignment": head.proto_presence.requires_grad,
        "last_layer": head.class_connection_layer.weight.requires_grad,
    }


@pytest.mark.parametrize(
    "phase_index,expected",
    [
        (
            0,
            {
                "backbone": False,
                "add_on_layers": True,
                "prototypes": True,
                "slot_assignment": True,
                "last_layer": False,
            },
        ),
        (
            1,
            {
                "backbone": True,
                "add_on_layers": True,
                "prototypes": True,
                "slot_assignment": True,
                "last_layer": False,
            },
        ),
        (
            3,
            {
                "backbone": False,
                "add_on_layers": False,
                "prototypes": False,
                "slot_assignment": False,
                "last_layer": True,
            },
        ),
    ],
)
def test_protopool_schedule_requires_grad(
    model, dataloader, loss, phase_index, expected
):
    schedule = make_schedule(model, dataloader, loss)
    phase = schedule.phases[phase_index].train
    with phase.phase_settings(model):
        assert _requires_grad(model) == expected


def test_train_protopool_layers_toggles_slot_assignment(model):
    TrainProtoPoolLayers(
        train_backbone=False,
        train_add_on_layers=False,
        train_prototype_layer=False,
        train_prototype_prediction_head=True,
        train_slot_assignment=False,
    ).set_training_layers(model, None)
    head = model.prototype_prediction_head
    assert not head.proto_presence.requires_grad
    assert head.class_connection_layer.weight.requires_grad


def test_protopool_backprop_phase_sets_gumbel_scale_and_trains(model, dataloader, loss):
    schedule = make_schedule(
        model, dataloader, loss, gumbel_schedule=GumbelScaleSchedule(gumbel_epochs=4)
    )
    warm = schedule.phases[0].train
    assert isinstance(warm, ProtoPoolBackpropPhase)
    assert warm.mixup_alpha == 0.5

    presence_before = model.prototype_prediction_head.proto_presence.detach().clone()
    model.train()
    with warm.phase_settings(model):
        metrics = warm.run_step(model, StepContext(global_step=1, step_in_phase=1))

    assert model.prototype_prediction_head.gumbel_scale == pytest.approx(
        GumbelScaleSchedule(gumbel_epochs=4)(1)
    )
    assert "mixup_cross_entropy" in metrics
    assert "protopool_cluster" in metrics
    assert "slot_orthogonality" in metrics
    assert not torch.equal(
        presence_before, model.prototype_prediction_head.proto_presence
    )

    eval_metrics = schedule.default_eval_phase.run_step(
        model, StepContext(global_step=1, step_in_phase=1)
    )
    assert "mixup_cross_entropy" in eval_metrics


def test_protopool_gumbel_scale_frozen_after_projection(model, dataloader, loss):
    schedule = make_schedule(model, dataloader, loss)
    model.project(dataloader)
    model.prototype_prediction_head.gumbel_scale = 7.0
    joint = schedule.phases[1].train
    joint._pre_dataloader_step_init(model, StepContext(global_step=3, step_in_phase=0))
    assert model.prototype_prediction_head.gumbel_scale == 7.0
