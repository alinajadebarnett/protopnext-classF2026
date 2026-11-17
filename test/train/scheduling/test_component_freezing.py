"""Tests for component-level freezing during training stages (change 1)."""

from unittest.mock import Mock

import pytest
import torch

from protopnet.train.scheduling import (
    ClassificationBackpropPhase,
    TrainLayersByComponent,
    TrainOptimizerLayers,
)


class TinyProtoPNet(torch.nn.Module):
    """A real (not mocked) model with the four ProtoPNet components and BatchNorm."""

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(4, 6), torch.nn.BatchNorm1d(6), torch.nn.ReLU()
        )
        self.add_on_layers = torch.nn.Sequential(
            torch.nn.Linear(6, 5), torch.nn.BatchNorm1d(5)
        )
        self.prototype_layer = torch.nn.Module()
        self.prototype_layer.prototype_tensors = torch.nn.Parameter(torch.randn(3, 5))
        self.prototype_prediction_head = torch.nn.Module()
        self.prototype_prediction_head.class_connection_layer = torch.nn.Linear(3, 2)

    def forward(self, x):
        z = self.add_on_layers(self.backbone(x))
        sims = z @ self.prototype_layer.prototype_tensors.T
        return self.prototype_prediction_head.class_connection_layer(sims)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyProtoPNet()


def requires_grad_by_component(model):
    out = {}
    for name, param in model.named_parameters():
        out.setdefault(name.split(".")[0], set()).add(param.requires_grad)
    return {k: v.pop() if len(v) == 1 else v for k, v in out.items()}


def backprop_phase(set_training_layers_fn, optimizer):
    loss = Mock()
    loss.batch_loss.required_forward_results.return_value = []
    return ClassificationBackpropPhase(
        name="test",
        dataloader=Mock(),
        device="cpu",
        loss=loss,
        optimizer=optimizer,
        set_training_layers_fn=set_training_layers_fn,
    )


# --------------------------------------------------------------------------
# TrainOptimizerLayers with a real optimizer (regression test for the bug fix)
# --------------------------------------------------------------------------


def test_train_optimizer_layers_with_real_optimizer_uses_all_param_groups(model):
    """
    A real torch optimizer stores tensors, not names, and often has several
    param groups. Previously every parameter ended up frozen in this situation.
    """
    optimizer = torch.optim.Adam(
        [
            {"params": model.add_on_layers.parameters(), "lr": 1e-3},
            {"params": [model.prototype_layer.prototype_tensors], "lr": 1e-2},
        ]
    )
    phase = backprop_phase(TrainOptimizerLayers(), optimizer)

    TrainOptimizerLayers().set_training_layers(model, phase)

    assert requires_grad_by_component(model) == {
        "backbone": False,
        "add_on_layers": True,
        "prototype_layer": True,
        "prototype_prediction_head": False,
    }


# --------------------------------------------------------------------------
# TrainLayersByComponent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trainable",
    [
        ("backbone",),
        ("add_on_layers", "prototype_layer"),
        ("prototype_prediction_head",),
        ("backbone", "add_on_layers", "prototype_layer", "prototype_prediction_head"),
    ],
)
def test_sets_requires_grad_for_listed_components_only(model, trainable):
    TrainLayersByComponent(trainable=trainable).set_training_layers(model)

    states = requires_grad_by_component(model)
    for component, requires_grad in states.items():
        assert requires_grad is (component in trainable), component


def test_from_flags_matches_explicit_list(model):
    from_flags = TrainLayersByComponent.from_flags(
        backbone=False,
        add_on_layers=True,
        prototype_layer=True,
        prototype_prediction_head=False,
    )
    explicit = TrainLayersByComponent(trainable=["prototype_layer", "add_on_layers"])
    assert from_flags == explicit


def test_trainable_is_normalized_and_hashable():
    a = TrainLayersByComponent(trainable=["prototype_layer", "backbone", "backbone"])
    assert a.trainable == ("backbone", "prototype_layer")
    hash(a)  # frozen dataclass must stay hashable


def test_single_string_is_rejected():
    with pytest.raises(TypeError, match="collection of component names"):
        TrainLayersByComponent(trainable="backbone")


def test_unknown_component_raises_in_strict_mode(model):
    with pytest.raises(ValueError, match="backbon"):
        TrainLayersByComponent(trainable=["backbon"]).set_training_layers(model)


def test_unknown_component_ignored_when_not_strict(model):
    TrainLayersByComponent(
        trainable=["backbone", "conv_offset"], strict=False
    ).set_training_layers(model)
    assert requires_grad_by_component(model)["backbone"] is True


def test_describe(model):
    layers = TrainLayersByComponent(trainable=["prototype_layer"])
    assert layers.describe(model) == {
        "add_on_layers": False,
        "backbone": False,
        "prototype_layer": True,
        "prototype_prediction_head": False,
    }


def test_freeze_batchnorm_only_affects_frozen_components(model):
    model.train()
    TrainLayersByComponent(
        trainable=["add_on_layers"], freeze_batchnorm=True
    ).set_training_layers(model)

    assert model.backbone[1].training is False  # frozen component -> BN eval
    assert model.add_on_layers[1].training is True  # trainable component -> BN train


def test_batchnorm_untouched_by_default(model):
    model.train()
    TrainLayersByComponent(trainable=["add_on_layers"]).set_training_layers(model)
    assert model.backbone[1].training is True


# --------------------------------------------------------------------------
# Integration with ClassificationBackpropPhase
# --------------------------------------------------------------------------


def test_phase_settings_restores_requires_grad_and_module_modes(model):
    model.train()
    before_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    before_modes = {n: m.training for n, m in model.named_modules()}

    optimizer = torch.optim.Adam(model.prototype_prediction_head.parameters())
    phase = backprop_phase(
        TrainLayersByComponent(
            trainable=["prototype_prediction_head"], freeze_batchnorm=True
        ),
        optimizer,
    )

    with phase.phase_settings(model):
        assert requires_grad_by_component(model)["backbone"] is False
        assert model.backbone[1].training is False

    assert {n: p.requires_grad for n, p in model.named_parameters()} == before_grad
    assert {n: m.training for n, m in model.named_modules()} == before_modes


def test_frozen_parameters_and_batchnorm_stats_do_not_change_during_a_step(model):
    """The end-to-end property users care about: frozen means unchanged."""
    model.train()
    optimizer = torch.optim.SGD(
        list(model.add_on_layers.parameters())
        + [model.prototype_layer.prototype_tensors],
        lr=0.1,
    )
    phase = backprop_phase(
        TrainLayersByComponent(
            trainable=["add_on_layers", "prototype_layer"], freeze_batchnorm=True
        ),
        optimizer,
    )

    snapshot = {k: v.clone() for k, v in model.state_dict().items()}

    with phase.phase_settings(model):
        x = torch.randn(8, 4) * 3 + 1  # off-center data so BN stats would move
        y = torch.randint(0, 2, (8,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = model.state_dict()

    for key, value in snapshot.items():
        component = key.split(".")[0]
        changed = not torch.equal(value, after[key])
        if component in ("backbone", "prototype_prediction_head"):
            # weights *and* BatchNorm running stats stay fixed
            assert not changed, f"{key} changed but its component is frozen"

    # trainable components actually learned something
    assert not torch.equal(
        snapshot["prototype_layer.prototype_tensors"],
        after["prototype_layer.prototype_tensors"],
    )
    assert not torch.equal(
        snapshot["add_on_layers.0.weight"], after["add_on_layers.0.weight"]
    )
    # frozen components received no gradient at all
    assert model.backbone[0].weight.grad is None
