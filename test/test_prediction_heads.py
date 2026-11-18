import pytest
import torch
import torch.nn.functional as F

from protopnet.prediction_heads import (
    LinearClassPrototypePredictionHead,
    ProtoPoolPredictionHead,
)


def test_topk():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes, num_classes))

    input = torch.randn(10, num_classes, 224, 224)
    prev_activations = None
    for k in range(1, 4):
        prototype_config = {
            "prototype_class_identity": prototype_class_identity,
            "k_for_topk": k,
        }

        prediction_head = LinearClassPrototypePredictionHead(**prototype_config)
        activations = prediction_head.forward(
            input, return_similarity_score_to_each_prototype=True
        )["similarity_score_to_each_prototype"]

        if prev_activations is not None:
            assert (prev_activations >= activations).all()

        prev_activations = activations


def _original_prototype_chooser_logits(
    distances, proto_presence_param, last_layer_weight, epsilon=1e-4
):
    """
    PrototypeChooser.forward from gmum/ProtoPool (model.py) with gumbel_scale=0, applied to precomputed distances.
    """
    proto_presence = torch.softmax(proto_presence_param, dim=1)
    min_distances = -F.max_pool2d(
        -distances, kernel_size=(distances.size()[2], distances.size()[3])
    ).squeeze()
    avg_dist = F.avg_pool2d(
        distances, kernel_size=(distances.size()[2], distances.size()[3])
    ).squeeze()
    min_mixed_distances = torch.einsum("bp,cpn->bcn", min_distances, proto_presence)
    avg_mixed_distances = torch.einsum("bp,cpn->bcn", avg_dist, proto_presence)

    def distance_2_similarity(d):
        return torch.log((d + 1) / (d + epsilon))

    x = distance_2_similarity(min_mixed_distances) - distance_2_similarity(
        avg_mixed_distances
    )
    return F.linear(x.flatten(start_dim=1), last_layer_weight), min_distances


@pytest.fixture
def protopool_head():
    torch.manual_seed(0)
    return ProtoPoolPredictionHead(num_classes=4, num_prototypes=6, num_descriptive=3)


def _protopool_inputs(batch_size=5, num_prototypes=6):
    distances = torch.rand(batch_size, num_prototypes, 7, 7) * 10
    return torch.log((distances + 1) / (distances + 1e-4)), distances


def test_protopool_head_shapes(protopool_head):
    activations, distances = _protopool_inputs()
    output = protopool_head(
        activations,
        prototype_distances=distances,
        return_min_distances=True,
        return_proto_presence=True,
        return_similarity_score_to_each_prototype=True,
    )
    assert output["logits"].shape == (5, 4)
    assert output["min_distances"].shape == (5, 6)
    assert output["proto_presence"].shape == (4, 6, 3)
    assert output["similarity_score_to_each_prototype"].shape == (5, 6)
    assert protopool_head.class_connection_layer.weight.shape == (4, 12)
    assert protopool_head.slot_class_identity.shape == (12, 4)
    assert torch.equal(
        protopool_head.class_connection_layer.weight,
        protopool_head.slot_class_identity.T,
    )


def test_protopool_head_batch_of_one(protopool_head):
    activations, distances = _protopool_inputs(batch_size=1)
    output = protopool_head(
        activations, prototype_distances=distances, return_min_distances=True
    )
    assert output["logits"].shape == (1, 4)
    assert output["min_distances"].shape == (1, 6)


def test_protopool_head_requires_distances(protopool_head):
    activations, _ = _protopool_inputs()
    with pytest.raises(AssertionError):
        protopool_head(activations)


def test_protopool_head_softmax_when_scale_zero(protopool_head):
    protopool_head.gumbel_scale = 0
    for training in [True, False]:
        protopool_head.train(training)
        assert torch.allclose(
            protopool_head.slot_assignment(),
            torch.softmax(protopool_head.proto_presence, dim=1),
        )


def test_protopool_head_gumbel_in_training(protopool_head):
    protopool_head.gumbel_scale = 5.0
    protopool_head.train()
    first = protopool_head.slot_assignment()
    second = protopool_head.slot_assignment()
    assert torch.allclose(first.sum(dim=1), torch.ones(4, 3))
    assert not torch.allclose(first, second)


def test_protopool_head_eval_is_noise_free_softmax_over_tau(protopool_head):
    protopool_head.gumbel_scale = 5.0
    protopool_head.eval()
    expected = torch.softmax(
        protopool_head.proto_presence * 5.0 / protopool_head.tau, dim=1
    )
    assert torch.allclose(protopool_head.slot_assignment(), expected)
    assert torch.allclose(protopool_head.slot_assignment(), expected)


def test_protopool_head_hard_assignment(protopool_head):
    protopool_head.set_hard_assignment()
    assignment = protopool_head.slot_assignment()
    expected_indices = torch.argmax(protopool_head.proto_presence, dim=1)
    assert torch.equal(assignment.argmax(dim=1), expected_indices)
    assert torch.equal(assignment.sum(dim=1), torch.ones(4, 3))
    assert set(assignment.unique().tolist()) <= {0.0, 1.0}
    assert torch.equal(protopool_head.slot_prototype_indices(), expected_indices)


def test_protopool_head_matches_original_prototype_chooser(protopool_head):
    activations, distances = _protopool_inputs()
    protopool_head.gumbel_scale = 0
    with torch.no_grad():
        protopool_head.class_connection_layer.weight.normal_()

    output = protopool_head(
        activations, prototype_distances=distances, return_min_distances=True
    )
    expected_logits, expected_min_distances = _original_prototype_chooser_logits(
        distances,
        protopool_head.proto_presence,
        protopool_head.class_connection_layer.weight,
    )

    assert torch.allclose(output["logits"], expected_logits, atol=1e-5)
    assert torch.allclose(output["min_distances"], expected_min_distances)


def test_protopool_head_effective_connections_under_hard_assignment(protopool_head):
    activations, distances = _protopool_inputs()
    with torch.no_grad():
        protopool_head.class_connection_layer.weight.normal_()
    protopool_head.set_hard_assignment()

    output = protopool_head(
        activations,
        prototype_distances=distances,
        return_similarity_score_to_each_prototype=True,
    )
    effective = protopool_head.effective_class_connections()
    assert effective.shape == (4, 6)
    assert torch.allclose(
        output["logits"],
        output["similarity_score_to_each_prototype"] @ effective.T,
        atol=1e-5,
    )


def test_protopool_head_prune_remaps_slots():
    head = ProtoPoolPredictionHead(num_classes=2, num_prototypes=4, num_descriptive=2)
    head.set_hard_assignment(torch.tensor([[0, 3], [3, 2]]))
    head.prune_prototypes_by_index([1])
    assert head.num_prototypes == 3
    assert head.proto_presence.shape == (2, 3, 2)
    assert head.slot_prototype_index.tolist() == [[0, 2], [2, 1]]

    with pytest.raises(AssertionError):
        head.prune_prototypes_by_index([2])


def test_protopool_head_extra_state_round_trip(protopool_head):
    protopool_head.gumbel_scale = 42.0
    protopool_head.set_hard_assignment()
    restored = ProtoPoolPredictionHead(
        num_classes=4, num_prototypes=6, num_descriptive=3
    )
    restored.load_state_dict(protopool_head.state_dict())
    assert restored.hard_assignment
    assert restored.gumbel_scale == 42.0
    assert torch.equal(
        restored.slot_prototype_index, protopool_head.slot_prototype_index
    )
