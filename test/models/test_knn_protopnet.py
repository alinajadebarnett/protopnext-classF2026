import torch
from torch import nn

from protopnet.datasets.fashionmnist import balanced_split_indices
from protopnet.models.knn_protopnet import (
    PrototypeProfileKNN,
    classification_accuracy,
    profile_neighbor_votes,
    replace_learned_prototypes_with_neighbors,
)
from protopnet.prediction_heads import KNNPrototypePredictionHead


def test_knn_head_votes_for_the_nearest_class():
    # Three prototypes. The first two are species 0 and the third is species 1.
    # k=2, so only the two closest prototypes vote.
    head = KNNPrototypePredictionHead(
        prototype_labels=torch.tensor([0, 0, 1]),
        num_classes=2,
        k_neighbors=2,
    )

    # One score per prototype. The two species-0 prototypes are the closest.
    activations = torch.tensor([0.9, 0.8, 0.1]).view(1, 3, 1, 1)
    logits = head(activations)["logits"]

    assert logits.shape == (1, 2)
    assert torch.allclose(logits, torch.tensor([[1.0, 0.0]]))
    assert int(logits.argmax(dim=1)) == 0


def test_knn_head_splits_a_tie_between_two_classes():
    head = KNNPrototypePredictionHead(
        prototype_labels=torch.tensor([0, 0, 1]),
        num_classes=2,
        k_neighbors=2,
    )

    # Closest prototypes are species 0 and species 1, one vote each.
    activations = torch.tensor([0.9, 0.1, 0.8]).view(1, 3, 1, 1)
    logits = head(activations)["logits"]

    assert torch.allclose(logits, torch.tensor([[0.5, 0.5]]))


def test_knn_head_does_not_average_class_ids():
    # Species ids are names. Averaging id 0 and id 2 would invent species 1.
    # The vote must leave species 1 at zero and split the vote between 0 and 2.
    head = KNNPrototypePredictionHead(
        prototype_labels=torch.tensor([0, 1, 2]),
        num_classes=3,
        k_neighbors=2,
    )

    activations = torch.tensor([0.9, 0.1, 0.8]).view(1, 3, 1, 1)
    logits = head(activations)["logits"]

    assert torch.allclose(logits, torch.tensor([[0.5, 0.0, 0.5]]))
    assert int(logits.argmax(dim=1)) != 1


def test_knn_head_uses_every_prototype_when_k_is_larger_than_the_set():
    head = KNNPrototypePredictionHead(
        prototype_labels=torch.tensor([0, 0, 1]),
        num_classes=2,
        k_neighbors=10,
    )

    activations = torch.tensor([0.3, 0.2, 0.1]).view(1, 3, 1, 1)
    logits = head(activations)["logits"]

    # All three prototypes vote: two for species 0 and one for species 1.
    assert torch.allclose(logits, torch.tensor([[2.0 / 3.0, 1.0 / 3.0]]))


class _PrototypeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Two learned prototypes, which the replacement must discard.
        self.prototype_tensors = nn.Parameter(torch.zeros(2, 4, 1, 1))
        self.register_buffer("prototype_class_identity", torch.zeros(2, 2))
        self.prototype_dimension = (1, 1)
        self._prototype_meta = []
        self._sample_id_to_prototype_indices = {}


class _FakeTrainedModel(nn.Module):
    """A stand-in that only implements the pieces the replacement reads."""

    def __init__(self, latent_batches):
        super().__init__()
        self.prototype_layer = _PrototypeLayer()
        self.prototype_prediction_head = nn.Module()
        self.prototype_prediction_head.k_for_topk = 1
        self._latent_batches = list(latent_batches)

    def embed(self, images):
        latent_vectors = self._latent_batches.pop(0)
        assert latent_vectors.shape[0] == images.shape[0]
        return {"latent_vectors": latent_vectors}


def test_replacement_stores_one_prototype_per_training_image():
    latent_batches = [
        torch.ones(2, 4, 2, 2),
        torch.zeros(1, 4, 2, 2),
    ]
    model = _FakeTrainedModel(latent_batches)
    # Two batches, three training images, species ids 0, 1, and 0.
    dataloader = [
        {"img": torch.zeros(2, 3, 8, 8), "target": torch.tensor([0, 1])},
        {"img": torch.zeros(1, 3, 8, 8), "target": torch.tensor([0])},
    ]

    labels = replace_learned_prototypes_with_neighbors(
        model, dataloader, num_classes=2, k_neighbors=10
    )

    assert labels.tolist() == [0, 1, 0]
    assert model.prototype_layer.prototype_tensors.shape == (3, 4, 2, 2)
    assert model.prototype_layer.prototype_dimension == (2, 2)
    assert model.prototype_prediction_head.prototype_labels.tolist() == [0, 1, 0]
    assert model.prototype_prediction_head.k_neighbors == 10
    assert model.prototype_layer.prototype_class_identity.shape == (3, 2)
    assert torch.equal(
        model.prototype_layer.prototype_class_identity.cpu(),
        torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32),
    )
    # The paper does not keep training the prototypes after this swap.
    assert model.prototype_layer.prototype_tensors.requires_grad is False


class _PredictsFirstClass(nn.Module):
    def forward(self, x, **kwargs):
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 0] = 1
        return {"logits": logits}


def test_classification_accuracy_counts_correct_predictions():
    model = _PredictsFirstClass()
    loader = [
        {"img": torch.zeros(2, 1, 4, 4), "target": torch.tensor([0, 1])},
        {"img": torch.zeros(1, 1, 4, 4), "target": torch.tensor([0])},
    ]

    metrics = classification_accuracy(model, loader)

    assert metrics["n_examples"] == 3
    assert metrics["n_correct"] == 2
    assert metrics["accuracy"] == 2 / 3


def test_balanced_split_keeps_the_same_count_per_class():
    targets = torch.arange(10).repeat(5)
    generator = torch.Generator().manual_seed(0)

    train_indices, val_indices = balanced_split_indices(
        targets, per_class=4, val_fraction=0.25, generator=generator
    )

    assert len(train_indices) == 30
    assert len(val_indices) == 10
    assert len(set(train_indices).intersection(val_indices)) == 0


def test_profile_vote_uses_the_closer_prototype_profile():
    train_profiles = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    train_labels = torch.tensor([0, 1])
    query = torch.tensor([[0.9, 0.1]])

    votes = profile_neighbor_votes(
        query, train_profiles, train_labels, k_neighbors=1, num_classes=2
    )

    assert torch.allclose(votes, torch.tensor([[1.0, 0.0]]))


def test_profile_vote_is_a_majority_of_the_neighbors():
    train_profiles = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_labels = torch.tensor([0, 0, 0, 1, 1])
    query = torch.tensor([[1.0, 0.0]])

    votes = profile_neighbor_votes(
        query, train_profiles, train_labels, k_neighbors=5, num_classes=2
    )

    assert torch.allclose(votes, torch.tensor([[0.6, 0.4]]))
    assert int(votes.argmax(dim=1)) == 0


class _ProfileModel(nn.Module):
    """Returns a two-prototype activation taken from the first pixel."""

    def __init__(self):
        super().__init__()
        self.prototype_layer = nn.Module()
        self.prototype_layer.prototype_tensors = nn.Parameter(torch.ones(2, 1, 1, 1))
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, images, return_prototype_layer_output_dict=False):
        value = images.reshape(images.shape[0], -1)[:, 0]
        activations = torch.stack([value, 1 - value], dim=1).view(-1, 2, 1, 1)
        return {"prototype_activations": activations, "logits": activations.view(-1, 2)}


def test_profile_knn_keeps_the_learned_prototypes():
    model = _ProfileModel()
    prototypes_before = model.prototype_layer.prototype_tensors.detach().clone()
    train_loader = [
        {"img": torch.ones(1, 1, 2, 2), "target": torch.tensor([0])},
        {"img": torch.zeros(1, 1, 2, 2), "target": torch.tensor([1])},
    ]
    test_loader = [
        {"img": torch.full((1, 1, 2, 2), 0.8), "target": torch.tensor([0])},
    ]

    knn = PrototypeProfileKNN(k_neighbors=1)
    knn.fit(model, train_loader, num_classes=2)
    metrics = knn.accuracy(model, test_loader, device=torch.device("cpu"))

    assert torch.equal(
        model.prototype_layer.prototype_tensors.detach(), prototypes_before
    )
    assert knn.profiles.shape == (2, 2)
    assert knn.labels.tolist() == [0, 1]
    assert metrics["n_correct"] == 1
    assert metrics["accuracy"] == 1.0
