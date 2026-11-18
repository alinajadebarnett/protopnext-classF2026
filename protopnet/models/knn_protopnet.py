"""
Replace a trained ProtoPNet's learned prototypes with one prototype per training image.

This is the kNN replacement step from Tang et al., Section 3.0.5: after the
embedding has been trained, every training sample becomes a prototype and the
linear class head is discarded. Prediction is then the vote implemented by
KNNPrototypePredictionHead.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..prediction_heads import KNNPrototypePredictionHead
from ..prototype_layers import PrototypeRandomInitMeta
from ..prototypical_part_model import ProtoPNet


def replace_learned_prototypes_with_neighbors(
    model: ProtoPNet,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    k_neighbors: int = 10,
) -> torch.Tensor:
    """
    Store one prototype per training image and install the kNN head.

    Returns the class id of each new prototype, in prototype order.
    """
    device = model.prototype_layer.prototype_tensors.device
    # Keep the spatial pooling setting the linear head was already using.
    # This is not the paper's k. k_neighbors is the paper's k.
    k_for_topk = getattr(model.prototype_prediction_head, "k_for_topk", 1)

    embeddings = []
    labels = []

    # Walk the training set once. Each image's embedding becomes one prototype,
    # which is what the paper does when it sets prototype_tensors to f(x_i).
    for batch in dataloader:
        images = batch["img"].to(device)
        latent_vectors = model.embed(images)["latent_vectors"].detach()
        embeddings.append(latent_vectors.cpu())

        target = batch["target"]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        # CUB labels are species ids. A 1D batch is expected.
        labels.append(target.detach().cpu().view(-1).long())

    if len(embeddings) == 0:
        raise ValueError("The dataloader produced no training images.")

    prototype_tensors = torch.cat(embeddings, dim=0)
    prototype_labels = torch.cat(labels, dim=0)
    if prototype_labels.shape[0] != prototype_tensors.shape[0]:
        raise ValueError("Each prototype embedding needs exactly one class id.")

    # The learned prototype parameter has one row per learned prototype, usually
    # 10 per class. The paper changes that count to one row per training image,
    # so the parameter itself has to be replaced rather than copied into.
    model.prototype_layer.prototype_tensors = nn.Parameter(
        prototype_tensors.to(device), requires_grad=False
    )
    model.prototype_layer.prototype_dimension = (
        prototype_tensors.shape[-2],
        prototype_tensors.shape[-1],
    )
    # Metadata from the learned prototypes no longer matches this list.
    model.prototype_layer._prototype_meta = [
        PrototypeRandomInitMeta(initialization_strategy="training_sample")
        for _ in range(prototype_tensors.shape[0])
    ]
    model.prototype_layer._sample_id_to_prototype_indices = {}

    # Record which species each training image belongs to. The kNN head reads
    # the same ids. The layer copy keeps the class of each prototype visible to
    # the rest of the library.
    if hasattr(model.prototype_layer, "prototype_class_identity"):
        class_identity = F.one_hot(prototype_labels, num_classes=num_classes).to(
            device=device,
            dtype=model.prototype_layer.prototype_class_identity.dtype,
        )
        model.prototype_layer.register_buffer(
            "prototype_class_identity", class_identity
        )

    model.prototype_prediction_head = KNNPrototypePredictionHead(
        prototype_labels=prototype_labels.to(device),
        num_classes=num_classes,
        k_neighbors=k_neighbors,
        k_for_topk=k_for_topk,
    ).to(device)

    return prototype_labels


@torch.no_grad()
def classification_accuracy(
    model: nn.Module,
    dataloader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Accuracy of the current prediction head.

    After the kNN replacement, this is the fraction of images whose neighbor
    vote matches the true class. ``n_correct`` and ``n_examples`` are included
    so the fraction can be checked.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    was_training = model.training
    model.eval()
    n_correct = 0
    n_examples = 0
    try:
        for batch in dataloader:
            images = batch["img"].to(device)
            target = batch["target"]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target)
            target = target.detach().to(device).view(-1).long()

            predicted = model(images)["logits"].argmax(dim=-1)
            n_correct += int((predicted == target).sum().item())
            n_examples += int(target.shape[0])
    finally:
        model.train(was_training)

    if n_examples == 0:
        raise ValueError("The dataloader produced no images.")

    return {
        "accuracy": n_correct / n_examples,
        "n_correct": n_correct,
        "n_examples": n_examples,
    }


class _NoGradPhaseMixin:
    """Eval mode and no gradients, matching the other one-step training phases."""

    @contextmanager
    def phase_settings(self, model: ProtoPNet):
        original_training_state = model.training
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            model.train(original_training_state)


@dataclass(frozen=True)
class KNNReplacePhase(_NoGradPhaseMixin):
    """
    One no-gradient step that performs the paper's kNN replacement.

    Run this after ProtoPNet training has finished. The learned prototypes and
    the linear head are not used again after this step.
    """

    dataloader: torch.utils.data.DataLoader
    num_classes: int
    k_neighbors: int = 10
    name: str = "knn_replace"

    def run_step(
        self, model: ProtoPNet, step_context
    ) -> Optional[Dict[str, Any]]:
        labels = replace_learned_prototypes_with_neighbors(
            model,
            self.dataloader,
            num_classes=self.num_classes,
            k_neighbors=self.k_neighbors,
        )
        return {"n_prototypes": int(labels.shape[0])}


def prototype_profile(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    One similarity per learned prototype.

    The paper defines this as the strongest match between a prototype and any
    patch in the image. The learned prototypes stay in place.
    """
    output = model(images, return_prototype_layer_output_dict=True)
    activations = output["prototype_activations"]
    # (batch, num_prototypes, height, width) -> (batch, num_prototypes)
    return activations.flatten(start_dim=2).max(dim=-1).values


def profile_neighbor_votes(
    query_profiles: torch.Tensor,
    train_profiles: torch.Tensor,
    train_labels: torch.Tensor,
    k_neighbors: int,
    num_classes: int,
) -> torch.Tensor:
    """
    Majority vote of the k training profiles closest in Euclidean distance.

    Returns the fraction of those neighbors in each class, shape
    (batch, num_classes). The predicted class is the largest entry.
    """
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be at least 1.")
    if train_profiles.shape[0] == 0:
        raise ValueError("The profile index has no training images.")

    # ||s_a - s_b||^2, the distance used by ProtoKNN.
    distances = torch.cdist(query_profiles, train_profiles, p=2).square()
    neighbor_k = min(k_neighbors, train_profiles.shape[0])
    _, neighbor_indices = torch.topk(distances, neighbor_k, largest=False, dim=-1)
    neighbor_labels = train_labels[neighbor_indices]
    votes = F.one_hot(neighbor_labels, num_classes=num_classes).to(
        dtype=query_profiles.dtype
    )
    return votes.mean(dim=1)


class PrototypeProfileKNN:
    """
    kNN over prototype profiles.

    Fit stores one profile and one class id per training image. The prototype
    tensors are not changed. Predict finds the k nearest stored profiles.
    """

    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.profiles: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.num_classes: Optional[int] = None

    @torch.no_grad()
    def fit(self, model: nn.Module, dataloader, num_classes: int) -> None:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        profiles = []
        labels = []
        try:
            for batch in dataloader:
                images = batch["img"].to(device)
                profiles.append(prototype_profile(model, images).cpu())
                target = batch["target"]
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                labels.append(target.detach().cpu().view(-1).long())
        finally:
            model.train(was_training)

        if len(profiles) == 0:
            raise ValueError("The dataloader produced no training images.")

        self.profiles = torch.cat(profiles, dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.num_classes = num_classes
        if self.labels.shape[0] != self.profiles.shape[0]:
            raise ValueError("Each prototype profile needs exactly one class id.")

    @torch.no_grad()
    def votes(self, model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        if self.profiles is None or self.labels is None or self.num_classes is None:
            raise RuntimeError("Call fit before predicting.")
        queries = prototype_profile(model, images)
        return profile_neighbor_votes(
            queries,
            self.profiles.to(queries.device),
            self.labels.to(queries.device),
            k_neighbors=self.k_neighbors,
            num_classes=self.num_classes,
        )

    @torch.no_grad()
    def accuracy(
        self, model: nn.Module, dataloader, device: torch.device
    ) -> Dict[str, float]:
        was_training = model.training
        model.eval()
        n_correct = 0
        n_examples = 0
        try:
            for batch in dataloader:
                images = batch["img"].to(device)
                target = batch["target"]
                if not isinstance(target, torch.Tensor):
                    target = torch.tensor(target)
                target = target.detach().to(device).view(-1).long()
                predicted = self.votes(model, images).argmax(dim=-1)
                n_correct += int((predicted == target).sum().item())
                n_examples += int(target.shape[0])
        finally:
            model.train(was_training)

        if n_examples == 0:
            raise ValueError("The dataloader produced no images.")

        return {
            "accuracy": n_correct / n_examples,
            "n_correct": n_correct,
            "n_examples": n_examples,
        }
