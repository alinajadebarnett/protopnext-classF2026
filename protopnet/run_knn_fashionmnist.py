"""
Train a small ProtoPNet on Fashion-MNIST, then score profile kNN.

The learned prototypes stay in place. Each image is summarized by its similarity
to those prototypes, and the predicted class is the majority vote of the k
nearest training profiles. The default keeps 100 training images and 100 test
images per class, trains for 15 epochs at batch size 32, and uses k=5. k is
capped at the number of training images per class so the vote cannot be forced
to include other classes. Pass ``None`` for either cap to use the full official
split.
"""

import logging
from typing import Optional

import torch
from torch import nn

from protopnet.activations import CosPrototypeActivation
from protopnet.datasets.fashionmnist import train_dataloaders
from protopnet.embedding import AddonLayers, EmbeddedBackbone
from protopnet.models.knn_protopnet import PrototypeProfileKNN
from protopnet.models.vanilla_protopnet import VanillaProtoPNet

log = logging.getLogger(__name__)


class _TinyFashionBackbone(nn.Module):
    """Two pooling stages: a 28x28 image becomes a 32 x 7 x 7 feature map."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _linear_head_accuracy(model: nn.Module, dataloader, device: torch.device) -> float:
    """Accuracy of the trained linear head, before the profile vote."""
    was_training = model.training
    model.eval()
    n_correct = 0
    n_examples = 0
    try:
        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(device)
                target = batch["target"].to(device).view(-1).long()
                predicted = model(images)["logits"].argmax(dim=-1)
                n_correct += int((predicted == target).sum().item())
                n_examples += int(target.shape[0])
    finally:
        model.train(was_training)
    if n_examples == 0:
        raise ValueError("The dataloader produced no images.")
    return n_correct / n_examples


def run_knn_fashionmnist(
    data_path: str = "data/FashionMNIST",
    train_per_class: Optional[int] = 100,
    test_per_class: Optional[int] = 100,
    epochs: int = 15,
    k_neighbors: int = 5,
    batch_size: int = 32,
    seed: int = 1234,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_sizes = {
        "train": batch_size,
        "project": batch_size,
        "val": batch_size,
        "test": batch_size,
    }
    splits = train_dataloaders(
        data_path=data_path,
        image_size=(28, 28),
        batch_sizes=batch_sizes,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        val_fraction=0.0,
        seed=seed,
    )
    if train_per_class is not None and k_neighbors > train_per_class:
        log.info(
            "capping k from %s to %s so neighbors can stay inside one class",
            k_neighbors,
            train_per_class,
        )
        k_neighbors = train_per_class

    backbone = EmbeddedBackbone(
        _TinyFashionBackbone(), input_channels=(3, 28, 28)
    )
    add_on_layers = AddonLayers(
        num_prototypes=splits.num_classes * 2,
        input_channels=backbone.latent_dimension[0],
        num_addon_layers=0,
    )
    model = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=CosPrototypeActivation(),
        num_classes=splits.num_classes,
        num_prototypes_per_class=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in splits.train_loader:
            images = batch["img"].to(device)
            target = batch["target"].to(device).view(-1).long()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(images)["logits"], target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        log.info(
            "epoch %s/%s train_loss %.4f",
            epoch + 1,
            epochs,
            total_loss / max(n_batches, 1),
        )

    linear_accuracy = _linear_head_accuracy(model, splits.test_loader, device)
    knn = PrototypeProfileKNN(k_neighbors=k_neighbors)
    knn.fit(model, splits.project_loader, num_classes=splits.num_classes)
    metrics = knn.accuracy(model, splits.test_loader, device=device)
    metrics["linear_accuracy"] = linear_accuracy
    metrics["n_prototypes"] = int(model.prototype_layer.num_prototypes)
    metrics["k_neighbors"] = k_neighbors
    metrics["train_per_class"] = train_per_class
    metrics["test_per_class"] = test_per_class
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = run_knn_fashionmnist()
    print(
        "Fashion-MNIST profile kNN accuracy: "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['n_correct']}/{metrics['n_examples']}), "
        f"linear_head={metrics['linear_accuracy']:.4f}, "
        f"k={metrics['k_neighbors']}, "
        f"prototypes={metrics['n_prototypes']}, "
        f"train_per_class={metrics['train_per_class']}, "
        f"test_per_class={metrics['test_per_class']}"
    )


if __name__ == "__main__":
    main()
