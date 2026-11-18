"""Fashion-MNIST loaders that return the same dict batches as the other datasets."""

import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .torch_extensions import VisionDictMixin, seed_worker_for_reproducability


class DictFashionMNIST(VisionDictMixin, datasets.FashionMNIST):
    """Fashion-MNIST samples as dicts with ``img``, ``target``, and ``sample_id``."""

    pass


def balanced_split_indices(
    targets: torch.Tensor,
    per_class: Optional[int],
    val_fraction: float,
    generator: torch.Generator,
) -> tuple:
    """
    Pick the same number of images from each class, then hold out a validation slice.

    ``per_class`` of None keeps every image of that class.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")

    targets = targets.long().view(-1)
    train_indices = []
    val_indices = []
    for label in range(int(targets.max().item()) + 1):
        class_indices = torch.where(targets == label)[0]
        order = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[order]
        if per_class is not None:
            class_indices = class_indices[:per_class]

        n_val = int(round(val_fraction * class_indices.numel()))
        val_indices.extend(class_indices[:n_val].tolist())
        train_indices.extend(class_indices[n_val:].tolist())

    return train_indices, val_indices


def _channel_mean_std(dataset, indices: Sequence[int]) -> tuple:
    total = torch.zeros(3)
    squared = torch.zeros(3)
    n_pixels = 0
    for index in indices:
        image = dataset[index]["img"]
        total += image.sum(dim=(1, 2))
        squared += image.square().sum(dim=(1, 2))
        n_pixels += image.shape[1] * image.shape[2]

    mean = total / n_pixels
    variance = squared / n_pixels - mean.square()
    std = torch.sqrt(variance.clamp(min=1e-6))
    return mean.tolist(), std.tolist()


def _loader(dataset, batch_size, shuffle, generator):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        worker_init_fn=seed_worker_for_reproducability,
    )


@dataclass
class FashionMNISTSplitDataloaders:
    data_path: str
    num_classes: int
    image_size: tuple
    batch_sizes: dict
    train_loader: DataLoader
    val_loader: DataLoader
    project_loader: DataLoader
    test_loader: DataLoader


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get(
        "FASHIONMNIST_DIR", "FashionMNIST"
    ),
    train_dir: str = "train",
    val_dir: str = "val",
    project_dir: str = "project",
    image_size=(28, 28),
    batch_sizes=None,
    train_per_class: Optional[int] = None,
    test_per_class: Optional[int] = None,
    val_fraction: float = 0.2,
    seed: int = 1234,
):
    """
    Fashion-MNIST splits.

    ``train_dir``, ``val_dir``, and ``project_dir`` are unused. Fashion-MNIST is
    downloaded as official train and test archives, not as image folders.
    Grayscale digits are repeated to three channels so the existing image
    backbones can read them.

    ``train_per_class`` and ``test_per_class`` cap how many images are kept
    from each class. Leave them unset to use the full official splits.
    """
    del train_dir, val_dir, project_dir

    if batch_sizes is None:
        batch_sizes = {"train": 64, "project": 64, "val": 64, "test": 64}

    generator = torch.Generator()
    generator.manual_seed(seed)

    to_tensor = transforms.Compose(
        [
            transforms.Resize(size=(image_size[0], image_size[1])),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
        ]
    )
    raw_train = DictFashionMNIST(
        root=data_path, train=True, download=True, transform=to_tensor
    )
    train_indices, val_indices = balanced_split_indices(
        raw_train.targets,
        per_class=train_per_class,
        val_fraction=val_fraction,
        generator=generator,
    )
    mean, std = _channel_mean_std(raw_train, train_indices)
    normalize = transforms.Normalize(mean=mean, std=std)
    eval_transform = transforms.Compose([to_tensor, normalize])

    train_dataset = DictFashionMNIST(
        root=data_path, train=True, download=True, transform=eval_transform
    )
    test_dataset = DictFashionMNIST(
        root=data_path, train=False, download=True, transform=eval_transform
    )
    test_indices, _ = balanced_split_indices(
        test_dataset.targets,
        per_class=test_per_class,
        val_fraction=0.0,
        generator=generator,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

    return FashionMNISTSplitDataloaders(
        data_path=str(data_path),
        num_classes=10,
        image_size=image_size,
        batch_sizes=batch_sizes,
        train_loader=_loader(
            train_subset, batch_sizes["train"], shuffle=True, generator=generator
        ),
        val_loader=_loader(
            val_subset,
            batch_sizes["val"],
            shuffle=False,
            generator=generator,
        ),
        project_loader=_loader(
            train_subset,
            batch_sizes["project"],
            shuffle=False,
            generator=generator,
        ),
        test_loader=_loader(
            test_subset,
            batch_sizes.get("test", batch_sizes["val"]),
            shuffle=False,
            generator=generator,
        ),
    )
