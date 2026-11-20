import matplotlib

# The tests draw figures, some in background threads (the factory's analyses); the macOS display
# backend aborts Python there, so use the non-display backend, as on a headless cluster.
matplotlib.use("Agg")

import torch  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402


class TensorDatasetDict(TensorDataset):
    """TensorDataset that yields protopnext-style dict samples."""

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {"img": img, "target": target, "sample_id": index}


# ---- shared by the metric and figure tests: 8x8 activation maps and a last layer

H = W = 8


class LastLayerHead(torch.nn.Module):
    """A prediction head holding only the last layer, with the given (classes, prototypes) weight."""

    def __init__(self, weight):
        super().__init__()
        self.class_connection_layer = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False
        )
        self.class_connection_layer.weight.data = weight


def act_map(rows, cols):
    a = torch.zeros(H, W)
    a[rows, cols] = 1.0
    return a


# 4 of 64 pixels active (just over the top 5%), inside / outside a 4x4 lesion in the top-left corner
INSIDE = act_map(slice(0, 2), slice(0, 2))
OUTSIDE = act_map(slice(6, 8), slice(6, 8))


# ---- figure stubs: prototype j's activation map is channel j of the image


class ChannelsAsActivations(torch.nn.Module):
    """Prototype j's activation map is channel j of the input image."""

    def __init__(self, weight):
        super().__init__()
        self.prototype_prediction_head = LastLayerHead(weight)

    def forward(self, img, return_prototype_layer_output_dict=False):
        return {"prototype_activations": img}


class Dataset(list):
    """A list of items, optionally with .classes like an ImageFolder."""

    def __init__(self, items, classes=None):
        super().__init__(items)
        if classes is not None:
            self.classes = classes


def lesion_mask():
    lesion = torch.zeros(H, W, dtype=torch.bool)
    lesion[:4, :4] = True
    return lesion


def item(channels, target, mask=None):
    if mask is None:
        mask = (~lesion_mask()).float().unsqueeze(0)  # 0 on the lesion
    return {
        "img": torch.stack(channels),
        "target": torch.tensor(target),
        "fine_annotation": mask,
    }


# Prototypes 0 and 1 belong to class 0, prototype 2 to class 1.
W_BEFORE = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


# The edit removed prototype 1 (its last-layer column is now all zeros).
W_AFTER = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


BEFORE, AFTER = ChannelsAsActivations(W_BEFORE), ChannelsAsActivations(W_AFTER)


ITEMS = [
    item([INSIDE, OUTSIDE, OUTSIDE], 0),  # image 0: removing proto 1 helps
    item([OUTSIDE, INSIDE, OUTSIDE], 0),  # image 1: removing proto 1 hurts
    item([INSIDE, INSIDE, INSIDE], 1),  # image 2: class 1 was not edited
]


DS = Dataset(ITEMS, classes=["cat", "dog"])
