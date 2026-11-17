# ====== MODEL AND DATA LOADING ======
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torchmetrics import Metric


# ======== HELPER FUNCTIONS ========
def in_bbox(loc, bbox):
    return (
        loc[0] >= bbox[0]
        and loc[0] <= bbox[1]
        and loc[1] >= bbox[2]
        and loc[1] <= bbox[3]
    )


def add_gaussian_noise(norm_img, generator, std=0.2, eps=0.25):
    """
    Adds noise to an image tensor, with constraints on the noise's magnitude.

    Parameters:
    - norm_img (torch.Tensor): Normalized image tensor.
    - std (float): Standard deviation of the Gaussian noise.
    - eps (float): Maximum absolute value of the noise elements.
    - generator (torch.Generator): Generator for random numbers, specifying device.

    Returns:
    - torch.Tensor: Perturbed image.
    """
    # Create Gaussian noise with the specified standard deviation using the provided generator
    noise = torch.zeros(norm_img.shape, device=norm_img.device).normal_(
        mean=0, std=std, generator=generator
    )
    # Constrain the noise to be within the specified epsilon limits
    noise = torch.clamp(noise, min=-eps, max=eps)
    # Add the constrained noise to the original image
    perturbed_img = norm_img + noise
    return perturbed_img

def smoothed_edf(outcomes: torch.Tensor, groups: torch.Tensor,
                 num_classes: int, concentration: float = 1.0) -> torch.Tensor:
    """
    Compute the smoothed empirical distribution function (EDF) for fairness metrics (Foulds et al., 2019).
    For outcomes y and groups s: P(y|s) = (N_{y,s} + α) / (N_s + concentration)
    where α = concentration / num_classes. 
    The smoothed EDF is then computed as the maximum difference in log probabilities across groups.
    
    Parameters:
    - outcomes (torch.Tensor): Tensor of model predictions or true labels.
    - groups (torch.Tensor): Tensor indicating the group membership of each sample.
    - num_classes (int): Number of classes in the classification task.
    - concentration (float): Concentration parameter for smoothing. Default is 1.0.
    """

    group_ids, inverse = torch.unique(groups, return_inverse=True)
    counts = torch.zeros(len(group_ids), num_classes,
                         dtype=torch.float64, device=outcomes.device)
    
    # Count occurrences of each outcome for each group
    counts.index_put_(
        (inverse, outcomes.long()),
        torch.ones_like(outcomes, dtype=torch.float64),
        accumulate=True,
    )
    alpha = concentration / num_classes
    probs = (counts + alpha) / (counts.sum(dim=1, keepdim=True) + concentration)
    log_p = probs.log()
    
    return (log_p.max(dim=0).values - log_p.min(dim=0).values).max()


# ============ METRICS ============
class InterpMetrics(Metric):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        """
        Base class for PartStabilityScore and PartConsistencyScore

        Parameters:
        - num_classes (int): The number of classes in the dataset.
        - part_num (int): The number of distinct parts in the dataset.
        - proto_per_class (int): The number of prototypes per class.
        - img_sz (int): Input image size to the network, default 224.
        - half_size (int): half of size of the prototype bounding box, defaults to 36.
        - dist_sync_on_step (bool): torchmetrics parameter.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.half_size = half_size
        self.proto_per_class = proto_per_class
        self.img_sz = img_sz
        self.part_num = part_num
        self.uncropped = uncropped

    def filter_proto_acts(self, proto_acts, targets):
        """
        Select correct prototype activations according to ground truth labels
        output shape is batch_size, prototypes_per_class, feature_size, feature_size
        """
        feature_size = proto_acts.shape[-1]
        proto_acts_reshaped = proto_acts.view(
            proto_acts.shape[0],
            self.num_classes,
            self.proto_per_class,
            feature_size,
            feature_size,
        )
        proto_acts_selected = proto_acts_reshaped[
            torch.arange(proto_acts_reshaped.size(0)), targets
        ]
        return proto_acts_selected, targets

    def proto2part_and_masks(
        self,
        all_proto_acts,
        all_targets,
        all_sample_parts_centroids,
        all_sample_bounding_box,
    ):
        all_proto_acts = torch.cat(all_proto_acts, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_sample_bounding_box = torch.cat(all_sample_bounding_box, dim=0)

        # Enumerate all the classes, thus enumerate all the prototypes
        all_proto_to_part, all_proto_part_mask = [], []
        # `all_proto_to_part` each element indicates the corresponding object parts of a prototype on the images.
        # `all_proto_part_mask`each element indicates the existing (non-masked) object parts on the images of a prototype.
        for test_image_label in range(self.num_classes):
            arr_ids = torch.argwhere(all_targets == test_image_label).flatten().tolist()
            if len(arr_ids) == 0:
                continue
            class_proto_acts = all_proto_acts[arr_ids]
            selected_parts_centroids = [all_sample_parts_centroids[i] for i in arr_ids]
            selected_bboxes = all_sample_bounding_box[arr_ids]

            # Get part annotations on all the images of current class
            class_part_labels, class_part_masks = [], []
            """
            `class_part_labels` save the part labels of images in this class.
            `class_part_masks` save the part masks of images in this class.
            """
            for idx in range(len(class_proto_acts)):
                # Get part annotations
                part_labels, part_mask = [], torch.zeros((self.part_num,))
                bbox = selected_bboxes[idx]
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                parts_centroids = selected_parts_centroids[idx]
                for part_centroid in parts_centroids:
                    part_id = int(
                        (part_centroid[0] - 1).item()
                    )  # The id of current object part (begin from 0)
                    part_mask[
                        part_id
                    ] = 1  # The current object part exists in current image
                    if self.uncropped:
                        part_labels.append(
                            [
                                part_id,
                                int(part_centroid[1] * self.img_sz),
                                int(part_centroid[2] * self.img_sz),
                            ]
                        )
                    else:
                        centroid_x, centroid_y = (
                            part_centroid[1] - bbox_x1,
                            part_centroid[2] - bbox_y1,
                        )
                        ratio_x, ratio_y = centroid_x / (
                            bbox_x2 - bbox_x1
                        ), centroid_y / (
                            bbox_y2 - bbox_y1
                        )  # Fit the bounding boxes' coordinates to the cropped images
                        re_centroid_x, re_centroid_y = int(self.img_sz * ratio_x), int(
                            self.img_sz * ratio_y
                        )
                        part_labels.append([part_id, re_centroid_x, re_centroid_y])

                class_part_labels.append(part_labels)
                class_part_masks.append(part_mask)

            for proto_idx in range(self.proto_per_class):
                img_num = len(class_proto_acts)
                proto_to_part = torch.zeros(
                    (img_num, self.part_num)
                )  # Element = 1 -> the prototype corresponds to this object part on this image, element = 0 otherwise
                for img_idx in range(img_num):
                    part_labels = class_part_labels[
                        img_idx
                    ]  # Get the part labels of current image
                    activation_map = class_proto_acts[
                        img_idx, proto_idx
                    ]  # Get the activation map of current prototype on current image
                    activation_map = activation_map.unsqueeze(0).unsqueeze(0)
                    upsampled_activation_map = (
                        transforms.Resize(
                            size=(self.img_sz, self.img_sz),
                            interpolation=transforms.InterpolationMode.BICUBIC,
                        )(activation_map)
                        .squeeze(0)
                        .squeeze(0)
                    )

                    max_value = upsampled_activation_map.max()
                    max_indices = (upsampled_activation_map == max_value).nonzero(
                        as_tuple=True
                    )

                    # to deal with a very unlikely (unless artificial) special case
                    # where max is not unique
                    if len(max_indices[0]) > 1 and len(max_indices[1]) > 1:
                        max_indices = (max_indices[0][:1], max_indices[0][:1])

                    region_pred = (
                        max(0, max_indices[0] - self.half_size),
                        min(self.img_sz, max_indices[0] + self.half_size),
                        max(0, max_indices[1] - self.half_size),
                        min(self.img_sz, max_indices[1] + self.half_size),
                    )  # Get the corresponding region of current prototype, (y1, y2, x1, x2)
                    # Get the corresponding object parts of current prototype
                    for part_label in part_labels:
                        part_id, centroid_x_gt, centroid_y_gt = (
                            part_label[0],
                            part_label[1],
                            part_label[2],
                        )
                        if in_bbox((centroid_y_gt, centroid_x_gt), region_pred):
                            proto_to_part[img_idx, part_id] = 1

                # class_part_masks = torch.stack(class_part_masks)
                all_proto_to_part.append(proto_to_part)
                all_proto_part_mask.append(class_part_masks)

        all_proto_part_mask = [torch.stack(o) for o in all_proto_part_mask]
        return all_proto_to_part, all_proto_part_mask


class PartConsistencyScore(InterpMetrics):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        part_thresh=0.8,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        super().__init__(
            num_classes=num_classes,
            part_num=part_num,
            proto_per_class=proto_per_class,
            img_sz=img_sz,
            half_size=half_size,
            dist_sync_on_step=dist_sync_on_step,
            uncropped=uncropped,
        )
        self.part_thresh = part_thresh
        self.add_state("all_proto_acts", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_parts_centroids", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_bounding_box", default=[], dist_reduce_fx="cat")

    def update(self, proto_acts, targets, sample_parts_centroids, sample_bounding_box):
        batch_proto_acts, batch_targets = self.filter_proto_acts(proto_acts, targets)
        self.all_proto_acts.append(batch_proto_acts)
        self.all_targets.append(batch_targets)
        self.all_sample_parts_centroids.extend(sample_parts_centroids)
        self.all_sample_bounding_box.append(sample_bounding_box)

    def compute(self):
        all_proto_to_part, all_proto_part_mask = self.proto2part_and_masks(
            self.all_proto_acts,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )
        all_proto_consis = []
        # Enumerate all the prototypes to calculate consistency score
        for proto_idx in range(len(all_proto_to_part)):
            proto_to_part = all_proto_to_part[proto_idx]
            proto_part_mask = all_proto_part_mask[proto_idx]
            assert (
                (1.0 - proto_part_mask) * proto_to_part
            ).sum() == 0  # Assert that the prototype does not correspond to an object part that cannot be visualized (not in the part annotations)
            proto_to_part_sum = proto_to_part.sum(axis=0)
            proto_part_mask_sum = proto_part_mask.sum(axis=0)
            proto_part_mask_sum = torch.where(
                proto_part_mask_sum == 0, proto_part_mask_sum + 1, proto_part_mask_sum
            )  # Eliminate the 0 elements in all_part_mask_sum~(prevent 0 from being denominator), it doesn't affect the evaluation result
            mean_part_float = proto_to_part_sum / proto_part_mask_sum
            mean_part = (
                mean_part_float >= self.part_thresh
            ).float()  # The prototope is determined to be non-consistent if  no element in the averaged corresponding object parts exceeds `part_thresh`

            if mean_part.sum() == 0:
                all_proto_consis.append(0.0)
            else:
                all_proto_consis.append(1.0)

        all_proto_consis = torch.tensor(all_proto_consis).float()
        consistency_score = all_proto_consis.mean()

        return consistency_score


class PartStabilityScore(InterpMetrics):
    def __init__(
        self,
        num_classes,
        part_num,
        proto_per_class,
        img_sz=224,
        half_size=36,
        dist_sync_on_step=False,
        uncropped=True,
    ):
        super().__init__(
            num_classes=num_classes,
            part_num=part_num,
            proto_per_class=proto_per_class,
            img_sz=img_sz,
            half_size=half_size,
            dist_sync_on_step=dist_sync_on_step,
            uncropped=uncropped,
        )

        self.add_state("all_proto_acts", default=[], dist_reduce_fx="cat")
        self.add_state("all_proto_acts_noisy", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_parts_centroids", default=[], dist_reduce_fx="cat")
        self.add_state("all_sample_bounding_box", default=[], dist_reduce_fx="cat")

    def update(
        self,
        proto_acts,
        proto_acts_noisy,
        targets,
        sample_parts_centroids,
        sample_bounding_box,
    ):
        batch_proto_acts, batch_targets = self.filter_proto_acts(proto_acts, targets)
        self.all_proto_acts.append(batch_proto_acts)
        self.all_targets.append(batch_targets)
        self.all_sample_parts_centroids.extend(sample_parts_centroids)
        self.all_sample_bounding_box.append(sample_bounding_box)

        batch_proto_acts_noisy, _ = self.filter_proto_acts(proto_acts_noisy, targets)
        self.all_proto_acts_noisy.append(batch_proto_acts_noisy)

    def compute(self):
        all_proto_to_part, _ = self.proto2part_and_masks(
            self.all_proto_acts,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )
        all_proto_to_part_noisy, _ = self.proto2part_and_masks(
            self.all_proto_acts_noisy,
            self.all_targets,
            self.all_sample_parts_centroids,
            self.all_sample_bounding_box,
        )

        all_proto_stability = []
        for proto_idx in range(len(all_proto_to_part)):
            proto_to_part = all_proto_to_part[proto_idx]
            proto_to_part_noise = all_proto_to_part_noisy[proto_idx]
            # Determine whether the elements in `proto_to_part` and `proto_to_part_perturb` are equal
            difference_sum = torch.abs(proto_to_part - proto_to_part_noise).sum(dim=-1)
            is_equal = difference_sum == 0
            is_equal = is_equal.float()
            proto_stability = (
                is_equal.mean()
            )  # The ratio of elements that keep unchanged under perturbation
            all_proto_stability.append(proto_stability)

        all_proto_stability = torch.tensor(all_proto_stability).float()
        stability_score = all_proto_stability.mean()

        return stability_score


class FairnessMetrics(Metric):
    """
    Base class for fairness metrics.
    This class wraps the torchmetrics.Metric class and provides a common interface for fairness metrics. 

    A positive outcome is defined as a correct prediction. 

    Attributes:
    - num_classes (int): The number of classes in the dataset.
    - dist_sync_on_step (bool): torchmetrics parameter.
    """

    full_state_update = False

    def __init__(self, num_classes, dist_sync_on_step=False):
        """
        Parameters:
        - num_classes (int): The number of classes in the dataset.
        - dist_sync_on_step (bool): torchmetrics parameter.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, num_classes = num_classes)
        self.add_state("all_preds", default=[], dist_reduce_fx="cat")
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")

    def update(self, logits, targets):
        # Shared by all subclasses; override only if a metric needs different state.
        self.all_preds.append(logits.argmax(dim=1))
        self.all_targets.append(targets)

    def compute(self):
        raise NotImplementedError("Subclasses must implement compute().")
    
    def get_group_acc(self):
        predictions = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)

        priv_classes = self.privileged_classes.to(targets.device)
        unpriv_classes = self.unprivileged_classes.to(targets.device)

        is_privileged = torch.isin(targets, priv_classes)
        is_unprivileged = torch.isin(targets, unpriv_classes)

        if is_privileged.sum() == 0 or is_unprivileged.sum() == 0:
            nan_val = torch.tensor(float("nan"), device=predictions.device)
            return nan_val, nan_val, nan_val, predictions
        
        correct_all = (predictions == targets).double()
        acc_priv = correct_all[is_privileged].mean()
        acc_unpriv = correct_all[is_unprivileged].mean()

        return correct_all, acc_priv, acc_unpriv, predictions


class DisparateImpact(FairnessMetrics):
    """
    A class for a disparate impact metrics. Child of FairnessMetrics. 

    Disparate impact (DI) is defined as:
    acc(unprivileged) / acc(privileged). (Feldman et al., 2014) 
    Fair value = 1, 
    while values less than 1 indicate bias against the unprivileged group, 
    and values greater than 1 indicate bias against the privileged group.

    Attributes:
    - num_classes (int): 
    - priviledged_classes (list): numerical list of the priviledged classes, corresponding to the class lables
    - unprivileged_classes (list): numerical list of the unpriviledged classes, corresponding to the class lables
    - dist_sync_on_step (bool): torchmetrics parameter. 
    """

    def __init__(

        self,
        num_classes,
        privileged_classes,
        unprivileged_classes,
        dist_sync_on_step=False,
    ):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step)
        self.register_buffer("privileged_classes", torch.tensor(privileged_classes))
        self.register_buffer("unprivileged_classes", torch.tensor(unprivileged_classes))

    def update(self, logits, targets):
        super().update(logits, targets)

    def compute(self):
        _, acc_priv, acc_unpriv, predictions = self.get_group_acc()

        if torch.isnan(acc_priv) or torch.isnan(acc_unpriv):
            return torch.tensor(float("nan"), device=predictions.device)

        if acc_priv > 0:
            return (acc_unpriv / acc_priv).item()
        else:
            return torch.tensor(float("nan"), device=predictions.device)
        


class EqualOpportunityDifference(FairnessMetrics):
    """
    A class for a equal opportunity difference metrics. Child of FairnessMetrics.

    acc(unprivileged) - acc(privileged). Fair value = 0. (Hardt et al., 2016)
    Positive values indicate bias against the privileged group, 
    while negative values indicate bias against the unprivileged group.

    Attributes:
    - num_classes (int): 
    - priviledged_classes (list): numerical list of the priviledged classes, corresponding to the class lables
    - unprivileged_classes (list): numerical list of the unpriviledged classes, corresponding to the class lables
    - dist_sync_on_step (bool): torchmetrics parameter. 
    """

    def __init__(
        self,
        num_classes,
        privileged_classes,
        unprivileged_classes,
        dist_sync_on_step=False,
    ):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step)
        self.register_buffer("privileged_classes", torch.tensor(privileged_classes))
        self.register_buffer("unprivileged_classes", torch.tensor(unprivileged_classes))

    def update(self, logits, targets):
        super().update(logits, targets)

    def compute(self):
        _, acc_priv, acc_unpriv, predictions = self.get_group_acc()
        if torch.isnan(acc_priv) or torch.isnan(acc_unpriv):
            return torch.tensor(float("nan"), device=predictions.device)
        else:
            return (acc_unpriv - acc_priv).item()


class DifferentialFairness(FairnessMetrics):
    """
    (Islam et al., 2020)
    Differential fairness (DF) is defined as:
    Smoothed empirical differential fairness epsilon (see smoothed_edf()) of the correct/incorrect
    outcome across privileged vs. unprivileged class groups. Fair value = 0.

    Differential fairness is symmetrical, meaning it does not indicate which group is favored. 
    
    Attributes:
    - num_classes (int): The number of classes
    - priviledged_classes (list): numerical list of the priviledged classes, corresponding to the class lables
    - unprivileged_classes (list): numerical list of the unpriviledged classes, corresponding to the class lables
    - concentration (float): smoothing parameter for smoothed_edf().
    - dist_sync_on_step (bool): torchmetrics parameter. 
    """

    def __init__(
        self,
        num_classes,
        privileged_classes,
        unprivileged_classes,
        concentration=1.0,
        dist_sync_on_step=False,
    ):
        super().__init__(num_classes=num_classes, dist_sync_on_step=dist_sync_on_step)
        self.concentration = concentration
        self.register_buffer("privileged_classes", torch.tensor(privileged_classes))
        self.register_buffer("unprivileged_classes", torch.tensor(unprivileged_classes))

    def update(self, logits, targets):
        super().update(logits, targets)

    def compute(self):
        if len(self.all_preds) == 0:
            return torch.tensor(float("nan"))

        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)

        priv_classes = self.privileged_classes.to(targets.device)
        unpriv_classes = self.unprivileged_classes.to(targets.device)

        is_privileged = torch.isin(targets, priv_classes)
        is_unprivileged = torch.isin(targets, unpriv_classes)

        if not (is_privileged.any() and is_unprivileged.any()):
            return torch.tensor(float("nan"))

        # Drop samples whose class is in neither group
        # TODO: if not priv, unpriv? 
        keep = is_privileged | is_unprivileged
        outcomes = (preds == targets)[keep].long()   # 1 = correct
        groups = is_privileged[keep].long()          # 1 = privileged

        return smoothed_edf(
            outcomes, groups, num_classes=2, concentration=self.concentration
        )


