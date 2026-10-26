"""Overlap and per-pixel losses.

Every class is constructed directly. ``sigmoid=True`` (the default) reads
logits. ``reduction`` is ``"mean"``, ``"sum"``, or ``"none"``.
"""

import warnings
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import masked_reduction, prepare_pair, reduce_loss
from .registry import register


def _check_reduction(reduction: str) -> str:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}.")
    return reduction


def _activate(prediction: torch.Tensor, sigmoid: bool, softmax: bool, other_act: Callable | None) -> torch.Tensor:
    if sigmoid:
        return torch.sigmoid(prediction)
    if softmax:
        if prediction.shape[1] == 1:
            warnings.warn("single channel prediction, softmax=True is ignored.", stacklevel=3)
            return prediction
        return torch.softmax(prediction, dim=1)
    if other_act is not None:
        return other_act(prediction)
    return prediction


def _one_hot(target: torch.Tensor, classes: int) -> torch.Tensor:
    """Label map ``(N, 1, ...)`` of class indices to ``(N, C, ...)``."""
    if target.shape[1] != 1:
        raise ValueError(
            f"to_onehot_y=True expects one channel of class indices, got shape {tuple(target.shape)}."
        )
    index = target[:, 0].long()
    if int(index.min()) < 0 or int(index.max()) >= classes:
        raise ValueError(
            f"Class index {int(index.min())}..{int(index.max())} does not fit {classes} channels."
        )
    encoded = F.one_hot(index, num_classes=classes)
    return encoded.movedim(-1, 1).to(dtype=target.dtype)


def _overlap(prediction, target, axes, squared: bool, soft_label: bool):
    """True positive, false positive, and false negative, summed on ``axes``.

    Hard labels use the voxel sums behind Dice. ``squared=True`` squares the
    prediction in those sums. ``soft_label=True`` follows the soft-label Dice
    of Wang et al. (MICCAI 2023): the gap between prediction and target
    weights the false positive and false negative terms.
    """
    if soft_label:
        gap = (prediction - target).abs()
        if squared:
            gap = gap.square()
        true_positive = (prediction * target).sum(dim=axes)
        false_positive = (prediction * gap).sum(dim=axes)
        false_negative = (target * gap).sum(dim=axes)
        return true_positive, false_positive, false_negative

    scores = prediction.square() if squared else prediction
    true_positive = (scores * target).sum(dim=axes)
    false_positive = (scores * (1.0 - target)).sum(dim=axes)
    false_negative = ((1.0 - scores) * target).sum(dim=axes)
    return true_positive, false_positive, false_negative


@register
class DiceLoss(nn.Module):
    """Soft Dice between a prediction and a mask.

    The default is binary logits: sigmoid, then one Dice per image, then the
    mean. That value matches ``1 - (2 * intersection + smooth) / (sum + smooth)``.

    Multi-class maps keep their channel axis, ``(N, C, H, W)`` or
    ``(N, C, D, H, W)``. Set ``softmax=True`` for exclusive classes and
    ``to_onehot_y=True`` when ``target`` stores a class index in one channel.
    ``sigmoid=True`` with several channels is multi-label. A tensor of rank 3
    is still a batch of single-channel frames, the same rule as the other losses.
    """

    name = "dice"
    spatial = (2, 3)

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth: float = 1e-6,
        smooth_nr: float | None = None,
        smooth_dr: float | None = None,
        batch: bool = False,
        weight: float | list | tuple | torch.Tensor | None = None,
        soft_label: bool = False,
        ignore_index: int | None = None,
    ):
        super().__init__()
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be callable, got {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Set only one of sigmoid, softmax, and other_act.")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.reduction = _check_reduction(reduction)
        self.smooth_nr = float(smooth if smooth_nr is None else smooth_nr)
        self.smooth_dr = float(smooth if smooth_dr is None else smooth_dr)
        self.batch = batch
        self.soft_label = soft_label
        self.ignore_index = ignore_index
        if weight is None:
            self.class_weight = None
        else:
            class_weight = torch.as_tensor(weight, dtype=torch.float32)
            if torch.any(class_weight < 0):
                raise ValueError("class weights must be >= 0.")
            self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = prepare_pair(logits, target)
        prediction = _activate(logits, self.sigmoid, self.softmax, self.other_act)

        valid = None
        if self.ignore_index is not None:
            if target.shape[1] != 1:
                raise ValueError(
                    "ignore_index applies to a one-channel label map. "
                    f"Target shape is {tuple(target.shape)}."
                )
            valid = target != self.ignore_index
            target = torch.where(valid, target, torch.zeros_like(target))

        if self.to_onehot_y:
            if prediction.shape[1] == 1:
                warnings.warn("single channel prediction, to_onehot_y=True is ignored.", stacklevel=2)
            else:
                target = _one_hot(target, prediction.shape[1])

        if not self.include_background:
            if prediction.shape[1] == 1:
                warnings.warn("single channel prediction, include_background=False is ignored.", stacklevel=2)
            else:
                prediction = prediction[:, 1:]
                target = target[:, 1:]

        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}. "
                "For class indices set to_onehot_y=True."
            )

        if valid is not None:
            prediction = prediction * valid
            target = target * valid

        axes = list(range(2, prediction.ndim))
        if self.batch:
            axes = [0, *axes]
        true_positive, false_positive, false_negative = _overlap(
            prediction, target, axes, self.squared_pred, self.soft_label
        )
        # Halving the errors makes the Dice denominator equal the sum of the
        # prediction and the target. Jaccard keeps both errors, which is IoU.
        if not self.jaccard:
            false_positive = false_positive * 0.5
            false_negative = false_negative * 0.5
        numerator = 2.0 * true_positive + self.smooth_nr
        denominator = 2.0 * (true_positive + false_positive + false_negative) + self.smooth_dr
        loss = 1.0 - numerator / denominator

        channels = target.shape[1]
        if self.class_weight is not None and channels != 1:
            weights = self.class_weight.to(device=loss.device, dtype=loss.dtype)
            if weights.numel() == 1:
                weights = weights.reshape(1).expand(channels)
            if weights.shape[0] != channels:
                raise ValueError(
                    f"class weight has length {weights.shape[0]}, the loss has {channels} channels. "
                    "With include_background=False, omit the background weight."
                )
            loss = loss * weights
        return reduce_loss(loss, self.reduction)


@register
class BCELoss(nn.Module):
    """Binary cross-entropy. Several channels are several labels, not exclusive classes.

    ``sigmoid=True`` uses the logits form. ``pos_weight`` is the positive-class
    multiplier from ``binary_cross_entropy_with_logits`` and needs logits.
    ``ignore_index`` drops voxels whose label equals that value.
    """

    name = "bce"
    spatial = (2, 3)

    def __init__(
        self,
        sigmoid: bool = True,
        reduction: str = "mean",
        pos_weight: float | list | tuple | torch.Tensor | None = None,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = _check_reduction(reduction)
        self.ignore_index = ignore_index
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32).reshape(-1))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = prepare_pair(logits, target)
        if logits.shape[1] != target.shape[1]:
            raise ValueError(
                f"BCE target shape {tuple(target.shape)} does not match prediction shape {tuple(logits.shape)}. "
                "Exclusive classes belong in DiceLoss(softmax=True, to_onehot_y=True)."
            )
        valid = None
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            target = torch.where(valid, target, torch.zeros_like(target))
        if self.sigmoid:
            pos_weight = None
            if self.pos_weight is not None:
                pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
                if pos_weight.numel() == 1 and logits.shape[1] != 1:
                    pos_weight = pos_weight.expand(logits.shape[1])
                if pos_weight.numel() != logits.shape[1]:
                    raise ValueError(
                        f"pos_weight has length {pos_weight.numel()}, the prediction has {logits.shape[1]} channels."
                    )
            per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
        else:
            if self.pos_weight is not None:
                raise ValueError("pos_weight applies to logits. Keep sigmoid=True, or drop pos_weight.")
            per_voxel = F.binary_cross_entropy(logits, target, reduction="none")
        return masked_reduction(per_voxel, valid, self.reduction)


@register
class FocalLoss(nn.Module):
    """RetinaNet focal loss on logits, or on probabilities when ``sigmoid=False``.

    ``alpha`` scales the loss and ``gamma`` down-weights easy voxels.
    ``ignore_index`` drops voxels whose label equals that value.
    """

    name = "focal"
    spatial = (2, 3)

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        sigmoid: bool = True,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.sigmoid = sigmoid
        self.reduction = _check_reduction(reduction)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = prepare_pair(logits, target)
        if logits.shape[1] != target.shape[1]:
            raise ValueError(
                f"Focal target shape {tuple(target.shape)} does not match prediction shape {tuple(logits.shape)}."
            )
        valid = None
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            target = torch.where(valid, target, torch.zeros_like(target))
        if self.sigmoid:
            per_voxel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            probabilities = torch.sigmoid(logits)
        else:
            per_voxel = F.binary_cross_entropy(logits, target, reduction="none")
            probabilities = logits
        probability_of_truth = torch.where(target > 0.5, probabilities, 1.0 - probabilities)
        per_voxel = self.alpha * (1.0 - probability_of_truth).pow(self.gamma) * per_voxel
        return masked_reduction(per_voxel, valid, self.reduction)


@register
class CombinedLoss(nn.Module):
    """``bce_weight * BCE + dice_weight * Dice``. Both terms share ``sigmoid``."""

    name = "combined"
    spatial = (2, 3)

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = BCELoss(sigmoid=sigmoid)
        self.dice = DiceLoss(smooth=smooth, sigmoid=sigmoid)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)
