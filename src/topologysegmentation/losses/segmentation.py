"""Segmentation losses. Each forward accepts logits and an optional skeleton."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import align_pair
from .registry import LossSpec, register


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probs = torch.sigmoid(logits).flatten(1)
        truth = target.flatten(1)
        intersection = (probs * truth).sum(dim=1)
        denominator = probs.sum(dim=1) + truth.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice).mean()


class BCELoss(nn.Module):
    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        return F.binary_cross_entropy_with_logits(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probs, 1.0 - probs)
        return (self.alpha * (1.0 - pt).pow(self.gamma) * bce).mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, target, skeleton=None):
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


def _dice(**kwargs):
    return DiceLoss(smooth=float(kwargs.get("smooth", 1e-6)))


def _bce(**kwargs):
    return BCELoss()


def _focal(**kwargs):
    return FocalLoss(alpha=float(kwargs.get("alpha", 0.25)), gamma=float(kwargs.get("gamma", 2.0)))


def _combined(**kwargs):
    return CombinedLoss(
        bce_weight=float(kwargs.get("bce_weight", 0.5)),
        dice_weight=float(kwargs.get("dice_weight", 0.5)),
        smooth=float(kwargs.get("smooth", 1e-6)),
    )


register(LossSpec("dice", _dice, expects="logits", differentiable=True))
register(LossSpec("bce", _bce, expects="logits", differentiable=True))
register(LossSpec("focal", _focal, expects="logits", differentiable=True))
register(LossSpec("combined", _combined, expects="logits", differentiable=True))
