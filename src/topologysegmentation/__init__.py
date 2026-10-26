"""Losses and metrics for topology-aware segmentation.

The VessMAP training pipeline lives in ``experiments/`` and is not installed
with this package. The release number is the single line in ``VERSION``.
"""

from pathlib import Path

__version__ = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()

from topologysegmentation.losses import (
    BCELoss,
    BettiMatching3DLoss,
    BettiMatchingLoss,
    CLDiceLoss,
    CombinedLoss,
    CriticalPointLoss,
    DiceLoss,
    FocalLoss,
    HuTopoLoss,
    ProjectedPoolingLoss,
    SATLoss,
    SumLoss,
    TLoss,
    available_losses,
    loss_shapes,
)
from topologysegmentation.metrics import betti_numbers, centerline_dice, dice_iou, topology_errors

__all__ = [
    "BCELoss",
    "BettiMatching3DLoss",
    "BettiMatchingLoss",
    "CLDiceLoss",
    "CombinedLoss",
    "CriticalPointLoss",
    "DiceLoss",
    "FocalLoss",
    "HuTopoLoss",
    "ProjectedPoolingLoss",
    "SATLoss",
    "SumLoss",
    "TLoss",
    "available_losses",
    "betti_numbers",
    "centerline_dice",
    "dice_iou",
    "loss_shapes",
    "topology_errors",
]
