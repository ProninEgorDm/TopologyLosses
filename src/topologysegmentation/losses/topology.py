"""Topology losses. Each class is constructed directly.

Vendor modules are imported only when that loss is constructed. These losses
are binary. ``sigmoid=True`` (the default) reads logits; ``sigmoid=False``
reads probabilities in ``[0, 1]``.
"""

import math

import torch
import torch.nn as nn

from .common import align_pair, reduce_loss
from .registry import register


def _probabilities(logits: torch.Tensor, sigmoid: bool) -> torch.Tensor:
    if sigmoid:
        return torch.sigmoid(logits)
    return logits


def _require_rank(logits: torch.Tensor, loss_name: str, rank: int) -> None:
    spatial = logits.ndim - 2
    if spatial != rank:
        kind = "a 2D frame (N, 1, H, W)" if rank == 2 else "a volume (N, 1, D, H, W)"
        raise ValueError(f"Loss '{loss_name}' expects {kind}, got shape {tuple(logits.shape)}.")


def _require(module_name: str, loss_name: str, extra: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"Loss '{loss_name}' needs optional package '{module_name}'. {extra}")


@register
class CLDiceLoss(nn.Module):
    """Soft clDice, Shit et al., arXiv:2003.07311.

    ``alpha`` is the clDice share and ``1 - alpha`` is the soft Dice share.
    The skeleton is ten pooling iterations. ``smooth`` is the vendor constant
    on the skeleton terms. ``exclude_background`` drops channel 0 when the map
    has a background channel and a foreground channel.
    """

    name = "cldice"
    spatial = (2, 3)

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1.0,
        exclude_background: bool = False,
        sigmoid: bool = True,
    ):
        super().__init__()
        from topologysegmentation.losses.vendor.cldice.cldice import soft_dice_cldice

        self.sigmoid = sigmoid
        self.inner = soft_dice_cldice(alpha=float(alpha), smooth=float(smooth), exclude_background=exclude_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = align_pair(logits, target)
        return self.inner(target, _probabilities(logits, self.sigmoid))


def _student_t(prediction, target, nu, epsilon):
    """Per-image T-loss from González-Jiménez et al. when every lambda is 1.

    The returned tensor has one value per batch item, before the voxel-count
    normalization applied by :class:`TLoss`.
    """
    delta = (prediction - target).flatten(1)
    count = prediction.new_tensor(float(delta.shape[1]))
    eps = epsilon.to(device=prediction.device, dtype=prediction.dtype)
    nu = nu.to(dtype=prediction.dtype)
    dof = torch.exp(nu) + eps
    precision = torch.exp(1.0 + eps)
    residual = (delta.square().sum(dim=1) * precision) / dof
    scale = (dof + count) / 2.0
    return (
        -torch.lgamma(scale)
        + torch.lgamma(dof / 2.0)
        - 0.5 * count * (1.0 + eps)
        + (count / 2.0) * math.log(math.pi)
        + (count / 2.0) * (nu + eps)
        + scale * torch.log1p(residual)
    )


@register
class TLoss(nn.Module):
    """Student-t likelihood of a noisy mask, divided by the voxel count.

    ``nu`` is an ``nn.Parameter``. ``exp(nu)`` is the degrees of freedom, so
    an optimizer that receives ``criterion.parameters()`` learns it. The value
    is not an overlap score: a perfect mask is not zero.
    """

    name = "tloss"
    spatial = (2, 3)

    def __init__(self, nu: float = 1.0, epsilon: float = 1e-8, sigmoid: bool = True, reduction: str = "mean"):
        super().__init__()
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.nu = nn.Parameter(torch.tensor(float(nu)))
        self.register_buffer("epsilon", torch.tensor(float(epsilon)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = align_pair(logits, target)
        probabilities = _probabilities(logits, self.sigmoid)
        per_image = _student_t(probabilities, target, self.nu, self.epsilon)
        per_image = per_image / probabilities[0].numel()
        return reduce_loss(per_image, self.reduction)


@register
class SATLoss(nn.Module):
    """Spatially aware diagram matching, Wen et al., arXiv:2412.02076. Frames only.

    The transport cost is divided by the image side length, so a weight near
    0.1 stays a regularizer when the resolution changes. ``p`` is the ground
    distance order.
    """

    name = "sat"
    spatial = (2,)

    def __init__(self, p: int = 2, sigmoid: bool = True):
        super().__init__()
        _require("torch_topological", "sat", "SAT also imports gudhi and pot (module name: ot).")
        _require("gudhi", "sat", "Install gudhi before using the SAT loss.")
        _require("ot", "sat", "The persistence matching code imports pot as ot.")
        from types import SimpleNamespace

        from topologysegmentation.losses.vendor.sat.SATLoss import PDMatchingLoss

        self.sigmoid = sigmoid
        self.inner = PDMatchingLoss(SimpleNamespace(precal_PD=False), p=int(p))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = align_pair(logits, target)
        _require_rank(logits, "sat", 2)
        probabilities = _probabilities(logits, self.sigmoid)
        height, width = probabilities.shape[-2:]
        side = (float(height) * float(width)) ** 0.5
        return self.inner(probabilities, target) / side


@register
class BettiMatchingLoss(nn.Module):
    """Betti matching, Stucki et al., ICML 2023. Frames only.

    ``filtration`` is ``superlevel`` (the default), ``sublevel``, or ``bothlevel``.
    ``relative`` asks the matcher for relative homology. ``comparison`` selects
    how the two diagrams are paired; ``union`` is the published training
    objective. ``construction`` ``V`` is the vertex grid the fast path implements.
    """

    name = "bm"
    spatial = (2,)

    def __init__(
        self,
        filtration: str = "superlevel",
        relative: bool = False,
        comparison: str = "union",
        construction: str = "V",
        sigmoid: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        _require("gudhi", "bm", "Install gudhi before using the pure-Python Betti matching loss.")
        if filtration not in {"superlevel", "sublevel", "bothlevel"}:
            raise ValueError("filtration must be 'superlevel', 'sublevel', or 'bothlevel'.")
        self.filtration = filtration
        self.relative = relative
        self.comparison = comparison
        self.construction = construction
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        from topologysegmentation.losses.vendor.bm.loss_functions import compute_BettiMatchingLoss

        logits, target = align_pair(logits, target)
        _require_rank(logits, "bm", 2)
        values = []
        for one_logits, one_target in zip(logits, target):
            value = compute_BettiMatchingLoss(
                (one_logits, one_target),
                sigmoid=self.sigmoid,
                filtration=self.filtration,
                relative=self.relative,
                comparison=self.comparison,
                construction=self.construction,
            )
            if not torch.is_tensor(value):
                value = torch.as_tensor(float(value), device=logits.device, dtype=logits.dtype)
            values.append(value)
        return reduce_loss(torch.stack(values), self.reduction)


@register
class BettiMatching3DLoss(nn.Module):
    """Betti matching through the C++ extension. Frames and volumes.

    Build https://github.com/nstucki/Betti-Matching-3D (MIT) and import
    ``betti_matching``. ``filtration`` is ``superlevel`` or ``sublevel``.
    """

    name = "bm3d"
    spatial = (2, 3)

    def __init__(self, filtration: str = "superlevel", sigmoid: bool = True):
        super().__init__()
        _require(
            "betti_matching",
            "bm3d",
            "Build https://github.com/nstucki/Betti-Matching-3D (MIT) and import the module betti_matching.",
        )
        from topologysegmentation.losses.vendor.bm3d.loss import BettiMatching3DLoss as Backend

        self.inner = Backend(filtration=filtration, sigmoid=sigmoid)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        logits, target = align_pair(logits, target)
        return self.inner(logits, target)
