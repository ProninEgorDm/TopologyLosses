"""Thin adapters around vendored topology losses.

Vendor modules are imported only when that loss is constructed.
"""

from types import SimpleNamespace

import math

import torch
import torch.nn as nn

from .common import align_pair
from .registry import LossSpec, register


class CLDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, iter_: int = 3):
        super().__init__()
        from topologysegmentation.losses.vendor.cldice.cldice import soft_dice_cldice

        self.inner = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=1.0)

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probabilities = torch.sigmoid(logits)
        return self.inner(target, probabilities)


class TLossAdapter(nn.Module):
    def __init__(self, image_size: int, nu: float = 1.0, device: str = "cpu"):
        super().__init__()
        from topologysegmentation.losses.vendor.tloss.tloss import TLoss

        config = SimpleNamespace(
            data=SimpleNamespace(image_size=int(image_size)),
            device=torch.device(device),
        )
        self.image_size = int(image_size)
        self.inner = TLoss(config, nu=float(nu))

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probabilities = torch.sigmoid(logits)
        # The published code uses a constant precision of 1 at every voxel, so the
        # likelihood depends on the voxel count and the sum of squared errors.
        # That form is valid for a 2D frame and a 3D volume. Divide by the voxel
        # count: the vendored value is an unnormalized negative log-likelihood.
        return _student_t(probabilities, target, self.inner.nu, self.inner.epsilon) / probabilities[0].numel()


def _student_t(prediction, target, nu, epsilon):
    """Per-image T-loss from González-Jiménez et al. when every lambda is 1."""
    delta = (prediction - target).flatten(1)
    count = prediction.new_tensor(float(delta.shape[1]))
    eps = epsilon.to(device=prediction.device, dtype=prediction.dtype)
    nu = nu.to(dtype=prediction.dtype)
    dof = torch.exp(nu) + eps
    precision = torch.exp(1.0 + eps)
    residual = (delta.square().sum(dim=1) * precision) / dof
    scale = (dof + count) / 2.0
    value = (
        -torch.lgamma(scale)
        + torch.lgamma(dof / 2.0)
        - 0.5 * count * (1.0 + eps)
        + (count / 2.0) * math.log(math.pi)
        + (count / 2.0) * (nu + eps)
        + scale * torch.log1p(residual)
    )
    return value.mean()


def _require_rank(logits: torch.Tensor, loss_name: str, rank: int) -> None:
    spatial = logits.ndim - 2
    if spatial != rank:
        kind = "a 2D frame (N, 1, H, W)" if rank == 2 else "a volume (N, 1, D, H, W)"
        raise ValueError(f"Loss '{loss_name}' expects {kind}, got shape {tuple(logits.shape)}.")


def _require(module_name: str, loss_name: str, extra: str) -> None:
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(
            f"Loss '{loss_name}' needs optional package '{module_name}'. {extra}"
        )


class SATLossAdapter(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        _require("torch_topological", "sat", "SAT also imports gudhi and pot (module name: ot).")
        _require("gudhi", "sat", "Install gudhi before using the SAT loss.")
        _require("ot", "sat", "The persistence matching code imports pot as ot.")
        from topologysegmentation.losses.vendor.sat.SATLoss import PDMatchingLoss

        self.inner = PDMatchingLoss(SimpleNamespace(precal_PD=False), p=p)

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        _require_rank(logits, "sat", 2)
        probabilities = torch.sigmoid(logits)
        height, width = probabilities.shape[-2:]
        # Transport costs are already in [0, 1], but the sum over bars has a much
        # steeper gradient than Dice. Divide by the image side so a config weight
        # near 0.1 stays a regularizer when the resolution changes.
        side = (float(height) * float(width)) ** 0.5
        return self.inner(probabilities, target) / side


class BettiMatchingAdapter(nn.Module):
    """Pure-Python Betti matching from Stucki et al., ICML 2023.

    https://proceedings.mlr.press/v202/stucki23a
    https://github.com/nstucki/Betti-matching
    """

    def __init__(self, filtration: str = "superlevel", relative: bool = False):
        super().__init__()
        _require("gudhi", "bm", "Install gudhi before using the pure-Python Betti matching loss.")
        self.filtration = filtration
        self.relative = relative

    def forward(self, logits, target, skeleton=None):
        from topologysegmentation.losses.vendor.bm.loss_functions import compute_BettiMatchingLoss

        logits, target = align_pair(logits, target)
        _require_rank(logits, "bm", 2)
        values = []
        for one_logits, one_target in zip(logits, target):
            value = compute_BettiMatchingLoss(
                (one_logits, one_target),
                sigmoid=True,
                filtration=self.filtration,
                relative=self.relative,
            )
            if not torch.is_tensor(value):
                value = torch.as_tensor(float(value), device=logits.device, dtype=logits.dtype)
            values.append(value)
        return torch.stack(values).mean()


class BettiMatching3DAdapter(nn.Module):
    """C++ Betti matching from Stucki, Bürgin, Paetzold and Bauer.

    https://arxiv.org/abs/2407.04683
    https://github.com/nstucki/Betti-Matching-3D
    """

    def __init__(self, filtration: str = "superlevel"):
        super().__init__()
        _require(
            "betti_matching",
            "bm3d",
            "Build https://github.com/nstucki/Betti-Matching-3D (MIT) and import the module betti_matching.",
        )
        from topologysegmentation.losses.vendor.bm3d.loss import BettiMatching3DLoss

        self.inner = BettiMatching3DLoss(filtration=filtration)

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        return self.inner(logits, target)


def _reject_unused(kwargs, allowed):
    unknown = set(kwargs) - allowed - {"device", "image_size"}
    if unknown:
        raise TypeError(f"Unexpected loss arguments: {sorted(unknown)}")


def _cldice(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"alpha", "iter_"})
    return CLDiceLoss(alpha=float(kwargs.get("alpha", 0.5)), iter_=int(kwargs.get("iter_", 3)))


def _tloss(device="cpu", image_size=256, **kwargs):
    _reject_unused(kwargs, {"nu"})
    return TLossAdapter(image_size=int(image_size), nu=float(kwargs.get("nu", 1.0)), device=device)


def _sat(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"p"})
    return SATLossAdapter(p=int(kwargs.get("p", 2)))


def _bm(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"filtration", "relative"})
    return BettiMatchingAdapter(
        filtration=kwargs.get("filtration", "superlevel"),
        relative=bool(kwargs.get("relative", False)),
    )


def _bm3d(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"filtration"})
    return BettiMatching3DAdapter(filtration=kwargs.get("filtration", "superlevel"))


def _hu(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"threshold", "epsilon"})
    from topologysegmentation.losses.hu import HuTopoLoss

    return HuTopoLoss(
        threshold=float(kwargs.get("threshold", 0.5)),
        epsilon=float(kwargs.get("epsilon", 1e-6)),
    )


def _pooling(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, set())
    from topologysegmentation.losses.pooling import ProjectedPoolingLoss

    return ProjectedPoolingLoss()


def _critical(device=None, image_size=None, **kwargs):
    _reject_unused(kwargs, {"sigma"})
    from topologysegmentation.losses.critical import CriticalPointLoss

    return CriticalPointLoss(sigma=float(kwargs.get("sigma", 0.5)))


register(LossSpec("cldice", _cldice, expects="probabilities", differentiable=True, spatial=(2, 3)))
register(LossSpec("tloss", _tloss, expects="probabilities", differentiable=True, needs_skeleton=False, spatial=(2, 3)))
register(LossSpec("sat", _sat, expects="probabilities", differentiable=True, spatial=(2,)))
register(LossSpec("bm", _bm, expects="logits", differentiable=True, spatial=(2,)))
register(LossSpec("bm3d", _bm3d, expects="logits", differentiable=True, spatial=(2, 3)))
register(LossSpec("hu", _hu, expects="logits", differentiable=True, spatial=(2, 3)))
register(LossSpec("pooling", _pooling, expects="logits", differentiable=True, spatial=(2, 3)))
register(LossSpec("critical", _critical, expects="logits", differentiable=True, spatial=(2, 3)))
