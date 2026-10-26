"""Name-to-loss registry. Importing this module does not import vendor code."""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass(frozen=True)
class LossSpec:
    name: str
    builder: Callable[..., nn.Module]
    expects: str
    differentiable: bool
    needs_skeleton: bool = False
    # Spatial rank: 2 is (N, 1, H, W), 3 is (N, 1, D, H, W).
    spatial: tuple[int, ...] = (2, 3)


REGISTRY: dict[str, LossSpec] = {}

METRIC_ONLY = {
    "betti",
    "voi",
    "hausdorff",
    "composite",
    "betti_matching",
}


def register(spec: LossSpec) -> None:
    REGISTRY[spec.name] = spec


def available_losses() -> list[str]:
    _ensure_registered()
    return sorted(REGISTRY)


def loss_shapes() -> dict[str, tuple[int, ...]]:
    """Spatial ranks accepted by each trainable loss. 2 is a frame, 3 is a volume."""
    _ensure_registered()
    return {name: REGISTRY[name].spatial for name in sorted(REGISTRY)}


def create_loss(name: str, **kwargs) -> nn.Module:
    _ensure_registered()
    key = name.strip().lower()
    if key in METRIC_ONLY:
        raise ValueError(
            f"'{key}' has no gradient in this repository. It is computed as a validation metric, "
            f"not as a training loss. Trainable losses: {sorted(REGISTRY)}"
        )
    if key not in REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[key].builder(**kwargs)


def _ensure_registered() -> None:
    import topologysegmentation.losses.segmentation  # noqa: F401
    import topologysegmentation.losses.topology  # noqa: F401
