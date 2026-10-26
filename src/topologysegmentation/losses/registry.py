"""Short name of each loss class. Importing this module does not import vendor code."""

import torch.nn as nn


REGISTRY: dict[str, type[nn.Module]] = {}

METRIC_ONLY = {
    "betti",
    "voi",
    "hausdorff",
    "composite",
    "betti_matching",
}


def register(cls):
    """Record a loss class. The class carries ``name`` and ``spatial``."""
    REGISTRY[cls.name] = cls
    return cls


def available_losses() -> list[str]:
    _ensure_registered()
    return sorted(REGISTRY)


def loss_shapes() -> dict[str, tuple[int, ...]]:
    """Spatial ranks accepted by each trainable loss. 2 is a frame, 3 is a volume."""
    _ensure_registered()
    return {name: REGISTRY[name].spatial for name in sorted(REGISTRY)}


def create_loss(name: str, **kwargs) -> nn.Module:
    """Build a loss from its short name.

    The classes are the supported way to construct a loss. This lookup exists
    so a config file can still name ``dice`` or ``cldice``. ``device`` and
    ``image_size`` are ignored: the tensors decide both.
    """
    _ensure_registered()
    key = name.strip().lower()
    if key in METRIC_ONLY:
        raise ValueError(
            f"'{key}' has no gradient in this repository. It is computed as a validation metric, "
            f"not as a training loss. Trainable losses: {sorted(REGISTRY)}"
        )
    if key not in REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(REGISTRY)}")
    kwargs.pop("device", None)
    kwargs.pop("image_size", None)
    return REGISTRY[key](**kwargs)


def _ensure_registered() -> None:
    import topologysegmentation.losses.critical  # noqa: F401
    import topologysegmentation.losses.hu  # noqa: F401
    import topologysegmentation.losses.pooling  # noqa: F401
    import topologysegmentation.losses.segmentation  # noqa: F401
    import topologysegmentation.losses.topology  # noqa: F401
