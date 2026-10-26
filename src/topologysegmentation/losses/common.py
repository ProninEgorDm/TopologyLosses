"""Shared tensor layout for every registered loss."""

import torch


def align_pair(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a batch of binary logits and a target on the same device and dtype.

    2D maps are ``(N, 1, H, W)``. Volumes are ``(N, 1, D, H, W)``. A leading
    channel is added when it was omitted, and a batch axis is added for a
    single 2D map. A 3D tensor is a batch of channel-free 2D maps.
    """
    logits = _with_batch_and_channel(logits)
    target = _with_batch_and_channel(target)
    if logits.ndim not in (4, 5) or logits.ndim != target.ndim:
        raise ValueError(
            f"Expected matching 2D (N, 1, H, W) or 3D (N, 1, D, H, W) tensors, "
            f"got {tuple(logits.shape)} and {tuple(target.shape)}"
        )
    if logits.shape[1] != 1:
        logits = logits[:, :1]
    if target.shape[1] != 1:
        target = target[:, :1]
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"Logits and target differ in batch or spatial size: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    return logits, target.to(device=logits.device, dtype=logits.dtype)


def prepare_pair(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Like :func:`align_pair`, but every channel is kept.

    A multi-class prediction is ``(N, C, H, W)`` or ``(N, C, D, H, W)``. The
    target may use the same channels or a single channel (a label map).
    """
    prediction = _with_batch_and_channel(prediction)
    target = _with_batch_and_channel(target)
    if prediction.ndim not in (4, 5) or prediction.ndim != target.ndim:
        raise ValueError(
            f"Expected matching 2D (N, C, H, W) or 3D (N, C, D, H, W) tensors, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"Prediction and target differ in batch or spatial size: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if target.shape[1] not in (1, prediction.shape[1]):
        raise ValueError(
            f"Target has {target.shape[1]} channels and the prediction has {prediction.shape[1]}. "
            "Use the same channels, or a one-channel label map."
        )
    return prediction, target.to(device=prediction.device, dtype=prediction.dtype)


def reduce_loss(value: torch.Tensor, reduction: str) -> torch.Tensor:
    """``mean``, ``sum``, or ``none`` over every remaining element."""
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    if reduction == "none":
        return value
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}.")


def masked_reduction(value: torch.Tensor, valid: torch.Tensor | None, reduction: str) -> torch.Tensor:
    """Reduce a per-voxel map. ``valid`` is 1 on voxels that count."""
    if valid is None:
        return reduce_loss(value, reduction)
    value = value * valid
    if reduction == "sum":
        return value.sum()
    if reduction == "none":
        return value
    if reduction == "mean":
        return value.sum() / valid.sum().clamp(min=1).to(dtype=value.dtype)
    raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}.")


def _with_batch_and_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor
