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


def _with_batch_and_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor
