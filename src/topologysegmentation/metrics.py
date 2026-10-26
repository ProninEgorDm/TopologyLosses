"""Overlap scores and eval-only topology counts.

Betti numbers come from the cubical complex in ``persistence``: foreground
pixels that share a corner are one component, and a feature counts only when
its bar contains the threshold.
The centerline score is the Dice of morphological skeletons.
"""

import numpy as np
import torch

from topologysegmentation.persistence import betti_numbers


def _batched(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


def dice_iou(probabilities: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    prediction = _batched((probabilities > threshold).float())
    truth = _batched(target.float())
    dims = tuple(range(1, prediction.ndim))
    intersection = (prediction * truth).sum(dim=dims)
    predicted = prediction.sum(dim=dims)
    labeled = truth.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (predicted + labeled + eps)
    union = predicted + labeled - intersection
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def centerline_dice(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice between morphological skeletons, one score per batch item."""
    prediction = _batched((probabilities > threshold).float())
    truth = _batched(target.float())
    predicted = _skeleton(prediction)
    labeled = _skeleton(truth)
    dims = tuple(range(1, predicted.ndim))
    intersection = (predicted * labeled).sum(dim=dims)
    return (2.0 * intersection + eps) / (predicted.sum(dim=dims) + labeled.sum(dim=dims) + eps)


def topology_errors(probabilities: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    prediction = _batched((probabilities > threshold).detach())
    truth = _batched(target.detach().float())
    prediction_np = prediction.cpu().numpy()
    truth_np = truth.cpu().numpy()
    rank = 2
    errors = [[], []]
    per_image = []
    for pred_mask, true_mask in zip(prediction_np, truth_np):
        pred_betti = betti_numbers(pred_mask[0] > 0)
        true_betti = betti_numbers(true_mask[0] > 0)
        rank = max(rank, len(pred_betti), len(true_betti))
        while len(errors) < rank:
            errors.append([])
        record = {}
        for dimension in range(rank):
            predicted = pred_betti[dimension] if dimension < len(pred_betti) else 0
            labeled = true_betti[dimension] if dimension < len(true_betti) else 0
            errors[dimension].append(abs(predicted - labeled))
            record[f"betti_pred_b{dimension}"] = predicted
            record[f"betti_target_b{dimension}"] = labeled
        per_image.append(record)
    lines = centerline_dice(prediction.float(), truth, threshold=0.0)
    for record, score in zip(per_image, lines):
        record["centerline_dice"] = float(score)
    report = {
        "centerline_dice": float(lines.mean()) if len(per_image) else 0.0,
        "per_image_topology": per_image,
    }
    for dimension, values in enumerate(errors):
        report[f"betti_b{dimension}_mae"] = float(np.mean(values) if values else 0.0)
    return report


def _skeleton(mask: torch.Tensor) -> torch.Tensor:
    """Binary morphological skeleton for a 2D frame or a 3D volume.

    The structuring element is the one implied by a padded max-pool: erosion
    keeps a voxel only when its in-bounds axis neighbors are set, and dilation
    is the full 3-cube. The loop runs in NumPy because a pool kernel per
    iteration spends more time on dispatch than on the mask.
    """
    binary = np.ascontiguousarray(mask.detach().cpu().numpy() > 0.5)
    spatial = binary[:, 0]
    remaining = spatial.copy()
    skeleton = np.zeros(spatial.shape, dtype=bool)
    for _ in range(int(max(spatial.shape[1:]))):
        if not remaining.any():
            break
        eroded = _erode(remaining)
        skeleton |= remaining & ~_dilate(eroded)
        remaining = eroded
    result = torch.from_numpy(np.ascontiguousarray(skeleton[:, None]))
    return result.to(device=mask.device, dtype=torch.float32)


def _erode(image: np.ndarray) -> np.ndarray:
    out = image.copy()
    for axis in range(1, image.ndim):
        higher = [slice(None)] * image.ndim
        source = [slice(None)] * image.ndim
        higher[axis] = slice(1, None)
        source[axis] = slice(None, -1)
        out[tuple(higher)] &= image[tuple(source)]
        lower = [slice(None)] * image.ndim
        source[axis] = slice(1, None)
        lower[axis] = slice(None, -1)
        out[tuple(lower)] &= image[tuple(source)]
    return out


def _dilate(image: np.ndarray) -> np.ndarray:
    spatial = image.ndim - 1
    padded = np.pad(image, [(0, 0)] + [(1, 1)] * spatial)
    acc = np.zeros(image.shape, dtype=bool)
    for offset in np.ndindex((3,) * spatial):
        slices = [slice(None)]
        for axis, shift in enumerate(offset):
            length = image.shape[axis + 1]
            slices.append(slice(int(shift), int(shift) + length))
        acc |= padded[tuple(slices)]
    return acc
