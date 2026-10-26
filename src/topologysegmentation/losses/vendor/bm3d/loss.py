"""Differentiable Betti matching loss on top of the C++ matcher.

Authors: Nico Stucki, Vincent Bürgin, Johannes C. Paetzold, Ulrich Bauer.
Paper: https://arxiv.org/abs/2407.04683
Code: https://github.com/nstucki/Betti-Matching-3D (MIT). Build that repository so
``import betti_matching`` works. The extension is not vendored here: it is C++
and has to be compiled with pybind11.

The matching itself is discrete. The loss follows the training objective of the
pure-Python Betti matching (https://proceedings.mlr.press/v202/stucki23a):
squared filtration error on matched pairs, and squared persistence of the
prediction's unmatched pairs. Values are read back from the torch volume at the
birth and death coordinates, so the gradient reaches those voxels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _by_dimension(result, name: str, counts_name: str) -> list[np.ndarray]:
    value = getattr(result, name, None)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [np.asarray(part) for part in value]
    array = np.asarray(value)
    counts = getattr(result, counts_name, None)
    if counts is None:
        return [array]
    parts = []
    start = 0
    for count in np.asarray(counts).astype(int).tolist():
        parts.append(array[start : start + count])
        start += count
    return parts


def _index(volume: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    if coords.size == 0:
        return volume.new_zeros((0,))
    columns = [
        torch.as_tensor(coords[:, axis], dtype=torch.long, device=volume.device)
        for axis in range(coords.shape[1])
    ]
    return volume[tuple(columns)]


def _pair_values(left: torch.Tensor, left_coords, right: torch.Tensor, right_coords):
    left_coords = np.asarray(left_coords)
    right_coords = np.asarray(right_coords)
    if left_coords.size == 0 or right_coords.size == 0:
        empty = left.new_zeros((0,))
        return empty, empty
    if left_coords.ndim == 1:
        left_coords = left_coords.reshape(1, -1)
    if right_coords.ndim == 1:
        right_coords = right_coords.reshape(1, -1)
    count = min(len(left_coords), len(right_coords))
    left_coords = left_coords[:count]
    right_coords = right_coords[:count]
    valid = np.all(left_coords >= 0, axis=1) & np.all(right_coords >= 0, axis=1)
    for axis in range(left_coords.shape[1]):
        valid &= left_coords[:, axis] < left.shape[axis]
        valid &= right_coords[:, axis] < right.shape[axis]
    return _index(left, left_coords[valid]), _index(right, right_coords[valid])


def matching_loss(prediction: torch.Tensor, target: torch.Tensor, result) -> torch.Tensor:
    """Squared Betti-matching cost. ``prediction`` carries the gradient."""
    matched_birth = _by_dimension(result, "input1_matched_birth_coordinates", "num_matched")
    matched_death = _by_dimension(result, "input1_matched_death_coordinates", "num_matched")
    matched_birth_target = _by_dimension(result, "input2_matched_birth_coordinates", "num_matched")
    matched_death_target = _by_dimension(result, "input2_matched_death_coordinates", "num_matched")
    unmatched_birth = _by_dimension(result, "input1_unmatched_birth_coordinates", "num_unmatched_input1")
    unmatched_death = _by_dimension(result, "input1_unmatched_death_coordinates", "num_unmatched_input1")

    total = prediction.new_zeros(())
    dimensions = max(
        len(matched_birth),
        len(matched_death),
        len(unmatched_birth),
        len(unmatched_death),
        0,
    )
    for index in range(dimensions):
        if index < len(matched_birth) and index < len(matched_death):
            birth, birth_target = _pair_values(
                prediction,
                matched_birth[index],
                target,
                matched_birth_target[index] if index < len(matched_birth_target) else [],
            )
            death, death_target = _pair_values(
                prediction,
                matched_death[index],
                target,
                matched_death_target[index] if index < len(matched_death_target) else [],
            )
            count = min(birth.numel(), death.numel(), birth_target.numel(), death_target.numel())
            if count:
                total = total + 2 * (
                    (birth[:count] - birth_target[:count]).square().sum()
                    + (death[:count] - death_target[:count]).square().sum()
                )
        if index < len(unmatched_birth) and index < len(unmatched_death):
            birth, death = _pair_values(
                prediction, unmatched_birth[index], prediction, unmatched_death[index]
            )
            count = min(birth.numel(), death.numel())
            if count:
                total = total + (birth[:count] - death[:count]).square().sum()
    return total


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


class BettiMatching3DLoss(nn.Module):
    """Logits in, Betti matching loss out. Requires the ``betti_matching`` extension."""

    def __init__(self, filtration: str = "superlevel", sigmoid: bool = True):
        super().__init__()
        if filtration not in {"superlevel", "sublevel"}:
            raise ValueError("filtration must be 'superlevel' or 'sublevel'.")
        import betti_matching

        self.filtration = filtration
        self.sigmoid = bool(sigmoid)
        self._backend = betti_matching

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits) if self.sigmoid else logits
        losses = []
        for prediction, truth in zip(probabilities, target):
            prediction = _spatial(prediction)
            truth = _spatial(truth).to(dtype=prediction.dtype)
            if self.filtration == "superlevel":
                # The C++ matcher builds sublevel sets. Inverting turns that into a
                # superlevel filtration of the original probabilities.
                prediction = 1.0 - prediction
                truth = 1.0 - truth
            result = self._backend.compute_matching(
                np.ascontiguousarray(prediction.detach().cpu().numpy(), dtype=np.float64),
                np.ascontiguousarray(truth.detach().cpu().numpy(), dtype=np.float64),
                include_input1_unmatched_pairs=True,
                include_input2_unmatched_pairs=False,
            )
            losses.append(matching_loss(prediction, truth, result))
        return torch.stack(losses).mean()
