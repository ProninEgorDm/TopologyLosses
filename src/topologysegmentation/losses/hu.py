"""Diagram-matching loss on critical pixels.

The objective follows Hu, Li, Samaras and Chen, NeurIPS 2019
(arXiv:1906.05404): in each homology dimension, the predicted bars closest
to birth 1 and death 0 are matched to that corner, one per ground-truth
feature, and every other bar is matched to the diagonal. The code is an
independent implementation on a Gudhi cubical complex. Gradients reach only
the birth and death pixels of those bars.
"""

import torch
import torch.nn as nn

from topologysegmentation.losses.common import align_pair
from topologysegmentation.persistence import cubical_bars


class HuTopoLoss(nn.Module):
    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probabilities = torch.sigmoid(logits)
        values = [
            self._one(probability.squeeze(0), truth.squeeze(0))
            for probability, truth in zip(probabilities, target)
        ]
        return torch.stack(values).mean()

    def _one(self, probability: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        predicted = cubical_bars(probability.detach().cpu().numpy(), epsilon=self.epsilon)
        reference = cubical_bars(truth.detach().cpu().numpy(), epsilon=self.epsilon)
        total = probability.sum() * 0.0
        expected = 0
        for dimension in range(probability.ndim):
            bars = [bar for bar in predicted if bar.dimension == dimension]
            keep_count = sum(
                1 for bar in reference if bar.dimension == dimension and bar.alive(self.threshold)
            )
            expected += keep_count
            if not bars:
                continue
            births = _sample(probability, [bar.birth for bar in bars])
            deaths = _sample(probability, [bar.death for bar in bars])
            distance = (births.detach() - 1.0).pow(2) + deaths.detach().pow(2)
            order = torch.argsort(distance)
            keep = order[:keep_count]
            remove = order[keep_count:]
            if keep.numel():
                total = total + (births[keep] - 1.0).pow(2).sum() + deaths[keep].pow(2).sum()
            if remove.numel():
                total = total + 0.5 * (births[remove] - deaths[remove]).pow(2).sum()
        return total / float(max(expected, 1))


def _sample(probability: torch.Tensor, points: list) -> torch.Tensor:
    """Read filtration values at critical voxels. The gather stays in the graph."""
    spatial = probability.shape
    chosen = []
    for point in points:
        inside = (
            point is not None
            and len(point) == len(spatial)
            and all(0 <= int(coord) < size for coord, size in zip(point, spatial))
        )
        chosen.append(tuple(int(coord) for coord in point) if inside else None)
    if not any(point is not None for point in chosen):
        return probability.new_zeros(len(points))
    axes = [[] for _ in spatial]
    slots = []
    for slot, point in enumerate(chosen):
        if point is None:
            continue
        slots.append(slot)
        for axis, coord in enumerate(point):
            axes[axis].append(coord)
    gathered = probability.new_zeros(len(points))
    index = tuple(torch.as_tensor(axis, device=probability.device) for axis in axes)
    gathered[torch.as_tensor(slots, device=probability.device)] = probability[index]
    return gathered
