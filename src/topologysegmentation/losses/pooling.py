"""Soft multi-scale occupancy loss.

The construction follows the projected pooling idea of Fu et al.
(SPIE Medical Imaging 2023; Journal of Medical Imaging 2024): max-pool a
segmentation at several kernel sizes and compare those descriptions. Their
volume method projects onto three planes and counts nonzero cells. On a
binary mask that count equals the sum of the pooled values. This version
uses the sum, so a soft prediction has a gradient. A frame also takes the two
axis max-projections, and a volume takes the three orthogonal ones. That is
the same idea as their projections:
a thin break changes the projected occupancy more than it changes Dice.
Kernel sizes start at a quarter of the side and halve, as in their schedule.
"""

import torch
import torch.nn.functional as F
from torch import nn

from topologysegmentation.losses.common import align_pair
from topologysegmentation.losses.registry import register


@register
class ProjectedPoolingLoss(nn.Module):
    """Multi-scale occupancy after max-pooling. ``sigmoid=False`` reads probabilities."""

    name = "pooling"
    spatial = (2, 3)

    def __init__(self, sigmoid: bool = True):
        super().__init__()
        self.sigmoid = sigmoid

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probabilities = torch.sigmoid(logits) if self.sigmoid else logits
        if probabilities.ndim == 5:
            prediction, truth = _describe_volume(probabilities), _describe_volume(target)
        else:
            prediction, truth = _describe_image(probabilities), _describe_image(target)
        return (prediction - truth).abs().mean()


def _describe_image(maps: torch.Tensor) -> torch.Tensor:
    height, width = maps.shape[-2:]
    parts = [
        _occupancy2d(maps, _kernel_sizes(min(height, width))),
        _occupancy1d(maps.amax(dim=2), _kernel_sizes(width)),
        _occupancy1d(maps.amax(dim=3), _kernel_sizes(height)),
    ]
    return torch.cat(parts, dim=0)


def _describe_volume(maps: torch.Tensor) -> torch.Tensor:
    depth, height, width = maps.shape[-3:]
    parts = [_occupancy3d(maps, _kernel_sizes(min(depth, height, width)))]
    projections = (
        (2, min(height, width)),
        (3, min(depth, width)),
        (4, min(depth, height)),
    )
    for axis, side in projections:
        parts.append(_occupancy2d(maps.amax(dim=axis), _kernel_sizes(side)))
    return torch.cat(parts, dim=0)


def _kernel_sizes(length: int) -> list[int]:
    size = max(int(length) // 4, 1)
    sizes = []
    while size >= 1:
        sizes.append(size)
        if size == 1:
            break
        size //= 2
    return sizes


def _occupancy2d(maps: torch.Tensor, kernels: list[int]) -> torch.Tensor:
    pooled = [_pool2d(maps, kernel).mean(dim=(2, 3)) for kernel in kernels]
    return torch.stack(pooled, dim=0)


def _occupancy1d(line: torch.Tensor, kernels: list[int]) -> torch.Tensor:
    pooled = [_pool1d(line, kernel).mean(dim=-1) for kernel in kernels]
    return torch.stack(pooled, dim=0)


def _occupancy3d(maps: torch.Tensor, kernels: list[int]) -> torch.Tensor:
    pooled = [_pool3d(maps, kernel).mean(dim=(2, 3, 4)) for kernel in kernels]
    return torch.stack(pooled, dim=0)


def _pool3d(maps: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = min(int(kernel), maps.shape[-3], maps.shape[-2], maps.shape[-1])
    if kernel <= 1:
        return maps
    return F.max_pool3d(maps, kernel_size=kernel, stride=kernel, ceil_mode=True)


def _pool2d(maps: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = min(int(kernel), maps.shape[-2], maps.shape[-1])
    if kernel <= 1:
        return maps
    return F.max_pool2d(maps, kernel_size=kernel, stride=kernel, ceil_mode=True)


def _pool1d(line: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = min(int(kernel), line.shape[-1])
    if kernel <= 1:
        return line
    return F.max_pool1d(line, kernel_size=kernel, stride=kernel, ceil_mode=True)
