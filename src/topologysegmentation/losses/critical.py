"""Neighborhood loss for endpoints, junctions and isolated pixels.

Thin structures fail in places where the local degree is wrong: a vessel
ends, splits, or a speck appears. This loss compares three soft maps of that
degree between the prediction and the mask. The neighbor count is one
convolution: 8 neighbors on a frame, 26 on a volume. It is an independent
formulation of the critical-pixel idea, not a set of hand-written templates.
"""

import torch
import torch.nn.functional as F
from torch import nn

from topologysegmentation.losses.common import align_pair
from topologysegmentation.losses.registry import register


@register
class CriticalPointLoss(nn.Module):
    """Endpoints, junctions, and isolated voxels.

    ``sigma`` is the width of the soft neighbor-count peaks. Eight neighbors
    on a frame, twenty-six on a volume. ``sigmoid=False`` reads probabilities.
    """

    name = "critical"
    spatial = (2, 3)

    def __init__(self, sigma: float = 0.5, sigmoid: bool = True):
        super().__init__()
        kernel = torch.ones(1, 1, 3, 3)
        kernel[0, 0, 1, 1] = 0.0
        self.register_buffer("neighbor_kernel", kernel)
        self.sigma = float(sigma)
        self.sigmoid = sigmoid

    def forward(self, logits, target, skeleton=None):
        logits, target = align_pair(logits, target)
        probabilities = torch.sigmoid(logits) if self.sigmoid else logits
        predicted = self._maps(probabilities)
        reference = self._maps(target)
        return torch.stack([(left - right).abs().mean() for left, right in zip(predicted, reference)]).mean()

    def _maps(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim == 5:
            kernel = torch.ones(1, 1, 3, 3, 3, device=image.device, dtype=image.dtype)
            kernel[0, 0, 1, 1, 1] = 0.0
            neighbors = F.conv3d(image, kernel, padding=1)
        else:
            kernel = self.neighbor_kernel.to(device=image.device, dtype=image.dtype)
            neighbors = F.conv2d(image, kernel, padding=1)
        width = 2.0 * self.sigma * self.sigma
        endpoint = image * torch.exp(-(neighbors - 1.0).pow(2) / width)
        junction = image * torch.sigmoid(4.0 * (neighbors - 2.5))
        isolated = image * torch.exp(-neighbors.pow(2) / width)
        return endpoint, junction, isolated
