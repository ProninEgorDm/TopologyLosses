"""Channel-first augmentations. Validation is deterministic."""

import torch


def augment_sample(image: torch.Tensor, label: torch.Tensor, skeleton: torch.Tensor | None, train: bool):
    """Flip and rotate image, label, and skeleton together. Intensity changes hit the image only."""
    if not train:
        return image, label, skeleton

    def _flip(tensor, dim):
        return torch.flip(tensor, [dim])

    if torch.rand(1).item() < 0.5:
        image = _flip(image, 1)
        label = _flip(label, 1)
        if skeleton is not None:
            skeleton = _flip(skeleton, 1)
    if torch.rand(1).item() < 0.5:
        image = _flip(image, 2)
        label = _flip(label, 2)
        if skeleton is not None:
            skeleton = _flip(skeleton, 2)
    if torch.rand(1).item() < 0.4:
        turns = int(torch.randint(1, 4, (1,)).item())
        image = torch.rot90(image, turns, [1, 2])
        label = torch.rot90(label, turns, [1, 2])
        if skeleton is not None:
            skeleton = torch.rot90(skeleton, turns, [1, 2])
    if torch.rand(1).item() < 0.5:
        shift = float(torch.empty(1).uniform_(-0.15, 0.15))
        image = (image + shift).clamp(0.0, 1.0)
    if torch.rand(1).item() < 0.1:
        image = (image + torch.randn_like(image) * 0.01).clamp(0.0, 1.0)
    if torch.rand(1).item() < 0.2:
        gamma = float(torch.empty(1).uniform_(0.8, 1.2))
        image = image.clamp(min=1e-6).pow(gamma).clamp(0.0, 1.0)
    return image, label, skeleton
