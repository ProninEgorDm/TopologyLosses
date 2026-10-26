"""Weighted sum of registered losses, with each term kept for logging."""

import torch
import torch.nn as nn

from .registry import create_loss


class LossComposer(nn.Module):
    def __init__(self, terms: list[tuple[str, float, nn.Module]]):
        super().__init__()
        modules = {}
        self.weights: dict[str, float] = {}
        for name, weight, module in terms:
            key = name
            index = 2
            while key in modules:
                key = f"{name}_{index}"
                index += 1
            modules[key] = module
            self.weights[key] = float(weight)
        if not modules:
            raise ValueError("At least one loss term is required.")
        self.terms = nn.ModuleDict(modules)

    def forward(self, logits, target, skeleton=None):
        total = torch.zeros((), device=logits.device, dtype=logits.dtype)
        parts: dict[str, torch.Tensor] = {}
        for name, module in self.terms.items():
            value = module(logits, target, skeleton)
            weighted = self.weights[name] * value
            parts[name] = weighted
            total = total + weighted
        return total, parts


def _terms_from_config(loss_cfg: dict) -> list[dict]:
    if not loss_cfg:
        return [{"name": "dice", "weight": 1.0}]
    if "terms" in loss_cfg:
        return list(loss_cfg["terms"])
    terms = []
    for group in ("segmentation", "topological"):
        for entry in loss_cfg.get(group) or []:
            name, params = next(iter(entry.items()))
            params = dict(params or {})
            weight = float(params.pop("weight", 1.0))
            terms.append({"name": name, "weight": weight, **params})
    if terms:
        return terms
    if "segmentation_loss" in loss_cfg:
        terms.append({"name": loss_cfg["segmentation_loss"], "weight": 1.0})
        topo = loss_cfg.get("topological_loss")
        if topo:
            terms.append(
                {
                    "name": topo,
                    "weight": float(loss_cfg.get("topological_weight", 0.1)),
                }
            )
    return terms


def build_loss(loss_cfg: dict, *, device: str, image_size: int) -> LossComposer:
    built = []
    for entry in _terms_from_config(loss_cfg):
        params = dict(entry)
        name = str(params.pop("name"))
        weight = float(params.pop("weight", 1.0))
        module = create_loss(name, device=device, image_size=image_size, **params)
        built.append((name, weight, module))
    composer = LossComposer(built)
    composer.to(torch.device(device))
    return composer
