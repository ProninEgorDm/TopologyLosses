"""Weighted sum of loss modules."""

import torch
import torch.nn as nn

from .registry import create_loss


class SumLoss(nn.Module):
    """Weighted sum of loss modules. ``forward`` returns one scalar.

    A term is a module, a ``(module, weight)`` pair, or a
    ``(module, weight, name)`` triple. The name defaults to the class's
    ``name`` (``dice``, ``cldice``, ...). A repeated name becomes ``dice_2``.

    After ``forward``, ``parts`` maps each name to that term already multiplied
    by its weight. Assign ``weights[name]`` between epochs to change a weight.
    A weight of ``0`` skips that term, so an expensive loss can stay in the
    module and start later.

    Example::

        criterion = SumLoss((DiceLoss(), 1.0), (CLDiceLoss(), 0.1))
        loss = criterion(logits, target)
        loss.backward()
        criterion.parts
    """

    def __init__(self, *terms: nn.Module | tuple):
        super().__init__()
        modules = {}
        self.weights: dict[str, float] = {}
        for term in terms:
            module, weight, name = _parse_term(term)
            key = name
            index = 2
            while key in modules:
                key = f"{name}_{index}"
                index += 1
            if not key.isidentifier():
                raise ValueError(f"Loss name {key!r} must be a Python identifier, for example 'dice' or 'cldice_2'.")
            modules[key] = module
            self.weights[key] = float(weight)
        if not modules:
            raise ValueError("At least one loss term is required.")
        self.terms = nn.ModuleDict(modules)
        self.parts: dict[str, torch.Tensor] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, skeleton=None) -> torch.Tensor:
        total = logits.sum() * 0.0
        parts: dict[str, torch.Tensor] = {}
        for name, module in self.terms.items():
            weight = self.weights[name]
            if weight == 0.0:
                parts[name] = total.detach() * 0.0
                continue
            value = module(logits, target)
            if value.ndim != 0:
                raise ValueError(
                    f"Loss '{name}' returned shape {tuple(value.shape)}. "
                    "SumLoss adds scalars, so leave reduction at 'mean' or 'sum'."
                )
            weighted = weight * value
            parts[name] = weighted
            total = total + weighted
        self.parts = parts
        return total


def _parse_term(term) -> tuple[nn.Module, float, str]:
    if isinstance(term, nn.Module):
        return term, 1.0, _term_name(term)
    if isinstance(term, tuple) and len(term) == 2 and isinstance(term[0], nn.Module):
        module, weight = term
        return module, float(weight), _term_name(module)
    if isinstance(term, tuple) and len(term) == 3 and isinstance(term[0], nn.Module):
        module, weight, name = term
        return module, float(weight), str(name)
    raise TypeError(
        "A SumLoss term is a loss module, a (module, weight) pair, or a (module, weight, name) triple."
    )


def _term_name(module: nn.Module) -> str:
    name = getattr(module, "name", None)
    if isinstance(name, str) and name:
        return name
    return module.__class__.__name__


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


def build_loss(loss_cfg: dict, *, device: str, image_size: int) -> SumLoss:
    """Build a :class:`SumLoss` from a training config.

    ``image_size`` is accepted so existing configs keep the same call. The
    loss reads its shape from the tensors.
    """
    del image_size
    built = []
    for entry in _terms_from_config(loss_cfg):
        params = dict(entry)
        name = str(params.pop("name"))
        weight = float(params.pop("weight", 1.0))
        built.append((create_loss(name, **params), weight, name))
    composer = SumLoss(*built)
    composer.to(torch.device(device))
    return composer
