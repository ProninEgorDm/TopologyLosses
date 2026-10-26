"""Time every trainable loss on several 2D and 3D shapes.

Run from the repository root:

    python tests/bench_losses.py

The numbers include the backward pass. They are one machine, not a bound.
``bm3d`` is skipped when the Betti-Matching-3D extension is not importable.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from topologysegmentation import loss_shapes
from topologysegmentation.losses import create_loss
from topologysegmentation.metrics import centerline_dice
from topologysegmentation.persistence import betti_numbers


def main() -> None:
    cases = {
        "dice": ((2, 1, 64, 64), (2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "bce": ((2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "focal": ((2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "combined": ((2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "cldice": ((2, 1, 64, 64), (2, 1, 256, 256), (1, 1, 32, 32, 32)),
        "pooling": ((2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "critical": ((2, 1, 256, 256), (2, 1, 32, 32, 32)),
        "tloss": ((2, 1, 256, 256), (1, 1, 32, 32, 32)),
        "hu": ((1, 1, 64, 64), (1, 1, 16, 16, 16)),
        "bm": ((1, 1, 32, 32), (1, 1, 64, 64)),
        "sat": ((1, 1, 32, 32), (1, 1, 64, 64)),
        "bm3d": ((1, 1, 32, 32), (1, 1, 16, 16, 16)),
    }
    shapes = loss_shapes()
    print(f"{'loss':10} {'shape':22} {'ms':>8}  check")
    for name, sizes in cases.items():
        for shape in sizes:
            rank = len(shape) - 2
            if rank not in shapes[name]:
                print(f"{name:10} {str(shape):22} {'—':>8}  rank {rank} is rejected")
                continue
            print(f"{name:10} {str(shape):22} {_loss_row(name, shape)}")
    _metrics()


def _loss_row(name: str, shape: tuple[int, ...]) -> str:
    try:
        module = create_loss(name, image_size=max(shape))
    except ModuleNotFoundError as error:
        return f"{'—':>8}  {error}"
    torch.manual_seed(0)
    logits = torch.randn(*shape, requires_grad=True)
    target = (torch.rand(*shape) > 0.7).float()
    try:
        value = module(logits, target)
        value.backward()
    except Exception as error:
        return f"{'—':>8}  {type(error).__name__}: {error}"
    grad = float(logits.grad.abs().sum())
    samples = []
    for _ in range(3):
        logits.grad = None
        start = time.perf_counter()
        value = module(logits, target)
        value.backward()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    finite = bool(torch.isfinite(value))
    status = "ok" if finite and grad > 0 else f"finite={finite} grad={grad:.3g}"
    return f"{samples[1]:8.1f}  loss={float(value.detach()):.4f} grad={grad:.3f} {status}"


def _metrics() -> None:
    rng = np.random.default_rng(0)
    betti_numbers(np.zeros((4, 4), dtype=np.uint8))
    for label, mask in (
        ("betti 256x256", rng.random((256, 256)) > 0.7),
        ("betti 32^3", rng.random((32, 32, 32)) > 0.7),
    ):
        start = time.perf_counter()
        betti_numbers(mask)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"{label:10} {'':22} {elapsed:8.1f}  {_betti_label(mask)}")
    frame = torch.from_numpy((rng.random((2, 1, 256, 256)) > 0.7).astype(np.float32))
    start = time.perf_counter()
    centerline_dice(frame, frame)
    print(f"{'centerline':10} {'(2, 1, 256, 256)':22} {(time.perf_counter() - start) * 1000:8.1f}  identical masks")


def _betti_label(mask: np.ndarray) -> str:
    return "count=" + str(betti_numbers(mask))


if __name__ == "__main__":
    main()
