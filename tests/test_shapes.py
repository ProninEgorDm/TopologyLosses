"""2D and 3D contracts: finite loss, a gradient, and the published edge cases."""

import os
import subprocess
import sys
import unittest

import numpy as np
import torch

from topologysegmentation import loss_shapes
from topologysegmentation.losses import create_loss
from topologysegmentation.metrics import _skeleton, centerline_dice, dice_iou, topology_errors
from topologysegmentation.persistence import betti_numbers, cubical_bars


def _logits(shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(shape, generator=generator, requires_grad=True)
    target = torch.zeros(shape)
    target.view(shape[0], -1)[:, ::7] = 1.0
    return logits, target


class ShapeTests(unittest.TestCase):
    def test_fast_losses_accept_frames_and_volumes(self):
        for shape in ((2, 1, 32, 32), (2, 1, 8, 16, 16), (32, 32), (2, 32, 32)):
            for name in ("dice", "bce", "focal", "combined", "cldice", "pooling", "critical"):
                logits, target = _logits(shape)
                loss = create_loss(name)(logits, target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_hard_masks_are_near_zero(self):
        frame = torch.full((1, 1, 32, 32), -20.0)
        frame[:, :, 8:24, 14:18] = 20.0
        volume = torch.full((1, 1, 8, 16, 16), -20.0)
        volume[:, :, 2:6, 4:12, 6:10] = 20.0
        for logits in (frame, volume):
            logits.requires_grad_(True)
            target = (logits.detach() > 0).float()
            for name in ("dice", "pooling", "critical"):
                loss = create_loss(name)(logits, target)
                self.assertLess(float(loss.detach()), 1e-3, f"{name} {tuple(logits.shape)}")

    def test_metrics_match_rank(self):
        for shape in ((1, 1, 32, 32), (1, 1, 8, 12, 12)):
            probabilities = torch.zeros(shape)
            probabilities[..., 2:-2, 4:7] = 1.0
            dice, iou = dice_iou(probabilities, probabilities)
            self.assertAlmostEqual(float(dice), 1.0, places=5)
            self.assertAlmostEqual(float(iou), 1.0, places=5)
            self.assertGreater(float(centerline_dice(probabilities, probabilities)), 0.5)

    def test_volume_betti_and_hu(self):
        solid = torch.zeros(12, 12, 12)
        solid[3:9, 3:9, 3:9] = 1.0
        self.assertEqual(betti_numbers(solid.numpy()), (1, 0, 0))
        split = solid.clone()
        split[:, :, 5:7] = 0.0
        self.assertEqual(betti_numbers(split.numpy()), (2, 0, 0))
        self.assertEqual(betti_numbers(torch.zeros(8, 8, 8).numpy()), (0, 0, 0))

        target = solid.view(1, 1, 12, 12, 12)
        clean_logits = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        clean = create_loss("hu")(clean_logits, target)
        extra = clean_logits.detach().clone()
        extra[:, :, 0:2, 0:2, 0:2] = 6.0
        extra.requires_grad_(True)
        dirty = create_loss("hu")(extra, target)
        dirty.backward()
        self.assertLess(float(clean.detach()), 1e-3)
        self.assertGreater(float(dirty.detach()), float(clean.detach()))
        self.assertGreater(float(extra.grad[:, :, 0:2, 0:2, 0:2].abs().sum()), 0.0)

    def test_two_dimensional_losses_reject_volumes(self):
        logits, target = _logits((1, 1, 8, 16, 16))
        for name in ("sat", "bm"):
            with self.assertRaises(ValueError):
                create_loss(name)(logits, target)

    def test_tloss_matches_any_shape_and_trains_nu(self):
        for shape in ((2, 1, 16, 16), (1, 1, 8, 8, 8)):
            logits, target = _logits(shape)
            module = create_loss("tloss", image_size=16)
            loss = module(logits, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(module.inner.nu.grad)
            self.assertLess(float(loss.detach()), 20.0)

    def test_dtype_and_device_follow_the_logits(self):
        logits = torch.randn(1, 1, 16, 16, dtype=torch.float64, requires_grad=True)
        target = torch.zeros(1, 1, 16, 16)
        loss = create_loss("dice")(logits, target)
        self.assertEqual(loss.dtype, torch.float64)
        loss.backward()
        errors = topology_errors(torch.sigmoid(logits.detach()), target)
        self.assertIn("betti_b0_mae", errors)
        self.assertIn("centerline_dice", errors)

    def test_declared_ranks_match_the_forward(self):
        shapes = loss_shapes()
        self.assertEqual(shapes["bm"], (2,))
        self.assertEqual(shapes["sat"], (2,))
        self.assertEqual(shapes["bm3d"], (2, 3))
        frame, volume = (1, 1, 16, 18), (1, 1, 6, 8, 7)
        for name, ranks in shapes.items():
            if name == "bm3d":
                continue
            for rank, shape in ((2, frame), (3, volume)):
                logits, target = _logits(shape, seed=rank)
                if rank in ranks:
                    value = create_loss(name, image_size=16)(logits, target)
                    value.backward()
                    self.assertTrue(torch.isfinite(value), name)
                else:
                    with self.assertRaises(ValueError):
                        create_loss(name, image_size=16)(logits, target)

    def test_unit_sides_stay_frames_or_volumes(self):
        self.assertEqual(betti_numbers(np.ones((1, 1))), (1, 0))
        self.assertEqual(betti_numbers(np.ones((1, 6))), (1, 0))
        self.assertEqual(betti_numbers(np.zeros((1, 1, 1))), (0, 0, 0))
        self.assertEqual(betti_numbers(np.ones((1, 4, 5))), (1, 0, 0))
        frame = torch.ones(1, 1, 1, 5)
        errors = topology_errors(frame, frame)
        self.assertEqual(errors["betti_b0_mae"], 0.0)
        self.assertNotIn("betti_b2_mae", errors)
        slab = torch.ones(1, 1, 1, 4, 5)
        volume_errors = topology_errors(slab, slab)
        self.assertIn("betti_b2_mae", volume_errors)
        self.assertEqual(volume_errors["betti_b0_mae"], 0.0)
        strip = torch.zeros(1, 1, 1, 8)
        strip[0, 0, 0, 2:5] = 1.0
        logits = torch.where(strip > 0.5, torch.full_like(strip, 12.0), torch.full_like(strip, -12.0))
        logits.requires_grad_(True)
        loss = create_loss("bm")(logits, strip)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.detach()), 1e-4)

    def test_betti_matches_the_diagram_on_random_masks(self):
        rng = np.random.default_rng(0)
        for _ in range(6):
            mask = rng.random((18, 21)) > 0.62
            self.assertEqual(betti_numbers(mask), _bars(mask))
        for _ in range(4):
            mask = rng.random((8, 9, 10)) > 0.6
            self.assertEqual(betti_numbers(mask), _bars(mask))
        hollow = np.zeros((10, 10, 10))
        hollow[2:8, 2:8, 2:8] = 1
        hollow[3:7, 3:7, 3:7] = 0
        tunnel = hollow.copy()
        tunnel[2:8, 2:8, 2:8] = 1
        tunnel[4:6, 4:6, 2:8] = 0
        self.assertEqual(betti_numbers(hollow), (1, 0, 1))
        self.assertEqual(betti_numbers(tunnel), (1, 1, 0))
        self.assertEqual(betti_numbers(np.ones((5, 5)), threshold=0.0), (0, 0))
        self.assertEqual(betti_numbers(np.ones((5, 5)), threshold=1.5), (0, 0))

    def test_numpy_fallback_matches_the_diagram(self):
        script = (
            "import os, numpy as np\n"
            "os.environ['TOPOLOGYSEGMENTATION_NATIVE']='0'\n"
            "from topologysegmentation.persistence import betti_numbers, cubical_bars\n"
            "mask = np.zeros((8, 8)); mask[2:6, 2:6] = 1; mask[3:5, 3:5] = 0\n"
            "bars = cubical_bars(mask)\n"
            "counts = [0, 0]\n"
            "for bar in bars:\n"
            "    if bar.alive(0.5): counts[bar.dimension] += 1\n"
            "assert betti_numbers(mask) == tuple(counts) == (1, 1), betti_numbers(mask)\n"
        )
        env = os.environ.copy()
        env["TOPOLOGYSEGMENTATION_NATIVE"] = "0"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_numpy_skeleton_matches_max_pool(self):
        rng = np.random.default_rng(1)
        for shape in ((2, 1, 19, 23), (1, 1, 9, 8, 10)):
            mask = torch.from_numpy((rng.random(shape) > 0.55).astype(np.float32))
            self.assertTrue(torch.equal(_skeleton(mask) > 0.5, _pool_skeleton(mask) > 0.5))


def _bars(mask: np.ndarray) -> tuple[int, ...]:
    counts = [0] * mask.ndim
    for bar in cubical_bars(mask):
        if bar.dimension < len(counts) and bar.alive(0.5):
            counts[bar.dimension] += 1
    return tuple(counts)


def _pool_skeleton(mask: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    remaining = mask.float()
    skeleton = torch.zeros_like(remaining)
    for _ in range(int(max(remaining.shape[2:]))):
        if float(remaining.max()) <= 0.0:
            break
        if remaining.ndim == 5:
            eroded = torch.amin(
                torch.stack(
                    [
                        -F.max_pool3d(-remaining, (3, 1, 1), stride=1, padding=(1, 0, 0)),
                        -F.max_pool3d(-remaining, (1, 3, 1), stride=1, padding=(0, 1, 0)),
                        -F.max_pool3d(-remaining, (1, 1, 3), stride=1, padding=(0, 0, 1)),
                    ]
                ),
                dim=0,
            )
            opened = F.max_pool3d(eroded, 3, stride=1, padding=1)
        else:
            vertical = -F.max_pool2d(-remaining, (3, 1), stride=1, padding=(1, 0))
            horizontal = -F.max_pool2d(-remaining, (1, 3), stride=1, padding=(0, 1))
            eroded = torch.minimum(vertical, horizontal)
            opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        skeleton = torch.maximum(skeleton, (remaining - opened).clamp(min=0.0))
        remaining = eroded
    return skeleton


if __name__ == "__main__":
    unittest.main()
