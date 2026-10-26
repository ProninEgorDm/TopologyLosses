import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.model import build_model
from topologysegmentation.losses import build_loss
from topologysegmentation.losses.registry import METRIC_ONLY, create_loss


def _target(size=32):
    target = torch.zeros(2, 1, size, size)
    target[:, :, 8:20, 8:24] = 1
    return target


class ContractTest(unittest.TestCase):
    def test_model_returns_decoder_logits(self):
        model = build_model(
            {"out_channels": 128, "block_num": 1, "mlp_dim": 64, "head_num": 4, "patch_dim": 16},
            image_size=64,
        )
        model.eval()
        images = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            logits = model(images)
            encoded = model.encoder(images)
            decoded = model.decoder(*encoded)
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        self.assertTrue(torch.allclose(logits, decoded))

    def test_model_train_step_has_gradient(self):
        model = build_model(
            {"out_channels": 128, "block_num": 1, "mlp_dim": 64, "head_num": 4, "patch_dim": 16},
            image_size=64,
        )
        model.train()
        images = torch.randn(2, 1, 64, 64)
        logits = model(images)
        logits.mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertGreater(sum(item.abs().sum().item() for item in gradients), 0)

    def test_dice_does_not_import_gudhi_and_has_gradient(self):
        self.assertNotIn("gudhi", sys.modules)
        composer = build_loss({"terms": [{"name": "dice", "weight": 1.0}]}, device="cpu", image_size=32)
        self.assertNotIn("gudhi", sys.modules)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        total, parts = composer(logits, _target())
        total.backward()
        self.assertIn("dice", parts)
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_metric_only_names_are_rejected(self):
        for name in sorted(METRIC_ONLY):
            with self.assertRaises(ValueError):
                create_loss(name, device="cpu", image_size=32)


class DataTest(unittest.TestCase):
    def test_split_is_stable_and_sample_has_logits_layout(self):
        from experiments.data.dataset import VessMapDataset, list_image_ids
        from experiments.data.splits import load_or_create_split
        from experiments.paths import default_dataset_dir

        dataset_dir = default_dataset_dir()
        if not dataset_dir.exists():
            self.skipTest("VessMAP is not on disk")
        ids = list_image_ids(dataset_dir)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            first = load_or_create_split(ids, path, seed=42, test_size=0.2, holdout_test_fraction=0.4)
            second = load_or_create_split(ids, path, seed=1, test_size=0.5, holdout_test_fraction=0.5)
        self.assertEqual(first["train"], second["train"])
        self.assertEqual(len(first["train"]) + len(first["val"]) + len(first["test"]), len(ids))
        sample = VessMapDataset(first["val"][:1], dataset_dir, image_size=64, train=False)[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 64, 64))
        self.assertEqual(tuple(sample["label"].shape), (1, 64, 64))
        self.assertGreaterEqual(float(sample["image"].min()), 0.0)
        self.assertLessEqual(float(sample["image"].max()), 1.0)
        self.assertTrue(set(torch.unique(sample["label"]).tolist()).issubset({0.0, 1.0}))

    def test_annotator_agreement_floor(self):
        from experiments.agreement import annotator_agreement
        from experiments.paths import default_dataset_dir

        if not default_dataset_dir().exists():
            self.skipTest("VessMAP is not on disk")
        report = annotator_agreement()
        self.assertEqual(report["shared_frames"], 20)
        self.assertGreater(report["mean_dice"], 0.0)
        self.assertLessEqual(report["mean_dice"], 1.0)

    def test_betti_error_is_eval_only(self):
        from topologysegmentation.metrics import topology_errors

        probabilities = torch.zeros(2, 1, 16, 16)
        probabilities[0, 0, 2:6, 2:6] = 1
        target = torch.zeros(2, 1, 16, 16)
        target[0, 0, 2:6, 2:6] = 1
        target[1, 0, 1:4, 1:3] = 1
        report = topology_errors(probabilities, target)
        self.assertEqual(report["betti_b0_mae"], 0.5)
        self.assertIn("centerline_dice", report)
        self.assertIn("per_image_topology", report)

    def test_cubical_betti_matches_the_complex(self):
        import numpy as np

        from topologysegmentation.metrics import betti_numbers, centerline_dice

        from topologysegmentation.persistence import cubical_bars

        diagonal = np.zeros((6, 6))
        diagonal[2, 2] = 1
        diagonal[3, 3] = 1
        self.assertEqual(betti_numbers(diagonal), (1, 0))
        separated = np.zeros((6, 6))
        separated[2, 2] = 1
        separated[2, 4] = 1
        self.assertEqual(betti_numbers(separated), (2, 0))
        ring = np.zeros((8, 8))
        ring[2:6, 2:6] = 1
        ring[3:5, 3:5] = 0
        self.assertEqual(betti_numbers(ring), (1, 1))
        self.assertEqual(betti_numbers(np.zeros((6, 6))), (0, 0))
        spot = np.zeros((4, 6))
        spot[1, 3] = 1
        alive = [bar for bar in cubical_bars(spot) if bar.alive(0.5)]
        self.assertEqual([bar.birth for bar in alive], [(1, 3)])
        mask = torch.zeros(1, 1, 8, 8)
        mask[0, 0, 2:6, 3:5] = 1
        score = centerline_dice(mask, mask)
        self.assertGreater(float(score), 0.99)


class TopologyLossTest(unittest.TestCase):
    def _backward(self, name, size, **kwargs):
        module = create_loss(name, device="cpu", image_size=size, **kwargs)
        logits = torch.randn(2, 1, size, size, requires_grad=True)
        value = module(logits, _target(size))
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        return module

    def test_cldice_gradient(self):
        self._backward("cldice", 32, alpha=0.5)

    def test_tloss_gradient_reaches_nu(self):
        module = self._backward("tloss", 32, nu=1.0)
        self.assertIsNotNone(module.inner.nu.grad)
        before = float(module.inner.nu.detach())
        optimizer = torch.optim.Adam([module.inner.nu], lr=0.1)
        optimizer.step()
        self.assertNotEqual(before, float(module.inner.nu.detach()))

    def test_tloss_scale_matches_overlap_losses(self):
        module = create_loss("tloss", device="cpu", image_size=32, nu=1.0)
        value = module(torch.randn(2, 1, 32, 32), _target(32))
        self.assertLess(float(value.detach()), 20.0)

    def test_bce_focal_combined(self):
        for name in ("bce", "focal", "combined"):
            self._backward(name, 16)

    def test_bm_is_available_or_explains_why_not(self):
        try:
            module = create_loss("bm", device="cpu", image_size=32)
        except ModuleNotFoundError as error:
            self.assertIn("gudhi", str(error))
            return
        self.assertTrue(hasattr(module, "forward"))

    def test_bm_is_zero_on_a_recovered_mask_and_penalizes_an_extra_component(self):
        try:
            from topologysegmentation.losses.vendor.bm.BettiMatching import CubicalPersistence

            module = create_loss("bm", device="cpu", image_size=16)
        except ModuleNotFoundError:
            self.skipTest("gudhi is not installed")
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 6:10] = 1
        clean = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        clean.requires_grad_(True)
        matched = module(clean, target)
        matched.backward()
        self.assertLess(float(matched.detach()), 1e-6)
        self.assertEqual(float(clean.grad.abs().sum()), 0.0)

        extra = clean.detach().clone()
        extra[:, :, 1:3, 1:3] = 6
        extra.requires_grad_(True)
        dirty = module(extra, target)
        dirty.backward()
        self.assertGreater(float(dirty.detach()), 0.5)
        self.assertGreater(float(extra.grad[:, :, 1:3, 1:3].abs().sum()), 0.0)
        self.assertEqual(float(extra.grad[:, :, 4:, :].abs().sum()), 0.0)

        # Vertex construction: pixels that only touch at a corner are two components.
        diagonal = torch.zeros(8, 8)
        diagonal[2, 2] = 1
        diagonal[3, 3] = 1
        self.assertEqual(CubicalPersistence(diagonal, filtration="superlevel").get_Betti_numbers(), [2, 0])
        ring = torch.zeros(10, 10)
        ring[2:8, 2:8] = 1
        ring[3:7, 3:7] = 0
        self.assertEqual(CubicalPersistence(ring, filtration="superlevel").get_Betti_numbers(), [1, 1])

    def test_sat_is_available_or_explains_why_not(self):
        try:
            module = create_loss("sat", device="cpu", image_size=32)
        except ModuleNotFoundError as error:
            self.assertIn("sat", str(error))
            return
        self.assertTrue(hasattr(module, "forward"))

    def test_sat_square_maps_keep_the_border(self):
        try:
            from topologysegmentation.losses.vendor.sat.SATLoss import PDMatchingLoss
        except ModuleNotFoundError:
            self.skipTest("SAT dependencies are not installed")
        loss = PDMatchingLoss(SimpleNamespace(precal_PD=False), p=2)
        image = torch.zeros(1, 1, 8, 8)
        image[:, :, 3:5, 3:5] = 1
        prepared, height, width = loss._prepare_map(image)
        self.assertEqual((height, width), (10, 10))
        border = torch.cat(
            (prepared[:, :, 0, :].reshape(-1), prepared[:, :, -1, :].reshape(-1), prepared[:, :, :, 0].reshape(-1), prepared[:, :, :, -1].reshape(-1))
        )
        self.assertTrue(torch.all(border == 0))

        wide = torch.zeros(1, 1, 8, 4)
        prepared, height, width = loss._prepare_map(wide)
        self.assertEqual((height, width), (10, 10))

    def test_sat_empty_diagrams_skip_transport(self):
        try:
            from torch_topological.nn import PersistenceInformation
            from topologysegmentation.losses.vendor.sat.PDMatching import SpatialAware_WassersteinDistance
        except ModuleNotFoundError:
            self.skipTest("SAT dependencies are not installed")
        empty = PersistenceInformation(
            pairing=torch.zeros(0, 4),
            diagram=torch.zeros(0, 2, requires_grad=True),
            dimension=1,
        )
        distance = SpatialAware_WassersteinDistance(p=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = distance(empty, empty, 8, 8)
        value.backward()
        self.assertEqual(float(value.detach()), 0.0)
        self.assertFalse(any("infeasible" in str(item.message).lower() for item in caught))

    def test_bm3d_matching_loss_reaches_critical_pixels(self):
        import numpy as np

        from topologysegmentation.losses.vendor.bm3d.loss import matching_loss

        prediction = torch.zeros(4, 4, requires_grad=True)
        target = torch.zeros(4, 4)
        target[0, 0] = 1.0
        result = SimpleNamespace(
            input1_matched_birth_coordinates=[np.array([[0, 0]])],
            input1_matched_death_coordinates=[np.array([[1, 1]])],
            input2_matched_birth_coordinates=[np.array([[0, 0]])],
            input2_matched_death_coordinates=[np.array([[3, 3]])],
            num_matched=np.array([1]),
            input1_unmatched_birth_coordinates=[np.array([[2, 2]])],
            input1_unmatched_death_coordinates=[np.array([[2, 3]])],
            num_unmatched_input1=np.array([1]),
        )
        value = matching_loss(prediction, target, result)
        value.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[3, 0]), 0.0)

    def test_bm3d_is_available_or_explains_why_not(self):
        try:
            module = create_loss("bm3d", device="cpu", image_size=32)
        except ModuleNotFoundError as error:
            self.assertIn("bm3d", str(error))
            return
        self.assertTrue(hasattr(module, "forward"))

    def test_pooling_and_critical_have_gradient(self):
        self._backward("pooling", 16)
        self._backward("critical", 16)

    def test_pooling_vanishes_when_the_mask_is_recovered(self):
        module = create_loss("pooling", device="cpu", image_size=16)
        target = _target(16)[:1]
        logits = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        self.assertLess(float(module(logits, target).detach()), 1e-4)

    def test_hu_penalizes_an_extra_component(self):
        try:
            module = create_loss("hu", device="cpu", image_size=16)
        except ModuleNotFoundError as error:
            self.assertIn("gudhi", str(error))
            return
        target = torch.zeros(1, 1, 16, 16)
        target[0, 0, 2:6, 2:6] = 1
        clean = torch.where(target > 0.5, torch.full_like(target, 20.0), torch.full_like(target, -20.0))
        matched = module(clean, target)
        self.assertLess(float(matched.detach()), 1e-3)
        extra = clean.detach().clone()
        extra[0, 0, 10:13, 10:13] = 6.0
        extra.requires_grad_()
        value = module(extra, target)
        value.backward()
        self.assertGreater(float(value.detach()), float(matched.detach()))
        self.assertGreater(float(extra.grad[0, 0, 10:13, 10:13].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
