"""Compare the two VessMAP annotators on the frames both of them labeled."""

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.paths import default_dataset_dir, resolve_path


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask.astype(np.float32) / 255.0 > 0.5


def _dice(left: np.ndarray, right: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.logical_and(left, right).sum()
    return float((2.0 * intersection + eps) / (left.sum() + right.sum() + eps))


def annotator_agreement(dataset_dir: Path | None = None) -> dict:
    root = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    first = {path.stem for path in (root / "annotator1" / "labels").glob("*.png")}
    second = {path.stem for path in (root / "annotator2" / "labels").glob("*.png")}
    shared = sorted(first & second)
    per_image = []
    for image_id in shared:
        left = _read_mask(root / "annotator1" / "labels" / f"{image_id}.png")
        right = _read_mask(root / "annotator2" / "labels" / f"{image_id}.png")
        per_image.append({"image_id": image_id, "dice": _dice(left, right)})
    scores = [item["dice"] for item in per_image]
    return {
        "dataset": str(root),
        "shared_frames": len(shared),
        "annotator1_frames": len(first),
        "annotator2_frames": len(second),
        "mean_dice": float(np.mean(scores) if scores else 0.0),
        "min_dice": float(np.min(scores) if scores else 0.0),
        "per_image": per_image,
    }


def write_agreement(output: Path | None = None, dataset_dir: Path | None = None) -> dict:
    report = annotator_agreement(dataset_dir)
    destination = resolve_path(output) if output else resolve_path("runs/annotator_agreement.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(
        f"annotator agreement on {report['shared_frames']} shared frames: "
        f"mean dice {report['mean_dice']:.4f}, min dice {report['min_dice']:.4f}"
    )
    print(f"wrote {destination}")
    return report
