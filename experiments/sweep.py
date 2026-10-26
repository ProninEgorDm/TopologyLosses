"""Train several configs and collect one summary. A failed config does not stop the rest."""

import csv
import json
from datetime import datetime
from pathlib import Path

from experiments.engine import train_from_file
from experiments.paths import repo_root, resolve_path

DEFAULT_CONFIGS = [
    "configs/experiments/dice.yaml",
    "configs/experiments/dice_cldice_w0.1.yaml",
    "configs/experiments/dice_cldice_w0.3.yaml",
    "configs/experiments/dice_tloss.yaml",
    "configs/experiments/dice_bm.yaml",
]


def run_sweep(config_paths: list[str], overrides: dict | None = None, summary_dir: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = summary_dir or (resolve_path("runs") / "sweeps" / stamp)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_path in config_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo_root() / path
        row = {"config": str(path), "status": "ok"}
        try:
            run_dir = train_from_file(path, overrides)
            with (run_dir / "history.json").open() as handle:
                history = json.load(handle)
            row.update(
                {
                    "run_dir": str(run_dir),
                    "best_val_dice": history.get("best_val_dice"),
                    "test_dice": history.get("test_dice"),
                    "test_iou": history.get("test_iou"),
                    "test_loss": history.get("test_loss"),
                }
            )
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"sweep skipped {path.name}: {row['error']}")
        rows.append(row)
    with (destination / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    with (destination / "summary.csv").open("w", newline="") as handle:
        fieldnames = ["config", "status", "best_val_dice", "test_dice", "test_iou", "test_loss", "run_dir", "error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"sweep summary written to {destination}")
    return destination
