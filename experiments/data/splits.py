"""Fixed train, validation, and test id lists."""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from experiments.paths import resolve_path


def split_ids(ids, seed: int, test_size: float, holdout_test_fraction: float):
    ordered = sorted(ids)
    train_ids, holdout_ids = train_test_split(
        ordered,
        test_size=test_size,
        shuffle=True,
        random_state=seed,
    )
    val_ids, test_ids = train_test_split(
        sorted(holdout_ids),
        test_size=holdout_test_fraction,
        shuffle=True,
        random_state=seed,
    )
    return {
        "seed": seed,
        "test_size": test_size,
        "holdout_test_fraction": holdout_test_fraction,
        "train": [str(item) for item in sorted(train_ids)],
        "val": [str(item) for item in sorted(val_ids)],
        "test": [str(item) for item in sorted(test_ids)],
    }


def load_or_create_split(ids, path, seed: int, test_size: float, holdout_test_fraction: float) -> dict:
    split_path = resolve_path(path)
    available = set(ids)
    if split_path.exists():
        with split_path.open() as handle:
            saved = json.load(handle)
        filtered = {}
        for part in ("train", "val", "test"):
            kept = [item for item in saved[part] if item in available]
            filtered[part] = kept
        missing = [part for part in ("train", "val", "test") if not filtered[part]]
        if missing:
            raise RuntimeError(
                f"Split {split_path} has no remaining ids for: {', '.join(missing)}"
            )
        saved.update(filtered)
        return saved

    created = split_ids(ids, seed, test_size, holdout_test_fraction)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w") as handle:
        json.dump(created, handle, indent=2)
    return created
