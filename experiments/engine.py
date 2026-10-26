"""One training run: data, logits, registered loss, checkpoint, run directory."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.data.dataset import VessMapDataset, list_image_ids
from experiments.data.splits import load_or_create_split
from experiments.model import build_model
from experiments.paths import default_dataset_dir, repo_root, resolve_path
from topologysegmentation.losses import build_loss
from topologysegmentation.losses.registry import REGISTRY, _ensure_registered
from topologysegmentation.metrics import dice_iou, topology_errors


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    return value


def prepare_run(config: dict, config_path: Path | None = None) -> tuple[dict, Path]:
    data_cfg = config.setdefault("data", {})
    train_cfg = config.setdefault("train", {})
    data_cfg["dataset_path"] = str(resolve_path(data_cfg.get("dataset_path"), default_dataset_dir()))
    seed = int(config.get("seed", 42))
    split_file = data_cfg.get("split_file") or f"splits/vessmap_seed{seed}.json"
    data_cfg["split_file"] = str(resolve_path(split_file))
    name = config.get("name") or (config_path.stem if config_path else "run")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve_path(train_cfg.get("run_root") or "runs") / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()
    with (run_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config, run_dir


def make_loaders(config: dict):
    data_cfg = config["data"]
    dataset_dir = Path(data_cfg["dataset_path"])
    ids = list_image_ids(dataset_dir)
    split = load_or_create_split(
        ids,
        data_cfg["split_file"],
        seed=int(config.get("seed", 42)),
        test_size=float(data_cfg.get("test_size", 0.2)),
        holdout_test_fraction=float(data_cfg.get("holdout_test_fraction", 0.4)),
    )
    image_size = int(data_cfg.get("image_size", 256))
    annotator = data_cfg.get("annotator", "annotator1")
    _ensure_registered()
    needs_skeleton = False
    for entry in config.get("loss", {}).get("terms", []):
        spec = REGISTRY.get(str(entry.get("name", "")).lower())
        if spec and spec.needs_skeleton:
            needs_skeleton = True
    if data_cfg.get("return_skeleton"):
        needs_skeleton = True

    def _dataset(part, train):
        return VessMapDataset(
            split[part],
            dataset_dir=dataset_dir,
            image_size=image_size,
            train=train,
            annotator=annotator,
            return_skeleton=needs_skeleton,
        )

    batch_size = int(data_cfg.get("batch_size", 4))
    num_workers = int(data_cfg.get("num_workers", 0))
    loaders = {}
    for part, train, shuffle in (
        ("train", True, True),
        ("val", False, False),
        ("test", False, False),
    ):
        loaders[part] = DataLoader(
            _dataset(part, train),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders, split


def _move_batch(batch, device):
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    skeleton = batch["skeleton"].to(device) if "skeleton" in batch else None
    return images, labels, skeleton, list(batch["image_id"])


def run_epoch(model, criterion, loader, optimizer, device, max_steps, threshold, train: bool):
    model.train(train)
    totals = []
    part_sums: dict[str, float] = {}
    dice_values = []
    iou_values = []
    per_image = []
    topology_batches = []
    steps = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        progress = tqdm(loader, desc="train" if train else "eval", leave=False)
        for batch in progress:
            images, labels, skeleton, image_ids = _move_batch(batch, device)
            logits = model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                raise RuntimeError(
                    f"Model spatial size {tuple(logits.shape[-2:])} does not match labels {tuple(labels.shape[-2:])}"
                )
            loss, parts = criterion(logits, labels, skeleton)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            probabilities = torch.sigmoid(logits.detach())
            dice, iou = dice_iou(probabilities, labels, threshold=threshold)
            totals.append(float(loss.detach()))
            for name, value in parts.items():
                part_sums[name] = part_sums.get(name, 0.0) + float(value.detach())
            dice_values.append(dice.detach().cpu())
            iou_values.append(iou.detach().cpu())
            for image_id, one_dice, one_iou in zip(image_ids, dice.detach().cpu(), iou.detach().cpu()):
                per_image.append({"image_id": image_id, "dice": float(one_dice), "iou": float(one_iou)})
            if not train:
                topology_batches.append((probabilities.cpu(), labels.cpu()))
            steps += 1
            progress.set_postfix(loss=f"{np.mean(totals):.4f}", dice=f"{torch.cat(dice_values).mean().item():.4f}")
            if max_steps is not None and steps >= max_steps:
                break
    dice_all = torch.cat(dice_values) if dice_values else torch.zeros(1)
    iou_all = torch.cat(iou_values) if iou_values else torch.zeros(1)
    summary = {
        "loss": float(np.mean(totals) if totals else 0.0),
        "dice": float(dice_all.mean()),
        "iou": float(iou_all.mean()),
        "parts": {name: value / max(steps, 1) for name, value in part_sums.items()},
        "per_image": per_image,
        "steps": steps,
    }
    if topology_batches and max_steps is None:
        probabilities = torch.cat([item[0] for item in topology_batches], dim=0)
        labels = torch.cat([item[1] for item in topology_batches], dim=0)
        topo = topology_errors(probabilities, labels, threshold=threshold)
        summary["betti_b0_mae"] = topo["betti_b0_mae"]
        summary["betti_b1_mae"] = topo["betti_b1_mae"]
        summary["centerline_dice"] = topo["centerline_dice"]
        for record, topo_record in zip(summary["per_image"], topo["per_image_topology"]):
            record.update(topo_record)
    return summary


def train_from_config(config: dict, config_path: Path | None = None) -> Path:
    device_name = config.setdefault("train", {}).get("device", "cpu")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA was requested ({device_name}) but is unavailable. Using cpu.")
        device_name = "cpu"
        config["train"]["device"] = device_name
    config, run_dir = prepare_run(config, config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device(device_name)

    loaders, split = make_loaders(config)
    with (run_dir / "split.json").open("w") as handle:
        json.dump(split, handle, indent=2)

    image_size = int(config["data"]["image_size"])
    model = build_model(config.get("model", {}), image_size).to(device)
    criterion = build_loss(config.get("loss", {}), device=device_name, image_size=image_size)
    parameters = [parameter for parameter in list(model.parameters()) + list(criterion.parameters()) if parameter.requires_grad]
    train_cfg = config["train"]
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(train_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps else None
    threshold = float(train_cfg.get("metric_threshold", 0.5))

    history = []
    best_dice = -1.0
    for epoch in range(epochs):
        train_summary = run_epoch(
            model, criterion, loaders["train"], optimizer, device, max_steps, threshold, train=True
        )
        val_limit = 1 if max_steps else None
        val_summary = run_epoch(
            model, criterion, loaders["val"], optimizer, device, val_limit, threshold, train=False
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": train_summary["loss"],
            "train_dice": train_summary["dice"],
            "train_parts": train_summary["parts"],
            "val_loss": val_summary["loss"],
            "val_dice": val_summary["dice"],
            "val_iou": val_summary["iou"],
            "val_parts": val_summary["parts"],
        }
        for key in ("betti_b0_mae", "betti_b1_mae"):
            if key in val_summary:
                row[key] = val_summary[key]
        history.append(row)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss {row['train_loss']:.4f}  "
            f"train_dice {row['train_dice']:.4f}  val_dice {row['val_dice']:.4f}  parts {row['train_parts']}"
        )
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_dice": val_summary["dice"],
        }
        torch.save(state, run_dir / "checkpoints" / "last.pt")
        if val_summary["dice"] > best_dice:
            best_dice = val_summary["dice"]
            torch.save(state, run_dir / "checkpoints" / "best.pt")
            with (run_dir / "val_per_image.json").open("w") as handle:
                json.dump(_json_ready(val_summary["per_image"]), handle, indent=2)
        if max_steps:
            break

    test_limit = 1 if max_steps else None
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_summary = run_epoch(
        model, criterion, loaders["test"], optimizer, device, test_limit, threshold, train=False
    )
    result = {
        "run_dir": str(run_dir),
        "best_val_dice": best_dice,
        "test_dice": test_summary["dice"],
        "test_iou": test_summary["iou"],
        "test_loss": test_summary["loss"],
        "history": history,
        "smoke": bool(max_steps),
    }
    for key in ("betti_b0_mae", "betti_b1_mae"):
        if key in test_summary:
            result[key] = test_summary[key]
    with (run_dir / "history.json").open("w") as handle:
        json.dump(_json_ready(result), handle, indent=2)
    with (run_dir / "test_per_image.json").open("w") as handle:
        json.dump(_json_ready(test_summary["per_image"]), handle, indent=2)
    print(f"run saved to {run_dir}")
    print(f"best val dice {best_dice:.4f}  test dice {test_summary['dice']:.4f}")
    return run_dir


def train_from_file(path: str | Path, overrides: dict | None = None) -> Path:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    config = load_config(config_path)
    for section, values in (overrides or {}).items():
        if section in {"seed", "name"}:
            config[section] = values
            continue
        config.setdefault(section, {}).update(values)
    return train_from_config(config, config_path)
