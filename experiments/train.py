"""Command line for one run, a sweep, or the annotator-agreement floor.

Examples:
    python train.py --config configs/experiments/dice.yaml
    python train.py --config configs/experiments/dice.yaml --max-steps 1 --image-size 64 --batch-size 2
    python train.py sweep --max-steps 1 --image-size 64 --batch-size 2
    python train.py agreement
"""

import argparse
from pathlib import Path

from experiments.agreement import write_agreement
from experiments.engine import train_from_file
from experiments.sweep import DEFAULT_CONFIGS, run_sweep


def _overrides(args) -> dict:
    train = {}
    data = {}
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.max_steps is not None:
        train["max_steps"] = args.max_steps
    if args.device is not None:
        train["device"] = args.device
    if args.batch_size is not None:
        data["batch_size"] = args.batch_size
    if args.image_size is not None:
        data["image_size"] = args.image_size
    if args.num_workers is not None:
        data["num_workers"] = args.num_workers
    overrides = {}
    if train:
        overrides["train"] = train
    if data:
        overrides["data"] = data
    return overrides


def _add_run_flags(parser):
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Batches per split. Also limits training to one epoch.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)


def train_main(argv) -> None:
    parser = argparse.ArgumentParser(description="Train one VessMAP experiment")
    parser.add_argument("--config", default="configs/experiments/dice.yaml")
    _add_run_flags(parser)
    args = parser.parse_args(argv)
    train_from_file(args.config, _overrides(args))


def sweep_main(argv) -> None:
    parser = argparse.ArgumentParser(description="Train each experiment config and write a summary")
    parser.add_argument("--config", action="append", dest="configs")
    _add_run_flags(parser)
    args = parser.parse_args(argv)
    run_sweep(args.configs or DEFAULT_CONFIGS, _overrides(args))


def agreement_main(argv) -> None:
    parser = argparse.ArgumentParser(description="Dice between annotator1 and annotator2 on shared frames")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    dataset = Path(args.dataset) if args.dataset else None
    output = Path(args.output) if args.output else None
    write_agreement(output, dataset)


def main(argv=None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "sweep":
        sweep_main(argv[1:])
        return
    if argv and argv[0] == "agreement":
        agreement_main(argv[1:])
        return
    train_main(argv)


if __name__ == "__main__":
    main()
