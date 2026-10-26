#!/usr/bin/env python3
"""Developer entry point. The installed package does not include this pipeline.

    python train.py --config configs/experiments/dice.yaml
    python train.py sweep --max-steps 1 --image-size 64 --batch-size 2
    python train.py agreement
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.train import main

if __name__ == "__main__":
    main()
