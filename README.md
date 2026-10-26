# topologysegmentation

[![PyPI](https://img.shields.io/badge/pypi-topologysegmentation-3775A9)](https://pypi.org/project/topologysegmentation)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2B%20Apache--2.0-green)](https://github.com/ProninEgorDm/TopologyLosses/blob/main/LICENSE)

PyTorch losses and metrics for topology-aware segmentation. Overlap losses stay high when a vessel or airway is broken in the middle. These terms penalize that: soft skeletons, persistence diagrams, and Betti matching, for a 2D frame or a 3D volume.

| Paper | Loss |
| --- | --- |
| Shit et al., [clDice](https://arxiv.org/abs/2003.07311) | `cldice` |
| Hu, Li, Samaras, Chen, [NeurIPS 2019](https://arxiv.org/abs/1906.05404) | `hu` |
| Stucki et al., [Betti matching, ICML 2023](https://proceedings.mlr.press/v202/stucki23a) | `bm`, `bm3d` |
| Wen et al., [SAT](https://arxiv.org/abs/2412.02076) | `sat` |
| González-Jiménez et al., [T-Loss](https://github.com/Digital-Dermatology/t-loss) | `tloss` |

The package is on PyPI: [pypi.org/project/topologysegmentation](https://pypi.org/project/topologysegmentation).

```bash
pip install topologysegmentation          # Dice, clDice, pooling, critical points, T-Loss
pip install "topologysegmentation[ph]"    # also Hu, Betti matching, and SAT
```

```python
import torch
from topologysegmentation import DiceLoss, create_loss, loss_shapes, topology_errors

logits = torch.randn(2, 1, 64, 64)   # frame (N, 1, H, W) or volume (N, 1, D, H, W)
target = torch.zeros(2, 1, 64, 64)
target[:, :, 16:48, 16:48] = 1

dice = DiceLoss()
cldice = create_loss("cldice")
value = dice(logits, target) + 0.1 * cldice(logits, target)
report = topology_errors(logits.sigmoid(), target)
loss_shapes()   # {"dice": (2, 3), "sat": (2,), ...}
```

Every loss applies sigmoid itself. Pass logits. `topology_errors` is eval-only and expects probabilities. Python 3.12 is required. `bm3d` needs a local build of [Betti-Matching-3D](https://github.com/nstucki/Betti-Matching-3D): put its `betti_matching` module on `PYTHONPATH`.

Third-party licenses are listed in [NOTICE](https://github.com/ProninEgorDm/TopologyLosses/blob/main/NOTICE). A short install walkthrough is in [QUICKSTART.md](https://github.com/ProninEgorDm/TopologyLosses/blob/main/QUICKSTART.md).

## Installation

The base package depends only on `torch` and `numpy`. Homology lives in the `ph` extra.

| Command | What you get |
| --- | --- |
| `pip install topologysegmentation` | `dice`, `bce`, `focal`, `combined`, `cldice`, `tloss`, `pooling`, `critical`, Dice/IoU, centerline Dice |
| `pip install "topologysegmentation[ph]"` | also `hu`, `bm`, `sat`, and Betti numbers |
| `pip install "topologysegmentation[plot]"` | plots for Betti matching |
| `pip install "topologysegmentation[all]"` | both extras |

## Shapes

A frame is `(N, 1, H, W)`. A volume is `(N, 1, D, H, W)`. One frame may be `(H, W)` or a channel-free batch `(N, H, W)`. A volume keeps the channel axis: `(D, H, W)` is read as a batch of frames. A side of length 1 is still that axis: a `1×N` strip is a frame, and a slab of depth 1 is a volume.

`loss_shapes()` maps each trainable name to the spatial ranks it accepts. `2` is a frame, `3` is a volume.

| Ranks | Losses |
| --- | --- |
| 2 and 3 | `dice`, `bce`, `focal`, `combined`, `cldice`, `tloss`, `pooling`, `critical`, `hu`, `bm3d` |
| 2 | `sat`, `bm` |

`sat` and `bm` raise `ValueError` on a volume. For a volume, Betti matching is `bm3d`.

## Losses

Every loss is called the same way: `create_loss(name)(logits, target)`.

| Name | What it does | Dependency | Weight next to Dice |
| --- | --- | --- | --- |
| `dice`, `bce`, `focal`, `combined` | Overlap and per-pixel losses. Any spatial rank | package | 1 |
| `cldice` | Soft skeleton, Shit et al., [arXiv:2003.07311](https://arxiv.org/abs/2003.07311). Ten pooling iterations, frames and volumes | package | 0.1–0.3 |
| `tloss` | Robust Student-t likelihood for a noisy mask, per voxel. `nu` is learned. Any shape | package | 0.1 |
| `pooling` | Multi-scale occupancy after max-pooling. A frame adds two axis projections, a volume adds three | package | 1 |
| `critical` | Endpoints, junctions, and isolated voxels. Eight neighbors on a frame, twenty-six on a volume | package | 1 |
| `hu` | Diagram matching toward birth 1 and death 0, extra bars toward the diagonal. Hu et al., NeurIPS 2019. Frames and volumes | `ph` | 0.1 |
| `bm` | Betti matching, Stucki et al., ICML 2023. Frames | `ph` | 0.1 |
| `bm3d` | The same objective through the C++ extension. Frames and volumes | compiled `betti_matching` | 0.1 |
| `sat` | Spatially aware diagram matching, divided by the image side length. Frames | `ph` | 0.1 |

Foreground voxels that touch at a corner are one component. `betti_numbers` on a frame is `(β0, β1)`. On a volume it is `(β0, β1, β2)`. An empty mask is all zeros. A single foreground voxel is `(1, 0)` on a frame and `(1, 0, 0)` on a volume.

## Speed

One CPU (Apple M1), forward and backward together. Frames are VessMAP labels (`annotator1`, native 256×256), resized with nearest-neighbor interpolation. The prediction is that same mask dilated by one pixel, with 12% of the vessel pixels removed, passed as logits ±4. A leading number is the batch: `3×1024²` is three such frames. `sat` and `bm` are frames only, so this table has no volumes.

| Loss | 256² | 512² | 1024² | 3×1024² | 4×256² |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dice` | 0.4 ms | 0.6 ms | 3.1 ms | 7.8 ms | 0.7 ms |
| `bce` | 0.5 ms | 1.5 ms | 4.7 ms | 13 ms | 1.0 ms |
| `focal` | 1.2 ms | 2.8 ms | 12 ms | 35 ms | 3.1 ms |
| `combined` | 1.1 ms | 1.9 ms | 7.2 ms | 25 ms | 2.0 ms |
| `tloss` | 0.6 ms | 1.0 ms | 3.2 ms | 8.7 ms | 0.8 ms |
| `pooling` | 6.7 ms | 25 ms | 0.11 s | 0.15 s | 9.3 ms |
| `critical` | 3.3 ms | 9.3 ms | 47 ms | 0.12 s | 7.7 ms |
| `cldice` | 99 ms | 0.41 s | 1.9 s | 5.3 s | 0.32 s |
| `hu` | 0.26 s | 1.1 s | 4.5 s | 14 s | 1.0 s |
| `sat` | 0.24 s | 1.0 s | 4.2 s | 13 s | 0.98 s |
| `bm` | 0.64 s | 4.4 s | 73 s | 4.8 min | 2.8 s |

These masks have a handful of components and loops (for example `(β0, β1)` of `(5, 0)`, `(4, 7)`, `(19, 4)`). Times up to 512² are the median of three passes. At 1024² and for `3×1024²`, `cldice`, `hu`, `sat`, and `bm` are one pass after a warmup.

## Metrics

`dice_iou(probabilities, target)` returns per-batch Dice and IoU after a 0.5 threshold. Frames and volumes both work. On a mask compared with itself the score is 1, including a single voxel.

`centerline_dice(probabilities, target)` is the Dice score between morphological skeletons.

`betti_numbers(mask)` is `(β0, β1)` or `(β0, β1, β2)`. The binary count does not import `gudhi`. `hu` does, through the `ph` extra, because it needs the critical voxels of the diagram.

`topology_errors(probabilities, target)` reports the mean absolute Betti error (including `betti_b2_mae` on a volume), the centerline Dice, and a per-image breakdown. This is a validation report, not a loss. A frame does not grow a `β2` term when one side is 1; a one-slice volume does.

## Citation

GitHub shows a “Cite this repository” button from [`CITATION.cff`](https://github.com/ProninEgorDm/TopologyLosses/blob/main/CITATION.cff). If you use a bundled loss in a paper, also cite the paper in the table above.

## License

Original code is MIT, see [LICENSE](https://github.com/ProninEgorDm/TopologyLosses/blob/main/LICENSE). T-Loss stays under Apache-2.0, so the package license expression is `MIT AND Apache-2.0`. The file table is in [NOTICE](https://github.com/ProninEgorDm/TopologyLosses/blob/main/NOTICE).
