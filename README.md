# VessMAP Semantic Segmentation Pipeline

A comprehensive training and validation pipeline for the VessMAP vessel segmentation dataset, supporting multiple loss functions and topological metrics.

## Features

### 1. **Complete Training Pipeline**
- Full training/validation/test workflow
- Support for multiple segmentation loss functions
- Optional topological losses for preserving structure
- Checkpointing and early stopping
- TensorBoard logging

### 2. **Multiple Loss Functions**

#### Segmentation Losses:
- **BCE** - Binary Cross-Entropy loss
- **Dice** - Dice coefficient loss
- **Focal** - Focal loss for class imbalance
- **Combined** - Weighted combination of BCE and Dice

#### Topological Losses:
- **Betti Loss** - Preserves topological features (connected components, holes)
- **Betti Matching** - Component-aware matching loss
- **VOI (Variation of Information)** - Information-theoretic loss
- **Hausdorff Distance** - Boundary-aware loss
- **Composite** - Weighted combination of multiple topological losses

### 3. **Comprehensive Metrics**

#### Segmentation Metrics:
- Accuracy
- Precision & Recall
- F1 Score (Dice)
- IoU (Jaccard Index)
- Specificity

#### Topological Metrics:
- Betti Numbers (connectivity, holes)
- Connected Components
- Connectivity Preservation
- Skeleton Accuracy
- Hausdorff Distance

#### Vessel-Specific Metrics:
- Bifurcation Points
- Branch Count
- Endpoint Preservation

### 4. **Loss Function Comparison**
- Train models with multiple losses simultaneously
- Automated comparison and reporting
- Visualization of results

### 5. **Advanced Evaluation**
- Per-image metrics
- Prediction visualization
- Detailed performance reports
- Comparison analysis across loss functions

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install tensorboard tqdm pyyaml
pip install scikit-learn scikit-image
pip install monai
pip install opencv-python tifffile
pip install scipy numpy
pip install matplotlib seaborn
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
data:
  dataset_path: '/path/to/VessMAP/'
  image_size: 256
  batch_size: 8
  num_workers: 2

model:
  name: 'transunet'
  in_channels: 1
  out_channels: 128

training:
  epochs: 100
  learning_rate: 0.0003
  device: 'cuda'  # or 'cpu'
  checkpoint_dir: './checkpoints'
  log_dir: './logs'

loss:
  segmentation_loss: 'focal'  # 'bce', 'dice', 'focal', 'combined'
  topological_loss: None       # None, 'betti', 'voi', 'hausdorff', 'composite'
  topological_weight: 0.1
```

## Usage

### 1. Basic Training

```bash
# Train with default configuration
python train.py --config configs/default_config.yaml

# Train with specific loss function
# Modify configs/default_config.yaml first, then:
python train.py --config configs/default_config.yaml
```

### 2. Compare Multiple Loss Functions

```bash
# Train and compare multiple losses
python train.py --config configs/default_config.yaml \
                --compare dice,bce,focal,combined

# Results will be saved to logs/loss_comparison.json
```

### 3. Resume Training

```bash
# Resume from checkpoint
python train.py --config configs/default_config.yaml \
                --checkpoint checkpoints/best_model.pt
```

### 4. Evaluate Model

```bash
# Full evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt

# Generate prediction samples
python evaluate.py --checkpoint checkpoints/best_model.pt \
                   --generate-samples 10 \
                   --sample-split test

# Compute per-image metrics
python evaluate.py --checkpoint checkpoints/best_model.pt \
                   --per-image-metrics
```

### 5. Compare Loss Function Results

```bash
# Analyze loss comparison results
python evaluate.py --compare-losses logs/loss_comparison.json
```

## Programmatic Usage

```python
from Train.pipeline import SegmentationPipeline

# Create pipeline
pipeline = SegmentationPipeline('configs/default_config.yaml')

# Train
pipeline.train()

# Evaluate
results = pipeline.evaluate(pipeline.test_loader)
print(f"Test Dice: {results['metrics']['f1']:.4f}")
print(f"Test IoU: {results['metrics']['iou']:.4f}")

# Save and load checkpoints
pipeline.load_checkpoint('checkpoints/best_model.pt')
```

## Training with Topological Losses

To include topological losses in training:

```yaml
# configs/default_config.yaml
loss:
  segmentation_loss: 'dice'
  topological_loss: 'betti'        # Enable Betti loss
  topological_weight: 0.1           # Weight of topological loss
```

Combined loss = segmentation_loss + topological_weight * topological_loss

Available topological losses:
- `betti` - Basic Betti loss
- `betti_matching` - Enhanced component matching
- `voi` - Variation of Information
- `hausdorff` - Hausdorff distance
- `composite` - Combination of all topological losses

## Output Structure

```
checkpoints/
  checkpoint_epoch_000.pt    # Checkpoint every epoch
  best_model.pt              # Best model on validation

logs/
  loss_comparison.json       # Loss comparison results
  training_history.json      # Training metrics history
  loss_<name>/
    events.out.*             # TensorBoard logs
  comparisons/
    *.png                    # Comparison visualizations

predictions/
  test_sample_000.png        # Sample predictions
  test_sample_001.png
  ...
```

## Metrics Output

### Training History (training_history.json)
```json
{
  "train_loss": [...],
  "val_loss": [...],
  "train_metrics": {
    "accuracy": [...],
    "precision": [...],
    "recall": [...],
    "f1": [...],
    "iou": [...]
  },
  "val_metrics": {
    ...same structure...
  }
}
```

### Loss Comparison (loss_comparison.json)
```json
{
  "dice": {
    "test_loss": 0.123,
    "test_metrics": {
      "accuracy": 0.95,
      "f1": 0.87,
      "iou": 0.82,
      ...
    }
  },
  "bce": {...},
  "focal": {...},
  "combined": {...}
}
```

## Advanced Features

### Custom Loss Functions

Extend the base loss classes:

```python
from losses.segmentation_losses import DiceLoss
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        # Custom loss computation
        dice_loss = self.dice(pred, target)
        # Add your custom logic
        return dice_loss
```

### Custom Metrics

```python
from metrics.segmentation_metrics import SegmentationMetrics

class CustomMetrics(SegmentationMetrics):
    def compute(self):
        metrics = super().compute()
        # Add custom metrics
        metrics['custom_metric'] = ...
        return metrics
```

### Data Augmentation

Modify transforms in `data/transforms.py`:

```python
def get_train_transforms(input_shape=(256, 256)):
    return Compose([
        # Add your custom transforms
        ...
    ])
```

## Troubleshooting

### GPU Out of Memory
- Reduce `batch_size` in config
- Reduce `image_size`
- Use gradient accumulation

### Training Divergence
- Lower `learning_rate`
- Enable `topological_loss` to stabilize
- Check data normalization

### Poor Topology Preservation
- Increase `topological_weight`
- Use `composite` topological loss
- Check skeletal structures in data

## Performance Tips

1. **Data Loading**: Use more `num_workers` for faster loading
2. **Augmentation**: Enable augmentation only in training
3. **Model Size**: Adjust `out_channels` based on memory
4. **Loss Weighting**: Experiment with topological loss weights

## References

- Binary Cross-Entropy Loss
- Dice Loss: Milletari et al., 2016
- Focal Loss: Lin et al., 2017
- Variation of Information: Meila, 2007
- Hausdorff Distance: For boundary-based comparison
- Betti Numbers: Topological data analysis

## Directory Structure

```
TopologyLosses/
├── Train/
│   └── pipeline.py          # Main pipeline
├── data/
│   ├── dataset.py           # Dataset class
│   └── transforms.py        # Data augmentation
├── models/
│   ├── TransUnet.py
│   ├── vit.py
│   └── attention.py
├── losses/
│   ├── segmentation_losses/
│   │   └── losses.py
│   └── topological_losses/
│       └── losses.py
├── metrics/
│   ├── segmentation_metrics.py
│   └── topological_metrcis.py
├── configs/
│   └── default_config.yaml
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── USAGE_EXAMPLES.py        # Example usage
└── checkpoints/             # Saved models
```

## Future Extensions

- [ ] 3D vessel segmentation
- [ ] Multi-annotator consensus
- [ ] Confidence calibration
- [ ] Uncertainty estimation
- [ ] Multi-task learning (vessel + bifurcation detection)
- [ ] Active learning pipeline

## License

Project specific license

## Contact

For questions or issues, contact the development team.
