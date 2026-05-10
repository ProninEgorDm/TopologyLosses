"""
Example scripts demonstrating how to use the training and evaluation pipeline.
"""

# Example 1: Basic training with default configuration
# python train.py --config configs/default_config.yaml

# Example 2: Training with specific loss function
# python train.py --config configs/default_config.yaml

# Then modify configs/default_config.yaml to change:
# loss:
#   segmentation_loss: 'dice'  # or 'bce', 'focal', 'combined'
#   topological_loss: None      # or 'betti', 'voi', 'hausdorff', 'composite'

# Example 3: Compare multiple loss functions
# python train.py --config configs/default_config.yaml --compare dice,bce,focal,combined

# Example 4: Resume training from checkpoint
# python train.py --config configs/default_config.yaml --checkpoint checkpoints/best_model.pt

# Example 5: Evaluate trained model
# python evaluate.py --checkpoint checkpoints/best_model.pt

# Example 6: Evaluate and generate prediction samples
# python evaluate.py --checkpoint checkpoints/best_model.pt --generate-samples 10 --sample-split test

# Example 7: Per-image metrics computation
# python evaluate.py --checkpoint checkpoints/best_model.pt --per-image-metrics

# Example 8: Compare different loss functions results
# python evaluate.py --compare-losses logs/loss_comparison.json


# ============================================================================
# PROGRAMMATIC USAGE EXAMPLES
# ============================================================================

# Example: Training with custom pipeline
from Train.pipeline import SegmentationPipeline

# Create pipeline
pipeline = SegmentationPipeline('configs/default_config.yaml')

# Train the model
pipeline.train()

# Evaluate on validation set
val_results = pipeline.evaluate(pipeline.val_loader)
print(f"Val Loss: {val_results['loss']:.4f}")
print(f"Val Dice: {val_results['metrics']['f1']:.4f}")

# Evaluate on test set
test_results = pipeline.evaluate(pipeline.test_loader)
print(f"Test Loss: {test_results['loss']:.4f}")
print(f"Test IoU: {test_results['metrics']['iou']:.4f}")


# Example: Loss comparison
from Train.pipeline import MultiLossComparator

losses_to_compare = ['dice', 'bce', 'focal', 'combined']
comparator = MultiLossComparator('configs/default_config.yaml', losses_to_compare)
comparator.run_comparison()


# Example: Model evaluation
from evaluate import ModelEvaluator

evaluator = ModelEvaluator('checkpoints/best_model.pt')

# Full evaluation
results = evaluator.evaluate_full()

# Generate visualizations
evaluator.generate_prediction_samples(num_samples=10, split='test')

# Compute per-image metrics
per_image_metrics = evaluator.compute_per_image_metrics(split='val')


# Example: Loss comparison analysis
from evaluate import ComparisonAnalyzer

analyzer = ComparisonAnalyzer('logs/loss_comparison.json')
analyzer.print_summary()
analyzer.plot_comparison(save_path='comparison.png')
