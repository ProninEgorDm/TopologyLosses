"""
QUICK START GUIDE - VessMAP Semantic Segmentation Pipeline
===========================================================

This guide will help you get started with training and evaluating models
on the VessMAP vessel segmentation dataset.
"""

# ============================================================================
# STEP 1: VERIFY SETUP
# ============================================================================

"""
Before training, verify that all dependencies are installed and configured:

$ python verify_setup.py

This will check:
- All required Python packages
- Project structure and files
- Configuration file validity
- Dataset availability
- GPU support
"""

# ============================================================================
# STEP 2: BASIC TRAINING
# ============================================================================

"""
Train a model with the default configuration:

$ python train.py --config configs/default_config.yaml

This will:
- Load the VessMAP dataset
- Initialize the TransUNet model
- Train for 100 epochs (configurable)
- Save checkpoints and logs
- Output metrics every epoch

Configuration can be modified in configs/default_config.yaml
"""

# ============================================================================
# STEP 3: COMPARE DIFFERENT LOSS FUNCTIONS
# ============================================================================

"""
Compare performance of multiple loss functions:

$ python train.py --config configs/default_config.yaml \
                  --compare dice,bce,focal,combined

This will:
- Train separate models for each loss function
- Compare results on test set
- Generate comparison report: logs/loss_comparison.json
- Create visualizations

Available losses:
- dice: Dice Loss
- bce: Binary Cross-Entropy
- focal: Focal Loss
- combined: BCE + Dice (weighted)
"""

# ============================================================================
# STEP 4: INCLUDE TOPOLOGICAL LOSSES
# ============================================================================

"""
To preserve vessel topology during training, modify configs/default_config.yaml:

loss:
  segmentation_loss: 'dice'
  topological_loss: 'betti'      # Enable Betti loss
  topological_weight: 0.1         # Weight of topological component

Available topological losses:
- betti: Basic Betti number loss
- betti_matching: Enhanced component matching
- voi: Variation of Information
- hausdorff: Hausdorff distance
- composite: Combination of all
"""

# ============================================================================
# STEP 5: EVALUATE TRAINED MODEL
# ============================================================================

"""
Evaluate a trained model on test set:

$ python evaluate.py --checkpoint checkpoints/best_model.pt

This will:
- Compute metrics on all splits
- Generate evaluation report
- Save results to JSON file

Generate prediction visualizations:

$ python evaluate.py --checkpoint checkpoints/best_model.pt \
                     --generate-samples 20 \
                     --sample-split test

This creates 20 sample visualizations in predictions/ folder
"""

# ============================================================================
# STEP 6: ANALYZE LOSS COMPARISON RESULTS
# ============================================================================

"""
After running loss comparison, analyze the results:

$ python evaluate.py --compare-losses logs/loss_comparison.json

This will:
- Print comparison summary
- Create comparison visualization
- Identify best performing loss
"""

# ============================================================================
# PROGRAMMATIC USAGE
# ============================================================================

"""
Use the pipeline directly in Python code:

from Train.pipeline import SegmentationPipeline

# Create and train pipeline
pipeline = SegmentationPipeline('configs/default_config.yaml')
pipeline.train()

# Evaluate on test set
results = pipeline.evaluate(pipeline.test_loader)

# Access metrics
test_dice = results['metrics']['f1']
test_iou = results['metrics']['iou']
"""

# ============================================================================
# INTERACTIVE NOTEBOOK
# ============================================================================

"""
For interactive exploration and experimentation:

jupyter notebook pipeline_tutorial.ipynb

This notebook includes:
- Data loading and visualization
- Model initialization
- Training loop demonstration
- Metrics computation
- Results visualization
- Loss comparison
"""

# ============================================================================
# KEY OUTPUT FILES
# ============================================================================

"""
After training, you'll find:

checkpoints/
  - best_model.pt              # Best model on validation
  - checkpoint_epoch_*.pt      # Checkpoints every epoch

logs/
  - loss_comparison.json       # Loss comparison results
  - training_history.json      # Training metrics history
  - loss_<name>/events.*       # TensorBoard logs

predictions/
  - test_sample_*.png          # Generated predictions

evaluation_report.json         # Detailed evaluation metrics
"""

# ============================================================================
# COMMON CONFIGURATIONS
# ============================================================================

"""
Minimal (Quick Testing):
  epochs: 5
  batch_size: 16
  image_size: 128
  
Balanced (Standard):
  epochs: 100
  batch_size: 8
  image_size: 256
  
High Quality (Large GPU):
  epochs: 200
  batch_size: 4
  image_size: 512

Edit configs/default_config.yaml to change these settings.
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
GPU out of memory?
- Reduce batch_size
- Reduce image_size
- Reduce out_channels

Poor vessel topology?
- Add topological_loss to config
- Increase topological_weight
- Use 'composite' topological loss

Training not converging?
- Lower learning_rate
- Use different loss function
- Check data normalization
- Enable topological regularization

For more detailed troubleshooting, see README.md
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
1. Run verify_setup.py to ensure everything is installed
2. Read README.md for comprehensive documentation
3. Check pipeline_tutorial.ipynb for interactive examples
4. Start with basic training on small dataset (5 epochs)
5. Compare different loss functions
6. Evaluate best model and analyze predictions
7. Fine-tune configuration for your specific needs
8. Implement custom losses or metrics if needed
"""
