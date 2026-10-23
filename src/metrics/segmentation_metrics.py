"""
Comprehensive segmentation metrics for evaluating model performance.
Includes Dice, IoU, Precision, Recall, and other standard metrics.
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score


class SegmentationMetrics:
    """Compute standard segmentation metrics."""
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
        
    def update(self, pred, target):
        """
        Update metrics with new predictions and targets.
        
        Args:
            pred: (B, 1, H, W) or (B, H, W) - predictions
            target: (B, H, W) - ground truth
        """
        # Convert to numpy
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Handle different shapes
        if pred.ndim == 4:
            pred = pred.squeeze(1)
        
        # Binarize
        pred_bin = (pred > self.threshold).astype(np.uint8)
        target_bin = target.astype(np.uint8)
        
        self.preds.append(pred_bin)
        self.targets.append(target_bin)
        
    def compute(self):
        """
        Compute all segmentation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        preds = np.concatenate([p.flatten() for p in self.preds])
        targets = np.concatenate([t.flatten() for t in self.targets])
        
        # Standard metrics
        metrics = {
            'accuracy': np.mean(preds == targets),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
            'iou': jaccard_score(targets, preds, zero_division=0),
        }
        
        # Dice score (same as F1 for binary case, but computed directly)
        metrics['dice'] = self._compute_dice(preds, targets)
        
        # Specificity
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def _compute_dice(self, pred, target, smooth=1e-6):
        """
        Compute Dice coefficient.
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        """
        intersection = np.sum(pred * target)
        dice = (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
        return dice
    
    def compute_per_image(self, pred, target):
        """
        Compute metrics for a single image.
        
        Args:
            pred: (H, W) or (1, H, W) - prediction
            target: (H, W) - ground truth
        
        Returns:
            Dictionary with per-image metrics
        """
        # Handle shapes
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        if target.ndim == 3:
            target = target.squeeze(0)
        
        # Binarize
        pred_bin = (pred > self.threshold).astype(np.uint8).flatten()
        target_bin = target.astype(np.uint8).flatten()
        
        metrics = {
            'accuracy': np.mean(pred_bin == target_bin),
            'precision': precision_score(target_bin, pred_bin, zero_division=0),
            'recall': recall_score(target_bin, pred_bin, zero_division=0),
            'f1': f1_score(target_bin, pred_bin, zero_division=0),
            'iou': jaccard_score(target_bin, pred_bin, zero_division=0),
            'dice': self._compute_dice(pred_bin, target_bin),
        }
        
        return metrics