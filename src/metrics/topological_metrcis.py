"""
Topological metrics for analyzing segmentation quality.
Includes Betti numbers, connectivity analysis, and other topological features.
"""

import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from skimage import measure


class TopologicalMetrics:
    """Compute topological metrics for segmentation evaluation."""
    
    def __init__(self, threshold=0.5):
        """
        Initialize topological metrics.
        
        Args:
            threshold: Threshold for binarizing predictions
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset metrics."""
        self.preds = []
        self.targets = []
    
    def update(self, pred, target):
        """
        Update with new predictions and targets.
        
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
        Compute all topological metrics.
        
        Returns:
            Dictionary with topological metrics
        """
        metrics = {
            'betti_pred': self._compute_betti_numbers(np.concatenate(self.preds, axis=0)),
            'betti_target': self._compute_betti_numbers(np.concatenate(self.targets, axis=0)),
            'num_components_pred': self._count_components(np.concatenate(self.preds, axis=0)),
            'num_components_target': self._count_components(np.concatenate(self.targets, axis=0)),
            'connectivity_match': self._compute_connectivity_match(
                np.concatenate(self.preds, axis=0),
                np.concatenate(self.targets, axis=0)
            ),
            'skeleton_accuracy': self._compute_skeleton_accuracy(
                np.concatenate(self.preds, axis=0),
                np.concatenate(self.targets, axis=0)
            ),
            'genus_diff': 0.0,  # Placeholder
        }
        
        return metrics
    
    def _compute_betti_numbers(self, segmentation):
        """
        Compute Betti numbers (0th, 1st) for segmentation.
        
        Betti_0: Number of connected components
        Betti_1: Number of holes (cycles)
        
        Args:
            segmentation: (N, H, W) binary segmentation
        
        Returns:
            Dictionary with Betti numbers
        """
        betti = {'b0': 0, 'b1': 0}
        
        for mask in segmentation:
            # Betti 0: connected components
            labeled, num_components = ndimage_label(mask)
            betti['b0'] += num_components
            
            # Betti 1: compute using Euler characteristic
            # For 2D: χ = V - E + F = b0 - b1
            # We estimate holes by looking at connected components in complement
            inverted = 1 - mask
            labeled_inv, num_holes = ndimage_label(inverted)
            # Subtract 1 for the outer infinite face
            betti['b1'] += max(0, num_holes - 1)
        
        betti['b0'] = betti['b0'] / len(segmentation) if len(segmentation) > 0 else 0
        betti['b1'] = betti['b1'] / len(segmentation) if len(segmentation) > 0 else 0
        
        return betti
    
    def _count_components(self, segmentation):
        """Count connected components."""
        num_components = 0
        for mask in segmentation:
            _, n = ndimage_label(mask)
            num_components += n
        return num_components / len(segmentation) if len(segmentation) > 0 else 0
    
    def _compute_connectivity_match(self, pred, target):
        """
        Compute how well prediction preserves connectivity of target.
        
        Returns:
            Score between 0 and 1 (higher is better)
        """
        match_score = 0.0
        count = 0
        
        for p, t in zip(pred, target):
            if t.sum() == 0:  # Skip empty targets
                continue
            
            # Label connected components
            labeled_pred, num_pred = ndimage_label(p)
            labeled_target, num_target = ndimage_label(t)
            
            # Compute IoU for each target component
            scores = []
            for comp_id in range(1, num_target + 1):
                target_mask = labeled_target == comp_id
                # Find most overlapping prediction component
                overlaps = np.array([
                    np.sum((labeled_pred == pred_id) & target_mask)
                    for pred_id in range(1, num_pred + 1)
                ])
                if len(overlaps) > 0 and overlaps.max() > 0:
                    best_pred_id = overlaps.argmax() + 1
                    pred_mask = labeled_pred == best_pred_id
                    iou = np.sum(pred_mask & target_mask) / np.sum(pred_mask | target_mask)
                    scores.append(iou)
            
            if len(scores) > 0:
                match_score += np.mean(scores)
                count += 1
        
        return match_score / count if count > 0 else 0.0
    
    def _compute_skeleton_accuracy(self, pred, target):
        """
        Compute skeleton-based accuracy using medial axis.
        
        Returns:
            Fraction of target skeleton pixels correctly predicted
        """
        accuracy = 0.0
        count = 0
        
        for p, t in zip(pred, target):
            if t.sum() == 0:
                continue
            
            # Compute skeleton (medial axis transform)
            target_skeleton = ndimage.binary_erosion(t)
            
            # Check how many skeleton points are predicted
            if target_skeleton.sum() > 0:
                correct_skeleton = np.sum(p[target_skeleton.astype(bool)])
                skeleton_acc = correct_skeleton / target_skeleton.sum()
                accuracy += skeleton_acc
                count += 1
        
        return accuracy / count if count > 0 else 0.0
    
    def _compute_hausdorff_distance(self, pred, target):
        """
        Compute Hausdorff distance between prediction and target.
        
        Measures the maximum distance from any point in one set to the nearest point in the other.
        
        Returns:
            Average Hausdorff distance
        """
        from scipy.spatial.distance import directed_hausdorff
        
        distances = []
        
        for p, t in zip(pred, target):
            if t.sum() == 0 or p.sum() == 0:
                continue
            
            # Get edge points
            target_edges = ndimage.binary_erosion(t, iterations=1) ^ t
            pred_edges = ndimage.binary_erosion(p, iterations=1) ^ p
            
            if target_edges.sum() > 0 and pred_edges.sum() > 0:
                target_points = np.argwhere(target_edges)
                pred_points = np.argwhere(pred_edges)
                
                # Compute directed Hausdorff distances
                dist1 = directed_hausdorff(target_points, pred_points)[0]
                dist2 = directed_hausdorff(pred_points, target_points)[0]
                distances.append(max(dist1, dist2))
        
        return np.mean(distances) if len(distances) > 0 else float('inf')
    
    def betti_matching_loss(self, pred, target):
        """
        Compute loss based on Betti number mismatch.
        
        Lower values indicate better topological preservation.
        
        Returns:
            Loss value
        """
        pred_betti = self._compute_betti_numbers(np.concatenate(self.preds, axis=0))
        target_betti = self._compute_betti_numbers(np.concatenate(self.targets, axis=0))
        
        loss = (
            abs(pred_betti['b0'] - target_betti['b0']) +
            abs(pred_betti['b1'] - target_betti['b1'])
        )
        
        return loss


class VesselTopologyMetrics(TopologicalMetrics):
    """
    Specialized topological metrics for vessel segmentation.
    Includes metrics specific to vascular networks.
    """
    
    def compute_vessel_metrics(self, pred, target):
        """
        Compute vessel-specific topological metrics.
        
        Args:
            pred: Predicted segmentation
            target: Target segmentation
        
        Returns:
            Dictionary with vessel-specific metrics
        """
        metrics = {}
        
        # Convert to binary
        if torch.is_tensor(pred):
            pred = (pred > self.threshold).cpu().numpy()
        else:
            pred = (pred > self.threshold).astype(np.uint8)
        
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        else:
            target = target.astype(np.uint8)
        
        # Bifurcation analysis
        metrics['pred_bifurcations'] = self._count_bifurcations(pred)
        metrics['target_bifurcations'] = self._count_bifurcations(target)
        
        # Branch analysis
        metrics['pred_num_branches'] = self._count_branches(pred)
        metrics['target_num_branches'] = self._count_branches(target)
        
        # Endpoint preservation
        metrics['endpoint_preservation'] = self._compute_endpoint_preservation(pred, target)
        
        return metrics
    
    def _count_bifurcations(self, segmentation):
        """Count bifurcation points in vessel network."""
        if segmentation.sum() == 0:
            return 0
        
        # Use skeleton and analyze junction points
        skeleton = ndimage.binary_erosion(segmentation, iterations=1) ^ segmentation
        
        # Count junctions (pixels with multiple neighbors)
        bifurcations = 0
        h, w = skeleton.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if skeleton[i, j]:
                    neighbors = skeleton[i-1:i+2, j-1:j+2].sum() - 1  # Exclude center
                    if neighbors >= 3:  # Bifurcation or higher order junction
                        bifurcations += 1
        
        return bifurcations
    
    def _count_branches(self, segmentation):
        """Count number of vessel branches."""
        if segmentation.sum() == 0:
            return 0
        
        # Label connected components as separate branches
        labeled, num_branches = ndimage_label(segmentation)
        return num_branches
    
    def _compute_endpoint_preservation(self, pred, target):
        """
        Compute how well prediction preserves vessel endpoints.
        
        Returns:
            Fraction of target endpoints present in prediction
        """
        if target.sum() == 0:
            return 0.0
        
        # Find endpoints (pixels with exactly one neighbor)
        target_endpoints = 0
        preserved_endpoints = 0
        
        h, w = target.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                if target[i, j]:
                    neighbors = target[i-1:i+2, j-1:j+2].sum() - 1
                    if neighbors == 1:  # Endpoint
                        target_endpoints += 1
                        if pred[i, j]:
                            preserved_endpoints += 1
        
        return preserved_endpoints / max(target_endpoints, 1)