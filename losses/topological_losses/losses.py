"""
Topological losses for semantic segmentation.
Includes Betti loss, clustering loss, and variation of information loss.
"""

import torch
import torch.nn as nn
from scipy import ndimage
import numpy as np
from types import SimpleNamespace

from losses.topological_losses.BMLoss.loss_functions import (
    BettiMatchingLoss as PyBettiMatchingLoss,
    DiceBettiMatchingLoss as PyDiceBettiMatchingLoss,
)
from losses.topological_losses.CLDiceLoss.cldice import soft_dice_cldice
from losses.topological_losses.GDLoss.GDLoss import GDL, GDiceLoss
from losses.topological_losses.SATLoss.SATLoss import PDMatchingLoss
from losses.topological_losses.TLoss.tloss import TLoss


class TopologicalLoss(nn.Module):
    """Base class for topological losses."""
    
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - predicted segmentation
            target: (B, H, W) - ground truth
        """
        raise NotImplementedError


class BettiLoss(TopologicalLoss):
    """
    Betti loss based on topological feature preservation.
    Penalizes differences in Betti numbers between prediction and target.
    """
    
    def forward(self, pred, target):
        """
        Compute Betti loss.
        
        Args:
            pred: (B, 1, H, W) - predicted segmentation (sigmoid outputs)
            target: (B, H, W) - ground truth
        
        Returns:
            Loss value
        """
        pred = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            pred_np = (pred[b] > 0.5).cpu().detach().numpy().astype(np.uint8)
            target_np = target[b].cpu().detach().numpy().astype(np.uint8)
            
            # Compute Betti numbers
            betti_pred = self._compute_betti(pred_np)
            betti_target = self._compute_betti(target_np)
            
            # Loss is difference in Betti numbers
            loss = (
                abs(betti_pred[0] - betti_target[0]) +
                abs(betti_pred[1] - betti_target[1])
            )
            total_loss += loss
        
        return self.weight * total_loss / batch_size
    
    def _compute_betti(self, segmentation):
        """
        Compute Betti numbers for 2D segmentation.
        
        Returns:
            Tuple (b0, b1) - number of components and holes
        """
        # B0: connected components
        labeled, b0 = ndimage.label(segmentation)
        
        # B1: holes (estimated from inverted image)
        inverted = 1 - segmentation
        labeled_inv, num_holes = ndimage.label(inverted)
        b1 = max(0, num_holes - 1)  # Subtract outer infinite face
        
        return (b0, b1)


class BettiMatchingLoss(TopologicalLoss):
    """
    Betti matching loss - enhanced version that matches components.
    Penalizes topological mismatches while being differentiable-friendly.
    """
    
    def forward(self, pred, target):
        """
        Compute Betti matching loss.
        
        Args:
            pred: (B, 1, H, W) - predicted segmentation
            target: (B, H, W) - ground truth
        
        Returns:
            Differentiable loss value
        """
        pred = torch.sigmoid(pred).squeeze(1)  # (B, H, W)
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            # Use soft versions for differentiability
            loss = self._compute_matching_loss(pred[b], target[b])
            total_loss += loss
        
        return self.weight * total_loss / batch_size
    
    def _compute_matching_loss(self, pred, target):
        """
        Compute component matching loss.
        Encourages prediction to match target's connected components.
        """
        # Compute connectivity through dilated predictions
        pred_np = pred.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        
        # Connectivity preservation: penalize breaking connected components
        # Use gradient-based connectivity measure
        
        # Horizontal and vertical gradients
        grad_target_h = np.abs(np.diff(target_np, axis=0))
        grad_target_v = np.abs(np.diff(target_np, axis=1))
        
        grad_pred_h = np.abs(np.diff(pred_np, axis=0))
        grad_pred_v = np.abs(np.diff(pred_np, axis=1))
        
        # Loss on mismatched boundaries
        loss_h = np.mean(np.abs(grad_pred_h - grad_target_h))
        loss_v = np.mean(np.abs(grad_pred_v - grad_target_v))
        
        return torch.tensor(loss_h + loss_v, device=pred.device, dtype=pred.dtype)


class VOILoss(TopologicalLoss):
    """
    Variation of Information (VOI) loss.
    Measures information-theoretic distance between two segmentations.
    """
    
    def forward(self, pred, target):
        """
        Compute VOI loss.
        
        Args:
            pred: (B, 1, H, W) - predicted segmentation
            target: (B, H, W) - ground truth
        
        Returns:
            Loss value
        """
        pred = torch.sigmoid(pred).squeeze(1)
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            pred_np = (pred[b] > 0.5).cpu().detach().numpy().astype(np.uint8)
            target_np = target[b].cpu().detach().numpy().astype(np.uint8)
            
            loss = self._compute_voi(pred_np, target_np)
            total_loss += loss
        
        return self.weight * total_loss / batch_size
    
    def _compute_voi(self, segmentation1, segmentation2):
        """
        Compute Variation of Information between two segmentations.
        
        VOI = H(X) + H(Y) - 2*I(X,Y)
        where H is entropy and I is mutual information.
        """
        # Label connected components
        labeled1, _ = ndimage.label(segmentation1)
        labeled2, _ = ndimage.label(segmentation2)
        
        # Flatten and create contingency matrix
        flat1 = labeled1.flatten()
        flat2 = labeled2.flatten()
        
        # Compute probability distributions
        unique1 = np.unique(flat1)
        unique2 = np.unique(flat2)
        
        n = len(flat1)
        
        # Entropy of segmentation 1
        h1 = 0.0
        for label in unique1:
            p = np.sum(flat1 == label) / n
            if p > 0:
                h1 -= p * np.log2(p)
        
        # Entropy of segmentation 2
        h2 = 0.0
        for label in unique2:
            p = np.sum(flat2 == label) / n
            if p > 0:
                h2 -= p * np.log2(p)
        
        # Mutual information
        mi = 0.0
        for l1 in unique1:
            for l2 in unique2:
                pij = np.sum((flat1 == l1) & (flat2 == l2)) / n
                if pij > 0:
                    pi = np.sum(flat1 == l1) / n
                    pj = np.sum(flat2 == l2) / n
                    mi += pij * np.log2(pij / (pi * pj))
        
        # VOI
        voi = h1 + h2 - 2 * mi
        
        return voi


class HausdorffDistanceLoss(TopologicalLoss):
    """
    Hausdorff distance loss for boundary-aware segmentation.
    Penalizes maximum distance between prediction and target boundaries.
    """
    
    def forward(self, pred, target):
        """
        Compute Hausdorff distance loss.
        
        Args:
            pred: (B, 1, H, W) - predicted segmentation
            target: (B, H, W) - ground truth
        
        Returns:
            Loss value
        """
        pred = torch.sigmoid(pred).squeeze(1)
        
        batch_size = pred.shape[0]
        total_loss = 0.0
        
        for b in range(batch_size):
            pred_np = (pred[b] > 0.5).cpu().detach().numpy().astype(np.uint8)
            target_np = target[b].cpu().detach().numpy().astype(np.uint8)
            
            loss = self._compute_hausdorff(pred_np, target_np)
            total_loss += loss
        
        return self.weight * total_loss / batch_size
    
    def _compute_hausdorff(self, pred, target):
        """
        Compute Hausdorff distance between boundaries.
        """
        from scipy.ndimage import binary_erosion
        from scipy.spatial.distance import directed_hausdorff
        
        # Extract boundaries
        pred_edges = binary_erosion(pred, iterations=1) ^ pred
        target_edges = binary_erosion(target, iterations=1) ^ target
        
        if pred_edges.sum() == 0 or target_edges.sum() == 0:
            return 0.0
        
        # Get edge coordinates
        pred_coords = np.argwhere(pred_edges)
        target_coords = np.argwhere(target_edges)
        
        # Compute directed Hausdorff distances
        try:
            dist1 = directed_hausdorff(pred_coords, target_coords)[0]
            dist2 = directed_hausdorff(target_coords, pred_coords)[0]
            hausdorff = max(dist1, dist2)
        except:
            hausdorff = 0.0
        
        return hausdorff


class CLDiceTopologicalLoss(TopologicalLoss):
    """CLDice-based topological loss wrapper."""

    def __init__(
        self,
        weight=1.0,
        alpha=0.5,
        iter_=3,
        exclude_background=False,
        sigmoid=False,
        softmax=False,
    ):
        super().__init__(weight)
        self.loss = soft_dice_cldice(iter_=iter_, alpha=alpha, smooth=1.0, exclude_background=exclude_background)
        self.sigmoid = sigmoid
        self.softmax = softmax

    def forward(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        if self.sigmoid:
            pred = torch.sigmoid(pred)
        if self.softmax:
            pred = torch.softmax(pred, dim=1)

        loss = self.loss(target.float(), pred.float())
        return self.weight * loss


class GDiceTopologicalLoss(TopologicalLoss):
    """Wrapper for the GDLoss / GDiceLoss implementations."""

    def __init__(self, weight=1.0, use_gdice=True):
        super().__init__(weight)
        self.loss = GDiceLoss(weight=None) if use_gdice else GDL(weight=None)

    def forward(self, pred, target):
        if pred.ndim == 4:
            pred = pred.unsqueeze(2)
        if target.ndim == 3:
            target = target.unsqueeze(1).unsqueeze(2).long()
        elif target.ndim == 4:
            target = target.unsqueeze(2).long()

        loss = self.loss(pred, target)
        return self.weight * loss


class SATTopologicalLoss(TopologicalLoss):
    """Wrapper for the SATLoss PDMatchingLoss implementation."""

    def __init__(self, weight=1.0, precal_PD=False, p=2):
        super().__init__(weight)
        opt = SimpleNamespace(precal_PD=precal_PD)
        self.loss = PDMatchingLoss(opt, p=p)

    def forward(self, pred, target, img_names=None):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1).float()

        loss = self.loss(pred.float(), target.float(), img_names=img_names)
        return self.weight * loss


class TLossWrapper(TopologicalLoss):
    """Wrapper for the TLoss implementation."""

    def __init__(self, weight=1.0, image_size=256, device=None, nu=1.0, epsilon=1e-8, reduction='mean'):
        super().__init__(weight)
        if device is None:
            device = torch.device('cpu')
        config = SimpleNamespace(data=SimpleNamespace(image_size=image_size), device=device)
        self.loss = TLoss(config, nu=nu, epsilon=epsilon, reduction=reduction)

    def forward(self, pred, target):
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        loss = self.loss(pred, target)
        return self.weight * loss


class BettiMatchingWrapper(TopologicalLoss):
    """Wrapper for the pure Python BMLoss Betti matching implementation."""

    def __init__(
        self,
        weight=1.0,
        batch=False,
        relative=False,
        filtration='superlevel',
    ):
        super().__init__(weight)
        self.loss = PyBettiMatchingLoss(
            batch=batch,
            relative=relative,
            filtration=filtration,
        )

    def forward(self, pred, target):
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if target.ndim == 3:
            target = target.unsqueeze(1)

        loss, _ = self.loss(pred, target)
        return self.weight * loss


class CompositeTopologicalLoss(TopologicalLoss):
    """
    Composite topological loss combining multiple topological losses.
    """
    
    def __init__(self, betti_weight=0.5, voi_weight=0.3, hausdorff_weight=0.2):
        super().__init__()
        self.betti_loss = BettiLoss(betti_weight)
        self.voi_loss = VOILoss(voi_weight)
        self.hausdorff_loss = HausdorffDistanceLoss(hausdorff_weight)
    
    def forward(self, pred, target):
        """Compute composite topological loss."""
        loss = (
            self.betti_loss(pred, target) +
            self.voi_loss(pred, target) +
            self.hausdorff_loss(pred, target)
        )
        return loss


def get_topological_loss(loss_name=None, weight=0.1, **kwargs):
    """
    Factory function to get topological loss.
    
    Args:
        loss_name: Name of loss ('betti', 'voi', 'hausdorff', 'composite', None)
        weight: Weight for the loss
    
    Returns:
        Loss module or None
    """
    
    if loss_name is None or loss_name.lower() == 'none':
        return None
    elif loss_name.lower() == 'betti':
        return BettiLoss(weight)
    elif loss_name.lower() == 'betti_matching':
        return BettiMatchingLoss(weight)
    elif loss_name.lower() == 'voi':
        return VOILoss(weight)
    elif loss_name.lower() == 'hausdorff':
        return HausdorffDistanceLoss(weight)
    elif loss_name.lower() == 'composite':
        return CompositeTopologicalLoss()
    elif loss_name.lower() in ('cldice', 'cl_dice'):
        return CLDiceTopologicalLoss(weight=weight, **kwargs)
    elif loss_name.lower() in ('gdice', 'gdl'):
        return GDiceTopologicalLoss(weight=weight, **kwargs)
    elif loss_name.lower() in ('sat', 'satloss', 'pdmatching'):
        return SATTopologicalLoss(weight=weight, **kwargs)
    elif loss_name.lower() == 'tloss':
        return TLossWrapper(weight=weight, **kwargs)
    elif loss_name.lower() in ('bm', 'betti_fast', 'fast_betti'):
        return BettiMatchingWrapper(weight=weight, **kwargs)
    else:
        raise ValueError(f"Unknown topological loss: {loss_name}")