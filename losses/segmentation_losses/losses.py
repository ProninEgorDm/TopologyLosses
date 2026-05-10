# losses/segmentation_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - sigmoid outputs
            target: (B, H, W) - binary masks
        """
        pred = pred.squeeze(1)  # (B, H, W)
        
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - sigmoid outputs
            target: (B, H, W) - binary masks
        """
        pred = pred.squeeze(1)
        
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """Combined BCE + Dice loss."""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss() if not smooth else nn.BCELoss()
        self.dice = DiceLoss(smooth)
        
    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: (B, 1, H, W) - raw logits
            target: (B, H, W) - binary masks
        """
        pred_sigmoid = torch.sigmoid(pred_logits)
        
        bce_loss = self.bce(pred_sigmoid, target)
        dice_loss = self.dice(pred_sigmoid, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def get_segmentation_loss(loss_name='bce', **kwargs):
    """Factory function to get segmentation loss."""
    
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'combined':
        return CombinedSegmentationLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")