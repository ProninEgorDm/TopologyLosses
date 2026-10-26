"""Soft clDice.

Authors: Suprosanna Shit, Johannes C. Paetzold, Anjany Sekuboyina, Ivan Ezhov,
Alexander Unger, Andrey Zhylka, Josien P. W. Pluim, Ulrich Bauer, Bjoern H. Menze.
Paper: https://arxiv.org/abs/2003.07311
Code: https://github.com/jocpae/clDice
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeleton import SoftSkeletonize

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:]
            y_pred = y_pred[:, 1:]
        skel_pred, skel_true = _both_skeletons(self.soft_skeletonize, y_pred, y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        if self.exclude_background:
            y_true = y_true[:, 1:]
            y_pred = y_pred[:, 1:]
        dice = soft_dice(y_true, y_pred)
        skel_pred, skel_true = _both_skeletons(self.soft_skeletonize, y_pred, y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice


def _both_skeletons(skeletonize, prediction, target):
    """One skeletonization over the stacked pair. Pooling does not mix batch items."""
    skel_pred, skel_true = skeletonize(torch.cat((prediction, target), dim=0)).split(prediction.shape[0], dim=0)
    return skel_pred, skel_true
