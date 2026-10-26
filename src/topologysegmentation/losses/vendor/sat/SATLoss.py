"""Spatial-aware persistent feature matching (SAT).

Authors: Bo Wen, Haochen Zhang, Dirk-Uwe G. Bartsch, William R. Freeman,
Truong Q. Nguyen, Cheolhong An.
Paper: https://arxiv.org/abs/2412.02076
Code: https://github.com/JRC-VPLab/SATLoss (MIT), original https://github.com/bw05025/SATLoss
The one-pixel border follows Hu, Li, Samaras, NeurIPS 2019: https://arxiv.org/abs/1906.05404

Local fixes: a square image keeps that border, and an empty diagram is not
passed to the transport solver. The training adapter divides by the image side.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cubical_complex import CubicalComplex
from .PDMatching import SpatialAware_WassersteinDistance

class PDMatchingLoss(nn.Module):
    def __init__(self, opt, p=2):
        super().__init__()
        # Cubical complex constructor for persistent homology computation
        self.getPersistentInfo = CubicalComplex(dim=2)

        # distance between persistent diagrams
        self.criterion = SpatialAware_WassersteinDistance(p=p)

        # For precomputed ground truth persistent diagram.
        self.precal_PD = opt.precal_PD
        self.PD_target = {}
        self.pad_dims = (1, 1, 1, 1)

    def _pad_to_square(self, x1, x2, H, W):
        margin = abs(H - W)
        pad1, pad2 = margin // 2, margin - margin // 2

        if H > W:
            paddings = (pad1, pad2, 0, 0)
        else:
            paddings = (0, 0, pad1, pad2)

        if x1 is not None:
            x1 = F.pad(x1, paddings, "constant", 0.0)
        if x2 is not None:
            x2 = F.pad(x2, paddings, "constant", 0.0)

        return x1, x2

    def _prepare_map(self, image):
        """Border of ones, then a square pad. The cubical complex sees the inverted map.

        Hu et al. pad the frame with ones so the boundary is one component.
        Square images used to drop that frame: the padded tensors were built and
        then ignored whenever H == W.
        """
        image = F.pad(image, self.pad_dims, mode="constant", value=1)
        height, width = image.shape[-2:]
        if height != width:
            image, _ = self._pad_to_square(image, None, height, width)
            height, width = image.shape[-2:]
        image = image.clamp(0.0, 1.0)
        return 1.0 - image, int(height), int(width)

    def _pre_compute_PD(self, target, img_names):
        """
            Pre-compute PD for ground truth and save the training time.
        """
        padded_target, _, _ = self._prepare_map(target)

        for i in range(target.shape[0]):
            img = padded_target[i, 0].unsqueeze(0).unsqueeze(0)
            self.PD_target[img_names[i]] = self.getPersistentInfo(img)

    def forward(self, input, target, img_names=None):
        assert input.size() == target.size()
        assert input.device == target.device
        assert input.size(1) == 1

        input = input.to(torch.float32)
        target = target.to(torch.float32)
        input, height, width = self._prepare_map(input)
        target, target_height, target_width = self._prepare_map(target)
        assert (height, width) == (target_height, target_width)

        pi_x = self.getPersistentInfo(input)
        if self.precal_PD:  # read ground truth persistent diagram from pre-computed
            pi_y = [self.PD_target[img_names[0]][0]]
            for idx in range(1, input.shape[0]):
                pi_y.append(self.PD_target[img_names[idx]][0])
        else:
            pi_y = self.getPersistentInfo(target)

        costs = []
        for i in range(input.shape[0]):
            # 0-th persistent diagram (connected components) and 1-st (loops).
            # height, width are the padded map, so creator coordinates stay in [0, 1].
            costs.append(
                self.criterion(pi_x[i][0][0], pi_y[i][0][0], height, width)
                + self.criterion(pi_x[i][0][1], pi_y[i][0][1], height, width)
            )

        if not costs:
            return input.sum() * 0
        return torch.stack(costs).mean()
