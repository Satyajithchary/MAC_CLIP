"""
Loss Functions
==============
All segmentation loss functions used in MAC-CLIP.

  - TverskyLoss       : asymmetric Dice; penalises false negatives
  - IoULoss           : differentiable intersection-over-union
  - DiceLoss          : classic soft Dice
  - FocalLoss         : focal cross-entropy for hard-negative mining
  - BoundaryAwareLoss : BCE with Gaussian distance-weighted boundary emphasis
  - EdgeAwareLoss     : Sobel-based edge consistency
  - ContourLoss       : Laplacian-based contour fidelity
  - BoundaryMapLoss   : weighted BCE on boundary probability map
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class TverskyLoss(nn.Module):
    """Tversky loss with asymmetric FP/FN penalty.

    alpha controls FP penalty, beta controls FN penalty.
    Setting alpha < 0.5, beta > 0.5 penalises false negatives more —
    useful for small lesion segmentation.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p  = pred.contiguous().view(-1)
        t  = target.contiguous().view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        return 1 - (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth)


class IoULoss(nn.Module):
    """Soft intersection-over-union loss (1 − IoU)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p     = pred.contiguous().view(-1)
        t     = target.contiguous().view(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum() - inter
        return 1 - (inter + self.smooth) / (union + self.smooth)


class DiceLoss(nn.Module):
    """Standard soft Dice loss (1 − Dice)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p     = pred.contiguous().view(-1)
        t     = target.contiguous().view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + 1.0) / (p.sum() + t.sum() + 1.0)


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation (hard-negative mining)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            p   = pred.float().clamp(1e-7, 1 - 1e-7)
            t   = target.float()
            bce = F.binary_cross_entropy(p, t, reduction="none")
            pt  = p * t + (1 - p) * (1 - t)
            fw  = (1 - pt) ** self.gamma
            at  = self.alpha * t + (1 - self.alpha) * (1 - t)
            return (at * fw * bce).mean()


class BoundaryAwareLoss(nn.Module):
    """BCE weighted by Gaussian-decaying distance from ground-truth boundary.

    Pixels near the boundary receive weight ≈ 1.0; pixels far away receive weight ≈ 0.2.
    """

    def __init__(self, sigma: float = 8.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weights = []
            for i in range(target.shape[0]):
                gt  = target[i].cpu().numpy()
                bnd = cv2.Canny((gt * 255).astype(np.uint8), 50, 150) > 0
                if bnd.sum() > 10:
                    dist = distance_transform_edt(~bnd)
                    wmap = np.exp(-dist ** 2 / (2 * self.sigma ** 2))
                    wmap = 0.2 + 0.8 * wmap
                else:
                    wmap = np.ones_like(gt)
                weights.append(torch.from_numpy(wmap.astype(np.float32)).to(pred.device))
            weights = torch.stack(weights)

        with torch.cuda.amp.autocast(enabled=False):
            p   = pred.float().clamp(1e-7, 1 - 1e-7)
            bce = F.binary_cross_entropy(p, target.float(), reduction="none")
            return (bce * weights).mean()


class ContourLoss(nn.Module):
    """Laplacian-based contour fidelity loss."""

    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        lap5 = torch.tensor(
            [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, -16, 2, 1],
             [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
        self.register_buffer("lap",  lap.view(1, 1, 3, 3))
        self.register_buffer("lap5", lap5.view(1, 1, 5, 5))

    def _c(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.abs(F.conv2d(x, k, padding=k.shape[-1] // 2))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.unsqueeze(1)   if pred.dim()   == 3 else pred
        t = target.unsqueeze(1) if target.dim() == 3 else target
        return (F.mse_loss(self._c(p, self.lap).clamp(0, 1),
                           self._c(t, self.lap).clamp(0, 1)) +
                0.5 * F.mse_loss(self._c(p, self.lap5).clamp(0, 1),
                                  self._c(t, self.lap5).clamp(0, 1)))


class EdgeAwareLoss(nn.Module):
    """Sobel-based edge consistency loss (L1 on gradient magnitudes)."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        sy = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer("sx", sx.unsqueeze(0))
        self.register_buffer("sy", sy.unsqueeze(0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.unsqueeze(1)   if pred.dim()   == 3 else pred
        t = target.unsqueeze(1) if target.dim() == 3 else target
        pe = torch.sqrt(F.conv2d(p, self.sx, padding=1) ** 2 +
                        F.conv2d(p, self.sy, padding=1) ** 2 + 1e-6)
        te = torch.sqrt(F.conv2d(t, self.sx, padding=1) ** 2 +
                        F.conv2d(t, self.sy, padding=1) ** 2 + 1e-6)
        return F.l1_loss(pe, te)


class BoundaryMapLoss(nn.Module):
    """Weighted BCE on predicted boundary probability map vs. GT boundary."""

    def _gt_bmap(self, masks: torch.Tensor) -> torch.Tensor:
        m  = masks.unsqueeze(1).float()
        sx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=m.dtype, device=m.device)
        sy = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=m.dtype, device=m.device)
        ex = F.conv2d(m, sx, padding=1)
        ey = F.conv2d(m, sy, padding=1)
        e  = (torch.sqrt(ex ** 2 + ey ** 2 + 1e-6) > 0.1).float()
        return F.max_pool2d(e, 3, 1, 1).squeeze(1)

    def forward(self, boundary_pred: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        gt = self._gt_bmap(masks)
        with torch.cuda.amp.autocast(enabled=False):
            p  = boundary_pred.float().clamp(1e-7, 1 - 1e-7)
            pw = (gt.numel() / (gt.sum() + 1e-7)).clamp(1, 20)
            bce = F.binary_cross_entropy(p, gt.float(), reduction="none")
            w   = 1.0 + (pw - 1.0) * gt.float()
            return (bce * w).mean()
