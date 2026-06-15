# Utils/losses.py
#
# Fix for the issue raised by the reviewer (Section 3.7.2):
# the original HausdorffDTLoss only used the GT distance transforms and did not
# compute the predicted boundary distance transform. This file restores the
# canonical formulation from
#
#     Karimi & Salcudean, "Reducing the Hausdorff distance in medical image
#     segmentation with convolutional neural networks", IEEE TMI 2020
#
#     L_HD = (1/|Ω|) * Σ_x (p(x) - q(x))^2 * (d_p(x)^α + d_q(x)^α)
#
# where d_p is the Euclidean distance transform of the *ground-truth boundary*
# and d_q is the Euclidean distance transform of the *predicted boundary*.
# Distance transforms are computed under torch.no_grad() (DT is not
# differentiable); the gradient flows through the (p - q)^2 term, weighted by
# the per-pixel distance penalty. Pixels that are mislabeled far from the
# nearest boundary contribute much more than pixels near the boundary, which is
# the property that gives the surrogate its Hausdorff-aware behaviour.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as eucl_dist


def _signed_boundary_dt(binary_mask: np.ndarray) -> np.ndarray:
    """
    Distance to the nearest pixel on the boundary of `binary_mask`, for every
    pixel of the image (both inside the foreground and outside it).
    Returns a non-negative float array of the same shape as `binary_mask`.

    If the mask is empty or full, returns zeros (no boundary defined). This is
    safer than the previous behaviour, which silently used distance-to-image-
    corner as the weight.
    """
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.sum() == 0 or binary_mask.sum() == binary_mask.size:
        return np.zeros_like(binary_mask, dtype=np.float32)

    d_outside = eucl_dist(1 - binary_mask).astype(np.float32)
    d_inside = eucl_dist(binary_mask).astype(np.float32)
    # d_outside is 0 on the foreground, positive in the background;
    # d_inside is 0 in the background, positive in the foreground.
    # Their sum is the distance to the boundary from any pixel.
    return d_outside + d_inside


class HausdorffDTLoss(nn.Module):
    """
    Differentiable surrogate of the Hausdorff distance using both the GT and
    predicted boundary distance transforms (Karimi & Salcudean, 2020).

    Args:
        alpha:     power applied to the distance map; higher α amplifies the
                   penalty on remote errors. α = 2 is the value used in the
                   original paper and in this thesis.
        threshold: threshold used to binarise the predicted probability map
                   for the purpose of computing its distance transform.
        normalize: if True, the GT and predicted DTs are independently
                   normalized to [0, 1] per-sample before being raised to
                   alpha. This is what the implementation described in
                   Section 3.7.2 did and is kept here for backward
                   compatibility, but can be turned off.
    """

    def __init__(self, alpha: float = 2.0, threshold: float = 0.5, normalize: bool = True):
        super().__init__()
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.normalize = bool(normalize)

    def _weights(self, gt_np: np.ndarray, pred_bin_np: np.ndarray) -> np.ndarray:
        d_gt = _signed_boundary_dt(gt_np)
        d_pr = _signed_boundary_dt(pred_bin_np)

        if self.normalize:
            denom_gt = d_gt.max()
            denom_pr = d_pr.max()
            if denom_gt > 0.0:
                d_gt = d_gt / denom_gt
            if denom_pr > 0.0:
                d_pr = d_pr / denom_pr

        return (d_gt ** self.alpha) + (d_pr ** self.alpha)

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred_logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch in HausdorffDTLoss: pred {pred_logits.shape} vs target {target.shape}"
            )

        pred_prob = torch.sigmoid(pred_logits)
        batch_size = pred_prob.shape[0]
        loss = pred_prob.new_zeros(())

        for b in range(batch_size):
            p = pred_prob[b, 0]
            q = target[b, 0]

            with torch.no_grad():
                gt_np = q.detach().cpu().numpy().astype(np.uint8)
                pred_bin_np = (p.detach() > self.threshold).cpu().numpy().astype(np.uint8)
                weight = self._weights(gt_np, pred_bin_np)
                weight_t = torch.from_numpy(weight).to(pred_prob.device, dtype=pred_prob.dtype)

            err2 = (p - q.to(pred_prob.dtype)) ** 2
            loss = loss + (err2 * weight_t).mean()

        return loss / max(batch_size, 1)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 2.0 * (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class StructureLoss(nn.Module):
    def __init__(self, pool_kernel: int = 31, pos_weight=None):
        super().__init__()
        self.pool_kernel = int(pool_kernel)
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, pred, target):
        padding = self.pool_kernel // 2
        weit = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(target, kernel_size=self.pool_kernel, stride=1, padding=padding) - target
        )

        wbce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=self.pos_weight, reduction='none'
        )
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)

        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

        return (wbce + wiou).mean()


class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size: int = 5, smooth: float = 1e-5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.smooth = smooth

    def _boundary_map(self, mask):
        max_pool = F.max_pool2d(mask, self.kernel_size, stride=1, padding=self.padding)
        min_pool = -F.max_pool2d(-mask, self.kernel_size, stride=1, padding=self.padding)
        return (max_pool - min_pool).clamp_(0.0, 1.0)

    def forward(self, pred, target):
        pred_boundary = self._boundary_map(torch.sigmoid(pred))
        target_boundary = self._boundary_map(target)
        intersection = (pred_boundary * target_boundary).sum(dim=(2, 3))
        union = pred_boundary.sum(dim=(2, 3)) + target_boundary.sum(dim=(2, 3))
        boundary_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - boundary_dice.mean()


class JointLoss(nn.Module):
    """
    Weighted sum of the component losses. Lambda weights are read from `config`.
    All weights default to zero if absent so that ablations can switch terms off
    by simply setting the corresponding lambda to 0.0 in the Config.
    """

    def __init__(self, config, pos_weight=None):
        super().__init__()
        self.structure = StructureLoss(
            pool_kernel=getattr(config, 'STRUCTURE_POOL_KERNEL', 31),
            pos_weight=pos_weight,
        )
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(
            kernel_size=getattr(config, 'BOUNDARY_LOSS_KERNEL', 5),
        )
        self.hd = HausdorffDTLoss(
            alpha=getattr(config, 'HD_ALPHA', 2.0),
            threshold=getattr(config, 'HD_BINARIZE_THRESHOLD', 0.5),
            normalize=getattr(config, 'HD_NORMALIZE', True),
        )

        self.lambda_struct = float(getattr(config, 'LAMBDA_STRUCT', 0.0))
        self.lambda_dice = float(getattr(config, 'LAMBDA_DICE', 0.0))
        self.lambda_bce = float(getattr(config, 'LAMBDA_BCE', 0.0))
        self.lambda_boundary = float(getattr(config, 'LAMBDA_BOUNDARY', 0.0))
        self.lambda_hd = float(getattr(config, 'LAMBDA_HD', 0.0))

        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, pred, target):
        total = pred.new_zeros(())
        if self.lambda_struct > 0.0:
            total = total + self.lambda_struct * self.structure(pred, target)
        if self.lambda_dice > 0.0:
            total = total + self.lambda_dice * self.dice(pred, target)
        if self.lambda_bce > 0.0:
            total = total + self.lambda_bce * F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=self.pos_weight,
            )
        if self.lambda_boundary > 0.0:
            total = total + self.lambda_boundary * self.boundary(pred, target)
        if self.lambda_hd > 0.0:
            total = total + self.lambda_hd * self.hd(pred, target)
        return total
