# Utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_dist


class HausdorffDTLoss(nn.Module):

    def __init__(self, alpha=2.0, debug=False):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)  # probabilities in [0,1]

        batch_size = pred_prob.shape[0]
        loss = 0.0

        for b in range(batch_size):
            p_b = pred_prob[b, 0]   # (H,W)
            t_b = target[b, 0]      # (H,W)

            with torch.no_grad():
                gt_np = t_b.detach().cpu().numpy().astype(np.uint8)

                d_fg = eucl_dist(gt_np)        # distance inside GT
                d_bg = eucl_dist(1 - gt_np)    # distance outside GT

                # Normalize distances to [0,1] to keep scale stable
                if d_fg.max() > 0:
                    d_fg = d_fg / (d_fg.max() + 1e-6)
                if d_bg.max() > 0:
                    d_bg = d_bg / (d_bg.max() + 1e-6)

                w_bg = torch.from_numpy(d_bg ** self.alpha).float().to(pred.device)
                w_fg = torch.from_numpy(d_fg ** self.alpha).float().to(pred.device)

            # False positives (pred high where GT is background)
            term_fp = p_b * w_bg

            # False negatives (pred low where GT is foreground)
            term_fn = (1.0 - p_b) * w_fg

            loss += (term_fp.mean() + term_fn.mean())

        return loss / batch_size


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred:   logits (B,1,H,W)
        target: binary mask (B,1,H,W)
        """
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 2.0 * (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class StructureLoss(nn.Module):
    """
    Boundary-aware structure loss popular in polyp segmentation.
    Combines weighted BCE and weighted IoU so uncertain boundaries matter more.
    """
    def __init__(self, pool_kernel=31, pos_weight=None):
        super().__init__()
        self.pool_kernel = pool_kernel
        if pos_weight is not None:
            pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, pred, target):
        padding = self.pool_kernel // 2
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(target, kernel_size=self.pool_kernel, stride=1, padding=padding) - target
        )

        wbce = F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=self.pos_weight,
            reduction='none',
        )
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp_min(1e-6)

        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)

        return (wbce + wiou).mean()


class BoundaryLoss(nn.Module):
    """
    Lightweight differentiable boundary alignment loss.
    """
    def __init__(self, kernel_size=5, smooth=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
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

    def __init__(self, config, pos_weight=None):
        super(JointLoss, self).__init__()
        self.structure = StructureLoss(
            pool_kernel=getattr(config, 'STRUCTURE_POOL_KERNEL', 31),
            pos_weight=pos_weight,
        )
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss(
            kernel_size=getattr(config, 'BOUNDARY_LOSS_KERNEL', 5),
        )
        self.hd = HausdorffDTLoss()
        self.lambda_struct = getattr(config, 'LAMBDA_STRUCT', 1.0)
        self.lambda_dice = config.LAMBDA_DICE
        self.lambda_bce = config.LAMBDA_BCE
        self.lambda_boundary = getattr(config, 'LAMBDA_BOUNDARY', 0.0)
        self.lambda_hd = config.LAMBDA_HD

        if pos_weight is not None:
            pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, pred, target):
        loss_s = 0.0
        if self.lambda_struct > 0.0:
            loss_s = self.structure(pred, target)

        loss_d = self.dice(pred, target)
        loss_b = 0.0
        if self.lambda_bce > 0.0:
            loss_b = F.binary_cross_entropy_with_logits(
                pred,
                target,
                pos_weight=self.pos_weight,
            )

        loss_boundary = 0.0
        if self.lambda_boundary > 0.0:
            loss_boundary = self.boundary(pred, target)

        loss_h = 0.0
        if self.lambda_hd > 0.0:
            loss_h = self.hd(pred, target)

        return (self.lambda_struct * loss_s) + \
               (self.lambda_dice * loss_d) + \
               (self.lambda_bce * loss_b) + \
               (self.lambda_boundary * loss_boundary) + \
               (self.lambda_hd * loss_h)
