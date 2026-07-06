"""
AMR Loss Functions
===================

Loss functions for infrared small target detection:
1. FocalLoss - handles extreme class imbalance
2. ResidualReconstructionLoss - physically constrained decomposition loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
import math


class FocalLoss(nn.Module):
    """
    Focal Loss for extreme class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces the loss contribution from easy examples, forcing the model
    to focus on hard examples.

    Args:
        alpha: Positive/negative sample weight factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        alpha_factor = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ResidualReconstructionLoss(nn.Module):
    """
    Residual reconstruction loss with physically motivated constraints.

    Based on the physical prior: original_image = background + target.

    Four loss components:
    1. Global reconstruction: L1(rec_bg + rec_tgt, img)
    2. Target sparsity: L2((1-mask) * rec_tgt) -- target signal only in target regions
    3. Target content: MSE(mask * rec_tgt, mask * img) -- preserve target details
    4. Background inpainting: inpaint loss in target regions

    Args:
        alpha: Target sparsity weight (default: 0.1)
        beta: Target content weight (default: 2.0)
        gamma: Background inpainting weight (default: 0.5)
        inpaint_kernel_size: Inpainting kernel size, should exceed target size (default: 15)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 2.0,
        gamma: float = 0.5,
        inpaint_kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.inpaint_kernel_size = inpaint_kernel_size

    def forward(
        self,
        img: Tensor,
        rec_bg: Tensor,
        rec_tgt: Tensor,
        mask: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute residual reconstruction loss.

        Args:
            img: Original image [B, C, H, W]
            rec_bg: Background branch decoded image [B, C, H, W]
            rec_tgt: Target branch decoded image [B, C, H, W]
            mask: GT mask [B, 1, H, W] (0 or 1)

        Returns:
            Dict with all loss components
        """
        # 1. Global reconstruction
        loss_global = F.l1_loss(rec_bg + rec_tgt, img)

        # 2. Target sparsity: L2 on background regions (smoother gradients than L1)
        bg_region = 1 - mask
        loss_tgt_sparse = torch.mean((bg_region * rec_tgt) ** 2)

        # 3. Target content: MSE on target regions
        mask_sum = mask.sum()
        if mask_sum > 0:
            diff = (mask * (rec_tgt - img)).pow(2)
            loss_tgt_content = diff.sum() / (mask_sum + 1e-6)
        else:
            loss_tgt_content = torch.tensor(0.0, device=img.device, requires_grad=True)

        # 4. Background inpainting
        loss_bg_inpaint = self.background_inpainting_loss(img, rec_bg, mask)

        total_loss = (
            loss_global
            + self.alpha * loss_tgt_sparse
            + self.beta * loss_tgt_content
            + self.gamma * loss_bg_inpaint
        )

        return {
            "loss_residual_recon": total_loss,
            "loss_global": loss_global,
            "loss_tgt_sparse": loss_tgt_sparse,
            "loss_tgt_content": loss_tgt_content,
            "loss_bg_inpaint": loss_bg_inpaint,
        }

    def background_inpainting_loss(
        self, img: Tensor, rec_bg: Tensor, mask: Tensor
    ) -> Tensor:
        """
        Background inpainting loss: background in target regions should be
        a smooth extension of surrounding background.

        Uses local mean filtering to estimate expected background values.

        Args:
            img: Original image [B, C, H, W]
            rec_bg: Background reconstruction [B, C, H, W]
            mask: Target mask [B, 1, H, W]

        Returns:
            Inpainting loss value
        """
        kernel_size = self.inpaint_kernel_size
        padding = kernel_size // 2
        bg_mask = 1 - mask

        bg_sum = F.avg_pool2d(img * bg_mask, kernel_size, stride=1, padding=padding)
        bg_count = F.avg_pool2d(bg_mask, kernel_size, stride=1, padding=padding) + 1e-6
        local_bg_mean = bg_sum / bg_count

        mask_sum = mask.sum()
        if mask_sum > 0:
            diff = (mask * (rec_bg - local_bg_mean)).abs()
            return diff.sum() / (mask_sum + 1e-6)
        return torch.tensor(0.0, device=img.device, requires_grad=True)
