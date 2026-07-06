"""
MemISTD Small Target Detection Network
=======================================

Optimized for infrared small target detection (target size < 7x7 pixels)

Key Improvements:
1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
2. TinyTargetAttention module for small target enhancement
3. Enhanced feature extraction with larger receptive field control
4. Memory-augmented mechanism for target/background separation

Architecture:
    Input Image (B, C, H, W)
        |
    [U-Net Backbone] -> Multi-scale encoder features [E0, E1, E2]
        |
    [Feature Split Branch] -> Global, Target, Background
        |
    [Memory Modules] -> Target Memory + Background Memory
        |
    [Fusion Module] -> Attention-based fusion
        |
    [Multi-Scale FPN] -> [P0: H×W, P1: H/2×W/2, P2: H/4×W/4]
        |
    [Detection Heads] -> Detection at 3 scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor
from typing import Optional, Tuple, Dict, List
import math


# ── 小目标专用模块（内联自 tiny_target_modules.py）───────────────────────────

class TinyTargetAttention(nn.Module):
    """Tiny Target Attention Module — 针对 <7x7 像素小目标优化"""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        mid_channels = max(channels // reduction, 16)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        local_feat = self.local_conv(x)
        point_feat = self.point_conv(local_feat)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        enhanced = point_feat * spatial_weight
        return x + self.gamma * enhanced


class SmallTargetHead(nn.Module):
    """Specialized detection head for small targets — YOLOX-style"""

    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        hidden = in_channels
        self.reg_conv = self._make_branch(in_channels, hidden)
        self.obj_conv = self._make_branch(in_channels, hidden)
        self.cls_conv = self._make_branch(in_channels, hidden)
        self.reg_pred = nn.Conv2d(hidden, 4, 1)
        self.obj_pred = nn.Conv2d(hidden, 1, 1)
        self.cls_pred = nn.Conv2d(hidden, num_classes, 1)
        self._init_weights()

    def _make_branch(self, in_ch, hidden):
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        import math
        bias_val = -math.log((1 - 0.05) / 0.05)
        nn.init.constant_(self.obj_pred.bias, bias_val)
        nn.init.constant_(self.cls_pred.bias, bias_val)

    def forward(self, x: Tensor) -> dict:
        return {
            "reg": self.reg_pred(self.reg_conv(x)),
            "obj": self.obj_pred(self.obj_conv(x)),
            "cls": self.cls_pred(self.cls_conv(x)),
        }


def get_activation(act_type: str = "relu"):
    """Get activation function"""
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "silu" or act_type == "swish":
        return nn.SiLU(inplace=True)
    elif act_type == "mish":
        return nn.Mish(inplace=True)
    elif act_type == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def get_norm_layer(norm_type: str, num_channels: int, num_groups: int = 32):
    """Get normalization layer"""
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm_type == "gn":
        num_groups = min(num_groups, num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type == "in":
        return nn.InstanceNorm2d(num_channels)
    else:
        return nn.Identity()


class DoubleConv(nn.Module):
    """
    Double Convolution Block with residual connection

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        norm_type: Normalization type ('bn', 'gn', 'in')
        act_type: Activation type ('relu', 'silu', 'mish')
        use_residual: Whether to use residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
        )
        self.act = get_activation(act_type)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                get_norm_layer(norm_type, out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_residual:
            out = out + identity
        out = self.act(out)
        return out


class MemoryDecoder(nn.Module):
    """
    Memory Decoder Module

    Decodes memory features back to image space with an explicit and fixed
    structure. Since the spatial upsampling ratio is known in advance, the
    decoder is directly composed of:
    - one convolution refinement block
    - several transposed-convolution upsampling blocks
    - one final projection head

    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output image channels
        hidden_channels: Number of hidden channels after refinement
        upsample_scale: Overall spatial upsample ratio from latent space to image space
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 1,
        hidden_channels: int = 64,
        upsample_scale: int = 4,
    ) -> None:
        super().__init__()

        if upsample_scale not in (1, 2, 4, 8):
            raise ValueError(
                f"upsample_scale must be one of (1, 2, 4, 8), got {upsample_scale}"
            )

        self.upsample_scale = upsample_scale

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        if upsample_scale == 1:
            self.decoder = nn.Identity()
            final_in_channels = hidden_channels
        elif upsample_scale == 2:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            )
            final_in_channels = hidden_channels
        elif upsample_scale == 4:
            mid_channels = max(hidden_channels // 2, out_channels)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            final_in_channels = mid_channels
        else:  # upsample_scale == 8
            mid_channels_1 = max(hidden_channels // 2, out_channels)
            mid_channels_2 = max(mid_channels_1 // 2, out_channels)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(hidden_channels, mid_channels_1, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels_1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(mid_channels_1, mid_channels_2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels_2),
                nn.ReLU(inplace=True),
            )
            final_in_channels = mid_channels_2

        self.head = nn.Conv2d(final_in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input feature map [B, in_channels, H, W]

        Returns:
            Decoded image [B, out_channels, H_out, W_out], unbounded output
        """
        x = self.refine(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module

    Captures long-range spatial dependencies with low computational cost.
    Decomposes channel attention into two 1D feature encoding processes
    along horizontal and vertical directions.

    Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021

    Args:
        inp: Number of input channels
        oup: Number of output channels
        reduction: Reduction ratio for intermediate channels
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32) -> None:
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w

        return out


class _BaseMemoryBranch(nn.Module):
    """
    Base class for multi-scale memory branch

    Uses patch-based memory reading with unfold/fold operations.

    Args:
        in_channels: Number of input channels
        num_memories: Number of memory slots
        patch_size: Size of the patch for memory reading
        topk: Number of top memory slots used for sparse readout
    """

    def __init__(
        self, 
        in_channels: int, 
        num_memories: int, 
        patch_size: int,
        similarity_type: str = 'dot',
        topk: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.padding = patch_size // 2
        self.mem_dim = in_channels * patch_size * patch_size
        self.memory = nn.Parameter(torch.randn(num_memories, self.mem_dim))
        self.temperature = nn.Parameter(torch.ones(1) * 10.0)
        self.similarity_type = similarity_type
        self.topk = topk

    def _read_memory(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        """
        Read from memory using patch-based sparse top-k attention.
        
        Args:
            x: Input feature map [B, C, H, W]
        
        Returns:
            read_out: Memory read output [B, H*W, mem_dim]
            shape: Original shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=self.patch_size, padding=self.padding, stride=1)
        x_unfold = x_unfold.transpose(1, 2)
        
        if self.similarity_type == 'cosine':
            x_norm = F.normalize(x_unfold, dim=2)
            mem_norm = F.normalize(self.memory, dim=1)
            sim = torch.matmul(x_norm, mem_norm.t()) * self.temperature
        elif self.similarity_type == 'dot':
            scale = 1.0 / (self.mem_dim ** 0.5)
            sim = torch.matmul(x_unfold, self.memory.t()) * scale * self.temperature
        else:
            raise ValueError(f"Unsupported similarity_type: {self.similarity_type}")

        num_memories = self.memory.shape[0]
        topk = num_memories if self.topk is None else min(self.topk, num_memories)

        if topk < num_memories:
            sim_topk, topk_indices = torch.topk(sim, k=topk, dim=2)
            att = F.softmax(sim_topk, dim=2)
            selected_memory = self.memory[topk_indices]
            read_out = torch.sum(att.unsqueeze(-1) * selected_memory, dim=2)
        else:
            att = F.softmax(sim, dim=2)
            read_out = torch.matmul(att, self.memory)
        
        return read_out, (B, C, H, W)


class BackgroundBranch(_BaseMemoryBranch):
    """
    Background memory branch with fold-based reconstruction.

    Reconstructs background features by folding memory read output.
    """

    def forward(self, x: Tensor) -> Tensor:
        read_out, (B, C, H, W) = self._read_memory(x)
        read_out = read_out.transpose(1, 2)
        out_sum = F.fold(read_out, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        ones = torch.ones(1, 1, H, W, device=x.device)
        ones_unfold = F.unfold(ones, kernel_size=self.patch_size, padding=self.padding, stride=1)
        divisor = F.fold(ones_unfold, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        return out_sum / (divisor + 1e-8)


class TargetBranch(_BaseMemoryBranch):
    """
    Target memory branch with center-pixel extraction.

    Extracts center pixel from each patch for target detection.
    Feature normalization is applied in _read_memory to stabilize output value range.
    """

    # 采用中心对齐方案
    # def forward(self, x: Tensor) -> Tensor:
    #     read_out, (B, C, H, W) = self._read_memory(x)
    #     read_out = read_out.view(B, H * W, C, self.patch_size, self.patch_size)
    #     center_idx = self.patch_size // 2
    #     out_center = read_out[:, :, :, center_idx, center_idx]
    #     out_center = out_center.transpose(1, 2).view(B, C, H, W)
        
    #     return out_center
    
    def forward(self, x: Tensor) -> Tensor:  # 和背景一个处理逻辑
        read_out, (B, C, H, W) = self._read_memory(x)
        read_out = read_out.transpose(1, 2)
        out_sum = F.fold(read_out, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        ones = torch.ones(1, 1, H, W, device=x.device)
        ones_unfold = F.unfold(ones, kernel_size=self.patch_size, padding=self.padding, stride=1)
        divisor = F.fold(ones_unfold, output_size=(H, W), kernel_size=self.patch_size, padding=self.padding, stride=1)
        return out_sum / (divisor + 1e-8)


class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion module for combining different patch size results.

    Supports arbitrary number of scales (adaptive to input list length).

    Args:
        channels: Number of channels
        num_scales: Number of scales (default 3 for [3,5,7])
        fusion_type: Fusion type ('cat' or 'attention')
    """

    def __init__(self, channels: int, num_scales: int = 3, fusion_type: str = 'attention') -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        self.channels = channels

        if fusion_type == 'cat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(channels * num_scales, channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'attention':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(channels // 4, channels * num_scales)
            )
            self.softmax = nn.Softmax(dim=1)

    def forward(self, features: List[Tensor]) -> Tensor:
        """
        Forward pass with list of features.

        Args:
            features: List of feature tensors, each [B, C, H, W]

        Returns:
            Fused feature tensor [B, C, H, W]
        """
        num_features = len(features)
        if num_features == 0:
            raise ValueError("features list cannot be empty")

        if num_features != self.num_scales:
            if hasattr(self, 'fusion_conv'):
                self.fusion_conv = nn.Sequential(
                    nn.Conv2d(self.channels * num_features,
                              self.channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.channels),
                    nn.ReLU(inplace=True)
                ).to(features[0].device)
            if hasattr(self, 'fc'):
                self.fc = nn.Sequential(
                    nn.Linear(self.channels, self.channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.channels // 4, self.channels * num_features)
                ).to(features[0].device)
            self.num_scales = num_features

        if self.fusion_type == 'cat':
            return self.fusion_conv(torch.cat(features, dim=1))
        elif self.fusion_type == 'attention':
            B, C, H, W = features[0].shape
            stack = torch.stack(features, dim=1)
            U = sum(features)
            w = self.softmax(self.fc(self.avg_pool(U).view(B, C)).view(
                B, num_features, C)).unsqueeze(-1).unsqueeze(-1)
            return (stack * w).sum(dim=1)
        return features[0]


class ArithmeticFusion(nn.Module):
    """
    Arithmetic fusion with decoupled mutual-exclusive gate.

    Fuses global, background memory, and target memory features through
    spatial reweighting: target regions are enhanced while background
    clutter is suppressed, without self-suppression at boundaries.

    Args:
        channels: Number of input channels
    """

    def __init__(self, channels: int, strategy: str = 'spatial_reweight') -> None:
        super().__init__()

        self.target_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.background_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, f_global: Tensor, f_bc: Tensor, f_tg: Tensor) -> Tensor:
        """
        Args:
            f_global: Global feature [B, C, H, W]
            f_bc: Background memory feature [B, C, H, W]
            f_tg: Target memory feature [B, C, H, W]

        Returns:
            Fused feature [B, C, H, W]
        """
        target_weight = torch.sigmoid(self.target_proj(f_tg))
        background_weight = torch.sigmoid(self.background_proj(f_bc))

        # Decoupled mutual-exclusive suppression:
        # effective_bg = bg_weight * (1 - target_weight)
        #   - Weak target, strong bg → large suppression
        #   - Strong target, weak bg → no self-suppression
        #   - Strong target, strong bg → target enhancement dominates
        effective_bg_suppress = background_weight * (1.0 - target_weight)

        out = f_global * (1 + target_weight) * (1 - effective_bg_suppress)
        out = self.spatial_attention(out)

        return out


class DualMemorySystem(nn.Module):
    """
    Dual Memory System with multi-scale memory branches.

    Combines background and target memory branches with adaptive multi-scale fusion.

    Args:
        in_channels: Number of input channels
        num_bg_memories: Number of background memory slots (more for complex background)
        num_tg_memories: Number of target memory slots (fewer for simple targets)
        bg_patch_sizes: List of patch sizes for background memory branches (default: [3, 5, 7])
        tg_patch_sizes: List of patch sizes for target memory branches (default: [3, 5, 7])
        ms_fusion_type: Multi-scale fusion type ('cat' or 'attention')
        bg_topk: Number of top background memory slots used per patch
        tg_topk: Number of top target memory slots used per patch
    """

    def __init__(
        self,
        in_channels: int,
        num_bg_memories: int = 64,
        num_tg_memories: int = 8,
        bg_patch_sizes: List[int] = [3, 5, 7],
        tg_patch_sizes: List[int] = [3, 5, 7],
        ms_fusion_type: str = 'attention',
        bg_topk: int = 8,
        tg_topk: int = 4,
    ) -> None:
        super().__init__()

        self.bg_patch_sizes = bg_patch_sizes
        self.num_bg_scales = len(bg_patch_sizes)
        self.bg_branches = nn.ModuleList([
            BackgroundBranch(in_channels, num_bg_memories, ps, topk=bg_topk) for ps in bg_patch_sizes
        ])
        self.bg_fusion = MultiScaleFusion(
            in_channels, num_scales=self.num_bg_scales, fusion_type=ms_fusion_type)

        self.tg_patch_sizes = tg_patch_sizes
        self.num_tg_scales = len(tg_patch_sizes)
        self.tg_branches = nn.ModuleList([
            TargetBranch(in_channels, num_tg_memories, ps, topk=tg_topk) for ps in tg_patch_sizes
        ])
        self.tg_fusion = MultiScaleFusion(
            in_channels, num_scales=self.num_tg_scales, fusion_type=ms_fusion_type)

    def forward(self, bg: Tensor, tg: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            bg: Background input feature [B, C, H, W]
            tg: Target input feature [B, C, H, W]

        Returns:
            f_bc: Background feature
            f_tg: Target feature
        """
        bg_features = [branch(bg) for branch in self.bg_branches]
        f_bc = self.bg_fusion(bg_features)

        tg_features = [branch(tg) for branch in self.tg_branches]
        f_tg = self.tg_fusion(tg_features)

        return f_bc, f_tg


class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        attention = self.sigmoid(
            avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


class FeatureSplitBranch(nn.Module):
    """
    Feature Split Branch for separating global, target, and background features.

    Each branch now uses two stride-2 stages:
    1. stage1: spatial downsampling by 2x while keeping channels at C
    2. stage2: further downsampling by 2x while expanding channels to 2C

    Therefore the final branch outputs are downsampled by 4x with channel size
    equal to 2x the original input channels.

    Args:
        in_channels: Number of input channels
        raw_channels: Channels for global feature output
        target_channels: Channels for target feature output
        background_channels: Channels for background feature output
        use_attention: Whether to use channel attention
    """

    def __init__(
        self,
        in_channels: int = 512,
        raw_channels: int = 1024,
        target_channels: int = 1024,
        background_channels: int = 1024,
        use_attention: bool = True,
    ) -> None:
        super().__init__()

        mid_channels = in_channels

        self.global_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, raw_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(raw_channels),
            nn.ReLU(inplace=True),
        )

        self.target_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, target_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
        )

        self.background_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, background_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(background_channels),
            nn.ReLU(inplace=True),
        )

        self.target_attention = ChannelAttention(
            target_channels) if use_attention else nn.Identity()
        self.background_attention = ChannelAttention(
            background_channels) if use_attention else nn.Identity()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        global_feat = self.global_proj(x)
        target_feat = self.target_proj(x)
        target_feat = self.target_attention(target_feat)
        background_feat = self.background_proj(x)
        background_feat = self.background_attention(background_feat)

        return global_feat, target_feat, background_feat



class UNetBackboneSmallTarget(nn.Module):
    """
    U-Net Backbone optimized for small target detection

    Features:
    1. Encoder with 3 levels (E0, E1, E2) for P0/P1/P2 detection
    2. Skip connections preserved for high-resolution features
    3. Decoder with feature refinement

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        depth: Number of encoder levels (default: 3 for P0/P1/P2)
        norm_type: Normalization type
        act_type: Activation type
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        depth: int = 3,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.enc_channels = [base_channels * (2**i) for i in range(depth)]

        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.encoder_blocks.append(
            DoubleConv(
                in_channels, self.enc_channels[0], norm_type, act_type, use_residual)
        )

        for i in range(1, depth):
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_blocks.append(
                DoubleConv(
                    self.enc_channels[i - 1],
                    self.enc_channels[i],
                    norm_type,
                    act_type,
                    use_residual,
                )
            )

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i in range(depth - 1, 0, -1):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=False),
                    nn.Conv2d(
                        self.enc_channels[i],
                        self.enc_channels[i - 1],
                        kernel_size=1,
                        bias=False,
                    ),
                    get_norm_layer(norm_type, self.enc_channels[i - 1]),
                    get_activation(act_type),
                )
            )

            self.decoder_blocks.append(
                DoubleConv(
                    self.enc_channels[i - 1] * 2,
                    self.enc_channels[i - 1],
                    norm_type,
                    act_type,
                    use_residual,
                )
            )

        self.output_proj = nn.Sequential(
            nn.Conv2d(
                self.enc_channels[0],
                self.enc_channels[-1],
                kernel_size=1,
                bias=False,
            ),
            get_norm_layer(norm_type, self.enc_channels[-1]),
            get_activation(act_type),
        )

        self.tiny_attention = TinyTargetAttention(self.enc_channels[-1])

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        encoder_features = []

        feat = self.encoder_blocks[0](x)
        encoder_features.append(feat)

        for i in range(1, self.depth):
            feat = self.pools[i - 1](feat)
            feat = self.encoder_blocks[i](feat)
            encoder_features.append(feat)

        bottleneck = encoder_features[-1]

        decoder_features = []
        feat = bottleneck

        for i in range(len(self.upsamples)):
            feat = self.upsamples[i](feat)
            skip_idx = self.depth - 2 - i
            skip = encoder_features[skip_idx]
            feat = torch.cat([feat, skip], dim=1)
            feat = self.decoder_blocks[i](feat)
            decoder_features.append(feat)

        output_features = self.output_proj(feat)
        output_features = self.tiny_attention(output_features)

        return output_features
        # return {
        #     "features": output_features,
        #     "encoder_features": encoder_features,
        #     "bottleneck": bottleneck,
        #     "decoder_features": decoder_features,
        # }


class CoordAtt(nn.Module):
    """
    Coordinate Attention FPN - Lightweight FPN with CoordAtt

    Uses Coordinate Attention for feature enhancement instead of 
    traditional multi-scale fusion. Significantly reduces parameters.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: Reduction ratio for CoordAtt
        norm_type: Normalization type
        act_type: Activation type
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        reduction: int = 16,
        norm_type: str = "bn",
        act_type: str = "relu",
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm_type, out_channels),
            get_activation(act_type),
        )

        self.coord_att = CoordinateAttention(
            out_channels, out_channels, reduction)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv(x)
        x = self.coord_att(x)
        return {"P0": x}


class SingleScaleDetectionHead(nn.Module):
    """
    Single-Scale Detection Head for P0 only

    Detection head with:
    - Box regression branch
    - Objectness branch
    - Classification branch

    Args:
        in_channels: Number of input channels
        num_classes: Number of detection classes
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.head = SmallTargetHead(in_channels, num_classes)

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p0 = features["P0"]
        head_out = self.head(p0)

        return {
            "P0_reg": head_out["reg"],
            "P0_obj": head_out["obj"],
            "P0_cls": head_out["cls"],
        }


class MemISTDSmallTarget(nn.Module):
    """
    MemISTD Small Target Detection Network

    Optimized for infrared small target detection with:
    1. Multi-scale detection at P0/P1/P2 (full resolution, 1/2, 1/4)
    2. Memory-augmented target/background separation
    3. TinyTargetAttention for small target enhancement
    4. Feature refinement modules
    5. New DualMemorySystem with multi-scale memory branches

    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        num_classes: Number of detection classes
        base_channels: Base number of channels
        backbone_depth: Depth of backbone (default: 3 for P0/P1/P2)
        target_memory_slots: Number of target memory slots
        background_memory_slots: Number of background memory slots
        use_attention: Whether to use attention in feature split
        norm_type: Normalization type
        act_type: Activation type
        use_residual: Whether to use residual connections
        use_dual_memory: Whether to use new DualMemorySystem
        ms_fusion_type: Multi-scale fusion type ('cat' or 'attention')
        global_fusion_strategy: Arithmetic fusion strategy
    """

    def __init__(
        self,
        cfg: Optional[Dict] = None,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        backbone_depth: int = 3,
        target_memory_slots: int = 16,
        background_memory_slots: int = 64,
        use_attention: bool = True,
        norm_type: str = "bn",
        act_type: str = "relu",
        use_residual: bool = True,
        ms_fusion_type: str = 'attention',
        global_fusion_strategy: str = 'spatial_reweight',
        bg_patch_sizes: List[int] = [3, 5, 7],
        tg_patch_sizes: List[int] = [3, 5, 7],
        bg_topk: int = 8,
        tg_topk: int = 4,
        bg_decoder_hidden_channels: int = 64,
        tg_decoder_hidden_channels: int = 64,
        bg_decoder_upsample_scale: int = 4,
        tg_decoder_upsample_scale: int = 4,
    ) -> None:
        super().__init__()

        if cfg is not None:
            mc = cfg.get("model", {})
            in_channels = mc.get("in_channels", in_channels)
            num_classes = mc.get("num_classes", num_classes)
            base_channels = mc.get("base_channels", base_channels)
            backbone_depth = mc.get("backbone_depth", backbone_depth)
            target_memory_slots = mc.get("target_memory_slots", target_memory_slots)
            background_memory_slots = mc.get("background_memory_slots", background_memory_slots)
            use_attention = mc.get("use_attention", use_attention)
            norm_type = mc.get("norm_type", norm_type)
            act_type = mc.get("act_type", act_type)
            use_residual = mc.get("use_residual", use_residual)
            ms_fusion_type = mc.get("ms_fusion_type", ms_fusion_type)
            global_fusion_strategy = mc.get("global_fusion_strategy", global_fusion_strategy)
            bg_patch_sizes = mc.get("bg_patch_sizes", bg_patch_sizes)
            tg_patch_sizes = mc.get("tg_patch_sizes", tg_patch_sizes)
            bg_topk = mc.get("bg_topk", bg_topk)
            tg_topk = mc.get("tg_topk", tg_topk)
            bg_decoder_hidden_channels = mc.get("bg_decoder_hidden_channels", bg_decoder_hidden_channels)
            tg_decoder_hidden_channels = mc.get("tg_decoder_hidden_channels", tg_decoder_hidden_channels)
            bg_decoder_upsample_scale = mc.get("bg_decoder_upsample_scale", bg_decoder_upsample_scale)
            tg_decoder_upsample_scale = mc.get("tg_decoder_upsample_scale", tg_decoder_upsample_scale)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.backbone = UNetBackboneSmallTarget(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=backbone_depth,
            norm_type=norm_type,
            act_type=act_type,
            use_residual=use_residual,
        )

        backbone_channels = base_channels * (2 ** (backbone_depth - 1))
        mem_channels = backbone_channels * 2

        self.feature_split = FeatureSplitBranch(
            in_channels=backbone_channels,
            raw_channels=mem_channels,
            target_channels=mem_channels,
            background_channels=mem_channels,
            use_attention=use_attention,
        )

        self.dual_memory = DualMemorySystem(
            in_channels=mem_channels,
            num_bg_memories=background_memory_slots,
            num_tg_memories=target_memory_slots,
            bg_patch_sizes=bg_patch_sizes,
            tg_patch_sizes=tg_patch_sizes,
            ms_fusion_type=ms_fusion_type,
            bg_topk=bg_topk,
            tg_topk=tg_topk,
        )

        self.final_arithmetic = ArithmeticFusion(mem_channels, strategy=global_fusion_strategy)

        self.neck = CoordAtt(
            in_channels=mem_channels,
            out_channels=mem_channels,
            reduction=16,
            norm_type=norm_type,
            act_type=act_type,
        )

        self.detection_head = SingleScaleDetectionHead(
            in_channels=mem_channels,
            num_classes=num_classes,
        )

        self.target_decoder = MemoryDecoder(
            in_channels=mem_channels,
            out_channels=in_channels,
            hidden_channels=tg_decoder_hidden_channels,
            upsample_scale=tg_decoder_upsample_scale,
        )
        self.background_decoder = MemoryDecoder(
            in_channels=mem_channels,
            out_channels=in_channels,
            hidden_channels=bg_decoder_hidden_channels,
            upsample_scale=bg_decoder_upsample_scale,
        )

    def forward(
        self,
        x: Tensor,
        decode_image: bool = False,
        letterbox_info: Optional[Dict] = None,
    ) -> Dict[str, object]:
        """
        Forward pass

        Args:
            x: Input image tensor [B, C, H, W]
            decode_image: If True, decode memory features to image space (only for training).
            letterbox_info: Dict with keys 'orig_h', 'orig_w', 'scale', 'dx', 'dy' for restoring original size.

        Returns:
            Dictionary containing:
            - predictions: Detection predictions at each scale
            - multi_scale_features: Multi-scale features (P0, P1, P2)
            - Memory module outputs
            - Decoded images (if decode_image=True)
        """
        backbone_feat = self.backbone(x)  # x:([1, 3, 512, 512])->([1, 128, 512, 512])

        global_feat, target_feat, background_feat = self.feature_split(backbone_feat)
        # 经过两级 stride-2 投影后，三路特征均为 4x 下采样；第一级保持 C，第二级扩展到 2C
        background_feat_memory, target_feat_memory = self.dual_memory(background_feat, target_feat)

        target_recon_img, background_recon_img = None, None
        if decode_image:
            target_recon_img = self.target_decoder(target_feat_memory)
            background_recon_img = self.background_decoder(background_feat_memory)

        fused_feat = self.final_arithmetic(global_feat, background_feat_memory, target_feat_memory)

        multi_scale_features = self.neck(fused_feat)  # ([1, 256, 128, 128])

        predictions = self.detection_head(multi_scale_features)

        outputs = {
            "predictions": predictions,
            "target_feat_recon": target_feat_memory,
            "background_feat_recon": background_feat_memory,
            "target_recon_img": target_recon_img,
            "background_recon_img": background_recon_img,
            "original_img": x,
        }

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, object],
        labels: List[Tensor],
        yolox_loss: nn.Module,
        residual_recon_loss: Optional[nn.Module] = None,
        residual_recon_weight: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Compute all losses inside the model

        Args:
            outputs: Output from forward()
            labels: Ground truth labels (list of tensors)
            yolox_loss: YOLOLoss instance
            residual_recon_loss: ResidualReconstructionLoss instance (optional)
            residual_recon_weight: Weight for residual reconstruction loss

        Returns:
            Dictionary containing all loss components
        """
        predictions = outputs["predictions"]
        pred_list = self.get_predictions_list(predictions)

        loss_dict = yolox_loss(pred_list, labels, return_components=True)
        loss_yolo = loss_dict["total_loss"]

        total_loss = loss_yolo

        loss_residual_recon = torch.tensor(0.0, device=loss_yolo.device)
        loss_global = torch.tensor(0.0, device=loss_yolo.device)
        loss_tgt_sparse = torch.tensor(0.0, device=loss_yolo.device)
        loss_tgt_content = torch.tensor(0.0, device=loss_yolo.device)
        loss_bg_inpaint = torch.tensor(0.0, device=loss_yolo.device)

        if residual_recon_loss is not None and outputs.get("target_recon_img") is not None:
            target_mask = outputs.get("target_mask")
            if target_mask is not None:
                residual_losses = residual_recon_loss(
                    img=outputs["original_img"],
                    rec_bg=outputs["background_recon_img"],
                    rec_tgt=outputs["target_recon_img"],
                    mask=target_mask,
                )
                loss_residual_recon = residual_losses["loss_residual_recon"] * \
                    residual_recon_weight
                loss_global = residual_losses["loss_global"]
                loss_tgt_sparse = residual_losses["loss_tgt_sparse"]
                loss_tgt_content = residual_losses["loss_tgt_content"]
                loss_bg_inpaint = residual_losses["loss_bg_inpaint"]

                total_loss = total_loss + loss_residual_recon

        loss_dict_result = {
            "total_loss": total_loss,
            "yolo_loss": loss_yolo,
            "loss_box": loss_dict["loss_box"],
            "loss_obj": loss_dict["loss_obj"],
            "loss_cls": loss_dict["loss_cls"],
            "residual_recon_loss": loss_residual_recon,
            "loss_global": loss_global,
            "loss_tgt_sparse": loss_tgt_sparse,
            "loss_tgt_content": loss_tgt_content,
            "loss_bg_inpaint": loss_bg_inpaint,
            "num_fg": loss_dict["num_fg"],
        }

        return loss_dict_result

    @torch.no_grad()
    def detect(
        self,
        x: Tensor,
        conf_thres: float = 0.05,
        nms_thres: float = 0.5,
        max_detections: int = 300,
        debug: bool = False,
    ) -> Tensor:
        """
        Inference detection function

        Directly outputs detection results after NMS.

        Args:
            x: Input image tensor [B, C, H, W]
            conf_thres: Confidence threshold for filtering detections
            nms_thres: NMS IoU threshold
            max_detections: Maximum number of detections per image
            debug: If True, print debug information

        Returns:
            detections: [B, N, 6] detection results
                - Each detection: [x1, y1, x2, y2, score, class_id]
                - N is the number of detections (padded to max_detections if fewer)
                - If no detections, returns zeros tensor
        """
        from torchvision.ops import batched_nms

        self.eval()

        B = x.shape[0]
        device = x.device

        outputs = self.forward(x, decode_image=False)
        predictions = outputs["predictions"]

        # 单尺度检测，只有 P0，stride = 4（FeatureSplitBranch 4x 下采样）
        stride = 4
        all_detections = [[] for _ in range(B)]

        reg = predictions["P0_reg"]
        obj = predictions["P0_obj"]
        cls = predictions["P0_cls"]

        _, _, H, W = reg.shape

        yv, xv = torch.meshgrid(
            [torch.arange(H), torch.arange(W)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, H, W, 2).type_as(reg)
        grid = grid + 0.5
        grid = grid.expand(B, -1, -1, -1)

        reg = reg.permute(0, 2, 3, 1).contiguous()
        obj = obj.permute(0, 2, 3, 1).contiguous()
        if cls is not None:
            cls = cls.permute(0, 2, 3, 1).contiguous()

        cx = (reg[..., 0] + grid[..., 0]) * stride
        cy = (reg[..., 1] + grid[..., 1]) * stride
        w = torch.exp(reg[..., 2].clamp(max=10)) * stride
        h = torch.exp(reg[..., 3].clamp(max=10)) * stride

        w = torch.clamp(w, min=2.0, max=1000.0)
        h = torch.clamp(h, min=2.0, max=1000.0)

        obj_conf = obj[..., 0]

        if cls is not None:
            cls_conf = torch.sigmoid(cls)
            if cls.shape[-1] == 1:
                cls_conf = cls_conf.squeeze(-1)
                scores = obj_conf * cls_conf
            else:
                max_cls_conf, _ = torch.max(cls_conf, dim=-1)
                scores = obj_conf * max_cls_conf
        else:
            scores = obj_conf

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)

        valid_coord_mask = (
            torch.isfinite(x1) & torch.isfinite(y1) &
            torch.isfinite(x2) & torch.isfinite(y2) &
            (x1 >= 0) & (y1 >= 0) & (x2 > x1) & (y2 > y1) &
            (x1 < 10000) & (y1 < 10000) & (x2 < 10000) & (y2 < 10000)
        )

        x1 = torch.clamp(x1, min=0, max=10000)
        y1 = torch.clamp(y1, min=0, max=10000)
        x2 = torch.clamp(x2, min=0, max=10001)
        y2 = torch.clamp(y2, min=0, max=10001)
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        for b in range(B):
            img_scores = scores[b].view(-1)
            img_valid_mask = valid_coord_mask[b].view(-1)
            mask = (img_scores > conf_thres) & img_valid_mask

            if mask.sum() == 0:
                continue

            img_x1 = x1[b].view(-1)[mask]
            img_y1 = y1[b].view(-1)[mask]
            img_x2 = x2[b].view(-1)[mask]
            img_y2 = y2[b].view(-1)[mask]
            img_scores_filtered = img_scores[mask]

            boxes = torch.stack([img_x1, img_y1, img_x2, img_y2], dim=1)

            all_detections[b].append((boxes, img_scores_filtered))

        final_detections = []
        for b in range(B):
            if len(all_detections[b]) == 0:
                final_detections.append(torch.zeros(
                    max_detections, 6, device=device))
                continue

            boxes = torch.cat([d[0] for d in all_detections[b]], dim=0)
            scores = torch.cat([d[1] for d in all_detections[b]], dim=0)

            boxes = boxes.float()
            boxes = torch.clamp(boxes, min=0.0, max=10000.0)
            scores = torch.clamp(scores, min=0.0, max=1.0)

            valid_boxes_mask = (
                (boxes[:, 2] - boxes[:, 0] >= 1.0) &
                (boxes[:, 3] - boxes[:, 1] >= 1.0) &
                torch.isfinite(boxes).all(dim=1) &
                torch.isfinite(scores)
            )

            if valid_boxes_mask.sum() == 0:
                final_detections.append(torch.zeros(
                    max_detections, 6, device=device))
                continue

            boxes = boxes[valid_boxes_mask]
            scores = scores[valid_boxes_mask]

            labels = torch.zeros(
                scores.shape[0], dtype=torch.long, device=device)

            boxes_cpu = boxes.cpu()
            scores_cpu = scores.cpu()
            labels_cpu = labels.cpu()
            keep = batched_nms(boxes_cpu, scores_cpu, labels_cpu, nms_thres)
            keep = keep.to(device)

            if len(keep) > max_detections:
                keep = keep[:max_detections]

            kept_boxes = boxes[keep]
            kept_scores = scores[keep]
            kept_labels = labels[keep]

            n_detections = kept_boxes.shape[0]
            detections = torch.zeros(max_detections, 6, device=device)
            detections[:n_detections, :4] = kept_boxes
            detections[:n_detections, 4] = kept_scores
            detections[:n_detections, 5] = kept_labels.float()

            final_detections.append(detections)

        result = torch.stack(final_detections, dim=0)

        return result

    def get_predictions_list(self, predictions: Dict[str, Tensor]) -> List[Tensor]:
        """
        Convert predictions dict to list format for loss computation

        Args:
            predictions: Dictionary of predictions from detection head

        Returns:
            List of tensors for each scale [P0_pred]
        """
        pred_list = []
        for scale in ["P0"]:
            reg = predictions[f"{scale}_reg"]
            obj = predictions[f"{scale}_obj"]
            cls = predictions[f"{scale}_cls"]
            if cls is not None:
                scale_pred = torch.cat([reg, obj, cls], dim=1)
            else:
                scale_pred = torch.cat([reg, obj], dim=1)
            pred_list.append(scale_pred)
        return pred_list



# ==============================================================================
#  Factory Function
# ==============================================================================

def build_model(cfg):
    """Build AMR model from config dict. Returns MemISTDSmallTarget instance."""
    return MemISTDSmallTarget(cfg)
