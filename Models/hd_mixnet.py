# Models/hd_mixnet.py
#
# Adds ablation switches requested by the reviewer:
#
#   config.BRANCH_MODE        : "both" | "cnn_only" | "swin_only"
#                                Drops the unused branch from the forward pass
#                                entirely and replaces fusion with a pass-
#                                through of whichever branch remains.
#   config.USE_BAMF           : if False, fusion = concat + 1x1 conv. The
#                                BAMF modules are skipped (no edge gating).
#   config.USE_EDGE_SUP       : if False, no ED blocks, no edge head, no
#                                Boundary-Enhance block. Edge loss should also
#                                be turned off in the loss config.
#
# All three switches are independent, so combinations such as
# "Swin-only + no-BAMF + no-edge-sup" are valid Res2Net ablations.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from Models.Components.res2net import Res2NetBlock
from Models.Components.swin_transformer import BasicSwinLayer
from Models.Components.layers import (
    BoundaryEnhanceBlock,
    ConvBNAct,
    DecoderRefineBlock,
    EdgeDetectBlock,
    PyramidContextBlock,
    SqueezeExcite,
)


class BoundaryAwareMixFusion(nn.Module):
    def __init__(self, cnn_dim, trans_dim, out_dim, edge_dim=None):
        super().__init__()
        self.conv_cnn = ConvBNAct(cnn_dim, out_dim, kernel_size=1, padding=0)
        self.conv_trans = ConvBNAct(trans_dim, out_dim, kernel_size=1, padding=0)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )
        self.edge_gate = None
        if edge_dim is not None:
            self.edge_gate = nn.Sequential(
                ConvBNAct(edge_dim, out_dim, kernel_size=1, padding=0),
                nn.Conv2d(out_dim, out_dim, kernel_size=1),
                nn.Sigmoid(),
            )
        self.refine = nn.Sequential(
            ConvBNAct(out_dim, out_dim, kernel_size=3),
            ConvBNAct(out_dim, out_dim, kernel_size=3),
            SqueezeExcite(out_dim),
        )

    def forward(self, cnn_feat, trans_feat, edge_feat=None):
        trans_feat = F.interpolate(
            trans_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False,
        )
        c = self.conv_cnn(cnn_feat)
        t = self.conv_trans(trans_feat)

        gate = self.mix_gate(torch.cat([c, t], dim=1))
        fused = gate * c + (1.0 - gate) * t

        if edge_feat is not None and self.edge_gate is not None:
            if edge_feat.shape[2:] != fused.shape[2:]:
                edge_feat = F.interpolate(
                    edge_feat, size=fused.shape[2:], mode='bilinear', align_corners=False,
                )
            edge_weight = self.edge_gate(edge_feat)
            fused = fused * (1.0 + edge_weight) + c * edge_weight

        return self.refine(fused) + c


class NaiveConcatFusion(nn.Module):
    """Used when USE_BAMF=False. Pure concatenation + 1x1 conv, no edge gating."""

    def __init__(self, cnn_dim, trans_dim, out_dim, **_kwargs):
        super().__init__()
        self.proj_cnn = ConvBNAct(cnn_dim, out_dim, kernel_size=1, padding=0)
        self.proj_trans = ConvBNAct(trans_dim, out_dim, kernel_size=1, padding=0)
        self.fuse = ConvBNAct(2 * out_dim, out_dim, kernel_size=3)

    def forward(self, cnn_feat, trans_feat, edge_feat=None):  # edge_feat ignored
        trans_feat = F.interpolate(
            trans_feat, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False,
        )
        c = self.proj_cnn(cnn_feat)
        t = self.proj_trans(trans_feat)
        return self.fuse(torch.cat([c, t], dim=1))


class SingleBranchFusion(nn.Module):
    """Used when only one branch is active. Just projects the single branch."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = ConvBNAct(in_dim, out_dim, kernel_size=1, padding=0)
        self.refine = nn.Sequential(
            ConvBNAct(out_dim, out_dim, kernel_size=3),
            SqueezeExcite(out_dim),
        )

    def forward(self, feat, target_size=None):
        if target_size is not None and feat.shape[2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        return self.refine(self.proj(feat))


class HD_MixNet(nn.Module):
    def __init__(self, num_classes=1, img_size=224, config=None):
        super().__init__()

        base_channels = getattr(config, 'CNN_BASE_CHANNELS', 64)
        stage2_channels = base_channels * 2
        stage3_channels = base_channels * 4
        embed_dim = getattr(config, 'SWIN_EMBED_DIM', 96)
        window_size = getattr(config, 'SWIN_WINDOW_SIZE', 7)
        res2net_scale = getattr(config, 'RES2NET_SCALE', 4)
        heads_stage1 = getattr(config, 'SWIN_HEADS_STAGE1', 3)
        heads_stage2 = getattr(config, 'SWIN_HEADS_STAGE2', 6)
        mlp_ratio = getattr(config, 'SWIN_MLP_RATIO', 4.0)
        stage_depths = getattr(config, 'SWIN_STAGE_DEPTHS', (2, 2))
        drop_path = getattr(config, 'SWIN_DROP_PATH', 0.1)

        # Ablation switches
        self.branch_mode = getattr(config, 'BRANCH_MODE', 'both')  # both | cnn_only | swin_only
        self.use_bamf = bool(getattr(config, 'USE_BAMF', True))
        self.use_edge_sup = bool(getattr(config, 'USE_EDGE_SUP', True))
        if self.branch_mode not in ('both', 'cnn_only', 'swin_only'):
            raise ValueError(f"BRANCH_MODE must be one of 'both', 'cnn_only', 'swin_only'; got {self.branch_mode}")

        self.use_grad_checkpointing = bool(getattr(config, 'USE_GRAD_CHECKPOINTING', False))
        self.input_multiple = window_size * 8
        self.store_layer_outputs = bool(getattr(config, 'STORE_LAYER_OUTPUTS', False))
        self.layer_outputs = {}

        if embed_dim % heads_stage1 != 0:
            raise ValueError(
                f"SWIN_EMBED_DIM ({embed_dim}) must be divisible by SWIN_HEADS_STAGE1 ({heads_stage1})."
            )
        if (embed_dim * 2) % heads_stage2 != 0:
            raise ValueError(
                f"SWIN_EMBED_DIM*2 ({embed_dim * 2}) must be divisible by SWIN_HEADS_STAGE2 ({heads_stage2})."
            )

        dpr_stage1 = torch.linspace(0.0, drop_path, stage_depths[0]).tolist()
        dpr_stage2 = torch.linspace(
            dpr_stage1[-1] if dpr_stage1 else 0.0, drop_path, stage_depths[1],
        ).tolist()

        # --- CNN branch (built unless swin_only) ----------------------------
        if self.branch_mode != 'swin_only':
            self.cnn_stem = nn.Sequential(
                ConvBNAct(3, base_channels, kernel_size=3),
                ConvBNAct(base_channels, base_channels, kernel_size=3),
            )
            self.res2net1 = Res2NetBlock(base_channels, base_channels, scale=res2net_scale)
            self.res2net2 = Res2NetBlock(base_channels, stage2_channels, scale=res2net_scale, stride=2)
            self.res2net3 = Res2NetBlock(stage2_channels, stage3_channels, scale=res2net_scale, stride=2)

        # --- Swin branch (built unless cnn_only) ----------------------------
        self.embed_dim = embed_dim
        if self.branch_mode != 'cnn_only':
            self.patch_embed = nn.Sequential(
                nn.Conv2d(3, self.embed_dim, kernel_size=4, stride=4, bias=False),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU(),
            )
            self.swin1 = BasicSwinLayer(
                dim=self.embed_dim, depth=stage_depths[0], num_heads=heads_stage1,
                window_size=window_size, mlp_ratio=mlp_ratio, drop_path=dpr_stage1,
            )
            self.patch_merge = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim * 2, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(self.embed_dim * 2),
                nn.GELU(),
            )
            self.swin3 = BasicSwinLayer(
                dim=self.embed_dim * 2, depth=stage_depths[1], num_heads=heads_stage2,
                window_size=window_size, mlp_ratio=mlp_ratio, drop_path=dpr_stage2,
            )

        # --- Edge supervision modules (only when both branches and edge sup) ---
        if self.use_edge_sup and self.branch_mode != 'swin_only':
            self.ed1 = EdgeDetectBlock(base_channels, base_channels)
            self.ed2 = EdgeDetectBlock(stage2_channels, stage2_channels)

        # --- Fusion ---------------------------------------------------------
        if self.branch_mode == 'both':
            FusionCls = BoundaryAwareMixFusion if self.use_bamf else NaiveConcatFusion
            edge_dim_mid = stage2_channels if (self.use_bamf and self.use_edge_sup) else None
            self.bamf1 = FusionCls(stage2_channels, self.embed_dim, stage2_channels, edge_dim=edge_dim_mid)
            self.bamf2 = FusionCls(stage3_channels, self.embed_dim * 2, stage3_channels, edge_dim=None)
        elif self.branch_mode == 'cnn_only':
            self.single_mid = SingleBranchFusion(stage2_channels, stage2_channels)
            self.single_deep = SingleBranchFusion(stage3_channels, stage3_channels)
        else:  # swin_only
            self.single_mid = SingleBranchFusion(self.embed_dim, stage2_channels)
            self.single_deep = SingleBranchFusion(self.embed_dim * 2, stage3_channels)

        self.context = PyramidContextBlock(stage3_channels, stage3_channels)

        # --- Decoder --------------------------------------------------------
        # For swin_only we still need a "skip" with base_channels for dec2; we
        # synthesize one via a small conv on the original image. The cleanest
        # behaviour is to project the patch_embed back to spatial resolution.
        self.dec1 = DecoderRefineBlock(stage3_channels, stage2_channels, stage2_channels)
        self.dec2 = DecoderRefineBlock(stage2_channels, base_channels, base_channels)

        if self.branch_mode == 'swin_only':
            # x_c1 stand-in: project input image to base_channels at full res
            self.swin_only_skip = nn.Sequential(
                ConvBNAct(3, base_channels, kernel_size=3),
                ConvBNAct(base_channels, base_channels, kernel_size=3),
            )

        # --- Output heads ---------------------------------------------------
        if self.use_edge_sup and self.branch_mode != 'swin_only':
            self.be_block = BoundaryEnhanceBlock(base_channels, base_channels)
            self.edge_out_conv = nn.Conv2d(base_channels, 1, 1)
        else:
            self.be_block = None
            self.edge_out_conv = None

        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        self.aux_out_conv = nn.Conv2d(stage2_channels, num_classes, 1)

    def _fwd(self, fn, *args):
        if self.use_grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _pad_input(self, x):
        orig_h, orig_w = x.shape[-2:]
        pad_h = (self.input_multiple - orig_h % self.input_multiple) % self.input_multiple
        pad_w = (self.input_multiple - orig_w % self.input_multiple) % self.input_multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, (orig_h, orig_w)

    @staticmethod
    def _crop(x, size):
        return x[..., :size[0], :size[1]]

    def _maybe_store(self, name, t):
        if self.store_layer_outputs:
            self.layer_outputs[name] = t.detach()

    def forward(self, x):
        x_padded, orig_size = self._pad_input(x)

        # CNN branch
        x_c1 = x_c2 = x_c3 = None
        edge1 = edge2 = None
        if self.branch_mode != 'swin_only':
            x_c0 = self.cnn_stem(x_padded); self._maybe_store('cnn_stem', x_c0)
            x_c1 = self._fwd(self.res2net1, x_c0); self._maybe_store('res2net1', x_c1)
            x_c2 = self._fwd(self.res2net2, x_c1); self._maybe_store('res2net2', x_c2)
            x_c3 = self._fwd(self.res2net3, x_c2); self._maybe_store('res2net3', x_c3)
            if self.use_edge_sup:
                edge1 = self._fwd(self.ed1, x_c1); self._maybe_store('edge_detect1', edge1)
                edge2 = self._fwd(self.ed2, x_c2); self._maybe_store('edge_detect2', edge2)

        # Swin branch
        x_s1 = x_s2 = None
        if self.branch_mode != 'cnn_only':
            x_s0 = self.patch_embed(x_padded); self._maybe_store('patch_embed', x_s0)
            b, c, h, w = x_s0.shape
            x_s1_flat = self._fwd(lambda inp: self.swin1(inp, h, w), x_s0.flatten(2).transpose(1, 2))
            x_s1 = x_s1_flat.transpose(1, 2).reshape(b, c, h, w); self._maybe_store('swin1', x_s1)

            x_s2_in = self.patch_merge(x_s1)
            b2, c2, h2, w2 = x_s2_in.shape
            x_s2_flat = self._fwd(
                lambda inp: self.swin3(inp, h2, w2),
                x_s2_in.flatten(2).transpose(1, 2),
            )
            x_s2 = x_s2_flat.transpose(1, 2).reshape(b2, c2, h2, w2); self._maybe_store('swin3', x_s2)

        # Fusion
        if self.branch_mode == 'both':
            fused_mid = self._fwd(self.bamf1, x_c2, x_s1, edge2)
            fused_deep = self._fwd(
                lambda c_, t_: self.bamf2(c_, t_, None),
                x_c3, x_s2,
            )
        elif self.branch_mode == 'cnn_only':
            fused_mid = self.single_mid(x_c2)
            fused_deep = self.single_deep(x_c3)
        else:  # swin_only
            target_mid = (x_padded.shape[-2] // 2, x_padded.shape[-1] // 2)
            target_deep = (x_padded.shape[-2] // 4, x_padded.shape[-1] // 4)
            fused_mid = self.single_mid(x_s1, target_size=target_mid)
            fused_deep = self.single_deep(x_s2, target_size=target_deep)
        self._maybe_store('fused_mid', fused_mid)

        fused_deep = self._fwd(self.context, fused_deep)
        self._maybe_store('context_block', fused_deep)

        d1 = self._fwd(self.dec1, fused_deep, fused_mid); self._maybe_store('decoder1', d1)

        # dec2 skip
        if self.branch_mode == 'swin_only':
            skip_full = self.swin_only_skip(x_padded)
        else:
            skip_full = x_c1
        d2 = self._fwd(self.dec2, d1, skip_full); self._maybe_store('decoder2', d2)

        if self.be_block is not None and edge1 is not None:
            final_feat = self._fwd(self.be_block, d2, edge1)
        else:
            final_feat = d2
        self._maybe_store('boundary_enhance', final_feat)

        seg_out = self.final_conv(final_feat); self._maybe_store('seg_out', seg_out)

        if self.edge_out_conv is not None:
            edge_out = self.edge_out_conv(final_feat)
        else:
            # Return a zero tensor of the right shape so callers don't break.
            edge_out = torch.zeros_like(seg_out)
        self._maybe_store('edge_out', edge_out)

        aux_out = self.aux_out_conv(d1)
        aux_out = F.interpolate(aux_out, size=seg_out.shape[2:], mode='bilinear', align_corners=False)
        self._maybe_store('aux_out', aux_out)

        seg_out = self._crop(seg_out, orig_size)
        edge_out = self._crop(edge_out, orig_size)
        aux_out = self._crop(aux_out, orig_size)

        return seg_out, edge_out, aux_out
