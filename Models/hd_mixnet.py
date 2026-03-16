# Models/hd_mixnet.py
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
    """
    Fuse local CNN features and global Transformer features while
    letting edge evidence tilt the mixture toward sharper local detail.
    """
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
            trans_feat,
            size=cnn_feat.shape[2:],
            mode='bilinear',
            align_corners=False,
        )

        c = self.conv_cnn(cnn_feat)
        t = self.conv_trans(trans_feat)

        gate = self.mix_gate(torch.cat([c, t], dim=1))
        fused = gate * c + (1.0 - gate) * t

        if edge_feat is not None and self.edge_gate is not None:
            if edge_feat.shape[2:] != fused.shape[2:]:
                edge_feat = F.interpolate(
                    edge_feat,
                    size=fused.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            edge_weight = self.edge_gate(edge_feat)
            fused = fused * (1.0 + edge_weight) + c * edge_weight

        return self.refine(fused) + c


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

        self.use_grad_checkpointing = bool(getattr(config, 'USE_GRAD_CHECKPOINTING', False))
        self.input_multiple = window_size * 8

        if embed_dim % heads_stage1 != 0:
            raise ValueError(
                f"SWIN_EMBED_DIM ({embed_dim}) must be divisible by "
                f"SWIN_HEADS_STAGE1 ({heads_stage1})."
            )
        if (embed_dim * 2) % heads_stage2 != 0:
            raise ValueError(
                f"SWIN_EMBED_DIM*2 ({embed_dim * 2}) must be divisible by "
                f"SWIN_HEADS_STAGE2 ({heads_stage2})."
            )

        dpr_stage1 = torch.linspace(0.0, drop_path, stage_depths[0]).tolist()
        dpr_stage2 = torch.linspace(
            dpr_stage1[-1] if dpr_stage1 else 0.0,
            drop_path,
            stage_depths[1],
        ).tolist()

        self.cnn_stem = nn.Sequential(
            ConvBNAct(3, base_channels, kernel_size=3),
            ConvBNAct(base_channels, base_channels, kernel_size=3),
        )
        self.res2net1 = Res2NetBlock(base_channels, base_channels, scale=res2net_scale)
        self.res2net2 = Res2NetBlock(base_channels, stage2_channels, scale=res2net_scale, stride=2)
        self.res2net3 = Res2NetBlock(stage2_channels, stage3_channels, scale=res2net_scale, stride=2)

        self.embed_dim = embed_dim
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, self.embed_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
        )

        self.swin1 = BasicSwinLayer(
            dim=self.embed_dim,
            depth=stage_depths[0],
            num_heads=heads_stage1,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=dpr_stage1,
        )

        self.patch_merge = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim * 2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(self.embed_dim * 2),
            nn.GELU(),
        )
        self.swin3 = BasicSwinLayer(
            dim=self.embed_dim * 2,
            depth=stage_depths[1],
            num_heads=heads_stage2,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=dpr_stage2,
        )

        self.ed1 = EdgeDetectBlock(base_channels, base_channels)
        self.ed2 = EdgeDetectBlock(stage2_channels, stage2_channels)

        self.bamf1 = BoundaryAwareMixFusion(
            stage2_channels,
            self.embed_dim,
            stage2_channels,
            edge_dim=stage2_channels,
        )
        self.bamf2 = BoundaryAwareMixFusion(
            stage3_channels,
            self.embed_dim * 2,
            stage3_channels,
        )
        self.context = PyramidContextBlock(stage3_channels, stage3_channels)

        self.dec1 = DecoderRefineBlock(stage3_channels, stage2_channels, stage2_channels)
        self.dec2 = DecoderRefineBlock(stage2_channels, base_channels, base_channels)

        self.be_block = BoundaryEnhanceBlock(base_channels, base_channels)

        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        self.edge_out_conv = nn.Conv2d(base_channels, 1, 1)
        self.aux_out_conv = nn.Conv2d(stage2_channels, num_classes, 1)

    def _forward_with_checkpoint(self, fn, *args):
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
    def _crop_to_size(x, size):
        return x[..., :size[0], :size[1]]

    def forward(self, x):
        x, orig_size = self._pad_input(x)

        x_c0 = self.cnn_stem(x)
        x_c1 = self._forward_with_checkpoint(self.res2net1, x_c0)
        edge1 = self._forward_with_checkpoint(self.ed1, x_c1)

        x_c2 = self._forward_with_checkpoint(self.res2net2, x_c1)
        edge2 = self._forward_with_checkpoint(self.ed2, x_c2)

        x_c3 = self._forward_with_checkpoint(self.res2net3, x_c2)

        x_s0 = self.patch_embed(x)
        b, c, h, w = x_s0.shape
        x_s0_flat = x_s0.flatten(2).transpose(1, 2)

        x_s1_flat = self._forward_with_checkpoint(
            lambda inp: self.swin1(inp, h, w),
            x_s0_flat,
        )
        x_s1 = x_s1_flat.transpose(1, 2).reshape(b, c, h, w)

        x_s2_in = self.patch_merge(x_s1)
        b2, c2, h2, w2 = x_s2_in.shape
        x_s2_flat = self._forward_with_checkpoint(
            lambda inp: self.swin3(inp, h2, w2),
            x_s2_in.flatten(2).transpose(1, 2),
        )
        x_s2 = x_s2_flat.transpose(1, 2).reshape(b2, c2, h2, w2)

        fused_mid = self._forward_with_checkpoint(self.bamf1, x_c2, x_s1, edge2)
        fused_deep = self._forward_with_checkpoint(
            lambda cnn_feat, trans_feat: self.bamf2(cnn_feat, trans_feat, None),
            x_c3,
            x_s2,
        )
        fused_deep = self._forward_with_checkpoint(self.context, fused_deep)

        d1 = self._forward_with_checkpoint(self.dec1, fused_deep, fused_mid)
        d2 = self._forward_with_checkpoint(self.dec2, d1, x_c1)

        final_feat = self._forward_with_checkpoint(self.be_block, d2, edge1)

        seg_out = self.final_conv(final_feat)
        edge_out = self.edge_out_conv(final_feat)
        aux_out = self.aux_out_conv(d1)
        aux_out = F.interpolate(aux_out, size=seg_out.shape[2:], mode='bilinear', align_corners=False)

        seg_out = self._crop_to_size(seg_out, orig_size)
        edge_out = self._crop_to_size(edge_out, orig_size)
        aux_out = self._crop_to_size(aux_out, orig_size)

        return seg_out, edge_out, aux_out
