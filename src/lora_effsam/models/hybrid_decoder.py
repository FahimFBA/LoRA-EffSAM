# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Hybrid CNN-ViT decoder (U-Net style), an alternative to the plain conv-upsampling
`MaskDecoder` (selected with --decoder hybrid).

Why: the plain decoder rebuilds the mask from ONE ViT feature map at 1/16 of the
encoder input (e.g. 32x32 for a 512 image), so lesion boundaries and small lesions
are reconstructed from 16x16-pixel cells with no fine detail. This decoder adds the
detail back through skip connections, as in TransUNet / UNETR:

  * CNNStem: a small conv path on the raw image -> features at 1/1, 1/2, 1/4, 1/8.
  * Optional ViT skips (UNETR-lite): token maps from intermediate encoder blocks
    (default blocks 3, 6, 9 of 12), 1x1-projected and resized to the 1/2, 1/4, 1/8
    decoder levels, so mid-level transformer features reach the decoder too.
  * Decoder: start from the (LoRA-adapted, attention-weighted) ViT neck output at
    1/16, then 4x [upsample 2x -> concat skips -> 2 conv blocks] back to full size.
  * MC-Dropout (Step 3) sits exactly where it does in MaskDecoder: right before the
    final 1x1 head, so the uncertainty method is unchanged.
  * Optional deep supervision: 1x1 heads on the 1/8, 1/4, 1/2 decoder levels; their
    logits are only used for extra training loss terms, never for prediction.

GroupNorm throughout (small physical batches; see mask_decoder.py).
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    """2x (conv3x3 -> GroupNorm -> GELU); the first conv may downsample (stride 2)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False), _gn(out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), _gn(out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class CNNStem(nn.Module):
    """Raw image [B,3,H,W] -> [f1 (H), f2 (H/2), f4 (H/4), f8 (H/8)]."""

    def __init__(self, chans: Sequence[int] = (16, 32, 64, 128)):
        super().__init__()
        self.stages = nn.ModuleList([ConvBlock(3, chans[0])] + [
            ConvBlock(chans[i - 1], chans[i], stride=2) for i in range(1, len(chans))
        ])

    def forward(self, x) -> List[torch.Tensor]:
        x = (x - 0.5) / 0.25  # images arrive in [0, 1]
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class HybridDecoder(nn.Module):
    """
    forward(image, neck, taps) -> (logits [B,1,H,W], aux_logits list or None)

    image: [B,3,H,W] at image_size; neck: [B,C_neck,h,w] ViT neck output (any h, it is
    resized to H/16); taps: list of 3 ViT token maps [B,C_vit,h,w] for the 1/8, 1/4,
    1/2 levels (deepest first), or an empty list when ViT skips are off.
    """

    STEM = (16, 32, 64, 128)      # channels at 1/1, 1/2, 1/4, 1/8
    DEC = (128, 64, 32, 16)       # decoder channels at 1/8, 1/4, 1/2, 1/1
    TAP = (64, 32, 16)            # ViT-skip projection channels at 1/8, 1/4, 1/2

    def __init__(self, neck_channels: int, image_size: int, vit_channels: int = 0,
                 mc_dropout_p: float = 0.1, deep_supervision: bool = False):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError("hybrid decoder needs image_size divisible by 16")
        self.image_size = image_size
        self.use_vit_skips = vit_channels > 0
        self.deep_supervision = deep_supervision

        self.stem = CNNStem(self.STEM)
        if self.use_vit_skips:
            self.tap_proj = nn.ModuleList([
                nn.Sequential(nn.Conv2d(vit_channels, c, 1, bias=False), _gn(c), nn.GELU())
                for c in self.TAP
            ])

        # level k fuses: upsampled previous + stem skip (+ ViT skip for the first 3)
        stem_skips = (self.STEM[3], self.STEM[2], self.STEM[1], self.STEM[0])
        in_prev = neck_channels
        self.blocks = nn.ModuleList()
        for k, out_ch in enumerate(self.DEC):
            tap = self.TAP[k] if (self.use_vit_skips and k < 3) else 0
            self.blocks.append(ConvBlock(in_prev + stem_skips[k] + tap, out_ch))
            in_prev = out_ch

        self.dropout = nn.Dropout2d(mc_dropout_p)
        self.head = nn.Conv2d(self.DEC[-1], 1, kernel_size=1)
        if deep_supervision:  # on the 1/8, 1/4, 1/2 levels
            self.aux_heads = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in self.DEC[:3]])

    def forward(self, image: torch.Tensor, neck: torch.Tensor,
                taps: Sequence[torch.Tensor] = ()) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        H = self.image_size
        f1, f2, f4, f8 = self.stem(image)
        skips = [f8, f4, f2, f1]

        x = neck
        if x.shape[-1] != H // 16:
            x = F.interpolate(x, size=(H // 16, H // 16), mode="bilinear", align_corners=False)

        aux = []
        for k, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            parts = [x, skips[k]]
            if self.use_vit_skips and k < 3:
                t = self.tap_proj[k](taps[k])
                parts.append(F.interpolate(t, size=x.shape[-2:], mode="bilinear", align_corners=False))
            x = block(torch.cat(parts, dim=1))
            if self.deep_supervision and k < 3 and self.training:
                aux.append(self.aux_heads[k](x))

        logits = self.head(self.dropout(x))
        if aux:
            aux = [F.interpolate(a, size=(H, H), mode="bilinear", align_corners=False) for a in aux]
        return logits, (aux or None)
