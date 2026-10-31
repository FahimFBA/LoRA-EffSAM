# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Trainable decoder (Theta_decoder in Eq. 7) taking EfficientSAM's frozen encoder
feature map [B, 256, 64, 64] to full-resolution foreground logits [B, 1, image_size,
image_size]. NOT SAM's own promptable mask decoder (that expects point/box prompts;
we do dense automatic segmentation) — this is a plain conv-upsampling head, same
role as Theta_decoder in the methodology doc.

Uses GroupNorm, not BatchNorm: the physical batch is 1-2 on laptops / small GPUs
(gradient accumulation does not help BatchNorm statistics), and BatchNorm running
stats from such tiny batches make eval-mode predictions diverge from training.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


class MaskDecoder(nn.Module):
    def __init__(self, in_channels: int, feat_size: int, image_size: int,
                 mc_dropout_p: float = 0.1):
        super().__init__()
        ratio = max(image_size // feat_size, 1)
        n_ups = max(int(round(math.log2(ratio))), 0)
        chans = [in_channels]
        for _ in range(n_ups):
            chans.append(max(chans[-1] // 2, 16))

        self.ups = nn.ModuleList(
            [ConvUpBlock(chans[i], chans[i + 1]) for i in range(n_ups)]
        )
        self.dropout = nn.Dropout2d(mc_dropout_p)
        self.head = nn.Conv2d(chans[-1], 1, kernel_size=1)
        self.image_size = image_size

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = feat
        for up in self.ups:
            x = up(x)
        x = self.dropout(x)
        logits = self.head(x)
        if logits.shape[-1] != self.image_size:
            logits = F.interpolate(
                logits, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return logits
