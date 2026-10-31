# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Extra attention components (your request: "implement attention mechanism so the
result can be bumped"). These are ADDITIONS on top of the methodology docs, so they
are individually switchable (--no-feature-attention / --no-refine-head) for ablation.

Both modules are exactly the IDENTITY at initialisation (zero-initialised gates), so
the model starts from the pretrained EfficientSAM behaviour and only learns to use the
attention if it helps.

1. FeatureAttention  (CBAM: channel attention -> spatial attention)
   Applied to the frozen encoder's image embeddings [B, 256, h, w] just before the
   mask decoder. Lets the trainable side re-weight *which* medical-domain feature
   channels / spatial locations the decoder should listen to.

2. AttentionRefineHead  (boundary-attention-gated residual refinement)
   EfficientSAM's decoder emits low-resolution masks (input/4, e.g. 128x128 for a
   512 image) that are bicubic-upsampled, so boundaries are soft. This head looks at
   the full-resolution image + the coarse prediction and predicts a residual logit
   correction, gated by a spatial attention map that peaks on the predicted boundary
   (edge of the coarse probability map). It therefore concentrates its capacity on
   the boundary band, which is what the boundary-aware loss (Step 2) cares about.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )

    def forward(self, x):
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        return torch.sigmoid(avg + mx)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


class FeatureAttention(nn.Module):
    """out = x + gamma * (x * A_channel * A_spatial),  gamma initialised to 0."""

    def __init__(self, channels: int = 256, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        attended = x * self.ca(x)
        attended = attended * self.sa(attended)
        return x + self.gamma * attended


class AttentionRefineHead(nn.Module):
    """
    logits_out = logits + delta,   delta = out_conv(features) * boundary_attention
    out_conv is zero-initialised => identity at init.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),          # [gray image, coarse prob, coarse logit]
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GroupNorm(4, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        # Fixed Sobel filters -> edge magnitude of the coarse probability map
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer("sobel_x", kx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", kx.t().contiguous().view(1, 1, 3, 3), persistent=False)
        # learnable sharpening of the edge map into an attention gate in (0, 1)
        self.gate_scale = nn.Parameter(torch.tensor(8.0))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))

    def boundary_attention(self, prob):
        gx = F.conv2d(prob, self.sobel_x, padding=1)
        gy = F.conv2d(prob, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-8)
        # dilate the edge a little so the gate covers a band around the boundary
        edge = F.max_pool2d(edge, kernel_size=7, stride=1, padding=3)
        return torch.sigmoid(self.gate_scale * edge + self.gate_bias)

    def forward(self, images, logits):
        gray = images.mean(dim=1, keepdim=True)
        prob = torch.sigmoid(logits)
        feats = self.encoder(torch.cat([gray, prob, logits / 8.0], dim=1))
        gate = self.boundary_attention(prob.detach())
        return logits + self.out_conv(feats) * gate
