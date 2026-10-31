# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Loads the REAL EfficientSAM image encoder (frozen backbone) from the official repo.

Requires a checkout of https://github.com/yformer/EfficientSAM (fetched by
`scripts/fetch_efficientsam.sh` into third_party/EfficientSAM, or set
LORA_EFFSAM_EFFICIENTSAM_DIR; see lora_effsam.paths) with its `weights/efficient_sam_vitt.pt`
(the zipped ViT-S weights are unzipped automatically).

EfficientSAM was pretrained at 1024x1024 RGB input and outputs a [B, 256, S/16, S/16]
feature map (patch_size=16, neck_dims=[256,256]). Its own forward() hard-asserts
S == 1024, but its positional embedding is ALWAYS bicubic-interpolated from the
14x14 (224px) pretraining grid to whatever token grid it sees, so the architecture
itself works at any multiple of 16. `run_encoder` below re-implements that forward
without the assert, which allows a smaller `encoder_size` (e.g. 512 -> 1024 tokens
instead of 4096, ~16x less attention memory) on memory-limited machines such as an
8GB Apple-Silicon laptop. 1024 remains the default (matches pretraining).

Two memory optimizations are also applied, neither of which changes the math:
  * attention uses F.scaled_dot_product_attention instead of materialising the full
    softmax(QK^T) matrix by hand;
  * optional per-block gradient checkpointing (recompute activations in backward).
"""
import os
import sys
import types
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_VARIANTS = {
    "vitt": dict(encoder_patch_embed_dim=192, encoder_num_heads=3, ckpt="efficient_sam_vitt.pt"),
    "vits": dict(encoder_patch_embed_dim=384, encoder_num_heads=6, ckpt="efficient_sam_vits.pt"),
}

ENCODER_IMG_SIZE = 1024      # EfficientSAM's pretraining resolution (default encoder_size)
ENCODER_OUT_CHANNELS = 256   # neck_dims[-1]
ENCODER_PATCH_SIZE = 16
ENCODER_FEAT_SIZE = ENCODER_IMG_SIZE // ENCODER_PATCH_SIZE  # 64 at the default size


def _repo_root() -> str:
    from lora_effsam.paths import efficientsam_dir
    return str(efficientsam_dir())


def _sdpa_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for EfficientSAM's Attention.forward (identical math)."""
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj(x)


def _patch_attention_with_sdpa(encoder: nn.Module) -> int:
    n = 0
    for m in encoder.modules():
        if type(m).__name__ == "Attention" and hasattr(m, "qkv") and hasattr(m, "scale"):
            m.forward = types.MethodType(_sdpa_attention_forward, m)
            n += 1
    return n


def run_encoder(encoder: nn.Module, x: torch.Tensor, grad_checkpoint: bool = False,
                return_layers: Tuple[int, ...] = ()):
    """
    Same computation as EfficientSAM's ImageEncoderViT.forward, minus the
    `input size == 1024` assert, plus optional per-block gradient checkpointing.
    x: [B, 3, S, S] already normalized, S a multiple of 16. Returns [B, 256, S/16, S/16].

    return_layers: block indices whose token outputs are also returned, as
    [B, C_vit, S/16, S/16] maps -> (neck_out, [tap, ...]) in the given order.
    """
    from efficient_sam.efficient_sam_encoder import get_abs_pos  # type: ignore

    x = encoder.patch_embed(x).permute(0, 2, 3, 1)  # B H W C
    x = x + get_abs_pos(encoder.pos_embed, encoder.pretrain_use_cls_token,
                        [x.shape[1], x.shape[2]])
    B, h, w, C = x.shape
    x = x.reshape(B, h * w, C)
    use_ckpt = grad_checkpoint and torch.is_grad_enabled()
    taps = {}
    for i, blk in enumerate(encoder.blocks):
        x = checkpoint(blk, x, use_reentrant=False) if use_ckpt else blk(x)
        if i in return_layers:
            taps[i] = x.reshape(B, h, w, C).permute(0, 3, 1, 2)
    x = x.reshape(B, h, w, C).permute(0, 3, 1, 2)
    out = encoder.neck(x)
    if return_layers:
        return out, [taps[i] for i in return_layers]
    return out


def load_efficient_sam_encoder(variant: str = "vitt", checkpoint_path: str = None
                                ) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Returns (image_encoder, pixel_mean[1,3,1,1], pixel_std[1,3,1,1]).
    Only the image_encoder is kept; EfficientSAM's promptable prompt_encoder /
    mask_decoder are discarded (we do dense/automatic segmentation, not
    point/box-prompted segmentation, and Step 1 only requires the image encoder).
    """
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {list(_VARIANTS)}, got {variant!r}")
    spec = _VARIANTS[variant]

    repo = os.path.abspath(_repo_root())
    if not os.path.isdir(repo):
        raise ImportError(
            f"EfficientSAM not found at {repo}. From the repository root run:\n"
            f"  bash scripts/fetch_efficientsam.sh\n"
            f"or point LORA_EFFSAM_EFFICIENTSAM_DIR at an existing checkout "
            f"(weights ship inside its weights/ folder)."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(repo, "weights", spec["ckpt"])
        zipped = checkpoint_path + ".zip"  # the repo ships vits weights zipped
        if not os.path.isfile(checkpoint_path) and os.path.isfile(zipped):
            import zipfile
            with zipfile.ZipFile(zipped) as z:
                z.extract(spec["ckpt"], os.path.dirname(checkpoint_path))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"EfficientSAM checkpoint not found: {checkpoint_path}. "
            f"(For variant='vits', unzip weights/efficient_sam_vits.pt.zip first.)"
        )

    from efficient_sam.efficient_sam import build_efficient_sam  # type: ignore

    sam = build_efficient_sam(
        encoder_patch_embed_dim=spec["encoder_patch_embed_dim"],
        encoder_num_heads=spec["encoder_num_heads"],
        checkpoint=checkpoint_path,
    )

    encoder = sam.image_encoder
    pixel_mean = sam.pixel_mean.clone()
    pixel_std = sam.pixel_std.clone()
    del sam  # drop the (unused) prompt encoder + mask decoder, save memory

    assert encoder.img_size == ENCODER_IMG_SIZE
    if _patch_attention_with_sdpa(encoder) == 0:
        raise RuntimeError("Could not find EfficientSAM Attention modules to patch.")
    return encoder, pixel_mean, pixel_std
