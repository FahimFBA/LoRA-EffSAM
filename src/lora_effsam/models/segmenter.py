# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Full pipeline: real, frozen EfficientSAM ViT encoder + LoRA (Step 1) -> optional
CBAM feature attention -> conv decoder (Theta_decoder) -> optional boundary-attention
refinement head. Trained with the Step 2 loss; supports Step 3's MC-Dropout via
enable_mc_dropout().

Theta_train = {Theta_LoRA, Theta_decoder, Theta_attention}   (attention modules are
an addition on top of the doc's Eq. 7, switchable via use_feature_attention /
use_refine_head for ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import (
    load_efficient_sam_encoder, run_encoder,
    ENCODER_IMG_SIZE, ENCODER_OUT_CHANNELS, ENCODER_PATCH_SIZE,
)
from .lora import inject_lora_into_vit, count_trainable_parameters
from .mask_decoder import MaskDecoder
from .hybrid_decoder import HybridDecoder
from .attention import FeatureAttention, AttentionRefineHead

# ViT skip connections for the hybrid decoder tap these fractions of encoder depth
# (blocks 3, 6, 9 of 12), deepest first -> decoder levels 1/8, 1/4, 1/2.
VIT_SKIP_DEPTHS = (0.75, 0.5, 0.25)


class LoRAMedSegmenter(nn.Module):
    def __init__(self, image_size: int = 512, variant: str = "vitt",
                 checkpoint_path: str = None,
                 lora_r: int = 4, lora_alpha: int = 4, lora_dropout: float = 0.0,
                 lora_also_proj: bool = False,
                 use_feature_attention: bool = True, use_refine_head: bool = True,
                 mc_dropout_p: float = 0.1,
                 encoder_size: int = ENCODER_IMG_SIZE,
                 grad_checkpoint: bool = False,
                 decoder: str = "simple", vit_skips: bool = False,
                 deep_supervision: bool = False, lora_mlp: bool = False):
        super().__init__()
        if decoder not in ("simple", "hybrid"):
            raise ValueError(f"decoder must be 'simple' or 'hybrid', got {decoder!r}")
        if (vit_skips or deep_supervision) and decoder != "hybrid":
            raise ValueError("vit_skips / deep_supervision need decoder='hybrid'")
        self.decoder_type = decoder
        self.deep_supervision = deep_supervision
        self.aux_logits = None  # deep-supervision logits of the last training forward
        if encoder_size % ENCODER_PATCH_SIZE != 0:
            raise ValueError(f"encoder_size must be a multiple of {ENCODER_PATCH_SIZE}")
        self.image_size = image_size
        self.encoder_size = encoder_size
        self.grad_checkpoint = grad_checkpoint

        encoder, pixel_mean, pixel_std = load_efficient_sam_encoder(variant, checkpoint_path)
        self.register_buffer("pixel_mean", pixel_mean, persistent=False)
        self.register_buffer("pixel_std", pixel_std, persistent=False)

        # Step 1: LoRA on Q/K/V, everything else in the encoder frozen.
        inject_lora_into_vit(encoder, r=lora_r, alpha=lora_alpha, dropout=lora_dropout,
                              also_proj=lora_also_proj, also_mlp=lora_mlp)
        self.encoder = encoder
        self.lora_dropout = lora_dropout

        self.use_feature_attention = use_feature_attention
        if use_feature_attention:
            self.feature_attn = FeatureAttention(channels=ENCODER_OUT_CHANNELS)

        n_blocks = len(encoder.blocks)
        self.vit_skip_layers = tuple(int(round(f * n_blocks)) - 1 for f in VIT_SKIP_DEPTHS) \
            if vit_skips else ()
        if decoder == "hybrid":
            self.decoder = HybridDecoder(
                neck_channels=ENCODER_OUT_CHANNELS, image_size=image_size,
                vit_channels=encoder.pos_embed.shape[-1] if vit_skips else 0,
                mc_dropout_p=mc_dropout_p, deep_supervision=deep_supervision,
            )
        else:
            self.decoder = MaskDecoder(
                in_channels=ENCODER_OUT_CHANNELS, feat_size=encoder_size // ENCODER_PATCH_SIZE,
                image_size=image_size, mc_dropout_p=mc_dropout_p,
            )

        self.use_refine_head = use_refine_head
        if use_refine_head:
            self.refine_head = AttentionRefineHead()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # Resize to the encoder's working resolution regardless of --image-size, then
        # normalize with EfficientSAM's own ImageNet mean/std (matches its pretraining).
        s = self.encoder_size
        if x.shape[-2:] != (s, s):
            x_enc = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        else:
            x_enc = x
        x_enc = (x_enc - self.pixel_mean) / self.pixel_std
        return run_encoder(self.encoder, x_enc,
                           grad_checkpoint=self.grad_checkpoint and self.training,
                           return_layers=self.vit_skip_layers)

    def encode(self, x: torch.Tensor):
        """Encoder (+LoRA) + optional feature attention. Deterministic unless lora_dropout > 0.

        Returns the neck feature map, or (neck, [ViT skip maps]) for the hybrid
        decoder. Either way it is passed unchanged to decode(), so MC-Dropout can
        encode once and decode K times regardless of decoder type."""
        out = self._encode(x)
        neck, taps = out if self.vit_skip_layers else (out, [])
        if self.use_feature_attention:
            neck = self.feature_attn(neck)
        return (neck, taps) if self.decoder_type == "hybrid" else neck

    def decode(self, x: torch.Tensor, feat) -> torch.Tensor:
        """Decoder (holds the MC-Dropout layer) + optional refine head.

        With deep supervision, the auxiliary logits of a TRAINING forward are left in
        self.aux_logits for the loss (None otherwise); the return value is always
        just the main logits."""
        images_at_size = x if x.shape[-1] == self.image_size else F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        if self.decoder_type == "hybrid":
            neck, taps = feat
            logits, self.aux_logits = self.decoder(images_at_size, neck, taps)
        else:
            logits = self.decoder(feat)  # [B, 1, image_size, image_size]
        if self.use_refine_head:
            logits = self.refine_head(images_at_size, logits)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(x, self.encode(x))

    def trainable_parameter_count(self) -> int:
        return count_trainable_parameters(self)

    # ---- Step 3 support -------------------------------------------------
    def enable_mc_dropout(self):
        """Only nn.Dropout/Dropout2d go to train mode; everything else (frozen
        encoder, GroupNorm, LayerNorm) stays in eval mode."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @property
    def encoder_is_stochastic(self) -> bool:
        """True if MC-Dropout also perturbs the encoder (LoRA dropout), so the
        encoder output can't be computed once and reused across the K passes."""
        return self.lora_dropout > 0


def build_model_from_checkpoint(path: str, device: str, **fallback_model_args):
    """
    Loads a checkpoint written by train.py. New-format checkpoints carry their own
    `model_args`, so evaluation can't silently rebuild a mismatched model (e.g. a
    different dropout rate or encoder size than was trained). Old raw state_dicts
    fall back to `fallback_model_args`.
    Returns (model, checkpoint_dict).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt and "model_args" in ckpt:
        model_args, state = dict(ckpt["model_args"]), ckpt["model"]
    else:
        model_args, state, ckpt = dict(fallback_model_args), ckpt, {}
    model_args["grad_checkpoint"] = False  # inference only
    model = LoRAMedSegmenter(**model_args)
    model.load_state_dict(state, strict=True)
    return model.to(device), ckpt
