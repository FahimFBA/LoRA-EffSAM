# 1. Method

All values below are the final model's (`model/final_config.json`). Code paths are in
`src/lora_effsam/` (e.g. `src/lora_effsam/models/hybrid_decoder.py`).

## 1.1 Overview

```
image (512x512, gray → 3 ch)
   │
   ├──────────────► CNN stem ──► skips at 1/1, 1/2, 1/4, 1/8 ───────────────┐
   │                                                                        │
   └─► resize to 512, ImageNet norm ─► EfficientSAM ViT-S encoder (frozen)  │
           + LoRA on Q,K,V of all 12 blocks (trainable)                     │
           ├── blocks 3, 6, 9 token maps (ViT skips) ───────────────────────┤
           └── neck output [256, 32, 32] ─► CBAM feature attention ─►  hybrid decoder
                                                                            │  (4 × upsample+fuse)
                                                          deep-supervision heads (train only)
                                                                            │
                                                     MC-Dropout (p=0.2) ─► 1×1 head ─► logits
                                                                            │
                                             boundary-gated residual refine head ─► mask
```

Parameters (counted from the final checkpoint):

| Part | Parameters | Trainable |
|---|---|---|
| ViT-S encoder (incl. neck) | 22,353,664 | no (frozen) |
| LoRA adapters (Q,K,V × 12 blocks) | 73,728 | yes |
| Feature attention (CBAM) | 8,564 | yes |
| Hybrid decoder (CNN stem, ViT-skip projections, decoder, heads) | 1,218,564 | yes |
| Refine head | 5,171 | yes |
| **Total** | **23,659,691** | **1,306,027 (5.5%)** |

## 1.2 Encoder: EfficientSAM ViT-S with LoRA (Step 1)

- **Backbone:** EfficientSAM image encoder, ViT-Small variant (patch 16, embedding dim 384,
  6 heads, 12 transformer blocks, neck to 256 channels), official pretrained weights
  (`efficient_sam_vits.pt`). Input resized to 512×512 (32×32 tokens); the positional
  embedding is bicubically interpolated from the 14×14 pretraining grid, as in
  EfficientSAM itself. Pixel normalization uses EfficientSAM's ImageNet mean/std.
  Attention is computed with `scaled_dot_product_attention` (numerically identical).
- **LoRA** on the fused Q,K,V projection of every block:

  h = W₀x + (α / r) · B A x,   r = 4, α = 4

  W₀ frozen; A initialised Kaiming-uniform, B = 0, so training starts exactly at the
  pretrained encoder. The attention output projection and MLPs stay frozen.
- **Feature attention** (CBAM-style) on the 256-channel neck output:
  out = x + γ·(x ⊙ A_channel(x) ⊙ A_spatial(·)), with γ initialised to 0 (identity at start).
  Channel attention: shared MLP (reduction 16) on avg- and max-pooled features;
  spatial attention: 7×7 conv on channel-wise mean and max.

## 1.3 Hybrid CNN–ViT decoder

A U-Net-style decoder that restores spatial detail lost at the ViT's 1/16 resolution
(`models/hybrid_decoder.py`).

- **CNN stem** on the raw image (normalised as (x − 0.5)/0.25): four stages of
  [conv3×3 → GroupNorm → GELU] ×2, stride 2 on the first conv of stages 2–4,
  giving 16, 32, 64, 128 channels at 1/1, 1/2, 1/4, 1/8 resolution.
- **ViT skips** (UNETR-style): token maps after blocks 9, 6, 3 (of 12), each 1×1-projected
  (conv → GN → GELU) to 64, 32, 16 channels and bilinearly resized to the 1/8, 1/4,
  1/2 decoder levels respectively.
- **Decoding:** start from the attention-weighted neck output at 1/16 (256 ch). Four levels;
  at each level: bilinear ×2 upsampling → concatenate [previous, CNN skip, ViT skip] →
  2× (conv3×3 → GN → GELU). Output channels 128, 64, 32, 16 at 1/8, 1/4, 1/2, 1/1.
- **MC-Dropout layer:** `Dropout2d(p = 0.2)` on the 16-channel full-resolution features,
  directly before the 1×1 output conv (same position as in the original decoder).
- **Deep supervision** (training only): 1×1 conv heads on the 1/8, 1/4, 1/2 level
  outputs, bilinearly upsampled to full size.
- GroupNorm (≤ 8 groups) everywhere, because the physical batch is 4.

## 1.4 Boundary-gated refinement head

Residual correction on the decoder logits (`models/attention.py`):
logits_out = logits + Conv1×1(F) ⊙ G, where F is a 3-layer conv encoder (16 channels,
one dilated conv) of [gray image, sigmoid(logits), logits/8], and
G = σ(s · maxpool₇(‖Sobel(sigmoid(logits))‖) + b) is a boundary-attention gate with
learnable s (init 8) and b (init −1). The output conv is zero-initialised (identity at start).

## 1.5 Size-adaptive, boundary-aware loss (Step 2)

For each image with ground-truth lesion area A (pixels) and prediction p = σ(logits):

- Size gate: g(A) = σ((T − A) / τ)
- Size-adaptive false-negative weight: α(A) = α₀ + (α_max − α₀) · g(A)  (α₀ if A = 0)
- Boundary multiplier: ω(A) = 1 + γ · g(A)  (1 if A = 0)
- Soft Dice loss: L_Dice = 1 − (2Σpg + ε) / (Σp + Σg + ε)
- Size-adaptive Tversky: L_ST = 1 − (TP + ε) / (TP + α(A)·FN + β₀·FP + ε), with soft TP, FN, FP
- Size-weighted boundary loss: L_B* = ω(A) · Σ pᵢ D̃ᵢ / (Σ pᵢ + ε), where D̃ = D_GT / max(H, W)
  and D_GT is the Euclidean distance of each pixel to the nearest ground-truth
  contour pixel (inside and outside). The normalisation by image size keeps L_B* on the
  same scale as the Dice terms (without it, training is unstable).
- Total: **L = L_Dice + λ_T · L_ST + λ_B · L_B***, averaged over the batch.

| Symbol | Value | Note |
|---|---|---|
| α₀, α_max, β₀ | 0.3, 0.7, 0.3 | |
| γ | 2.0 | |
| T | 400 px | 100 px at 256×256, scaled ×4 for 512×512 |
| τ | 20 px | |
| λ_T, λ_B | 1.0, 1.0 | |
| ε | 1e-6 | |

**Deep supervision:** L_total = L(main) + 0.5·L(aux 1/2) + 0.25·L(aux 1/4) + 0.125·L(aux 1/8),
each term the same loss above, sharing one distance map per image. Logged losses are
the main term only, so train and validation losses are comparable.

## 1.6 MC-Dropout uncertainty (Step 3)

At test time all layers are in eval mode except the Dropout2d layer. With K = 20
stochastic passes (the encoder runs once; only the decoder is repeated):

- p̄ᵢ = (1/K) Σₖ pᵢ⁽ᵏ⁾,   ŷᵢ = 1[p̄ᵢ ≥ 0.5],   Uᵢ = (1/K) Σₖ (pᵢ⁽ᵏ⁾ − p̄ᵢ)²
- Region means of U over: boundary band B (3 px either side of the GT contour, via 3×3
  binary dilation/erosion), interior I (eroded lesion), background G (outside the dilated lesion).
- Size stratification: small vs large lesions (training threshold T, and area quartiles).

## 1.7 Flip test-time augmentation (inference only)

Optional. Pass k of the K MC passes sees the input under flip k mod 4 (none, horizontal,
vertical, both); its output is flipped back before averaging. Same K = 20 decoder passes
(4 encoder passes). With TTA, U mixes dropout variance with flip variance, so all
uncertainty analysis in the paper uses the **non-TTA** runs.

## 1.8 Training

AdamW (lr 1e-4, weight decay 1e-4) on trainable parameters only; cosine annealing over 60
epochs; batch 4 × gradient accumulation 2 (effective 8); mixed precision (fp16 autocast,
loss in fp32); early stopping on validation Dice with patience 15; the checkpoint with the
best validation Dice is used for testing. Final model, seeds 42 / 1 / 2: trained 60 / 56 / 52
epochs, best epochs 55 / 41 / 37, 2.2 ± 0.2 h per seed on one RTX 3060.
