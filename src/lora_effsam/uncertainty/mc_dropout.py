# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 3: Explainable Uncertainty Analysis via Monte Carlo Dropout.

    p_i^(k) = f_theta^(k)(x)_i,   k = 1..K          (Eq. 1)
    pbar_i  = (1/K) sum_k p_i^(k)                   (Eq. 2)
    yhat_i  = 1[pbar_i >= 0.5]                       (Eq. 3)
    U_i     = (1/K) sum_k (p_i^(k) - pbar_i)^2       (Eq. 4)

Then region-wise (Eq. 5-7) and size-stratified (Eq. 10-11) means of U, computed
ONLY at evaluation time and only using ground truth for analysis, never for
training or for producing the mean/uncertainty maps themselves.
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from lora_effsam.utils import region_masks, lesion_area, dice_score


@dataclass
class UncertaintyResult:
    mean_prob: torch.Tensor       # [B, 1, H, W]  pbar_i
    pred_mask: torch.Tensor       # [B, 1, H, W]  yhat_i
    uncertainty: torch.Tensor     # [B, 1, H, W]  U_i (normalized to [0,1] for viz)


_FLIPS = [(), (-1,), (-2,), (-2, -1)]  # identity, horizontal, vertical, both


@torch.no_grad()
def mc_dropout_predict(model, x: torch.Tensor, K: int = 20, tta: bool = False) -> UncertaintyResult:
    """
    Runs K stochastic forward passes with MC-Dropout enabled (dropout active,
    everything else in eval mode — see LoRAMedSegmenter.enable_mc_dropout) and
    returns the mean probability map, thresholded mask, and pixel-wise variance.

    tta=True: pass k sees the input under flip k mod 4 (none / h / v / h+v) and its
    output is flipped back, so the mean also averages over flips. Same K decoder
    passes; U_i then mixes dropout variance with flip variance.
    """
    model.enable_mc_dropout()
    flips = _FLIPS if tta else [()]

    # The only dropout layers sit after the encoder (unless LoRA dropout is on), so
    # the expensive ViT output is identical across passes: compute it once (per
    # flip) and re-run only the stochastic decoder K times (~K x faster, same result).
    views = [torch.flip(x, f) if f else x for f in flips]
    feats = [None if model.encoder_is_stochastic else model.encode(v) for v in views]
    probs = []
    for k in range(K):
        j = k % len(flips)
        v, feat = views[j], feats[j]
        logits = model(v) if feat is None else model.decode(v, feat)
        if flips[j]:
            logits = torch.flip(logits, flips[j])
        probs.append(torch.sigmoid(logits))
    probs = torch.stack(probs, dim=0)  # [K, B, 1, H, W]

    mean_prob = probs.mean(dim=0)                       # Eq. 2
    pred_mask = (mean_prob >= 0.5).float()               # Eq. 3
    uncertainty = ((probs - mean_prob.unsqueeze(0)) ** 2).mean(dim=0)  # Eq. 4

    # Normalize per-image to [0,1] for visualization only (doesn't affect the raw
    # variance values returned, callers can use either).
    return UncertaintyResult(mean_prob=mean_prob, pred_mask=pred_mask, uncertainty=uncertainty)


def normalize_uncertainty(u: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalization to [0,1], for visualization (not for stats)."""
    b = u.shape[0]
    flat = u.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    norm = (flat - lo) / (hi - lo + 1e-8)
    return norm.view_as(u)


def region_uncertainty(uncertainty: torch.Tensor, gt_mask: torch.Tensor,
                        band_px: int = 3) -> Dict[str, float]:
    """
    Eq. 5-7: mean uncertainty in boundary band B, interior I, background G, for a
    SINGLE image. uncertainty, gt_mask: [1, H, W] or [H, W].
    """
    u_np = uncertainty.squeeze().detach().cpu().numpy()
    m_np = gt_mask.squeeze().detach().cpu().numpy()

    boundary, interior, background = region_masks(m_np, band_px=band_px)

    def _mean(region):
        return float(u_np[region].mean()) if region.any() else float("nan")

    return {
        "U_boundary": _mean(boundary),
        "U_interior": _mean(interior),
        "U_background": _mean(background),
    }


def size_stratified_analysis(per_image_boundary_u: List[float],
                              areas: List[float], T: float) -> Dict[str, float]:
    """
    Eq. 8-11: split images into small (A_GT < T) / large (A_GT >= T) groups using
    the SAME threshold T as the training loss, and average each group's
    U_boundary (from `region_uncertainty`).
    """
    small = [u for u, a in zip(per_image_boundary_u, areas) if a < T and not np.isnan(u)]
    large = [u for u, a in zip(per_image_boundary_u, areas) if a >= T and not np.isnan(u)]
    return {
        "U_small": float(np.mean(small)) if small else float("nan"),
        "U_large": float(np.mean(large)) if large else float("nan"),
        "n_small": len(small),
        "n_large": len(large),
    }


def full_dataset_uncertainty_report(model, dataloader, K: int, T: float, tta: bool = False,
                                     band_px: int = 3, device: str = "cpu") -> Dict:
    """
    Runs the complete Step-3 analysis over a dataloader:
      - MC-Dropout mean/uncertainty maps per image (no ground truth used here)
      - boundary/interior/background means per image (ground truth used only here)
      - small vs. large lesion stratification of boundary uncertainty
    Returns a dict with per-image records and the aggregated small/large summary.
    """
    model.to(device)
    per_image = []

    for images, masks, labels in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        result = mc_dropout_predict(model, images, K=K, tta=tta)
        dice = dice_score(result.pred_mask, masks)

        for i in range(images.shape[0]):
            regions = region_uncertainty(result.uncertainty[i], masks[i], band_px=band_px)
            area = float(lesion_area(masks[i:i+1]).item())
            per_image.append({
                "area": area,
                "label": int(labels[i]) if torch.is_tensor(labels) else int(labels),
                "dice": float(dice[i].item()),
                **regions,
            })

    boundary_u = [r["U_boundary"] for r in per_image]
    areas = [r["area"] for r in per_image]
    stratified = size_stratified_analysis(boundary_u, areas, T)
    mean_dice = float(np.mean([r["dice"] for r in per_image])) if per_image else float("nan")

    return {"per_image": per_image, "size_stratified": stratified, "mc_mean_dice": mean_dice}
