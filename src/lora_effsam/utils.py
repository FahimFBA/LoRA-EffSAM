# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
import random
from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> str:
    """
    "auto" -> cuda if present, else Apple-Silicon "mps", else "cpu".
    An explicit "mps"/"cuda" that isn't available fails loudly instead of silently
    running on CPU (which would be ~10x slower and easy to miss).
    """
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if mps_ok else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    if name == "mps" and not mps_ok:
        raise RuntimeError(
            "--device mps requested but MPS is not available. Needs Apple Silicon, "
            "macOS 12.3+, and an arm64 (not Rosetta/x86) Python + PyTorch build."
        )
    return name


def empty_device_cache(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def lesion_area(mask: torch.Tensor) -> torch.Tensor:
    """A_GT per sample: mask is [B, 1, H, W] in {0,1} -> returns [B] pixel counts."""
    return mask.flatten(1).sum(dim=1)


def dice_score(prob: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample Dice on binary prob/mask, both [B, 1, H, W]. Returns [B]."""
    p = prob.flatten(1)
    g = mask.flatten(1)
    inter = (p * g).sum(dim=1)
    return (2 * inter + eps) / (p.sum(dim=1) + g.sum(dim=1) + eps)


def boundary_distance_map(mask_np: np.ndarray) -> np.ndarray:
    """
    D_GT for ONE mask [H, W] in {0,1}: Euclidean distance (in pixels) from every
    pixel to the nearest pixel on the other side of the ground-truth contour (lesion
    pixels: distance to the background; background pixels: distance to the lesion).
    It is 1 (up to sqrt(2)) for pixels touching the contour and grows both into the
    lesion interior and into the background.
    """
    if mask_np.sum() == 0:
        # No lesion in this slice: define boundary distance as distance to nothing
        # in particular -> return zeros so the boundary loss contributes nothing.
        return np.zeros_like(mask_np, dtype=np.float32)

    # distance_transform_edt(X) gives, for each NONZERO pixel of X, the distance to
    # the nearest ZERO pixel of X (and 0 for zero pixels themselves).
    fg_dist = distance_transform_edt(mask_np)      # nonzero only at fg: dist to nearest bg pixel
    bg_dist = distance_transform_edt(1 - mask_np)  # nonzero only at bg: dist to nearest fg pixel
    # Distance-to-boundary: for a fg pixel use fg_dist (its distance to bg, i.e. to
    # the boundary from inside); for a bg pixel use bg_dist (its distance to fg,
    # i.e. to the boundary from outside).
    dist_to_boundary = np.where(mask_np > 0, fg_dist, bg_dist)
    return dist_to_boundary.astype(np.float32)


def batch_boundary_distance_maps(mask: torch.Tensor) -> torch.Tensor:
    """mask: [B, 1, H, W] tensor in {0,1} -> [B, 1, H, W] float distance maps."""
    device = mask.device
    mask_np = mask.detach().cpu().numpy()
    out = np.zeros_like(mask_np, dtype=np.float32)
    for b in range(mask_np.shape[0]):
        out[b, 0] = boundary_distance_map(mask_np[b, 0])
    return torch.from_numpy(out).to(device)


def region_masks(mask_np: np.ndarray, band_px: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a binary lesion mask [H, W], returns three boolean masks:
      boundary band B (band_px on each side of the contour),
      interior I (lesion minus boundary band),
      background G (everything outside the lesion minus boundary band).
    Used for Step 3's boundary-centered uncertainty analysis.
    """
    mask_bool = mask_np.astype(bool)
    if not mask_bool.any():
        empty = np.zeros_like(mask_bool)
        return empty, empty, ~mask_bool  # everything is "background" if no lesion

    struct = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask_bool, structure=struct, iterations=band_px)
    dilated = binary_dilation(mask_bool, structure=struct, iterations=band_px)

    boundary = dilated & ~eroded          # band straddling the contour
    interior = eroded                     # lesion core, away from the boundary
    background = ~dilated                 # background, away from the boundary
    return boundary, interior, background
