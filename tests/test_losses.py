# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Size-adaptive boundary-aware loss (Step 2) and the metric/distance utilities."""
import numpy as np
import torch

from lora_effsam.config import LossConfig
from lora_effsam.losses.size_adaptive_loss import (SizeAdaptiveBoundaryLoss, size_adaptive_alpha,
                                             size_gate_omega)
from lora_effsam.utils import boundary_distance_map, dice_score, region_masks


def _disk(size=64, r=10, c=(32, 32)):
    yy, xx = np.mgrid[:size, :size]
    return ((yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r).astype(np.float32)


def test_size_gate_small_vs_large():
    area = torch.tensor([0.0, 50.0, 5000.0])
    omega = size_gate_omega(area, T=400, tau=20, gamma=2.0)
    alpha = size_adaptive_alpha(area, T=400, tau=20, alpha0=0.3, alpha_max=0.7)
    assert omega[0] == 1.0 and alpha[0] == 0.3            # empty mask: neutral weights
    assert torch.isclose(omega[1], torch.tensor(3.0), atol=1e-3)   # tiny lesion: 1 + gamma
    assert torch.isclose(alpha[1], torch.tensor(0.7), atol=1e-3)   # tiny lesion: alpha_max
    assert torch.isclose(omega[2], torch.tensor(1.0), atol=1e-3)   # large lesion: no boost


def test_perfect_prediction_beats_inverted_prediction():
    mask = torch.from_numpy(_disk())[None, None]
    loss_fn = SizeAdaptiveBoundaryLoss(LossConfig(T=400, tau=20))
    good, parts = loss_fn((mask * 2 - 1) * 20, mask)
    bad, _ = loss_fn(-(mask * 2 - 1) * 20, mask)
    assert parts["L_Dice"] < 1e-3 and parts["L_ST"] < 1e-3
    # The boundary term (Eq. 6) averages the distance to the contour over ALL predicted
    # pixels, interior included, so it is > 0 even for a perfect mask.
    assert parts["L_B_star"] > 0
    assert good < 0.5 * bad


def test_distance_map_grows_away_from_contour():
    m = _disk()
    d = boundary_distance_map(m)
    boundary, interior, background = region_masks(m, band_px=1)
    assert d[32, 32] > 5 and d[0, 0] > 5                  # far inside / far outside
    # pixels next to the contour are 1-sqrt(2) px from the nearest pixel across it
    assert 1.0 <= d[boundary].min() and d[boundary].max() <= 2.0
    assert not (interior & background).any()


def test_dice_score():
    m = torch.from_numpy(_disk())[None, None]
    assert torch.isclose(dice_score(m, m), torch.tensor([1.0]))
    assert dice_score(torch.zeros_like(m), m).item() < 1e-6
