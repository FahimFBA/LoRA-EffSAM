# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 2: Size-Adaptive, Boundary-Aware Loss.

    L_total = L_Dice + lambda_T * L_ST + lambda_B * L_B*

All symbols match the methodology doc 1:1 (see docstring on each function).
Boundary distance maps D_GT are computed once per mask via
`utils.batch_boundary_distance_maps` (pass them in precomputed, or let this module
compute them on the fly — see `SizeAdaptiveBoundaryLoss.forward`).
"""
import torch
import torch.nn as nn

from lora_effsam.utils import lesion_area, batch_boundary_distance_maps


def _sigmoid_gate(A_GT: torch.Tensor, T: float, tau: float) -> torch.Tensor:
    """sigma((T - A_GT) / tau), the shared smooth gate used by omega() and alpha()."""
    return torch.sigmoid((T - A_GT) / tau)


def size_gate_omega(A_GT: torch.Tensor, T: float, tau: float, gamma: float) -> torch.Tensor:
    """
    Eq. 2:
        omega(A_GT) = 1                                  if A_GT == 0
                    = 1 + gamma * sigma((T - A_GT)/tau)   if A_GT  > 0
    A_GT: [B] pixel counts. Returns [B].
    """
    gate = 1.0 + gamma * _sigmoid_gate(A_GT, T, tau)
    return torch.where(A_GT > 0, gate, torch.ones_like(gate))


def size_adaptive_alpha(A_GT: torch.Tensor, T: float, tau: float,
                         alpha0: float, alpha_max: float) -> torch.Tensor:
    """
    Eq. 3:
        alpha(A_GT) = alpha0                                        if A_GT == 0
                    = alpha0 + (alpha_max - alpha0)*sigma((T-A_GT)/tau)  if A_GT > 0
    Returns [B].
    """
    val = alpha0 + (alpha_max - alpha0) * _sigmoid_gate(A_GT, T, tau)
    return torch.where(A_GT > 0, val, torch.full_like(val, alpha0))


def dice_loss(prob: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard soft Dice loss, per-sample then averaged over the batch."""
    p = prob.flatten(1)
    g = mask.flatten(1)
    inter = (p * g).sum(dim=1)
    dice = (2 * inter + eps) / (p.sum(dim=1) + g.sum(dim=1) + eps)
    return (1 - dice).mean()


def size_adaptive_tversky_loss(prob: torch.Tensor, mask: torch.Tensor,
                                alpha_vec: torch.Tensor, beta0: float,
                                eps: float) -> torch.Tensor:
    """
    Eq. 4-5:
        TP, FN, FP accumulated per sample over all pixels.
        L_ST = 1 - (TP + eps) / (TP + alpha(A_GT)*FN + beta*FP + eps)
    prob, mask: [B, 1, H, W]. alpha_vec: [B]. Returns scalar (mean over batch).
    """
    p = prob.flatten(1)
    g = mask.flatten(1)
    TP = (p * g).sum(dim=1)
    FN = ((1 - p) * g).sum(dim=1)
    FP = (p * (1 - g)).sum(dim=1)
    tversky = (TP + eps) / (TP + alpha_vec * FN + beta0 * FP + eps)
    return (1 - tversky).mean()


def _normalize_dist_map(dist_map: torch.Tensor) -> torch.Tensor:
    """
    D_GT is in raw pixel units, which can be tens to hundreds for a 512x512 image.
    Eq. 6/7 as written (sum(p*D)/sum(p)) then has units of "pixels", while L_Dice
    and L_ST are unitless in [0,1] — so an unnormalized L_B*/L_B dominates
    L_total by 1-2 orders of magnitude and destabilizes training (verified
    empirically: with raw pixel distances, gradient descent on a synthetic disk
    target converges to Dice ~0.56 instead of ~1.0 at lambda_B=1; dividing by the
    image's longer side fixes it — confirmed same experiment reaches Dice ~1.0).
    We divide by max(H, W) so L_B/L_B* stay roughly comparable in scale to L_Dice
    and L_ST, which is what makes lambda_B=1 a sane default instead of requiring
    a dataset-specific rescale.
    """
    H, W = dist_map.shape[-2], dist_map.shape[-1]
    return dist_map / float(max(H, W))


def boundary_loss(prob: torch.Tensor, dist_map: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Eq. 6: L_B = sum(p_i * D_GT,i) / (sum(p_i) + eps), per sample then averaged.
    prob, dist_map: [B, 1, H, W]. dist_map is normalized internally (see
    `_normalize_dist_map`) before use.
    """
    d = _normalize_dist_map(dist_map)
    p = prob.flatten(1)
    d = d.flatten(1)
    num = (p * d).sum(dim=1)
    den = p.sum(dim=1) + eps
    return (num / den).mean()


def size_weighted_boundary_loss(prob: torch.Tensor, dist_map: torch.Tensor,
                                 omega_vec: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Eq. 7: L_B* = omega(A_GT) * L_B, applied per sample before averaging (so the
    gate scales each sample's own boundary term, not a single scalar over the batch).
    dist_map is normalized internally (see `_normalize_dist_map`).
    """
    d = _normalize_dist_map(dist_map)
    p = prob.flatten(1)
    d = d.flatten(1)
    num = (p * d).sum(dim=1)
    den = p.sum(dim=1) + eps
    per_sample_LB = num / den
    return (omega_vec * per_sample_LB).mean()


class SizeAdaptiveBoundaryLoss(nn.Module):
    """
    Eq. 8: L_total = L_Dice + lambda_T * L_ST + lambda_B * L_B*

    Usage:
        loss_fn = SizeAdaptiveBoundaryLoss(cfg.loss)
        loss, parts = loss_fn(logits, mask)   # mask: [B,1,H,W] binary ground truth
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits: torch.Tensor, mask: torch.Tensor,
                dist_map: torch.Tensor = None):
        prob = torch.sigmoid(logits)
        A_GT = lesion_area(mask)  # [B]

        if dist_map is None:
            dist_map = batch_boundary_distance_maps(mask)

        omega_vec = size_gate_omega(A_GT, self.cfg.T, self.cfg.tau, self.cfg.gamma)
        alpha_vec = size_adaptive_alpha(A_GT, self.cfg.T, self.cfg.tau,
                                         self.cfg.alpha0, self.cfg.alpha_max)

        l_dice = dice_loss(prob, mask, self.cfg.eps)
        l_st = size_adaptive_tversky_loss(prob, mask, alpha_vec, self.cfg.beta0, self.cfg.eps)
        l_b_star = size_weighted_boundary_loss(prob, dist_map, omega_vec, self.cfg.eps)

        total = l_dice + self.cfg.lambda_T * l_st + self.cfg.lambda_B * l_b_star

        parts = {
            "L_Dice": l_dice.detach(),
            "L_ST": l_st.detach(),
            "L_B_star": l_b_star.detach(),
            "L_total": total.detach(),
        }
        return total, parts
