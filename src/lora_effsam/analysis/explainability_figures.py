# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Explainability figures and statistics for the paper (inference only, no training).

    lora-effsam explainability           # -> results/figures + results/statistics

Writes to <out>/figures:
  fig_architecture.{png,pdf}         block diagram of the final model
  fig_explainability_maps.{png,pdf}  per case: input + contours, CBAM spatial attention,
                                     decoder outputs at 1/8, 1/4, 1/2 (deep-supervision heads)
                                     and full resolution, refine-head boundary gate,
                                     MC-Dropout uncertainty
  fig_uncertainty_analysis.{png,pdf} region uncertainty, U_boundary vs Dice, U by size quartile
  fig_dice_distribution.{png,pdf}    per-image Dice distribution by class, baseline vs final
and <out>/statistics/explainability_stats.{md,json} (learned attention/gate parameters and
attention statistics over the test set).

Cases for the maps use the same fixed rule as qualitative_figure.py (median case per
class, median of the smallest-size quartile, worst case) on the seed-42 final model.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from lora_effsam.config import TUMOR_CLASSES
from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.uncertainty.mc_dropout import mc_dropout_predict
from lora_effsam.analysis.qualitative_figure import pick_cases, crop_box
from lora_effsam.paths import data_dir, results_dir, runs_dir

FINAL = "img512_intaug_vits_hybrid_vsds"
FINAL_SEEDS = [FINAL, FINAL + "_s1", FINAL + "_s2"]
BASELINE = "enc512"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
C_BASE, C_FINAL, C_GT, C_PRED, C_FAIL = "#2a78d6", "#eda100", "#1baf7a", "#eb6834", "#e87ba4"


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- internals
@torch.no_grad()
def internals(model, x):
    """Deterministic forward (dropout off) that also returns the intermediate maps."""
    model.eval()
    neck, taps = model._encode(x)
    fa = model.feature_attn
    ca = fa.ca(neck)                       # [B, C, 1, 1]
    att = neck * ca
    sa = fa.sa(att)                        # [B, 1, h, w] spatial attention
    neck_att = neck + fa.gamma * (att * sa)

    dec = model.decoder
    H = dec.image_size
    f1, f2, f4, f8 = dec.stem(x)
    skips = [f8, f4, f2, f1]
    h = neck_att
    if h.shape[-1] != H // 16:
        h = F.interpolate(h, size=(H // 16, H // 16), mode="bilinear", align_corners=False)
    levels = []
    for k, block in enumerate(dec.blocks):
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        parts = [h, skips[k]]
        if dec.use_vit_skips and k < 3:
            t = dec.tap_proj[k](taps[k])
            parts.append(F.interpolate(t, size=h.shape[-2:], mode="bilinear", align_corners=False))
        h = block(torch.cat(parts, dim=1))
        if k < 3:
            levels.append(torch.sigmoid(dec.aux_heads[k](h)))   # 1/8, 1/4, 1/2 predictions
    coarse_logits = dec.head(h)                                   # decoder output, full res
    rh = model.refine_head
    gate = rh.boundary_attention(torch.sigmoid(coarse_logits))
    final_logits = rh(x, coarse_logits)
    return dict(sa=sa, ca=ca, gamma=float(fa.gamma), levels=levels,
                coarse=torch.sigmoid(coarse_logits), gate=gate, final=torch.sigmoid(final_logits))


def up(t, size):
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


# --------------------------------------------------------------------------- figures
def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc, ec=MUTED, fs=7.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1, dashed=False):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9,
                                    linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))

    frozen, train, infer = "#e9e8e2", "#fde9bf", "#d9e8fa"
    fs = 7
    box(1, 18, 8, 8, "MRI slice\n512×512", "#ffffff", fs=fs)
    box(12, 27, 22, 11, "EfficientSAM ViT-S encoder\n12 blocks, frozen (22.4M)\n+ LoRA r=4 on Q,K,V (74K)", frozen, fs=fs)
    box(12, 4, 22, 9, "CNN stem\nfeatures at 1/1, 1/2, 1/4, 1/8", train, fs=fs)
    box(38, 30, 11, 6, "CBAM feature\nattention", train, fs=fs)
    box(53, 16, 19, 16, "Hybrid U-Net decoder\n4 levels: 1/16 → 1/1\nconcat CNN + ViT skips\nper level", train, fs=fs)
    box(53, 2, 19, 8, "Deep-supervision heads\n1/8, 1/4, 1/2 (train only)", train, fs=fs)
    box(76, 25, 12, 7, "MC-Dropout\n(p = 0.2) + 1×1", infer, fs=fs)
    box(76, 13, 12, 7, "Refine head\n(boundary gate)", train, fs=fs)
    box(91, 18, 8, 8, "p̄ and U\n(K = 20)", infer, fs=fs)

    arrow(9, 22, 12, 32); arrow(9, 22, 12, 8.5)
    arrow(34, 33, 38, 33)
    arrow(49, 33, 53, 28)
    arrow(34, 29, 53, 22)
    ax.text(37, 24.2, "ViT skips (blocks 3, 6, 9)", fontsize=6.5, color=MUTED, rotation=-17)
    arrow(34, 8.5, 53, 19)
    arrow(62.5, 16, 62.5, 10, dashed=True)
    arrow(72, 27, 76, 28.5); arrow(82, 25, 82, 20); arrow(88, 16.5, 91, 21)
    for fc, lab, xx in ((frozen, "frozen", 2), (train, "trainable (1.31M total)", 12), (infer, "uncertainty / inference", 34)):
        ax.add_patch(FancyBboxPatch((xx, 40), 2.4, 2.4, boxstyle="round,pad=0.1", fc=fc, ec=MUTED, lw=0.8))
        ax.text(xx + 3.4, 41.2, lab, fontsize=7.5, va="center", color=INK)
    ax.text(1, 0.5, "Loss (train): L_Dice + λ_T·L_ST(size-adaptive Tversky) + λ_B·L_B*(size-weighted boundary), "
            "+ deep-supervision terms 0.5 / 0.25 / 0.125", fontsize=7, color=MUTED)
    save(fig, out, "fig_architecture")


def fig_maps(out, model, test, index, per):
    cases = pick_cases(per)
    cols = ["Input + contours", "CBAM spatial\nattention", "Decoder 1/8", "Decoder 1/4",
            "Decoder 1/2", "Decoder 1/1\n(before refine)", "Refine-head\nboundary gate", "Final p̄ (MC)", "Uncertainty U"]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(15, 1.85 * len(cases) + 0.6), dpi=150)
    stats_rows = []
    for r, (title, k) in enumerate(cases):
        rec = per[k]
        img, mask, _ = test[index[rec["id"]]]
        x = img.unsqueeze(0).cuda()
        it = internals(model, x)
        torch.manual_seed(0)
        mc = mc_dropout_predict(model, x, K=20)
        S = img.shape[-1]
        gt = mask[0].numpy() > 0.5
        pred = mc.pred_mask[0, 0].cpu().numpy() > 0.5
        y0, x0, side = crop_box(gt, pred, S, margin=60, min_side=160)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        gray = img[0].numpy()
        sa_full = up(it["sa"], S)
        maps = [None, sa_full] + [up(l, S) for l in it["levels"]] + [
            it["coarse"][0, 0].cpu().numpy(), it["gate"][0, 0].cpu().numpy(),
            mc.mean_prob[0, 0].cpu().numpy(), mc.uncertainty[0, 0].cpu().numpy()]
        # attention inside vs outside the tumor (for the stats file)
        stats_rows.append(dict(case=title, id=rec["id"], dice=rec["dice"],
                               sa_in=float(sa_full[gt].mean()), sa_out=float(sa_full[~gt].mean())))
        for c, m in enumerate(maps):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(gray[sl], cmap="gray", vmin=0, vmax=1)
                ax.contour(gt[sl], levels=[0.5], colors=C_GT, linewidths=1.2)
                if pred[sl].any():
                    ax.contour(pred[sl], levels=[0.5], colors=C_PRED, linewidths=1.2, linestyles="--")
                ax.set_ylabel(f"{title.replace(', median case', '').replace('Smallest-size quartile', 'Small lesion')}"
                              f"\nDice {rec['dice']:.2f}", fontsize=7.5, color=INK)
            elif c == 1:
                ax.imshow(gray[sl], cmap="gray", vmin=0, vmax=1)
                ax.imshow(m[sl], cmap="viridis", alpha=0.6)
                ax.contour(gt[sl], levels=[0.5], colors="white", linewidths=0.8)
            elif c == len(cols) - 1:
                ax.imshow(m[sl], cmap="magma", vmin=0, vmax=max(float(m[sl].max()), 1e-6))
                ax.contour(gt[sl], levels=[0.5], colors=C_GT, linewidths=0.8)
            else:
                ax.imshow(m[sl], cmap="viridis", vmin=0, vmax=1)
                ax.contour(gt[sl], levels=[0.5], colors="white", linewidths=0.8)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5, color=INK)
    fig.suptitle("How the final model builds its prediction (crops; white/green contour = ground truth, "
                 "dashed orange = prediction; maps 2–8 on a 0–1 scale, U per-image scale)",
                 fontsize=8.5, color=INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    save(fig, out, "fig_explainability_maps")
    return stats_rows


def load_report(run, name="uncertainty_report.json"):
    return json.load(open(os.path.join(str(runs_dir()), run, name)))["per_image"]


def fig_uncertainty(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=150)
    # (a) region means per seed, log scale
    ax = axes[0]; style(ax)
    regions = [("U_boundary", "Boundary band"), ("U_interior", "Interior"), ("U_background", "Background")]
    vals = np.array([[np.nanmean([p[k] for p in load_report(r)]) for k, _ in regions] for r in FINAL_SEEDS])
    xs = np.arange(3)
    ax.bar(xs, vals.mean(0), 0.55, color=C_FINAL, yerr=vals.std(0, ddof=1),
           error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels([l for _, l in regions], fontsize=8)
    ax.set_ylabel("Mean MC-Dropout variance U", fontsize=8.5, color=INK)
    ax.set_title("(a) Uncertainty by region (3 seeds)", fontsize=9, color=INK, loc="left")

    # (b) U_boundary vs Dice, seed 42, failures marked
    ax = axes[1]; style(ax, grid_axis="both")
    P = load_report(FINAL)
    d = np.array([p["dice"] for p in P]); ub = np.array([p["U_boundary"] for p in P])
    fail = d < 0.5
    ax.scatter(d[~fail], ub[~fail], s=9, color=C_FINAL, alpha=0.6, lw=0, label=f"Dice ≥ 0.5 (n={(~fail).sum()})")
    ax.scatter(d[fail], ub[fail], s=14, color=C_FAIL, alpha=0.9, lw=0, label=f"Failure, Dice < 0.5 (n={fail.sum()})")
    ax.set_xlabel("Per-image Dice", fontsize=8.5, color=INK)
    ax.set_ylabel("Mean U in GT boundary band", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("(b) Boundary uncertainty vs Dice (seed 42)", fontsize=9, color=INK, loc="left")

    # (c) U_boundary by size quartile, baseline vs final (seed 42)
    ax = axes[2]; style(ax)
    for j, (run, col, lab) in enumerate(((BASELINE, C_BASE, "Baseline A"), (FINAL, C_FINAL, "Final D"))):
        P = load_report(run)
        a = np.array([p["area"] for p in P]); u = np.array([p["U_boundary"] for p in P])
        q = np.digitize(a, np.quantile(a, [0.25, 0.5, 0.75]))
        m = [np.nanmean(u[q == i]) for i in range(4)]
        ax.bar(np.arange(4) + (j - 0.5) * 0.36, m, 0.34, color=col, label=lab)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nsmallest", "Q2", "Q3", "Q4\nlargest"], fontsize=8)
    ax.set_ylabel("Mean U in GT boundary band", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("(c) Boundary uncertainty by lesion size (seed 42)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig_uncertainty_analysis")


def fig_dice_dist(out):
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=150)
    style(ax)
    base = load_report(BASELINE, "uncertainty_report_ids.json")
    final = load_report(FINAL)
    groups = [("All", None)] + [(c.replace("Pituitary tumor", "Pituitary"), i) for i, c in enumerate(TUMOR_CLASSES)]
    data, pos, cols = [], [], []
    for g, (name, lab) in enumerate(groups):
        for j, (rep, col) in enumerate(((base, C_BASE), (final, C_FINAL))):
            data.append([p["dice"] for p in rep if lab is None or p["label"] == lab])
            pos.append(g * 3 + j); cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.75, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=1.2), flierprops=dict(marker="o", ms=2.5, alpha=0.4,
                                                                         mec=MUTED, mfc="none"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor(MUTED); patch.set_alpha(0.85)
    ax.set_xticks([g * 3 + 0.5 for g in range(len(groups))])
    ax.set_xticklabels([n for n, _ in groups], fontsize=8.5)
    ax.set_ylabel("Per-image test Dice", fontsize=8.5, color=INK)
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["Baseline A (seed 42)", "Final D (seed 42)"],
              frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Distribution of per-image test Dice (n = 477): the mean is pulled down by a tail of failures",
                 fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, out, "fig_dice_distribution")


@torch.no_grad()
def attention_stats(model, test, n_max=None):
    """CBAM spatial attention inside vs outside the tumor, over the whole test set."""
    ratios, inside, outside = [], [], []
    for i in range(len(test) if n_max is None else n_max):
        img, mask, _ = test[i]
        it = internals(model, img.unsqueeze(0).cuda())
        sa = up(it["sa"], img.shape[-1]); gt = mask[0].numpy() > 0.5
        inside.append(sa[gt].mean()); outside.append(sa[~gt].mean())
        ratios.append(sa[gt].mean() / max(sa[~gt].mean(), 1e-8))
    return dict(n=len(ratios), sa_inside_mean=float(np.mean(inside)), sa_outside_mean=float(np.mean(outside)),
                ratio_median=float(np.median(ratios)), frac_images_inside_gt_outside=float(np.mean(np.array(inside) > np.array(outside))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_dir()))
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "statistics"), exist_ok=True)

    fig_architecture(a.out)
    fig_uncertainty(a.out)
    fig_dice_dist(a.out)

    model, ck = build_model_from_checkpoint(os.path.join(str(runs_dir()), FINAL, "best.pt"), "cuda")
    s = ck["split"]
    ds = BrainTumorSegDataset(str(data_dir()), image_size=model.image_size, to_rgb=True)
    _, _, test = ds.split(s["val_split"], s["test_split"], seed=s["seed"])
    index = {test._sample_id(x): i for i, x in enumerate(test.samples)}
    per = load_report(FINAL)
    case_stats = fig_maps(a.out, model, test, index, per)
    att = attention_stats(model, test)

    learned = {}
    for run in FINAL_SEEDS:
        m, _ = build_model_from_checkpoint(os.path.join(str(runs_dir()), run, "best.pt"), "cpu")
        learned[run] = dict(feature_attention_gamma=float(m.feature_attn.gamma),
                            refine_gate_scale=float(m.refine_head.gate_scale),
                            refine_gate_bias=float(m.refine_head.gate_bias),
                            refine_out_conv_weight_norm=float(m.refine_head.out_conv.weight.norm()))
    res = dict(learned_parameters=learned, attention_over_test_set_seed42=att, cases=case_stats)
    json.dump(res, open(os.path.join(a.out, "statistics", "explainability_stats.json"), "w"), indent=2)

    md = ["# Explainability statistics (final model)", "",
          "## Learned parameters of the attention components (init: gamma 0, gate scale 8, gate bias -1, out-conv weights 0)", "",
          "| run | CBAM gamma | refine gate scale | refine gate bias | refine out-conv ‖w‖ |", "|---|---|---|---|---|"]
    md += [f"| {r} | {v['feature_attention_gamma']:.4f} | {v['refine_gate_scale']:.3f} | "
           f"{v['refine_gate_bias']:.3f} | {v['refine_out_conv_weight_norm']:.4f} |" for r, v in learned.items()]
    md += ["", "## CBAM spatial attention, test set (seed 42)", "",
           f"- images: {att['n']}",
           f"- mean attention inside the ground-truth tumor: {att['sa_inside_mean']:.4f}; outside: {att['sa_outside_mean']:.4f}",
           f"- median per-image ratio inside/outside: {att['ratio_median']:.3f}",
           f"- fraction of images with higher attention inside the tumor: {att['frac_images_inside_gt_outside']:.3f}",
           "", "## Cases in fig_explainability_maps", "",
           "| case | id | Dice | attention inside | attention outside |", "|---|---|---|---|---|"]
    md += [f"| {c['case']} | {c['id']} | {c['dice']:.3f} | {c['sa_in']:.4f} | {c['sa_out']:.4f} |" for c in case_stats]
    open(os.path.join(a.out, "statistics", "explainability_stats.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
