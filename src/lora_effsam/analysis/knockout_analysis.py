# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Component reliance of the trained final model by test-time knock-out (inference only).

Each component is disabled in the TRAINED model and the deterministic test Dice is
re-measured; the drop shows how much the trained network relies on it. This is not a
training ablation (the model was not retrained without the component), so it can
overstate the drop for components the network was trained to depend on.

    lora-effsam knockout                 # -> results/statistics/knockout.*, results/figures/fig_knockout.*

Knock-outs:
  cbam_off        feature-attention residual removed (gamma = 0)
  refine_off      refine head skipped (decoder logits used directly)
  lora_off        LoRA adapters disabled (pretrained EfficientSAM weights only)
  vit_skips_off   ViT skip features set to zero
  cnn_skips_off   CNN-stem skip features set to zero
Writes statistics/knockout.{md,json} and figures/fig_knockout.{png,pdf}.
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lora_effsam.config import TUMOR_CLASSES
from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.lora import LoRALinear
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.utils import dice_score
from lora_effsam.paths import data_dir, results_dir, runs_dir

RUNS = ["img512_intaug_vits_hybrid_vsds", "img512_intaug_vits_hybrid_vsds_s1",
        "img512_intaug_vits_hybrid_vsds_s2"]
KNOCKOUTS = ["full", "cbam_off", "refine_off", "lora_off", "vit_skips_off", "cnn_skips_off"]
LABELS = {"cbam_off": "CBAM attention off", "refine_off": "Refine head off", "lora_off": "LoRA off",
          "vit_skips_off": "ViT skips zeroed", "cnn_skips_off": "CNN skips zeroed"}
INK, MUTED, GRID, COLOR = "#1f1f1e", "#6b6a64", "#e4e3dd", "#eda100"


@contextlib.contextmanager
def knockout(model, kind):
    undo = []
    if kind == "cbam_off":
        g = model.feature_attn.gamma.data.clone()
        model.feature_attn.gamma.data.zero_()
        undo.append(lambda: model.feature_attn.gamma.data.copy_(g))
    elif kind == "refine_off":
        model.use_refine_head = False
        undo.append(lambda: setattr(model, "use_refine_head", True))
    elif kind == "lora_off":
        for m in model.modules():
            if isinstance(m, LoRALinear):
                s = m.scaling
                m.scaling = 0.0
                undo.append(lambda m=m, s=s: setattr(m, "scaling", s))
    elif kind == "vit_skips_off":
        for p in model.decoder.tap_proj:
            h = p.register_forward_hook(lambda mod, inp, out: torch.zeros_like(out))
            undo.append(h.remove)
    elif kind == "cnn_skips_off":
        h = model.decoder.stem.register_forward_hook(lambda mod, inp, out: [torch.zeros_like(f) for f in out])
        undo.append(h.remove)
    try:
        yield
    finally:
        for u in undo:
            u()


@torch.no_grad()
def per_image_dice(model, loader):
    model.eval()
    out = []
    for x, y, _ in loader:
        x, y = x.cuda(), y.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x)
        out += dice_score((torch.sigmoid(lo.float()) >= 0.5).float(), y).tolist()
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_dir()))
    a = ap.parse_args(argv)

    res = {}
    labels = areas = None
    for run in RUNS:
        model, ck = build_model_from_checkpoint(os.path.join(str(runs_dir()), run, "best.pt"), "cuda")
        s = ck["split"]
        ds = BrainTumorSegDataset(str(data_dir()), image_size=model.image_size, to_rgb=True)
        _, _, test = ds.split(s["val_split"], s["test_split"], seed=s["seed"])
        if labels is None:
            labels = np.array([x[2] for x in test.samples])
            areas = np.array([float(test[i][1].sum()) for i in range(len(test))])
        loader = DataLoader(test, batch_size=4, num_workers=2)
        res[run] = {}
        for k in KNOCKOUTS:
            with knockout(model, k):
                res[run][k] = per_image_dice(model, loader)
            print(run, k, round(float(res[run][k].mean()), 4), flush=True)

    q1 = areas <= np.percentile(areas, 25)
    groups = {"All": np.ones(len(labels), bool), "Glioma": labels == TUMOR_CLASSES.index("Glioma"),
              "Size Q1": q1}
    summary = {}
    for k in KNOCKOUTS:
        summary[k] = {}
        for g, m in groups.items():
            means = [float(res[r][k][m].mean()) for r in RUNS]
            drops = [float(res[r][k][m].mean() - res[r]["full"][m].mean()) for r in RUNS]
            summary[k][g] = dict(dice_mean=float(np.mean(means)), dice_sd=float(np.std(means, ddof=1)),
                                 delta_mean=float(np.mean(drops)), delta_sd=float(np.std(drops, ddof=1)))
    json.dump(dict(runs=RUNS, summary=summary), open(os.path.join(a.out, "statistics", "knockout.json"), "w"), indent=2)

    md = ["# Test-time knock-out of components (final model, 3 seeds)", "",
          "Deterministic test Dice (dropout off) of the TRAINED model with one component disabled; "
          "mean ± SD over seeds 42, 1, 2. Δ = change vs the full model. Not a training ablation.", "",
          "| Knock-out | All: Dice | Δ | Glioma: Δ | Size Q1: Δ |", "|---|---|---|---|---|"]
    for k in KNOCKOUTS:
        sm = summary[k]
        md.append(f"| {LABELS.get(k, 'Full model')} | {sm['All']['dice_mean']:.4f} ± {sm['All']['dice_sd']:.4f} | "
                  f"{sm['All']['delta_mean']:+.4f} | {sm['Glioma']['delta_mean']:+.4f} | {sm['Size Q1']['delta_mean']:+.4f} |")
    open(os.path.join(a.out, "statistics", "knockout.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    ko = KNOCKOUTS[1:]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    d = [summary[k]["All"]["delta_mean"] for k in ko]
    e = [summary[k]["All"]["delta_sd"] for k in ko]
    y = np.arange(len(ko))
    ax.barh(y, d, 0.55, color=COLOR, xerr=e, error_kw=dict(ecolor=INK, elinewidth=1, capsize=3))
    for yi, di, ei in zip(y, d, e):
        ax.text(min(di - ei, 0) - 0.006, yi, f"{di:+.4f}", va="center", ha="right",
                fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in ko], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Change in test Dice when the component is disabled at test time", fontsize=8.5, color=INK)
    ax.set_title(f"Component reliance of the trained final model (full model: "
                 f"{summary['full']['All']['dice_mean']:.3f}; mean ± SD over 3 seeds)",
                 fontsize=9, color=INK, loc="left")
    lo = min(d) - max(e) - 0.07
    ax.set_xlim(lo, 0.03)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, "figures", f"fig_knockout.{ext}"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
