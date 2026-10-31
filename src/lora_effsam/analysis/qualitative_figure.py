# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Representative qualitative figure for the paper (inference only).

Cases are chosen by a fixed rule from the run's per-image report, not by eye:
  * per tumor class: the test image whose Dice is closest to that class's median Dice
  * smallest lesions: the image closest to the median Dice of the smallest area quartile
  * failure: the lowest-Dice test image
Columns: full slice (crop box), crop with ground-truth and predicted contours,
MC-Dropout uncertainty U on the same crop.

    lora-effsam qualitative-figure       # final model (seed 42) -> results/figures/fig_qualitative.*
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from lora_effsam.config import TUMOR_CLASSES
from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.uncertainty.mc_dropout import mc_dropout_predict
from lora_effsam.utils import dice_score
from lora_effsam.paths import data_dir, results_dir, runs_dir

GT_COLOR, PRED_COLOR, INK = "#1baf7a", "#eb6834", "#1f1f1e"


def pick_cases(per_image):
    d = np.array([p["dice"] for p in per_image])
    lab = np.array([p["label"] for p in per_image])
    area = np.array([p["area"] for p in per_image])
    closest = lambda m: int(np.flatnonzero(m)[np.argmin(np.abs(d[m] - np.median(d[m])))])
    cases = [(f"{TUMOR_CLASSES[c]}, median case", closest(lab == c)) for c in range(len(TUMOR_CLASSES))]
    cases.append(("Smallest-size quartile, median case", closest(area <= np.percentile(area, 25))))
    cases.append(("Worst test case", int(np.argmin(d))))
    return cases


def crop_box(gt, pred, size, margin=40, min_side=128):
    ys, xs = np.nonzero(gt | pred)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 2 * margin
    side = int(min(max(side, min_side), size))
    y0 = int(np.clip(cy - side // 2, 0, size - side))
    x0 = int(np.clip(cx - side // 2, 0, size - side))
    return y0, x0, side


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(runs_dir() / "img512_intaug_vits_hybrid_vsds"))
    ap.add_argument("--data-root", default=str(data_dir()))
    ap.add_argument("--mc-passes", type=int, default=20)
    ap.add_argument("--out", default=str(results_dir() / "figures" / "fig_qualitative"),
                    help="output path without extension")
    a = ap.parse_args(argv)

    torch.manual_seed(0)
    model, ck = build_model_from_checkpoint(os.path.join(a.run, "best.pt"), "cuda")
    s = ck["split"]
    ds = BrainTumorSegDataset(a.data_root, image_size=model.image_size, to_rgb=True)
    _, _, test = ds.split(s["val_split"], s["test_split"], seed=s["seed"])
    index = {test._sample_id(x): i for i, x in enumerate(test.samples)}

    per = json.load(open(os.path.join(a.run, "uncertainty_report.json")))["per_image"]
    cases = pick_cases(per)

    fig, axes = plt.subplots(len(cases), 3, figsize=(7.2, 2.45 * len(cases)), dpi=150)
    for row, (title, k) in enumerate(cases):
        rec = per[k]
        img, mask, _ = test[index[rec["id"]]]
        res = mc_dropout_predict(model, img.unsqueeze(0).cuda(), K=a.mc_passes)
        pred = res.pred_mask[0, 0].cpu().numpy() > 0.5
        gt = mask[0].numpy() > 0.5
        u = res.uncertainty[0, 0].cpu().numpy()
        dice = dice_score(res.pred_mask.cpu(), mask.unsqueeze(0)).item()
        gray = img[0].numpy()
        y0, x0, side = crop_box(gt, pred, gray.shape[0])
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))

        ax = axes[row]
        ax[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        ax[0].add_patch(Rectangle((x0, y0), side, side, fill=False, ec="white", lw=1))
        ax[0].set_title(f"{title}\nid {rec['id']}", fontsize=8, color=INK, loc="left")
        ax[1].imshow(gray[sl], cmap="gray", vmin=0, vmax=1)
        ax[1].contour(gt[sl], levels=[0.5], colors=GT_COLOR, linewidths=1.6)
        if pred[sl].any():
            ax[1].contour(pred[sl], levels=[0.5], colors=PRED_COLOR, linewidths=1.6, linestyles="--")
        ax[1].set_title(f"Dice {dice:.3f}", fontsize=8, color=INK, loc="left")
        im = ax[2].imshow(u[sl], cmap="magma", vmin=0, vmax=max(float(u[sl].max()), 1e-6))
        ax[2].set_title("MC-Dropout uncertainty U", fontsize=8, color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        for x in ax:
            x.set_xticks([]); x.set_yticks([])

    fig.legend(handles=[Line2D([], [], color=GT_COLOR, lw=1.6, label="Ground truth"),
                        Line2D([], [], color=PRED_COLOR, lw=1.6, ls="--", label="Prediction")],
               loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    print("cases:", [(t, per[k]["id"], round(per[k]["dice"], 3)) for t, k in cases])


if __name__ == "__main__":
    main()
