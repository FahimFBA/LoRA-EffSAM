# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Failure analysis for the paper (inference only).

Reports the class mix of failed test slices (Dice < 0.3) for each seed of the final
model, and how early failures are decided: the Dice of the 1/8-resolution
deep-supervision prediction for failed vs other slices.

    lora-effsam failure-check            # -> results/statistics/failure_analysis.txt
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch

from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.analysis.explainability_figures import internals, up
from lora_effsam.paths import data_dir, results_dir, runs_dir
from lora_effsam.utils import dice_score

FINAL = "img512_intaug_vits_hybrid_vsds"
SEEDS = [FINAL, FINAL + "_s1", FINAL + "_s2"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(results_dir() / "statistics" / "failure_analysis.txt"))
    a = ap.parse_args(argv)
    runs = str(runs_dir())
    lines = []
    for run in SEEDS:
        per = json.load(open(os.path.join(runs, run, "uncertainty_report.json")))["per_image"]
        fails = [p for p in per if p["dice"] < 0.3]
        lines.append(f"{run} fails<0.3: {len(fails)} {dict(Counter(p['class'] for p in fails))}")

    model, ck = build_model_from_checkpoint(os.path.join(runs, FINAL, "best.pt"), "cuda")
    s = ck["split"]
    ds = BrainTumorSegDataset(str(data_dir()), image_size=model.image_size, to_rgb=True)
    _, _, test = ds.split(s["val_split"], s["test_split"], seed=s["seed"])
    idx = {test._sample_id(x): i for i, x in enumerate(test.samples)}
    per = json.load(open(os.path.join(runs, FINAL, "uncertainty_report.json")))["per_image"]
    rows = []
    for p in per:
        img, m, _ = test[idx[p["id"]]]
        it = internals(model, img.unsqueeze(0).cuda())
        pred8 = torch.tensor(up(it["levels"][0], model.image_size))[None, None] >= 0.5
        rows.append((p["dice"], dice_score(pred8.float(), m.unsqueeze(0)).item()))
    r = np.array(rows)
    f = r[:, 0] < 0.3
    lines += [
        f"seed42 failures n={f.sum()}: mean Dice at 1/8 level {r[f, 1].mean():.3f} "
        f"(final {r[f, 0].mean():.3f}); share with 1/8 Dice < 0.3: {(r[f, 1] < 0.3).mean():.2f}",
        f"non-failures: mean 1/8 Dice {r[~f, 1].mean():.3f}",
        f"corr final vs 1/8 Dice: {np.corrcoef(r[:, 0], r[:, 1])[0, 1]:.3f}",
    ]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
