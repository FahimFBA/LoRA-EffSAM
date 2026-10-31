# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 3 (MC-Dropout uncertainty) results for the paper, from uncertainty_report.json
(plain MC-Dropout, no TTA, so U is dropout variance only).

    lora-effsam uncertainty-stats        # -> results/statistics/uncertainty_stats.{md,json}

Per run: region means of U (boundary band / interior / background), boundary vs
interior paired test, small vs large lesions (smallest vs largest area quartile),
Spearman correlation of U_boundary with Dice, and how well U_boundary flags
segmentation failures (AUROC for Dice < 0.5).
"""
import argparse
import json
import os

import numpy as np
from scipy import stats
from lora_effsam.paths import results_dir, runs_dir


def auroc(score, positive):
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def analyse(run):
    rep = json.load(open(os.path.join(str(runs_dir()), run, "uncertainty_report.json")))
    P = rep["per_image"]
    dice = np.array([p["dice"] for p in P])
    ub = np.array([p["U_boundary"] for p in P])
    ui = np.array([p["U_interior"] for p in P])
    ug = np.array([p["U_background"] for p in P])
    area = np.array([p["area"] for p in P])
    ok = ~np.isnan(ub) & ~np.isnan(ui)
    q1, q4 = area <= np.percentile(area, 25), area > np.percentile(area, 75)
    ub_q1, ub_q4 = ub[q1 & ~np.isnan(ub)], ub[q4 & ~np.isnan(ub)]
    rho = stats.spearmanr(ub[ok], dice[ok])
    fail = dice < 0.5
    return dict(
        run=run, n=len(P), mean_dice=float(dice.mean()), median_dice=float(np.median(dice)),
        U_boundary=float(np.nanmean(ub)), U_interior=float(np.nanmean(ui)), U_background=float(np.nanmean(ug)),
        boundary_gt_interior=f"{int((ub[ok] > ui[ok]).sum())}/{int(ok.sum())}",
        boundary_vs_interior_p=float(stats.wilcoxon(ub[ok], ui[ok]).pvalue),
        U_boundary_smallQ1=float(ub_q1.mean()), U_boundary_largeQ4=float(ub_q4.mean()),
        small_vs_large_p=float(stats.mannwhitneyu(ub_q1, ub_q4).pvalue),
        spearman_Uboundary_dice=float(rho.statistic), spearman_p=float(rho.pvalue),
        n_fail_lt0_5=int(fail.sum()),
        auroc_Uboundary_flags_failure=auroc(np.nan_to_num(ub), fail),
        size_stratified_T=rep["size_stratified"],
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[
        "img512_intaug_vits_hybrid_vsds", "img512_intaug_vits_hybrid_vsds_s1",
        "img512_intaug_vits_hybrid_vsds_s2", "img512_intaug", "enc512"])
    ap.add_argument("--out", default=str(results_dir() / "statistics" / "uncertainty_stats.md"))
    a = ap.parse_args(argv)
    res = [analyse(r) for r in a.runs]
    json.dump(res, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=2)

    g = lambda v: f"{v:.2e}" if isinstance(v, float) and abs(v) < 1e-3 else (f"{v:.4f}" if isinstance(v, float) else str(v))
    keys = [k for k in res[0] if k != "size_stratified_T"]
    md = ["# MC-Dropout uncertainty analysis (K=20, p=0.2, no TTA)", "",
          "U = per-pixel variance of the K sigmoid outputs. Boundary band = 3 px either side of "
          "the ground-truth contour. Small/large = smallest/largest quartile of lesion area. "
          "AUROC: U_boundary as a score for detecting failed segmentations (Dice < 0.5).", "",
          "| metric | " + " | ".join(r["run"] for r in res) + " |", "|---|" + "---|" * len(res)]
    md += [f"| {k} | " + " | ".join(g(r[k]) for r in res) + " |" for k in keys if k != "run"]
    md += ["", "Size stratification with the training threshold T (as in the method's Eq. 10-11):", ""]
    md += [f"- {r['run']}: {r['size_stratified_T']}" for r in res]
    open(a.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
