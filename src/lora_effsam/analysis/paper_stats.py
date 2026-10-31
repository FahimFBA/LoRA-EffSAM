# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Significance tests and figures for the paper, from finished runs only (no training).

    lora-effsam paper-stats   # -> results/statistics/stats.{md,json}, results/figures/fig_{learning_curves,subgroups}.*

Configs are groups of runs named <config> (seed 42) and <config>_s<seed>, all on the
same test split (--split-seed 42). Per-image tests pair the same test image across
configs, using each image's Dice averaged over a config's seeds.
"""
import csv
import json
import os
import re
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lora_effsam.config import TUMOR_CLASSES
from lora_effsam.paths import results_dir, runs_dir

RUNS, OUT, FIG_OUT = str(runs_dir()), str(results_dir() / "statistics"), str(results_dir() / "figures")
CONFIGS = [  # (key, run-name prefix, report file or None, label, color) -- fixed
    # order/colors, validated palette. A report file means an inference-only variant
    # of the prefix's checkpoints (no separate learning curve).
    ("enc512", "enc512", None, "Baseline (256 px)", "#2a78d6"),
    ("img512", "img512", None, "512 px", "#eb6834"),
    ("img512_intaug", "img512_intaug", None, "512 px + intensity aug", "#1baf7a"),
    ("img512_intaug_gateQ1", "img512_intaug_gateQ1", None, "+ size gate T=1500", "#e87ba4"),
    ("img512_intaug_tta", "img512_intaug", "uncertainty_report_tta.json", "+ flip TTA", "#e87ba4"),
    ("img512_intaug_hybrid", "img512_intaug_hybrid", None, "+ hybrid decoder", "#e87ba4"),
    ("final", "img512_intaug_vits_hybrid_vsds", None, "Final: ViT-S + hybrid + ViT skips + DS", "#eda100"),
    ("final_tta", "img512_intaug_vits_hybrid_vsds", "uncertainty_report_tta.json", "Final + flip TTA", "#eda100"),
]
# Configs drawn in the figures (one color each, validated palette order).
FIG_CONFIGS = ["enc512", "img512", "img512_intaug", "final"]
# Each comparison pairs per-image Dice averaged over the seeds BOTH configs have.
COMPARISONS = [("img512", "enc512"), ("img512_intaug", "img512"),
               ("img512_intaug_gateQ1", "img512_intaug"), ("img512_intaug_tta", "img512_intaug"),
               ("img512_intaug_hybrid", "img512_intaug"), ("final", "img512_intaug"),
               ("final_tta", "img512_intaug_tta"), ("final_tta", "enc512")]
N_BOOT, RNG = 10000, np.random.default_rng(0)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def runs_of(config):
    names = sorted(d for d in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, d)))
    return [n for n in names if n == config or re.fullmatch(re.escape(config) + r"_s\d+", n)]


def load_per_image(run, report=None):
    """Per-image records sorted by image id (needs uncertainty_report_ids.json, see
    scripts/reeval_with_ids.sh), so index i is the same test image in every run."""
    path = os.path.join(RUNS, run, report or "uncertainty_report_ids.json")
    if not report and not os.path.isfile(path):  # evaluated after ids were added to the report
        path = os.path.join(RUNS, run, "uncertainty_report.json")
    recs = json.load(open(path))["per_image"]
    assert "id" in recs[0], f"{path} has no image ids; run scripts/reeval_with_ids.sh"
    return sorted(recs, key=lambda p: p["id"])


def load_metrics(run):
    with open(os.path.join(RUNS, run, "metrics.csv")) as f:
        return [{k: float(v) for k, v in r.items() if k in ("epoch", "train_dice", "val_dice")}
                for r in csv.DictReader(f)]


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def boot_ci(diff):
    idx = RNG.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main(argv=None):
    global OUT, FIG_OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT, help="statistics output folder")
    ap.add_argument("--fig-out", default=FIG_OUT, help="figure output folder")
    a = ap.parse_args(argv)
    OUT, FIG_OUT = a.out, a.fig_out
    os.makedirs(FIG_OUT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    prefix = {k: p for k, p, _, _, _ in CONFIGS}
    report = {k: rep for k, _, rep, _, _ in CONFIGS}
    cfg_runs = {k: runs_of(p) for k, p, _, _, _ in CONFIGS}

    # ---- per-image Dice, checked to be the same images in the same order ----------
    per = {c: [load_per_image(r, report[c]) for r in rs] for c, rs in cfg_runs.items()}
    ref = per[CONFIGS[0][0]][0]
    ids = [p["id"] for p in ref]
    labels = np.array([p["label"] for p in ref])
    for c, reps in per.items():
        for rep in reps:
            assert [p["id"] for p in rep] == ids, f"{c}: test images differ from baseline"
    # Size quartiles from GT area at 512 px (full resolution), same images everywhere.
    hi_res = next(c for c in cfg_runs if json.load(open(os.path.join(
        RUNS, cfg_runs[c][0], "config.json")))["model_args"]["image_size"] == 512)
    area = np.array([p["area"] for p in per[hi_res][0]], dtype=float)
    dice = {c: np.mean([[p["dice"] for p in rep] for rep in reps], axis=0) for c, reps in per.items()}
    seed_dice = {c: {re.search(r"_s(\d+)$", r).group(1) if "_s" in r[len(prefix[c]):] else "42":
                     np.array([p["dice"] for p in rep]) for r, rep in zip(cfg_runs[c], per[c])}
                 for c in per}
    n = len(labels)
    quart = np.digitize(area, np.quantile(area, [0.25, 0.5, 0.75]))  # 0..3

    groups = {"all": np.ones(n, bool)}
    for i, cls in enumerate(TUMOR_CLASSES):
        groups[cls] = labels == i
    for q in range(4):
        groups[f"size Q{q + 1}"] = quart == q

    # ---- tests -------------------------------------------------------------------
    results = []
    for a, b in COMPARISONS:
        common = sorted(set(seed_dice[a]) & set(seed_dice[b]))
        da = np.mean([seed_dice[a][s] for s in common], axis=0)
        db = np.mean([seed_dice[b][s] for s in common], axis=0)
        d = da - db
        w = stats.wilcoxon(da, db)
        lo, hi = boot_ci(d)
        seed_means = [(seed_dice[a][s].mean(), seed_dice[b][s].mean()) for s in common]
        t_p = (float(stats.ttest_rel([x for x, _ in seed_means], [y for _, y in seed_means]).pvalue)
               if len(common) > 1 else float("nan"))
        per_seed_w = {s: float(stats.wilcoxon(seed_dice[a][s], seed_dice[b][s]).pvalue) for s in common}
        sub = {}
        for g, m in groups.items():
            dg = d[m]
            glo, ghi = boot_ci(dg)
            sub[g] = dict(n=int(m.sum()), mean_a=float(da[m].mean()), mean_b=float(db[m].mean()),
                          diff=float(dg.mean()), ci95=[float(glo), float(ghi)],
                          wilcoxon_p=float(stats.wilcoxon(da[m], db[m]).pvalue))
        results.append(dict(
            a=a, b=b, n_images=n, mean_a=float(da.mean()), mean_b=float(db.mean()),
            diff=float(d.mean()), ci95=[float(lo), float(hi)],
            improved=int((d > 0).sum()), worse=int((d < 0).sum()), tied=int((d == 0).sum()),
            wilcoxon_p=float(w.pvalue), seeds=common,
            seed_paired_t_p=t_p, per_seed_wilcoxon_p=per_seed_w, subgroups=sub))
    for r, p in zip(results, holm([r["wilcoxon_p"] for r in results])):
        r["wilcoxon_p_holm"] = float(p)

    json.dump(dict(runs=cfg_runs, comparisons=results), open(os.path.join(OUT, "stats.json"), "w"), indent=2)

    name = {c: l for c, _, _, l, _ in CONFIGS}
    name.update({
        "img512_intaug_gateQ1": "512 px + intensity aug + size gate",
        "img512_intaug_tta": "512 px + intensity aug + flip TTA",
        "img512_intaug_hybrid": "512 px + intensity aug + hybrid decoder",
        "final": "Final model",
        "final_tta": "Final model + flip TTA",
    })
    fp = lambda p: "n/a" if p != p else ("<0.0001" if p < 1e-4 else f"{p:.4f}")
    md = ["# Statistical comparison of configurations", "",
          f"Test set: {n} images (patient-level split, identical for all runs). Runs per "
          "configuration: " + "; ".join(f"{name[c]}: {', '.join(rs)}" for c, rs in cfg_runs.items()) + ".",
          "",
          "Each comparison pairs the same test images and uses per-image Dice averaged over "
          "the seeds BOTH configurations have (column 'seeds'), so a single-seed model is "
          "compared seed-for-seed. Wilcoxon signed-rank test on paired per-image Dice; Holm "
          f"correction across the {len(results)} comparisons; 95% CI of the mean difference "
          f"by bootstrap over images ({N_BOOT} resamples). Seed-level: paired t-test on "
          "per-seed mean test Dice (needs >= 2 shared seeds).", "",
          "| Comparison | seeds | Mean Dice (A vs B) | Δ mean [95% CI] | images better / worse | Wilcoxon p | Holm p | seed-paired t p |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {name[r['a']]} vs {name[r['b']]} | {','.join(r['seeds'])} | "
                  f"{r['mean_a']:.4f} vs {r['mean_b']:.4f} | "
                  f"{r['diff']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
                  f"{r['improved']} / {r['worse']} | {fp(r['wilcoxon_p'])} | {fp(r['wilcoxon_p_holm'])} | "
                  f"{fp(r['seed_paired_t_p'])} |")
    for r in results:
        md += ["", f"## {name[r['a']]} vs {name[r['b']]}: subgroups (uncorrected, exploratory)", "",
               "| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |", "|---|---|---|---|"]
        for g, s in r["subgroups"].items():
            md.append(f"| {g} | {s['n']} | {s['diff']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
                      f"{fp(s['wilcoxon_p'])} |")
        md.append(f"\nPer-seed Wilcoxon p (same seed, A vs B): "
                  + ", ".join(f"seed {s}: {fp(p)}" for s, p in r["per_seed_wilcoxon_p"].items()))
    open(os.path.join(OUT, "stats.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- figure 1: validation Dice learning curves (mean ± std over seeds) --------
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    style(ax)
    fig_cfgs = [cf for cf in CONFIGS if cf[0] in FIG_CONFIGS]
    for c, _, rep, label, color in fig_cfgs:
        curves = [load_metrics(r) for r in cfg_runs[c]]
        max_ep = max(len(cv) for cv in curves)
        need = 2 if len(curves) > 1 else 1
        ep, mu, sd = [], [], []
        for e in range(1, max_ep + 1):
            vals = [cv[e - 1]["val_dice"] for cv in curves if len(cv) >= e]
            if len(vals) < need:  # multi-seed curves end where < 2 seeds still train
                break
            ep.append(e); mu.append(np.mean(vals))
            sd.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ep, mu, sd = map(np.array, (ep, mu, sd))
        if len(curves) > 1:
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.18, linewidth=0)
        ax.plot(ep, mu, color=color, linewidth=2,
                label=label + (f" (n={len(curves)} seeds)" if len(curves) > 1 else " (1 seed)"))
    ax.set_xlabel("Epoch", color=INK, fontsize=10)
    ax.set_ylabel("Validation Dice", color=INK, fontsize=10)
    ax.set_ylim(0.68, 0.82)
    ax.set_xlim(1, None)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    ax.set_title("Validation Dice by epoch (mean ± SD over seeds)", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_learning_curves.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: test Dice by subgroup (mean ± SD over seeds) -------------------
    gnames = list(groups)
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=150)
    style(ax)
    width = 0.8 / len(fig_cfgs) - 0.015
    x = np.arange(len(gnames))
    for k, (c, _, _, label, color) in enumerate(fig_cfgs):
        vals = np.array([[seed_dice[c][s][groups[g]].mean() for g in gnames] for s in seed_dice[c]])
        offset = (k - (len(fig_cfgs) - 1) / 2) * (width + 0.015)
        ax.bar(x + offset, vals.mean(0), width, color=color, label=label,
               yerr=vals.std(0, ddof=1) if len(vals) > 1 else None,
               error_kw=dict(ecolor=INK, elinewidth=1, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("Pituitary tumor", "Pituitary").replace("size ", "Size ")
                        .replace("all", "All")
                        + (f"\n(n={int(groups[g].sum())})") for g in gnames], fontsize=8.5)
    ax.set_ylabel("Test Dice", color=INK, fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left", labelcolor=INK)
    ax.set_title("Test Dice by tumor class and lesion-size quartile (mean ± SD over 3 seeds)",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_subgroups.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {OUT}/stats.md, stats.json, fig_learning_curves.*, fig_subgroups.*")


if __name__ == "__main__":
    main()
