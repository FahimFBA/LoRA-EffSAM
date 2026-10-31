# 3. Results

Test set: 477 slices, 34 patients, identical for every run. Dice = mean per-image Dice of
the MC-Dropout prediction (K = 20). “± SD” = sample SD over 3 training seeds (42, 1, 2).
Sources: `tables/all_runs.csv`, `tables/summary_by_config.md`, `statistics/stats.md`,
`statistics/uncertainty_stats.md`.

## 3.1 Main results (Table 1)

| # | Configuration | Seeds | Test Dice | + flip TTA | Median | Failures (<0.3) |
|---|---|---|---|---|---|---|
| A | Baseline: 256 px, ViT-T, simple decoder | 3 | 0.780 ± 0.006 | 0.791 ± 0.007 | 0.883 | 35.3 ± 2.1 |
| B | A + full resolution (512 px) | 3 | 0.800 ± 0.003 | 0.813 ± 0.005 | 0.890 | 29.7 ± 4.2 |
| C | B + intensity augmentation | 3 | 0.805 ± 0.003 | 0.814 ± 0.005 | 0.892 | 26.0 ± 2.0 |
| **D** | **C + ViT-S + hybrid decoder + ViT skips + deep supervision (final)** | 1 (42) | **0.819** | **0.825** | **0.903** | 26 |

Medians and failure counts for A–C are the seed-42 run (median) and the 3-seed mean
(failures). TTA ± SD computed from the per-seed TTA results in `tables/all_runs.csv`.

**Final model, by subgroup** (seed 42; Dice without / with TTA):

| Subgroup | n | Final | Final + TTA | Baseline A (3 seeds) |
|---|---|---|---|---|
| All | 477 | 0.819 | 0.825 | 0.780 |
| Meningioma | 120 | 0.904 | 0.909 | 0.887 |
| Glioma | 222 | 0.747 | 0.755 | 0.696 |
| Pituitary | 135 | 0.861 | 0.864 | 0.824 |
| Size Q1 (smallest) | 119 | 0.765 | 0.777 | 0.678 |
| Size Q2 | 119 | 0.831 | 0.836 | 0.814 |
| Size Q3 | 119 | 0.818 | 0.817 | 0.790 |
| Size Q4 (largest) | 120 | 0.862 | 0.868 | 0.839 |

Figures: `figures/fig_subgroups.pdf` (A, B, C, D by subgroup), `figures/fig_learning_curves.pdf`
(validation Dice per epoch), `figures/fig_qualitative.pdf` (final model; cases picked by
rule, not by eye: the median-Dice case of each class (0.944 / 0.849 / 0.906), the median case
of the smallest-size quartile (0.867), and the worst test case (Dice 0.0). Ground-truth and
predicted contours on a zoomed crop, and the MC-Dropout uncertainty map).

## 3.2 Statistical comparisons (Table 2)

Paired per-image Wilcoxon tests on the same 477 images, Holm-corrected over 8 comparisons.

| Comparison | Seeds | Δ Dice [95% CI] | Better / worse images | Holm p |
|---|---|---|---|---|
| B vs A (512 px) | 42, 1, 2 | +0.020 [+0.013, +0.027] | 312 / 165 | < 0.0001 |
| C vs B (intensity aug) | 42, 1, 2 | +0.005 [+0.001, +0.008] | 249 / 228 | 0.059 |
| C + TTA vs C | 42, 1, 2 | +0.009 [+0.005, +0.015] | 275 / 202 | 0.0004 |
| D vs C (final architecture) | 42 | +0.017 [+0.006, +0.027] | 274 / 203 | < 0.0001 |
| D + TTA vs C + TTA | 42 | +0.016 [+0.006, +0.025] | 283 / 194 | < 0.0001 |
| **D + TTA vs A** | 42 | **+0.042 [+0.030, +0.054]** | 320 / 157 | < 0.0001 |
| C + size gate vs C (negative) | 42, 1, 2 | −0.001 [−0.004, +0.002] | 196 / 281 | 0.006 |
| C + hybrid decoder only vs C (negative) | 42 | −0.004 [−0.013, +0.005] | 207 / 270 | 0.045 |

Note the CI and the Wilcoxon test can disagree in direction of significance (e.g. size
gate: CI of the mean includes 0, but more images get worse, so the signed-rank test is
significant). Report both.

## 3.3 What each component contributes

1. **Full resolution (512 px) — clear gain, +0.020.** Largest on the smallest lesions
   (+0.032) and gliomas (+0.027); significant in every class and size quartile.
2. **Intensity augmentation — modest, +0.005 (borderline, Holm p = 0.059).** Consistent
   across seeds (paired t-test over seeds p = 0.036); mainly gliomas (+0.008, p = 0.014,
   uncorrected). No gain on the validation set (0.781 vs 0.781).
3. **Final architecture (ViT-S + hybrid decoder + ViT skips + deep supervision) — +0.017.**
   Biggest effect on the smallest lesions (+0.054, p < 0.0001) and gliomas (+0.024,
   p = 0.0001); also the largest lesions (+0.018). Size Q2 is flat (−0.006, n.s.).
   Validation Dice also rises (0.805 vs 0.787). The model was still improving at the end
   (best epoch 55 of 60; `figures/fig_learning_curves.pdf`).
4. **Flip TTA — +0.009 to +0.016 at no training cost.** Helps every configuration
   (≈ +0.01 on A, B, C). Concentrated on the smallest lesions (+0.029 on C) and
   gliomas/pituitary; meningiomas unchanged. A single flipped view alone scores *lower*
   than the upright image (anatomy has a canonical orientation); the gain comes from averaging.

## 3.4 Negative results (worth reporting)

- **Hybrid decoder alone on ViT-T: −0.004** (seed 42). The skip-connection decoder only pays
  off together with ViT-S, ViT skips and deep supervision; which of these three matters
  most was not separated (see limitations).
- **Larger LoRA / encoder at 256 px (encoder input 1024, rank 8, lr 3e-4): 0.778 vs 0.784** for
  the seed-42 baseline. Capacity was not the bottleneck at 256 px.
- **Wider size gate (T = 1500, τ = 100) to cover the smallest-lesion quartile: −0.001**, and
  slightly *worse* on the smallest quartile (−0.006). With the default T = 400, τ = 20 the
  size gate is active (> 0.5) on only 8 of 477 test images; widening it did not help, so
  the small-lesion weakness was not a loss-weighting problem.

## 3.5 Uncertainty analysis (Step 3)

MC-Dropout, K = 20, p = 0.2, no TTA (`statistics/uncertainty_stats.md`). Final model:

| Quantity | Value |
|---|---|
| Mean U in boundary band / interior / background | 2.6e-3 / 9.8e-4 / 3.6e-5 |
| Images with U_boundary > U_interior | 429 / 477 (Wilcoxon p ≈ 7e-62) |
| U_boundary, smallest vs largest area quartile | 3.8e-3 vs 2.1e-3 (Mann–Whitney p = 0.048) |
| Spearman(U_boundary, Dice) | +0.22 (p < 1e-6) |
| AUROC of U_boundary for detecting failures (Dice < 0.5, n = 40) | 0.27 |

- **Supported:** uncertainty concentrates at lesion boundaries (2.7× interior, ~70×
  background), and small lesions are more uncertain than large ones. This holds for the
  baseline and intermediate models too.
- **Not supported:** boundary uncertainty does **not** flag failed segmentations. Failures
  have *lower* uncertainty in the true-boundary band (AUROC 0.27 < 0.5): when the model
  misses a tumor, it is confidently wrong there. The uncertainty map explains *where* the
  model hesitates along predicted boundaries, not *whether* it missed a lesion. The worst
  case in `figures/fig_qualitative.pdf` shows this directly: uncertainty traces the edges of
  the wrong predicted region and is zero at the missed tumor. Dropout is
  only in the decoder (the encoder is deterministic), which limits the epistemic
  uncertainty it can express.
- With the training threshold T = 400 px only 8 test lesions count as “small”, so report
  the quartile-based stratification (above) as the main size analysis.

## 3.6 Comparison with published work

Not done in this package. Many reported Dice values on this dataset use slice-level
(per-image) splits, which leak slices of the same patient into train and test, or pool
all pixels into one Dice score (dominated by large tumors). Our protocol is patient-level
with mean per-image Dice, which is stricter. Before writing a comparison table, check each
cited paper's split and metric definition.
