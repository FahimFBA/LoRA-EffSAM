# 3. Results

Test set: 477 slices, 34 patients, identical for every run. Dice = mean per-image Dice of
the MC-Dropout prediction (K = 20). “± SD” = sample SD over 3 training seeds (42, 1, 2).
Sources: `tables/all_runs.csv`, `tables/summary_by_config.md`, `statistics/stats.md`,
`statistics/uncertainty_stats.md`.

## 3.1 Main results (Table 1)

| # | Configuration | Seeds | Test Dice | + flip TTA | Val Dice | Median | Failures (<0.3) |
|---|---|---|---|---|---|---|---|
| A | Baseline: 256 px, ViT-T, simple decoder | 3 | 0.780 ± 0.006 | 0.791 ± 0.007 | 0.775 ± 0.008 | 0.883 | 35.3 ± 2.1 |
| B | A + full resolution (512 px) | 3 | 0.800 ± 0.003 | 0.813 ± 0.005 | 0.781 ± 0.008 | 0.890 | 29.7 ± 4.2 |
| C | B + intensity augmentation | 3 | 0.805 ± 0.003 | 0.814 ± 0.005 | 0.781 ± 0.006 | 0.892 | 26.0 ± 2.0 |
| **D** | **C + ViT-S + hybrid decoder + ViT skips + deep supervision (final)** | 3 | **0.819 ± 0.003** | **0.823 ± 0.003** | **0.810 ± 0.004** | **0.903** | 26.0 ± 0.0 |

Medians are the seed-42 run; failure counts are 3-seed means. Per-seed values are in
`tables/all_runs.csv`. Final model per seed (42 / 1 / 2): test Dice 0.8190 / 0.8215 / 0.8155,
with TTA 0.8246 / 0.8237 / 0.8196.

**Final model, by subgroup** (3-seed mean; Dice without / with TTA):

| Subgroup | n | Final | Final + TTA | Baseline A |
|---|---|---|---|---|
| All | 477 | 0.819 | 0.823 | 0.780 |
| Meningioma | 120 | 0.904 | 0.908 | 0.887 |
| Glioma | 222 | 0.747 | 0.750 | 0.696 |
| Pituitary | 135 | 0.861 | 0.866 | 0.824 |
| Size Q1 (smallest) | 119 | 0.763 | 0.774 | 0.678 |
| Size Q2 | 119 | 0.833 | 0.832 | 0.814 |
| Size Q3 | 119 | 0.816 | 0.816 | 0.790 |
| Size Q4 (largest) | 120 | 0.863 | 0.868 | 0.839 |

Figures: `figures/fig_subgroups.pdf` (A, B, C, D by subgroup, mean ± SD over 3 seeds),
`figures/fig_learning_curves.pdf` (validation Dice per epoch, mean ± SD band),
`figures/fig_qualitative.pdf` (final model, seed 42; cases picked by rule, not by eye: the
median-Dice case of each class (0.944 / 0.849 / 0.906), the median case of the
smallest-size quartile (0.867), and the worst test case (Dice 0.0). Ground-truth and
predicted contours on a zoomed crop, and the MC-Dropout uncertainty map).

## 3.2 Statistical comparisons (Table 2)

Paired per-image Wilcoxon tests on the same 477 images (per-image Dice averaged over the
seeds both configurations share), Holm-corrected over 8 comparisons; seed-level paired
t-test on per-seed means.

| Comparison | Seeds | Δ Dice [95% CI] | Better / worse images | Holm p | Seed t-test p |
|---|---|---|---|---|---|
| B vs A (512 px) | 42, 1, 2 | +0.020 [+0.013, +0.027] | 312 / 165 | < 0.0001 | 0.051 |
| C vs B (intensity aug) | 42, 1, 2 | +0.005 [+0.001, +0.008] | 249 / 228 | 0.059 | 0.036 |
| C + TTA vs C | 42, 1, 2 | +0.009 [+0.005, +0.015] | 275 / 202 | 0.0004 | 0.064 |
| **D vs C (final architecture)** | 42, 1, 2 | **+0.014 [+0.005, +0.023]** | 281 / 196 | < 0.0001 | **0.011** |
| D + TTA vs C + TTA | 42, 1, 2 | +0.008 [+0.001, +0.016] | 291 / 186 | < 0.0001 | 0.18 |
| **D + TTA vs A** | 42, 1, 2 | **+0.042 [+0.032, +0.053]** | 343 / 134 | < 0.0001 | **0.011** |
| C + size gate vs C (negative) | 42, 1, 2 | −0.001 [−0.004, +0.002] | 196 / 281 | 0.006 | 0.54 |
| C + hybrid decoder only vs C (negative) | 42 | −0.004 [−0.013, +0.005] | 207 / 270 | 0.045 | n/a |

The CI and the Wilcoxon test can disagree (e.g. size gate: the CI of the mean includes 0,
but more images get worse, so the signed-rank test is significant). Report both. With 3
seeds the seed-level t-test has little power; the per-image test is the main evidence.

## 3.3 What each component contributes

1. **Full resolution (512 px): clear gain, +0.020.** Largest on the smallest lesions
   (+0.032) and gliomas (+0.027); significant in every class and size quartile.
2. **Intensity augmentation: modest, +0.005 (borderline, Holm p = 0.059).** Consistent
   across seeds (seed t-test p = 0.036); mainly gliomas (+0.008, p = 0.014, uncorrected).
   No gain on the validation set (0.781 vs 0.781).
3. **Final architecture (ViT-S + hybrid decoder + ViT skips + deep supervision): +0.014,
   confirmed over 3 seeds** (per-image p < 0.0001, seed t-test p = 0.011). Biggest effect on
   the smallest lesions (+0.045, p < 0.0001) and gliomas (+0.016, p = 0.002); also
   meningiomas (+0.007, p = 0.0003) and the largest lesions (+0.014, p = 0.046). Size Q2 is
   flat (−0.006, n.s.). Validation Dice rises too (0.810 vs 0.781). Seeds 1 and 2 early-stopped
   (best epochs 41 and 37), seed 42 peaked at epoch 55; the 3-seed validation curve flattens
   after ~45 epochs (`figures/fig_learning_curves.pdf`).
4. **Flip TTA: +0.004 to +0.013 at no training cost.** Helps every configuration (+0.011,
   +0.013, +0.009 on A, B, C; only +0.004 on the final model, i.e. the stronger model
   gains less from TTA). Concentrated on the smallest lesions (+0.029 on C) and gliomas/pituitary;
   meningiomas unchanged. A single flipped view alone scores *lower* than the upright image
   (anatomy has a canonical orientation); the gain comes from averaging.
5. **Overall (A → D + TTA): +0.042**, with the largest gains on the smallest lesions
   (+0.097) and gliomas (+0.054); better on 343 of 477 test images.

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

MC-Dropout, K = 20, p = 0.2, no TTA (`statistics/uncertainty_stats.md`). Final model, per
seed (42 / 1 / 2):

| Quantity | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|
| Mean U: boundary band | 2.6e-3 | 1.9e-3 | 1.7e-3 |
| Mean U: interior | 9.8e-4 | 3.9e-4 | 5.7e-4 |
| Mean U: background | 3.6e-5 | 1.4e-5 | 1.9e-5 |
| Images with U_boundary > U_interior | 429 / 477 | 430 / 477 | 432 / 477 |
| U_boundary, smallest vs largest area quartile | 3.8e-3 vs 2.1e-3 (p = 0.048) | 2.2e-3 vs 1.7e-3 (p < 0.0001) | 2.2e-3 vs 1.6e-3 (p = 0.20) |
| Spearman(U_boundary, Dice) | +0.22 | +0.42 | +0.24 |
| Failures (Dice < 0.5) | 40 | 36 | 42 |
| AUROC of U_boundary for detecting failures | 0.27 | 0.14 | 0.28 |

- **Supported on every seed:** uncertainty concentrates at lesion boundaries (≈ 2.7–5× the
  interior, ≈ 70–140× the background; boundary > interior in ≈ 90% of images, p < 1e-60).
  Also holds for the baseline and intermediate models.
- **Partly supported:** small lesions have higher boundary uncertainty than large ones on all
  three seeds, but the difference is significant on only two (p = 0.048, < 0.0001, 0.20).
  It is clearer for the baseline (p ≈ 2e-14). State it as a trend for the final model.
- **Not supported, on every seed:** boundary uncertainty does **not** flag failed
  segmentations. Failures have *lower* uncertainty in the true-boundary band (AUROC 0.14–0.28,
  below 0.5): when the model misses a tumor, it is confidently wrong there. The uncertainty
  map explains *where* the model hesitates along predicted boundaries, not *whether* it
  missed a lesion. The worst case in `figures/fig_qualitative.pdf` shows this directly:
  uncertainty traces the edges of the wrong predicted region and is zero at the missed
  tumor. Dropout is only in the decoder (the encoder is deterministic), which limits the
  epistemic uncertainty it can express.
- With the training threshold T = 400 px only 8 test lesions count as “small”, so report
  the quartile-based stratification as the main size analysis.

## 3.6 Explainability of the trained model

Inference-only analyses of the final model (`statistics/knockout.md`,
`statistics/explainability_stats.md`, `statistics/failure_analysis.txt`; figures
`fig_explainability_maps`, `fig_knockout`, `fig_uncertainty_analysis`, `fig_dice_distribution`).

- **Coarse-to-fine prediction.** The tumor is located at 1/8 resolution and refined at
  each finer level. Failures are decided at the coarsest level: the 26 failed slices
  (Dice < 0.3, seed 42) already have Dice 0.078 at 1/8 (25/26 below 0.3), vs 0.835 for the
  rest; 1/8-level and final Dice correlate at r = 0.97. The 26 failures are the same count
  and class mix on every seed (21 glioma, 3 meningioma, 2 pituitary).
- **Component reliance (test-time knock-outs, 3 seeds):** LoRA off −0.282; ViT skips
  zeroed −0.070 (−0.165 on the smallest lesions); CNN skips zeroed −0.026 (−0.068 on the
  smallest); CBAM attention off −0.0001; refine head off −0.0006. These measure reliance of
  the trained network, not retraining without the component.
- **Attention modules are nearly unused.** CBAM gate γ learned 0.06–0.08 (from 0); its
  spatial attention is not tumor-focused (inside 0.672 vs outside 0.673; higher inside in
  46 % of slices). The refine head learned a boundary-shaped gate (scale ≈ 8.3,
  bias ≈ −0.7) but contributes −0.0006 Dice. Do not present these attention maps as
  explanations.

## 3.7 Comparison with published work

Not done in this package. Many reported Dice values on this dataset use slice-level
(per-image) splits, which leak slices of the same patient into train and test, or pool
all pixels into one Dice score (dominated by large tumors). Our protocol is patient-level
with mean per-image Dice, which is stricter. Before writing a comparison table, check each
cited paper's split and metric definition.
