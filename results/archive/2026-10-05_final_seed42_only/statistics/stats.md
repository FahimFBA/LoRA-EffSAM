# Statistical comparison of configurations

Test set: 477 images (patient-level split, identical for all runs). Runs per configuration: Baseline (256 px): enc512, enc512_s1, enc512_s2; 512 px: img512, img512_s1, img512_s2; 512 px + intensity aug: img512_intaug, img512_intaug_s1, img512_intaug_s2; 512 px + intensity aug + size gate: img512_intaug_gateQ1, img512_intaug_gateQ1_s1, img512_intaug_gateQ1_s2; 512 px + intensity aug + flip TTA: img512_intaug, img512_intaug_s1, img512_intaug_s2; 512 px + intensity aug + hybrid decoder: img512_intaug_hybrid; Final model: img512_intaug_vits_hybrid_vsds; Final model + flip TTA: img512_intaug_vits_hybrid_vsds.

Each comparison pairs the same test images and uses per-image Dice averaged over the seeds BOTH configurations have (column 'seeds'), so a single-seed model is compared seed-for-seed. Wilcoxon signed-rank test on paired per-image Dice; Holm correction across the 8 comparisons; 95% CI of the mean difference by bootstrap over images (10000 resamples). Seed-level: paired t-test on per-seed mean test Dice (needs >= 2 shared seeds).

| Comparison | seeds | Mean Dice (A vs B) | Δ mean [95% CI] | images better / worse | Wilcoxon p | Holm p | seed-paired t p |
|---|---|---|---|---|---|---|---|
| 512 px vs Baseline (256 px) | 1,2,42 | 0.8002 vs 0.7804 | +0.0199 [+0.0131, +0.0268] | 312 / 165 | <0.0001 | <0.0001 | 0.0509 |
| 512 px + intensity aug vs 512 px | 1,2,42 | 0.8048 vs 0.8002 | +0.0046 [+0.0009, +0.0083] | 249 / 228 | 0.0588 | 0.0588 | 0.0356 |
| 512 px + intensity aug + size gate vs 512 px + intensity aug | 1,2,42 | 0.8037 vs 0.8048 | -0.0011 [-0.0043, +0.0024] | 196 / 281 | 0.0018 | 0.0055 | 0.5385 |
| 512 px + intensity aug + flip TTA vs 512 px + intensity aug | 1,2,42 | 0.8142 vs 0.8048 | +0.0094 [+0.0047, +0.0146] | 275 / 202 | 0.0001 | 0.0004 | 0.0642 |
| 512 px + intensity aug + hybrid decoder vs 512 px + intensity aug | 42 | 0.7983 vs 0.8025 | -0.0042 [-0.0126, +0.0045] | 207 / 270 | 0.0223 | 0.0446 | n/a |
| Final model vs 512 px + intensity aug | 42 | 0.8190 vs 0.8025 | +0.0165 [+0.0061, +0.0271] | 274 / 203 | <0.0001 | <0.0001 | n/a |
| Final model + flip TTA vs 512 px + intensity aug + flip TTA | 42 | 0.8246 vs 0.8091 | +0.0155 [+0.0059, +0.0252] | 283 / 194 | <0.0001 | <0.0001 | n/a |
| Final model + flip TTA vs Baseline (256 px) | 42 | 0.8246 vs 0.7830 | +0.0416 [+0.0297, +0.0537] | 320 / 157 | <0.0001 | <0.0001 | n/a |

## 512 px vs Baseline (256 px): subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0199 [+0.0130, +0.0267] | <0.0001 |
| Meningioma | 120 | +0.0091 [+0.0016, +0.0171] | 0.0029 |
| Glioma | 222 | +0.0269 [+0.0152, +0.0392] | <0.0001 |
| Pituitary tumor | 135 | +0.0180 [+0.0052, +0.0298] | <0.0001 |
| size Q1 | 119 | +0.0317 [+0.0119, +0.0519] | <0.0001 |
| size Q2 | 119 | +0.0239 [+0.0105, +0.0373] | <0.0001 |
| size Q3 | 119 | +0.0140 [+0.0036, +0.0250] | 0.0119 |
| size Q4 | 120 | +0.0099 [+0.0018, +0.0191] | 0.0381 |

Per-seed Wilcoxon p (same seed, A vs B): seed 1: <0.0001, seed 2: 0.0010, seed 42: 0.0014

## 512 px + intensity aug vs 512 px: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0046 [+0.0009, +0.0082] | 0.0588 |
| Meningioma | 120 | +0.0002 [-0.0038, +0.0052] | 0.5823 |
| Glioma | 222 | +0.0079 [+0.0012, +0.0144] | 0.0143 |
| Pituitary tumor | 135 | +0.0031 [-0.0021, +0.0086] | 0.4605 |
| size Q1 | 119 | +0.0088 [-0.0016, +0.0188] | 0.1183 |
| size Q2 | 119 | +0.0010 [-0.0064, +0.0085] | 0.4080 |
| size Q3 | 119 | +0.0085 [+0.0022, +0.0155] | 0.0404 |
| size Q4 | 120 | +0.0001 [-0.0035, +0.0035] | 0.1931 |

Per-seed Wilcoxon p (same seed, A vs B): seed 1: 0.3088, seed 2: 0.0190, seed 42: 0.0131

## 512 px + intensity aug + size gate vs 512 px + intensity aug: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | -0.0011 [-0.0044, +0.0025] | 0.0018 |
| Meningioma | 120 | +0.0007 [-0.0020, +0.0034] | 0.9603 |
| Glioma | 222 | +0.0025 [-0.0038, +0.0093] | 0.2756 |
| Pituitary tumor | 135 | -0.0086 [-0.0133, -0.0043] | <0.0001 |
| size Q1 | 119 | -0.0062 [-0.0153, +0.0043] | 0.0004 |
| size Q2 | 119 | +0.0044 [-0.0015, +0.0116] | 0.7827 |
| size Q3 | 119 | -0.0041 [-0.0100, +0.0016] | 0.0701 |
| size Q4 | 120 | +0.0015 [-0.0014, +0.0045] | 0.9645 |

Per-seed Wilcoxon p (same seed, A vs B): seed 1: 0.0017, seed 2: 0.0423, seed 42: 0.2364

## 512 px + intensity aug + flip TTA vs 512 px + intensity aug: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0094 [+0.0047, +0.0146] | 0.0001 |
| Meningioma | 120 | -0.0014 [-0.0066, +0.0042] | 0.7296 |
| Glioma | 222 | +0.0120 [+0.0037, +0.0209] | 0.0031 |
| Pituitary tumor | 135 | +0.0148 [+0.0065, +0.0243] | 0.0003 |
| size Q1 | 119 | +0.0287 [+0.0147, +0.0445] | <0.0001 |
| size Q2 | 119 | +0.0065 [-0.0001, +0.0140] | 0.1293 |
| size Q3 | 119 | +0.0004 [-0.0083, +0.0094] | 0.3023 |
| size Q4 | 120 | +0.0020 [-0.0024, +0.0066] | 0.3261 |

Per-seed Wilcoxon p (same seed, A vs B): seed 1: 0.0125, seed 2: 0.0001, seed 42: 0.0111

## 512 px + intensity aug + hybrid decoder vs 512 px + intensity aug: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | -0.0042 [-0.0127, +0.0045] | 0.0223 |
| Meningioma | 120 | -0.0041 [-0.0095, +0.0019] | 0.0009 |
| Glioma | 222 | -0.0084 [-0.0234, +0.0072] | 0.0489 |
| Pituitary tumor | 135 | +0.0027 [-0.0139, +0.0197] | 0.2989 |
| size Q1 | 119 | +0.0088 [-0.0155, +0.0353] | 0.1086 |
| size Q2 | 119 | -0.0125 [-0.0283, +0.0013] | 0.0231 |
| size Q3 | 119 | -0.0103 [-0.0221, +0.0019] | 0.0009 |
| size Q4 | 120 | -0.0027 [-0.0163, +0.0095] | 0.2174 |

Per-seed Wilcoxon p (same seed, A vs B): seed 42: 0.0223

## Final model vs 512 px + intensity aug: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0165 [+0.0062, +0.0270] | <0.0001 |
| Meningioma | 120 | +0.0084 [+0.0004, +0.0167] | 0.0210 |
| Glioma | 222 | +0.0241 [+0.0043, +0.0432] | <0.0001 |
| Pituitary tumor | 135 | +0.0111 [-0.0052, +0.0291] | 0.1811 |
| size Q1 | 119 | +0.0535 [+0.0309, +0.0779] | <0.0001 |
| size Q2 | 119 | -0.0064 [-0.0307, +0.0167] | 0.9662 |
| size Q3 | 119 | +0.0010 [-0.0185, +0.0195] | 0.1831 |
| size Q4 | 120 | +0.0177 [+0.0024, +0.0335] | 0.0070 |

Per-seed Wilcoxon p (same seed, A vs B): seed 42: <0.0001

## Final model + flip TTA vs 512 px + intensity aug + flip TTA: subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0155 [+0.0060, +0.0251] | <0.0001 |
| Meningioma | 120 | +0.0157 [+0.0052, +0.0284] | 0.0002 |
| Glioma | 222 | +0.0226 [+0.0056, +0.0395] | 0.0005 |
| Pituitary tumor | 135 | +0.0038 [-0.0115, +0.0203] | 0.3761 |
| size Q1 | 119 | +0.0398 [+0.0174, +0.0631] | <0.0001 |
| size Q2 | 119 | -0.0070 [-0.0284, +0.0125] | 0.7403 |
| size Q3 | 119 | +0.0068 [-0.0091, +0.0235] | 0.2631 |
| size Q4 | 120 | +0.0224 [+0.0079, +0.0378] | 0.0033 |

Per-seed Wilcoxon p (same seed, A vs B): seed 42: <0.0001

## Final model + flip TTA vs Baseline (256 px): subgroups (uncorrected, exploratory)

| Subgroup | n | Δ mean [95% CI] | Wilcoxon p |
|---|---|---|---|
| all | 477 | +0.0416 [+0.0300, +0.0540] | <0.0001 |
| Meningioma | 120 | +0.0153 [+0.0055, +0.0270] | <0.0001 |
| Glioma | 222 | +0.0616 [+0.0396, +0.0846] | <0.0001 |
| Pituitary tumor | 135 | +0.0322 [+0.0178, +0.0479] | <0.0001 |
| size Q1 | 119 | +0.0928 [+0.0663, +0.1228] | <0.0001 |
| size Q2 | 119 | +0.0221 [-0.0017, +0.0465] | 0.0003 |
| size Q3 | 119 | +0.0232 [+0.0030, +0.0465] | 0.0106 |
| size Q4 | 120 | +0.0285 [+0.0119, +0.0467] | 0.0020 |

Per-seed Wilcoxon p (same seed, A vs B): seed 42: <0.0001
