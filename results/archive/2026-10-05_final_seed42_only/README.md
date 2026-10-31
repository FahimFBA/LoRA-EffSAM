# Results (archived): brain-tumor segmentation with LoRA-adapted EfficientSAM

Frozen on 2026-10-05. Method, data protocol,
results with statistics, the full experiment log, figures, raw per-epoch data, the
final checkpoint and a code snapshot. Experiments are concluded; the final model is
`img512_intaug_vits_hybrid_vsds` (seed 42).

## Headline result

Brain-tumor segmentation on the Figshare (Cheng et al.) T1-CE MRI dataset with a
**patient-level** split (477 test slices from 34 held-out patients):

| Model | Test Dice (mean per image) | Median | Smallest-lesion quartile | Glioma |
|---|---|---|---|---|
| Baseline (256 px, ViT-T, simple decoder), 3 seeds | 0.780 ± 0.006 | 0.883 | 0.678 | 0.696 |
| **Final model** (512 px, ViT-S, hybrid decoder, ViT skips, deep supervision, intensity aug), seed 42 | **0.819** | **0.903** | **0.765** | **0.747** |
| **Final model + flip TTA** | **0.825** | — | **0.777** | **0.755** |

Baseline median is from its seed-42 run. The final model with TTA beats the baseline by **+0.042 Dice** on the same seed (Wilcoxon on 477
paired images p < 0.0001; better on 320 images, worse on 157).

## Read in this order

| File | Contents | Paper section |
|---|---|---|
| [01_method.md](01_method.md) | Architecture, LoRA, loss, MC-Dropout, TTA — with equations and every hyperparameter | Methods |
| [02_data_and_protocol.md](02_data_and_protocol.md) | Dataset, preprocessing, patient-level split, augmentation, training and evaluation protocol, hardware | Methods / Experimental setup |
| [03_results.md](03_results.md) | Main results, ablations, statistics, subgroups, TTA, uncertainty analysis, negative results | Results / Discussion |
| [04_experiment_log.md](04_experiment_log.md) | Every run in order, what changed and why, bugs found and fixed | Supplementary / for the authors |
| [05_reproducibility.md](05_reproducibility.md) | Commands to reproduce every number and figure | Code availability |
| [06_limitations_and_notes.md](06_limitations_and_notes.md) | Caveats reviewers will ask about, and what to verify before submission | Limitations |

## Folder layout

```
paper_final/
├── README.md, 01..06 *.md       the write-up (above)
├── figures/                     fig_learning_curves.{png,pdf}, fig_subgroups.{png,pdf},
│                                fig_qualitative.{png,pdf} (median case per class, small lesion, worst case; contours + uncertainty)
├── tables/                      all_runs.{csv,md} (one row per run, full config + all test metrics)
│                                summary.{csv,md}, summary_by_config.md (mean ± SD over seeds)
├── statistics/                  stats.{md,json} (paired tests, CIs, subgroups)
│                                uncertainty_stats.{md,json} (MC-Dropout analysis)
├── experiment_data/<run>/       per run: config.json, metrics.csv (every epoch), test.json,
│                                uncertainty_report*.json (per-image results), train.log, samples.png
```
