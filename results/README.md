# Results: brain-tumor segmentation with LoRA-adapted EfficientSAM

Frozen on 2026-10-05; updated the same evening with seeds 1 and 2 of the final model
(the seed-42-only version is kept in `archive/2026-10-05_final_seed42_only/`).
Method, data protocol, results with statistics,
the full experiment log, figures and raw per-epoch data. Code is in `src/lora_effsam`; checkpoints are kept outside git.
Experiments are concluded; the final model is `img512_intaug_vits_hybrid_vsds`
(3 seeds: 42, 1, 2).

## Headline result

Brain-tumor segmentation on the Figshare (Cheng et al.) T1-CE MRI dataset with a
**patient-level** split (477 test slices from 34 held-out patients):

All rows: mean ± SD over 3 training seeds (42, 1, 2), same test set.

| Model | Test Dice (mean per image) | Median (seed 42) | Smallest-lesion quartile | Glioma |
|---|---|---|---|
| Baseline (256 px, ViT-T, simple decoder) | 0.780 ± 0.006 | 0.883 | 0.678 | 0.696 |
| **Final model** (512 px, ViT-S, hybrid decoder, ViT skips, deep supervision, intensity aug) | **0.819 ± 0.003** | **0.903** | **0.763** | **0.747** |
| **Final model + flip TTA** | **0.823 ± 0.003** | — | **0.774** | **0.750** |

The final model with TTA beats the baseline by **+0.042 Dice** [95% CI +0.032, +0.053]
(Wilcoxon on 477 paired images p < 0.0001, better on 343 images, worse on 134; seed-level
paired t-test p = 0.011).

## Read in this order

| File | Contents |
|---|---|
| [01_method.md](01_method.md) | Architecture, LoRA, loss, MC-Dropout, TTA — with equations and every hyperparameter |
| [02_data_and_protocol.md](02_data_and_protocol.md) | Dataset, preprocessing, patient-level split, augmentation, training and evaluation protocol, hardware |
| [03_results.md](03_results.md) | Main results, ablations, statistics, subgroups, TTA, uncertainty analysis, negative results |
| [04_experiment_log.md](04_experiment_log.md) | Every run in order, what changed and why, bugs found and fixed |
| [05_reproducibility.md](05_reproducibility.md) | Commands to reproduce every number and figure |
| [06_limitations_and_notes.md](06_limitations_and_notes.md) | Caveats reviewers will ask about, and what to verify before submission |

## Folder layout

```
results/
├── README.md, 01..06 *.md       the write-up (above)
├── figures/                     fig_architecture        model diagram
│                                fig_subgroups           Dice by class and lesion size
│                                fig_dice_distribution   per-image Dice, baseline vs final
│                                fig_learning_curves     validation Dice per epoch
│                                fig_explainability_maps attention, coarse-to-fine decoder, refine gate, uncertainty
│                                fig_knockout            component reliance (test-time knock-outs)
│                                fig_uncertainty_analysis uncertainty by region, vs Dice, by size
│                                fig_qualitative         rule-selected cases with contours + uncertainty
│                                (each as .png and .pdf)
├── tables/                      all_runs.{csv,md} (one row per run, full config + all test metrics)
│                                summary.{csv,md}, summary_by_config.md (mean ± SD over seeds)
├── statistics/                  stats.{md,json} (paired tests, CIs, subgroups)
│                                uncertainty_stats.{md,json} (MC-Dropout analysis)
│                                knockout.{md,json}, explainability_stats.{md,json},
│                                failure_analysis.txt (explainability)
├── experiment_data/<run>/       per run: config.json, metrics.csv (every epoch), test.json,
│                                uncertainty_report*.json (per-image results), train.log, samples.png
├── model/                       final_config.json (the checkpoint itself is not in git)
└── archive/                     earlier versions of the write-up/statistics (seed 42 only)
```

The code is the repository itself (`src/lora_effsam`, `scripts/`); regenerate this folder with
`bash scripts/reproduce_results.sh`. Checkpoints, the dataset and the EfficientSAM weights
are not in git (see `05_reproducibility.md`).
