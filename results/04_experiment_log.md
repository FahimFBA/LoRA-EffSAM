# 4. Experiment log

Every training run, in order, with why it was run and what it showed. All runs: same
data, same patient split, test set of 477 slices. Full configuration and every metric per
run: `tables/all_runs.csv`; per-epoch curves: `experiment_data/<run>/metrics.csv`.
Test Dice below = MC-Dropout mean prediction (K = 20); TTA = with flip TTA.

*Paths in this log are those of the original development layout. Since 2026-10-07 the
code lives in `src/lora_effsam`: `med_seg/train.py` → `lora-effsam train`, `evaluate_uncertainty.py` →
`lora-effsam evaluate`, `scripts/<name>.py` → `lora-effsam <subcommand>` (see the top-level README),
`scripts/test_architecture.py` → `tests/test_gpu_pipeline.py`, the queue scripts →
`scripts/experiments/run_all_experiments.sh`. No result changed in the move.*

## 4.1 Runs

| # | Run | Date | What changed (vs previous best) | Why | Epochs (best) | Test Dice | TTA |
|---|---|---|---|---|---|---|---|
| 1 | `enc512` | 10-03 | Baseline: 256 px images, ViT-T encoder at 512, rank 4, lr 1e-4, patience 8 | Original fast preset | 39 (31) | 0.783 | 0.788 |
| 2 | `enc1024_r8` | 10-03 | Encoder at 1024 (pretraining size), LoRA rank 8, lr 3e-4, patience 15 | Baseline looked like underfitting (train ≈ val Dice) | 24 (9) | 0.778 | 0.790 |
| 3 | `img512` | 10-03 | Back to #1 settings, images at full 512 px; size threshold T scaled 100 → 400 | Failures concentrated in gliomas and the smallest lesions (~9×9 px at 256) | 45 (30) | 0.797 | 0.807 |
| 4 | `img512_intaug` | 10-03 | + intensity augmentation (gamma, contrast, brightness) | Glioma failures are low-contrast cases | 60 (45) | 0.803 | 0.809 |
| 5–10 | `img512_s1/_s2`, `img512_intaug_s1/_s2`, `enc512_s1/_s2` | 10-03/04 | Seeds 1 and 2 of #1, #3, #4 (same split) | Mean ± SD for the paper | see table | see table | see table |
| 11–13 | `img512_intaug_gateQ1`, `_s1`, `_s2` | 10-04 | Size gate T = 1500, τ = 100 | Default gate (T = 400, τ = 20) fires on only 8/477 test images | 60, 60, 55 | 0.804 ± 0.001 | 0.815 ± 0.002 |
| 14 | `img512_intaug_hybrid` | 10-04/05 | Hybrid CNN–ViT decoder (ViT-T) | Simple decoder works from a 32×32 map only | 53 (38) | 0.798 | 0.813 |
| 15 | **`img512_intaug_vits_hybrid_vsds`** | 10-05 | ViT-S + hybrid decoder + ViT skips + deep supervision | Full architecture upgrade | 60 (55) | **0.819** | **0.825** |
| 16–17 | `img512_intaug_vits_hybrid_vsds_s1`, `_s2` | 10-05 | Seeds 1 and 2 of #15 (same split) | Mean ± SD for the final model | 56 (41), 52 (37) | 0.822, 0.816 | 0.824, 0.820 |

Seed results (MC test Dice, seeds 42 / 1 / 2):

| Configuration | Seed 42 | Seed 1 | Seed 2 | Mean ± SD |
|---|---|---|---|---|
| enc512 (A) | 0.7830 | 0.7734 | 0.7847 | 0.780 ± 0.006 |
| img512 (B) | 0.7970 | 0.8025 | 0.8012 | 0.800 ± 0.003 |
| img512_intaug (C) | 0.8025 | 0.8080 | 0.8040 | 0.805 ± 0.003 |
| img512_intaug_gateQ1 | 0.8029 | 0.8039 | 0.8044 | 0.804 ± 0.001 |
| **final** (img512_intaug_vits_hybrid_vsds) | 0.8190 | 0.8215 | 0.8155 | **0.819 ± 0.003** |

Not trained (proposed, not run): hybrid + ViT skips + deep supervision on ViT-T (to
separate the ViT-S effect), + LoRA on MLP layers (ablation), a longer schedule for the
final model. The code for all of them exists (flags
`--decoder hybrid --vit-skips --deep-supervision --lora-mlp`).

## 4.2 Inference-only analyses

- **Flip TTA** on every run (`uncertainty_report_tta.json`). An earlier quick check on 64
  images scored each flip separately and suggested TTA would not help; the full
  evaluation with averaging showed it does (+0.007 to +0.014 per run).
- **Re-evaluation with image IDs** (`uncertainty_report_ids.json`) for per-image pairing
  across runs; reproduces the original reports to within ±0.0002 (MC sampling noise).

## 4.3 Code and infrastructure changes during the experiments

None of these changed a reported number unless stated.

1. **Environment.** Code came from macOS; moved to WSL2 + conda (`torch` env, CUDA). Fixed a
   macOS-only `stat -f%z` call in `scripts/download_data.sh`.
2. **Experiment records.** From run #4 on, `train.py` writes `config.json`, `metrics.csv`
   (one row per epoch, appended), and `test.json`, and refuses to overwrite an existing
   run. Runs #1–3 were backfilled from their training logs: their `config.json` args are
   reconstructed from the run scripts (`args_reconstructed_from_run_script: true`), and
   their `metrics.csv` `lr` column is the post-epoch value (shifted by one epoch).
3. **Split seed.** `--split-seed` added before the seed repeats, so seeds change training
   randomness only and every run keeps the same test set.
4. **Test-image order bug (found and fixed 10-04).** `split_by_patient` iterated a Python
   set of patient-ID strings, whose order changes per process (hash randomisation). Test
   *membership* was always identical (verified for all runs: same multiset of
   (class, lesion area)), but image *order* differed, so per-image results could not be
   paired by index. Fixed by sorting; reports now carry image IDs; all runs re-evaluated.
   Seed-level results were unaffected.
5. **Flip TTA** (`evaluate_uncertainty.py --tta`).
6. **Size-gate flags** `--tau`, `--gamma` (defaults equal the original values).
7. **Architecture options** `--decoder hybrid`, `--vit-skips`, `--deep-supervision`,
   `--variant vits`, `--lora-mlp` (all off by default). Verified that the default path is
   unchanged: the run #4 checkpoint reproduces its recorded test Dice exactly (0.8024)
   after the change (`scripts/test_architecture.py`).
8. **Statistics tooling:** `scripts/summarize_runs.py`, `scripts/paper_stats.py`,
   `scripts/uncertainty_stats.py`, `scripts/export_paper_tables.py`.

Also, during setup the code files in `med_seg/data/` (`dataset.py`, `__init__.py`) were
briefly moved with the image data and then restored unchanged; no run used a different
version.
