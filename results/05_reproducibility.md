# 5. Reproducibility

Code: this repository (`src/lora_effsam`, `scripts/`, `tests/`). All commands run from the
repository root in a CUDA-enabled environment (see the top-level README for install).

## 5.1 Setup

```bash
conda create -n lora-effsam python=3.12 && conda activate lora-effsam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
bash scripts/fetch_efficientsam.sh   # EfficientSAM code + weights -> third_party/EfficientSAM
bash scripts/setup_data.sh           # dataset -> data/figshare_brain_tumor ("3064 slices from 233 patients")
lora-effsam verify-env
pytest                               # unit + GPU checks, no training (~1 min)
SMOKE=1 bash scripts/run_pipeline.sh # 1-epoch end-to-end check (~5 min)
```

Data, weights and runs can live elsewhere: set `LORA_EFFSAM_DATA_DIR`, `LORA_EFFSAM_EFFICIENTSAM_DIR`,
`LORA_EFFSAM_RUNS_DIR` (see `src/lora_effsam/paths.py`).

## 5.2 Train the final model

```bash
bash scripts/train_final.sh          # seeds 42, 1, 2 -> runs/img512_intaug_vits_hybrid_vsds{,_s1,_s2}
```

which, per seed, runs

```bash
bash scripts/run_experiment.sh img512_intaug_vits_hybrid_vsds \
  --image-size 512 --encoder-size 512 --batch-size 4 --grad-accum-steps 2 \
  --rank 4 --lr 1e-4 --small-lesion-threshold 400 --intensity-aug \
  --split-seed 42 --seed 42 \
  --variant vits --decoder hybrid --vit-skips --deep-supervision
lora-effsam evaluate --checkpoint runs/img512_intaug_vits_hybrid_vsds/best.pt --mc-passes 20 --tta \
  --out runs/img512_intaug_vits_hybrid_vsds/uncertainty_report_tta.json
```

`run_experiment.sh` adds `--epochs 60 --early-stop-patience 15`, trains, runs the
MC-Dropout evaluation and the sample figure, and updates `runs/summary.*`. The exact
command of every run is stored in its `config.json` (`"command"`; runs from before the
package refactor show the old `train.py` form with the same options). Re-running an
interrupted run resumes from its last epoch; a finished run is never overwritten.

All 17 experiments of `04_experiment_log.md`: `bash scripts/experiments/run_all_experiments.sh`
(~30 h on one RTX 3060).

Checkpoints are not in the repository (GitHub's 100 MB limit). Trained checkpoints are
not distributed; retrain with the commands above. To reuse existing runs, point
`LORA_EFFSAM_RUNS_DIR` at their folder.

## 5.3 Rebuild every table and figure

```bash
bash scripts/reproduce_results.sh    # ~20 min; needs the runs incl. the final model's checkpoints
```

which runs, in order: `lora-effsam summarize`, `export-tables`, `paper-stats`,
`uncertainty-stats`, `qualitative-figure`, `explainability`, `knockout`, `failure-check`
(each writes into `results/`; see `lora-effsam <subcommand> --help`). Paired per-image statistics
need per-image reports with image IDs (`uncertainty_report.json` of runs evaluated after
2026-10-04, or `uncertainty_report_ids.json` for the earlier ones).

## 5.4 Determinism

Seeds fix Python/NumPy/PyTorch RNGs, the split and data order, but cuDNN kernels and
AMP are not bitwise deterministic, so a retrain gives slightly different numbers. Across
seeds the SD of test Dice was 0.003–0.006; expect differences of that order.
