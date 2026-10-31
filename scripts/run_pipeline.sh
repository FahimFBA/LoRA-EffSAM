#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# End-to-end pipeline for the final model:
#   environment check -> EfficientSAM -> dataset -> train + evaluate (3 seeds) -> summary
#
#   bash scripts/run_pipeline.sh            # 3 seeds (~7 h on one RTX 3060)
#   SEEDS="42" bash scripts/run_pipeline.sh # one seed (~2.5 h)
#   SMOKE=1 bash scripts/run_pipeline.sh    # 1-epoch check on a tiny subset (~5 min)
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "== 1/4 environment";  "${LORA_EFFSAM[@]}" verify-env || { echo "No CUDA GPU found." >&2; exit 1; }
echo "== 2/4 EfficientSAM"; bash "$ROOT/scripts/fetch_efficientsam.sh"
echo "== 3/4 dataset";      bash "$ROOT/scripts/setup_data.sh"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "== 4/4 smoke run (runs/smoke)"
  rm -rf "$RUNS/smoke" "$RUNS/smoke.train.log"
  bash "$ROOT/scripts/run_experiment.sh" smoke --image-size 512 --encoder-size 512 \
    --batch-size 4 --grad-accum-steps 2 --variant vits --decoder hybrid --vit-skips \
    --deep-supervision --intensity-aug --small-lesion-threshold 400 \
    --epochs 1 --max-train-samples 32 --max-eval-samples 16
  exit 0
fi
echo "== 4/4 final model"
read -r -a SEED_LIST <<< "${SEEDS:-42 1 2}"
bash "$ROOT/scripts/train_final.sh" "${SEED_LIST[@]}"
echo "Done. Runs in $RUNS; rebuild results/ with: bash scripts/reproduce_results.sh"
