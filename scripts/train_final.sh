#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Train + evaluate the final model (paper configuration D) for one or more seeds.
#
#   bash scripts/train_final.sh            # seeds 42 1 2 (as in the paper, ~2.2 h each on an RTX 3060)
#   bash scripts/train_final.sh 42         # one seed
#
# Run names: img512_intaug_vits_hybrid_vsds (seed 42), ..._s1, ..._s2. All seeds use the
# same patient split (--split-seed 42). Each run is also evaluated with flip TTA.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

FINAL=(--image-size 512 --encoder-size 512 --batch-size 4 --grad-accum-steps 2
       --rank 4 --lr 1e-4 --small-lesion-threshold 400 --intensity-aug --split-seed 42
       --variant vits --decoder hybrid --vit-skips --deep-supervision)

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 1 2)
for s in "${SEEDS[@]}"; do
  name="img512_intaug_vits_hybrid_vsds"; [ "$s" != "42" ] && name="${name}_s$s"
  bash "$ROOT/scripts/run_experiment.sh" "$name" "${FINAL[@]}" --seed "$s"
  if [ ! -f "$RUNS/$name/uncertainty_report_tta.json" ]; then
    "${LORA_EFFSAM[@]}" evaluate --checkpoint "$RUNS/$name/best.pt" --mc-passes 20 --tta \
      --out "$RUNS/$name/uncertainty_report_tta.json"
  fi
done
