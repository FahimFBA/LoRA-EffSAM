#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Re-run every experiment reported in results/ (17 runs, ~30 h on one RTX 3060), with the
# exact arguments recorded in each run's config.json, then the flip-TTA evaluation of each.
# Finished runs are skipped; an interrupted run resumes. See results/04_experiment_log.md.
#
#   bash scripts/experiments/run_all_experiments.sh
set -euo pipefail
source "$(dirname "$0")/../_env.sh"
RUN="$ROOT/scripts/run_experiment.sh"

# A: baseline (original fast preset: 256 px images, ViT-T encoder at 512, patience 8)
A=(--image-size 256 --encoder-size 512 --batch-size 4 --grad-accum-steps 2 --rank 4 --lr 1e-4
   --small-lesion-threshold 100 --split-seed 42 --epochs 100 --early-stop-patience 8)
# B: full resolution; C: + intensity augmentation
B=(--image-size 512 --encoder-size 512 --batch-size 4 --grad-accum-steps 2 --rank 4 --lr 1e-4
   --small-lesion-threshold 400 --split-seed 42)
C=("${B[@]}" --intensity-aug)

for s in 42 1 2; do
  sfx=""; [ "$s" != "42" ] && sfx="_s$s"
  bash "$RUN" "enc512$sfx" "${A[@]}" --seed "$s"
  bash "$RUN" "img512$sfx" "${B[@]}" --seed "$s"
  bash "$RUN" "img512_intaug$sfx" "${C[@]}" --seed "$s"
  bash "$RUN" "img512_intaug_gateQ1$sfx" "${C[@]}" --small-lesion-threshold 1500 --tau 100 --seed "$s"
done
# single-seed ablations
bash "$RUN" enc1024_r8 --image-size 256 --encoder-size 1024 --batch-size 2 --grad-accum-steps 4 \
  --rank 8 --lr 3e-4 --split-seed 42 --seed 42
bash "$RUN" img512_intaug_hybrid "${C[@]}" --decoder hybrid --seed 42
# D: final model, 3 seeds (also runs its TTA evaluation)
bash "$ROOT/scripts/train_final.sh" 42 1 2

# Flip-TTA evaluation of every run
for d in "$RUNS"/*/; do
  run="${d%/}"
  [ -f "$run/best.pt" ] && [ ! -f "$run/uncertainty_report_tta.json" ] || continue
  "${LORA_EFFSAM[@]}" evaluate --checkpoint "$run/best.pt" --mc-passes 20 --tta \
    --out "$run/uncertainty_report_tta.json"
done
echo "All experiments done. Rebuild the results with: bash scripts/reproduce_results.sh"
