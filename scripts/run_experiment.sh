#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Run one named experiment: train -> MC-Dropout eval -> figure -> update runs/summary.{csv,md}.
#
#   bash scripts/run_experiment.sh <name> [lora-effsam train args...]
#   bash scripts/run_experiment.sh img512_intaug --image-size 512 --encoder-size 512 --intensity-aug
#
# Everything lands in runs/<name>/ (config.json, metrics.csv per epoch, test.json,
# uncertainty_report.json, samples.png, checkpoints) plus runs/<name>.train.log.
# A name that already finished is never overwritten; re-running an interrupted one
# resumes it (same args required).
set -euo pipefail
source "$(dirname "$0")/_env.sh"

if [ $# -lt 1 ]; then
  echo "usage: bash scripts/run_experiment.sh <name> [lora-effsam train args...]" >&2
  exit 1
fi
NAME="$1"; shift
RUN_DIR="$RUNS/$NAME"

if [ -f "$RUN_DIR/.train_done" ]; then
  echo "$RUN_DIR already trained; pick a new name for a new experiment." >&2
else
  mkdir -p "$RUN_DIR"
  RESUME=()
  if [ -f "$RUN_DIR/last.pt" ]; then RESUME=(--resume); echo "resuming from $RUN_DIR/last.pt"; fi
  "${LORA_EFFSAM[@]}" train --checkpoint-dir "$RUN_DIR" \
    --epochs 60 --early-stop-patience 15 "$@" \
    ${RESUME[@]+"${RESUME[@]}"} 2>&1 | tee -a "$RUN_DIR.train.log"
  touch "$RUN_DIR/.train_done"
fi

if [ ! -f "$RUN_DIR/uncertainty_report.json" ]; then
  "${LORA_EFFSAM[@]}" evaluate --checkpoint "$RUN_DIR/best.pt" \
    --mc-passes 20 --out "$RUN_DIR/uncertainty_report.json"
fi
if [ ! -f "$RUN_DIR/samples.png" ]; then
  "${LORA_EFFSAM[@]}" visualize --checkpoint "$RUN_DIR/best.pt" --num 4 --out "$RUN_DIR/samples.png"
fi
"${LORA_EFFSAM[@]}" summarize
echo "Done. Results in $RUN_DIR/"
