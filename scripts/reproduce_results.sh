#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Rebuild every table, statistic and figure in results/ from the runs (inference only,
# ~20 min on one RTX 3060). Needs the runs (incl. checkpoints of the final model's 3
# seeds), the dataset and EfficientSAM.
#
#   bash scripts/reproduce_results.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"${LORA_EFFSAM[@]}" summarize
cp "$RUNS"/summary.csv "$RUNS"/summary.md "$RUNS"/summary_by_config.md "$ROOT/results/tables/"
"${LORA_EFFSAM[@]}" export-tables
"${LORA_EFFSAM[@]}" paper-stats
"${LORA_EFFSAM[@]}" uncertainty-stats
"${LORA_EFFSAM[@]}" qualitative-figure
"${LORA_EFFSAM[@]}" explainability
"${LORA_EFFSAM[@]}" knockout
"${LORA_EFFSAM[@]}" failure-check
echo "results/ rebuilt."
