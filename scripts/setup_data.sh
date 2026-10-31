#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Download + convert the dataset in one step:
#   data/raw_mat/data/*.mat  ->  $LORA_EFFSAM_DATA_DIR (default data/figshare_brain_tumor)
#
#   bash scripts/setup_data.sh
set -euo pipefail
source "$(dirname "$0")/_env.sh"

OUT="${LORA_EFFSAM_DATA_DIR:-$ROOT/data/figshare_brain_tumor}"
if [ -f "$OUT/metadata.csv" ]; then
  echo "$OUT already prepared ($(($(wc -l < "$OUT/metadata.csv") - 1)) slices)."
  exit 0
fi
bash "$ROOT/scripts/download_data.sh" "$ROOT/data/raw_mat"
"${LORA_EFFSAM[@]}" prepare-data --mat-dir "$ROOT/data/raw_mat/data" --out-dir "$OUT"
echo "Dataset ready in $OUT (the raw .mat files in data/raw_mat can be deleted)."
