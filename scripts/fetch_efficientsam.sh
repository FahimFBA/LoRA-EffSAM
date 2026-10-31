#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Fetch the official EfficientSAM code + pretrained weights (ViT-T and zipped ViT-S)
# into $LORA_EFFSAM_EFFICIENTSAM_DIR (default third_party/EfficientSAM, gitignored).
#
#   bash scripts/fetch_efficientsam.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${LORA_EFFSAM_EFFICIENTSAM_DIR:-$ROOT/third_party/EfficientSAM}"

if [ -f "$DEST/weights/efficient_sam_vitt.pt" ]; then
  echo "EfficientSAM already present in $DEST"
else
  rm -rf "$DEST"
  git clone --depth 1 https://github.com/yformer/EfficientSAM.git "$DEST"
fi
# The ViT-S weights ship zipped; unzip once (the loader also does this on first use).
if [ ! -f "$DEST/weights/efficient_sam_vits.pt" ] && [ -f "$DEST/weights/efficient_sam_vits.pt.zip" ]; then
  (cd "$DEST/weights" && unzip -o -q efficient_sam_vits.pt.zip efficient_sam_vits.pt)
fi
ls -lh "$DEST/weights"/*.pt
