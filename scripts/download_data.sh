#!/usr/bin/env bash
# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Download the raw Cheng/Figshare brain tumor dataset (.mat, ~830MB) and unpack it to
# <OUT_DIR>/data/1.mat ... 3064.mat.
#
#   bash scripts/download_data.sh [OUT_DIR]      # default OUT_DIR=data/raw_mat
#
# Then convert to PNG + metadata.csv (or use scripts/setup_data.sh for both steps):
#   lora-effsam prepare-data --mat-dir data/raw_mat/data --out-dir data/figshare_brain_tumor
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUT="${1:-$ROOT/data/raw_mat}"
URL="https://ndownloader.figshare.com/articles/1512427/versions/5"
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

mkdir -p "$OUT"
cd "$OUT"

if [ "$(ls data/*.mat 2>/dev/null | wc -l | tr -d ' ')" = "3064" ]; then
  echo "$OUT/data already has 3064 .mat files, nothing to do."
  exit 0
fi

if [ ! -f parts.zip ] || [ "$(wc -c < parts.zip | tr -d ' ')" -lt 500000000 ]; then
  # figshare does not support resuming (HTTP range requests), so a partial file
  # from an interrupted download is useless: start over.
  rm -f parts.zip
  echo "Downloading (~830MB)..."
  # Browser User-Agent avoids figshare 403s.
  curl -L --fail -A "$UA" -o parts.zip "$URL"
fi
SIZE=$(wc -c < parts.zip | tr -d ' ')
if [ "$SIZE" -lt 500000000 ]; then
  echo "ERROR: parts.zip is only $SIZE bytes (expected ~830MB). Open $URL in a browser," >&2
  echo "save the file as $OUT/parts.zip, and re-run this script." >&2
  exit 1
fi

echo "Extracting outer archive..."
unzip -o -q parts.zip
PARTS=(brainTumorDataPublic_*)
if [ "${#PARTS[@]}" -ne 4 ]; then
  echo "ERROR: expected 4 brainTumorDataPublic_* parts, found ${#PARTS[@]}: ${PARTS[*]}" >&2
  exit 1
fi

mkdir -p data
# The parts usually unzip independently; if not, they're fragments of one archive
# that `zip -FF` can repair after concatenation.
ok=1
for p in "${PARTS[@]}"; do
  unzip -o -q "$p" -d data || ok=0
done
if [ "$ok" -ne 1 ]; then
  echo "Per-part unzip failed; repairing concatenated archive with zip -FF..."
  cat brainTumorDataPublic_* > temp.zip
  yes | zip -FF temp.zip --out data.zip
  unzip -o -q data.zip -d data
  rm -f temp.zip data.zip
fi

N=$(ls data/*.mat 2>/dev/null | wc -l | tr -d ' ')
echo "$N .mat files in $OUT/data"
if [ "$N" != "3064" ]; then
  echo "ERROR: expected 3064 .mat files." >&2
  exit 1
fi
rm -f parts.zip brainTumorDataPublic_*
echo "Done."
