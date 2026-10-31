# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
# Shared setup sourced by the scripts in this folder (not meant to be run directly).
#   ROOT    repository root
#   PY      Python of the active conda/venv env (falls back to python3)
#   LORA_EFFSAM   "python -m lora_effsam.cli" with src/ on PYTHONPATH, so no `pip install` is required
#   RUNS    runs directory (LORA_EFFSAM_RUNS_DIR or <repo>/runs), same as lora_effsam.paths.runs_dir()

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
  PY="$CONDA_PREFIX/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="$(command -v python3)"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LORA_EFFSAM=("$PY" -m lora_effsam.cli)
RUNS="${LORA_EFFSAM_RUNS_DIR:-$ROOT/runs}"
