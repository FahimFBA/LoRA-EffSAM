# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Locations of the dataset, run outputs, EfficientSAM checkout and results package.

Each can be overridden with an environment variable, so the pipeline can run against
data or checkpoints that live elsewhere (e.g. a WSL home directory) without copying:

    LORA_EFFSAM_DATA_DIR          processed dataset (class folders + metadata.csv)
    LORA_EFFSAM_RUNS_DIR          training runs (one folder per experiment)
    LORA_EFFSAM_EFFICIENTSAM_DIR  checkout of https://github.com/yformer/EfficientSAM
    LORA_EFFSAM_RESULTS_DIR       results package

Defaults are inside the repository (all except ``results/`` are gitignored).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    """Returns the path in environment variable ``var``, else ``default``.

    :param var: Environment variable name.
    :param default: Fallback path.
    :return: The resolved path.
    :rtype: Path
    """
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def data_dir() -> Path:
    """Processed Figshare brain-tumor dataset (output of ``lora-effsam prepare-data``).

    :return: Dataset directory.
    :rtype: Path
    """
    return _env_path("LORA_EFFSAM_DATA_DIR", REPO_ROOT / "data" / "figshare_brain_tumor")


def runs_dir() -> Path:
    """Directory holding one sub-folder per training run.

    :return: Runs directory.
    :rtype: Path
    """
    return _env_path("LORA_EFFSAM_RUNS_DIR", REPO_ROOT / "runs")


def efficientsam_dir() -> Path:
    """Checkout of the official EfficientSAM repository (code + weights).

    :return: EfficientSAM directory.
    :rtype: Path
    """
    return _env_path("LORA_EFFSAM_EFFICIENTSAM_DIR", REPO_ROOT / "third_party" / "EfficientSAM")


def results_dir() -> Path:
    """Results package (tables, statistics, figures).

    :return: Results directory.
    :rtype: Path
    """
    return _env_path("LORA_EFFSAM_RESULTS_DIR", REPO_ROOT / "results")
