# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Shared fixtures/markers. Heavy tests skip themselves when their requirements are missing:

* ``needs_cuda``          a CUDA GPU
* ``needs_efficientsam``  the EfficientSAM checkout (``scripts/fetch_efficientsam.sh``)
* ``needs_data``          the processed dataset (``scripts/setup_data.sh``)
* ``needs_final_runs``    the final model's runs incl. checkpoints (``scripts/train_final.sh``)
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from lora_effsam import paths  # noqa: E402

FINAL = "img512_intaug_vits_hybrid_vsds"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
needs_efficientsam = pytest.mark.skipif(
    not (paths.efficientsam_dir() / "weights" / "efficient_sam_vitt.pt").is_file(),
    reason="needs EfficientSAM (bash scripts/fetch_efficientsam.sh)")
needs_data = pytest.mark.skipif(not (paths.data_dir() / "metadata.csv").is_file(),
                                reason="needs the dataset (bash scripts/setup_data.sh)")
needs_final_runs = pytest.mark.skipif(
    not all((paths.runs_dir() / r / "best.pt").is_file() for r in (FINAL, FINAL + "_s1", FINAL + "_s2")),
    reason="needs the final model's runs (bash scripts/train_final.sh)")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
