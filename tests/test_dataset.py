# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""Patient-level split on the real dataset: sizes, no leakage, deterministic order."""
from conftest import needs_data

from lora_effsam import paths
from lora_effsam.data.dataset import BrainTumorSegDataset


def _patients(ds):
    return {ds._pid_by_sample_id[ds._sample_id(s)] for s in ds.samples}


@needs_data
def test_split_sizes_and_no_patient_leakage():
    ds = BrainTumorSegDataset(str(paths.data_dir()), image_size=64)
    assert len(ds) == 3064
    tr, va, te = ds.split(0.15, 0.15, seed=42)
    assert (len(tr), len(va), len(te)) == (2178, 409, 477)        # the paper's split
    assert not (_patients(tr) & _patients(va) or _patients(tr) & _patients(te)
                or _patients(va) & _patients(te))
    assert len(_patients(te)) == 34


@needs_data
def test_split_order_is_reproducible():
    a = BrainTumorSegDataset(str(paths.data_dir()), image_size=64).split(0.15, 0.15, seed=42)[2]
    b = BrainTumorSegDataset(str(paths.data_dir()), image_size=64).split(0.15, 0.15, seed=42)[2]
    assert [s[0] for s in a.samples] == [s[0] for s in b.samples]
