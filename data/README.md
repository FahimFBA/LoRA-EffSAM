<!--
Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
-->

# Data

Figshare brain-tumor dataset (Cheng et al.; https://figshare.com/articles/dataset/brain_tumor_dataset/1512427,
version 5): 3,064 T1-weighted contrast-enhanced MRI slices from 233 patients
(meningioma 708, glioma 1,426, pituitary tumor 930), each with a tumor mask.

The dataset is public and **not committed** (both folders below are gitignored).
Create it with:

```bash
bash scripts/setup_data.sh
```

which runs `scripts/download_data.sh` (~830 MB) and then

```bash
lora-effsam prepare-data --mat-dir data/raw_mat/data --out-dir data/figshare_brain_tumor
```

```
raw_mat/data/<n>.mat              # original MATLAB files (can be deleted after conversion)
figshare_brain_tumor/
  Meningioma/<id>.png, <id>_mask.png
  Glioma/<id>.png, <id>_mask.png
  Pituitary tumor/<id>.png, <id>_mask.png
  metadata.csv                    # id, class, patient_id -> patient-level (leakage-free) split
```

Expected output: "3064 slices from 233 patients". The patient-level split used in every
experiment (train 2,178 / val 409 / test 477 slices; 165 / 34 / 34 patients) is computed
from `metadata.csv` by `BrainTumorSegDataset.split(0.15, 0.15, seed=42)` and checked in
`tests/test_dataset.py`. Use another location with `LORA_EFFSAM_DATA_DIR`.
