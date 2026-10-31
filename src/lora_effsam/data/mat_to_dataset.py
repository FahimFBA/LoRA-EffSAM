# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Convert the raw Cheng/Figshare brain tumor dataset (.mat, MATLAB v7.3/HDF5 format)
into the PNG folder layout this codebase expects, WHILE preserving patient ID (PID)
in a metadata CSV — this is the piece that gets lost if you only keep the exported
PNGs, and it's required for a leakage-free (patient-disjoint) train/val/test split.

Each raw file stores a struct `cjdata` with fields:
    cjdata.label       -> 1=meningioma, 2=glioma, 3=pituitary tumor
    cjdata.PID         -> patient ID
    cjdata.image       -> the MRI slice (grayscale, varying intensity range)
    cjdata.tumorMask   -> binary tumor mask
    cjdata.tumorBorder -> polygon border coords (not used here; tumorMask is
                          already the pixel-accurate mask)

Get the raw data first (~830MB, 4 zip parts on Figshare):
    bash scripts/download_data.sh            # -> raw_mat/data/1.mat ... 3064.mat

Then run:
    python3 scripts/mat_to_dataset.py --mat-dir raw_mat/data --out-dir data

Output layout (matches BrainTumorSegDataset):
    data/
      Meningioma/<id>.png, <id>_mask.png
      Glioma/<id>.png, <id>_mask.png
      Pituitary tumor/<id>.png, <id>_mask.png
      metadata.csv   <- id, class, patient_id  (THIS is what enables patient-level splits)
"""
import argparse
import csv
import glob
import os

import h5py
import numpy as np
from PIL import Image

CLASS_NAMES = {1: "Meningioma", 2: "Glioma", 3: "Pituitary tumor"}


def _read_scalar(dataset) -> float:
    """cjdata.label is stored as a 2D 1x1 double array in the HDF5 struct."""
    return float(np.array(dataset)[0, 0])


def _read_pid(dataset) -> str:
    """
    cjdata.PID is a MATLAB *char array* (e.g. '100360'), which MATLAB v7.3/HDF5
    stores as a uint16 array of character codes -- NOT a number. Reading only
    element [0, 0] returns the code of the first character (e.g. 49 for '1'), which
    merges every patient whose ID starts with the same digit into one "patient" and
    breaks the patient-level split. Decode the whole array instead. A plain numeric
    PID (1x1 double) is also handled, just in case.
    """
    arr = np.array(dataset).flatten()
    if arr.dtype == np.uint16 or arr.size > 1:
        return "".join(chr(int(c)) for c in arr if int(c) != 0).strip()
    return str(int(arr[0]))


def convert_one(mat_path: str, out_dir: str) -> dict:
    with h5py.File(mat_path, "r") as f:
        cjdata = f["cjdata"]
        label = int(_read_scalar(cjdata["label"]))
        pid = _read_pid(cjdata["PID"])
        image = np.array(cjdata["image"]).T
        mask = np.array(cjdata["tumorMask"]).T

    cls = CLASS_NAMES.get(label)
    if cls is None:
        raise ValueError(f"{mat_path}: unexpected label {label}")

    # Normalize the raw (often uint16, non-8-bit-range) intensities to 0-255 for PNG.
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    img_u8 = ((img - lo) / (hi - lo + 1e-8) * 255.0).astype(np.uint8) if hi > lo else \
        np.zeros_like(img, dtype=np.uint8)

    mask_u8 = (mask > 0).astype(np.uint8) * 255

    sample_id = os.path.splitext(os.path.basename(mat_path))[0]
    class_dir = os.path.join(out_dir, cls)
    os.makedirs(class_dir, exist_ok=True)

    Image.fromarray(img_u8).save(os.path.join(class_dir, f"{sample_id}.png"))
    Image.fromarray(mask_u8).save(os.path.join(class_dir, f"{sample_id}_mask.png"))

    return {"id": sample_id, "class": cls, "patient_id": pid}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-dir", required=True, help="Folder of raw <n>.mat files")
    ap.add_argument("--out-dir", required=True, help="Output folder (becomes --data-root)")
    args = ap.parse_args(argv)

    mat_files = sorted(glob.glob(os.path.join(args.mat_dir, "*.mat")))
    if not mat_files:
        raise SystemExit(f"No .mat files found in {args.mat_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for i, mat_path in enumerate(mat_files, 1):
        try:
            rows.append(convert_one(mat_path, args.out_dir))
        except Exception as e:
            print(f"  [skip] {mat_path}: {e}")
        if i % 200 == 0 or i == len(mat_files):
            print(f"  {i}/{len(mat_files)} converted")

    csv_path = os.path.join(args.out_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)

    n_patients = len(set(r["patient_id"] for r in rows))
    print(f"\nDone: {len(rows)} slices from {n_patients} patients.")
    if n_patients < 100:
        print("WARNING: the Figshare dataset has 233 patients; far fewer means the "
              "PID field was not decoded correctly and the patient split is unreliable.")
    print(f"Wrote {csv_path} (id, class, patient_id) — needed for patient-level splitting.")


if __name__ == "__main__":
    main()
