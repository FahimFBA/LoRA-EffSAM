# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Dataset for the folder layout shown in the user's screenshots:

data/
  Meningioma/<id>.png, <id>_mask.png
  Glioma/<id>.png, <id>_mask.png
  Pituitary tumor/<id>.png, <id>_mask.png
"""
import os
import csv
import glob
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF

TUMOR_CLASSES = ("Meningioma", "Glioma", "Pituitary tumor")


def _find_pairs(class_dir: str) -> List[Tuple[str, str]]:
    """Pair <id>.png with <id>_mask.png inside one class folder."""
    pairs = []
    all_pngs = glob.glob(os.path.join(class_dir, "*.png"))
    mask_set = {os.path.basename(p) for p in all_pngs if p.endswith("_mask.png")}
    for p in all_pngs:
        base = os.path.basename(p)
        if base.endswith("_mask.png"):
            continue
        mask_name = base[:-4] + "_mask.png"
        if mask_name in mask_set:
            pairs.append((p, os.path.join(class_dir, mask_name)))
        # else: silently skip images with no matching mask (e.g. stray files)
    return sorted(pairs)


class BrainTumorSegDataset(Dataset):
    """
    Returns (image, mask, label) where:
      image: FloatTensor [1, H, W] in [0, 1]  (grayscale MRI; set to 3ch if your
             encoder expects RGB — see `to_rgb`)
      mask:  FloatTensor [1, H, W] in {0., 1.}
      label: int index into TUMOR_CLASSES (kept for stratified analysis / optional
             multi-class extension; not used by the binary loss by default)
    """

    def __init__(
        self,
        root: str,
        classes: Tuple[str, ...] = TUMOR_CLASSES,
        image_size: int = 512,
        to_rgb: bool = True,
        augment: bool = False,
        file_list: Optional[List[Tuple[str, str, int]]] = None,
    ):
        self.root = root
        self.classes = classes
        self.image_size = image_size
        self.to_rgb = to_rgb
        self.augment = augment
        self.intensity_aug = False  # set on the train split by train.py --intensity-aug

        # Optional metadata.csv (id, class, patient_id) written by
        # scripts/mat_to_dataset.py. Enables patient-level (leakage-free) splits.
        # Without it, `.split()` falls back to a per-image split and warns loudly,
        # since slices from the same patient can otherwise land in both train and
        # test and inflate validation/test Dice.
        self._pid_by_sample_id: Dict[str, str] = {}
        meta_path = os.path.join(root, "metadata.csv")
        if os.path.isfile(meta_path):
            with open(meta_path, newline="") as f:
                for row in csv.DictReader(f):
                    self._pid_by_sample_id[row["id"]] = str(row["patient_id"]).strip()
        self.has_patient_ids = len(self._pid_by_sample_id) > 0

        if file_list is not None:
            self.samples = file_list
        else:
            self.samples = []
            for label, cls in enumerate(classes):
                class_dir = os.path.join(root, cls)
                if not os.path.isdir(class_dir):
                    raise FileNotFoundError(
                        f"Expected class folder not found: {class_dir}. "
                        f"Check --data-root points at the folder containing "
                        f"{list(classes)}."
                    )
                for img_path, mask_path in _find_pairs(class_dir):
                    self.samples.append((img_path, mask_path, label))
            if not self.samples:
                raise RuntimeError(f"No (image, mask) pairs found under {root}")

    def __len__(self):
        return len(self.samples)

    def _sample_id(self, sample) -> str:
        img_path = sample[0]
        return os.path.splitext(os.path.basename(img_path))[0]

    def split(self, val_frac: float, test_frac: float, seed: int = 42):
        """
        Splits into train/val/test. Uses a PATIENT-LEVEL split (all slices from one
        patient go to the same split) if metadata.csv with patient IDs is present
        (see scripts/mat_to_dataset.py) — this is required to avoid leakage on
        datasets with multiple slices per patient, such as the Cheng/Figshare brain
        tumor dataset. Falls back to a per-image stratified split otherwise, with a
        loud warning, since that fallback CAN leak information across splits.
        """
        if self.has_patient_ids:
            return self.split_by_patient(val_frac, test_frac, seed)

        import warnings
        warnings.warn(
            "No metadata.csv with patient IDs found under "
            f"{self.root} — falling back to a PER-IMAGE split. If your dataset has "
            "multiple slices per patient (e.g. the Cheng/Figshare brain tumor "
            "dataset), this can leak near-duplicate slices across train/val/test "
            "and inflate your reported Dice score. Re-generate the dataset with "
            "scripts/mat_to_dataset.py to get patient-level splits instead.",
            stacklevel=2,
        )
        return self._split_by_image(val_frac, test_frac, seed)

    def _split_by_image(self, val_frac: float, test_frac: float, seed: int = 42):
        """Stratified split by tumor class only, ignoring patient identity."""
        rng = np.random.default_rng(seed)
        by_class = {i: [] for i in range(len(self.classes))}
        for s in self.samples:
            by_class[s[2]].append(s)

        train, val, test = [], [], []
        for label, items in by_class.items():
            items = items.copy()
            rng.shuffle(items)
            n = len(items)
            n_test = int(round(n * test_frac))
            n_val = int(round(n * val_frac))
            test += items[:n_test]
            val += items[n_test:n_test + n_val]
            train += items[n_test + n_val:]

        return self._make_subsets(train, val, test)

    def split_by_patient(self, val_frac: float, test_frac: float, seed: int = 42):
        """
        Group-aware, class-stratified split: every slice from one patient goes
        entirely into train, val, or test — never split across them. Stratifies at
        the PATIENT level by that patient's majority tumor class, so all three
        classes are still represented in each split.
        """
        if not self.has_patient_ids:
            raise RuntimeError(
                "split_by_patient() requires metadata.csv with patient IDs "
                "(see scripts/mat_to_dataset.py) — none found under this root."
            )

        rng = np.random.default_rng(seed)

        # Group sample indices by patient, and by (patient's majority class).
        patient_samples: Dict[str, list] = {}
        for s in self.samples:
            sid = self._sample_id(s)
            pid = self._pid_by_sample_id.get(sid)
            if pid is None:
                raise RuntimeError(
                    f"Sample id {sid} has no patient_id entry in metadata.csv."
                )
            patient_samples.setdefault(pid, []).append(s)

        # Assign each patient to the class they have the most slices of, purely to
        # stratify the PATIENT-level split across classes.
        patient_class: Dict[str, int] = {}
        for pid, items in patient_samples.items():
            labels = [s[2] for s in items]
            patient_class[pid] = max(set(labels), key=labels.count)

        by_class_patients: Dict[int, list] = {i: [] for i in range(len(self.classes))}
        for pid, cls in patient_class.items():
            by_class_patients[cls].append(pid)

        train, val, test = [], [], []
        for cls, pids in by_class_patients.items():
            pids = pids.copy()
            rng.shuffle(pids)
            n = len(pids)
            n_test = int(round(n * test_frac))
            n_val = int(round(n * val_frac))
            test_pids = set(pids[:n_test])
            val_pids = set(pids[n_test:n_test + n_val])
            train_pids = set(pids[n_test + n_val:])

            # sorted(): set iteration order of string ids changes per process
            # (hash randomization), which made image order differ between runs.
            # Membership is unaffected; only the order is now reproducible.
            for pid in sorted(test_pids):
                test += patient_samples[pid]
            for pid in sorted(val_pids):
                val += patient_samples[pid]
            for pid in sorted(train_pids):
                train += patient_samples[pid]

        return self._make_subsets(train, val, test)

    def subsample(self, n: int, seed: int = 42) -> "BrainTumorSegDataset":
        """Random (seeded) subset of n samples, for quick smoke-test runs on slow machines."""
        if n is None or n <= 0 or n >= len(self.samples):
            return self
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(self.samples), size=n, replace=False).tolist())
        sub = BrainTumorSegDataset(self.root, self.classes, self.image_size, self.to_rgb,
                                   self.augment, file_list=[self.samples[i] for i in idx])
        return sub

    def _make_subsets(self, train, val, test):
        make = lambda lst: BrainTumorSegDataset(
            self.root, self.classes, self.image_size, self.to_rgb, False, file_list=lst
        )
        return make(train), make(val), make(test)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        img_t = TF.to_tensor(img)                       # [1, H, W] in [0,1]
        mask_np = (np.array(mask) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        if self.augment:
            img_t, mask_t = _random_flip_rotate(img_t, mask_t)
        if self.intensity_aug:
            img_t = _random_intensity(img_t)

        if self.to_rgb:
            img_t = img_t.repeat(3, 1, 1)  # [3, H, W] for ViT encoders expecting RGB

        return img_t, mask_t, label


def _random_intensity(img_t):
    """Random gamma, contrast and brightness on a [1, H, W] image in [0, 1].

    MRI intensity varies between scanners/protocols, and low-contrast lesions
    (mostly gliomas) are where the model fails, so train on that variation.
    """
    if torch.rand(1).item() < 0.5:
        gamma = float(torch.empty(1).uniform_(0.7, 1.5))
        img_t = img_t.clamp(min=0) ** gamma
    if torch.rand(1).item() < 0.5:
        c = float(torch.empty(1).uniform_(0.75, 1.25))
        img_t = (img_t - img_t.mean()) * c + img_t.mean()
    if torch.rand(1).item() < 0.5:
        img_t = img_t + float(torch.empty(1).uniform_(-0.1, 0.1))
    return img_t.clamp(0, 1)


def _random_flip_rotate(img_t, mask_t):
    if torch.rand(1).item() < 0.5:
        img_t = TF.hflip(img_t)
        mask_t = TF.hflip(mask_t)
    if torch.rand(1).item() < 0.5:
        img_t = TF.vflip(img_t)
        mask_t = TF.vflip(mask_t)
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        img_t = torch.rot90(img_t, k, dims=[1, 2])
        mask_t = torch.rot90(mask_t, k, dims=[1, 2])
    return img_t, mask_t
