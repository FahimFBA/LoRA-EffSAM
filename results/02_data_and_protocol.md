# 2. Data and experimental protocol

## 2.1 Dataset

Figshare brain-tumor dataset (Cheng et al.; figshare article 1512427, version 5):
3,064 T1-weighted contrast-enhanced MRI slices from **233 patients**, each with a
tumor mask and a class label. Every slice contains a tumor.

| Class | Slices |
|---|---|
| Meningioma | 708 |
| Glioma | 1,426 |
| Pituitary tumor | 930 |
| **Total** | **3,064** |

Cite the dataset's original papers (Cheng et al., PLoS ONE 2015 / 2016); check the exact
references on the figshare page before submission.

## 2.2 Preprocessing

1. `.mat` → PNG (`lora-effsam prepare-data`, `src/lora_effsam/data/mat_to_dataset.py`): per-slice min–max scaling of the raw
   intensities to [0, 255]; mask = (tumorMask > 0). The patient ID (`PID`, a MATLAB char
   array) is decoded in full and written to `metadata.csv` (id, class, patient_id).
2. At load time: resize image to 512×512 (bilinear) and mask (nearest neighbour), scale
   to [0, 1], threshold mask at 0.5, repeat the gray channel to 3 channels.

## 2.3 Patient-level split (no leakage)

All slices of a patient are in exactly one split. Patients are assigned to a class by
their majority label, shuffled per class with a fixed seed (42), and split 70/15/15 by
patient. **The same split is used for every experiment** (`--split-seed 42`), so all
runs are tested on identical images; this was verified image-by-image.

| Split | Slices | Patients | Meningioma | Glioma | Pituitary |
|---|---|---|---|---|---|
| Train | 2,178 | 165 | 463 | 1,033 | 682 |
| Validation | 409 | 34 | 125 | 171 | 113 |
| Test | 477 | 34 | 120 | 222 | 135 |

The validation set is used for early stopping and checkpoint selection only; the test
set is used once per trained model.

## 2.4 Augmentation (training only)

- Geometric: horizontal flip (p = 0.5), vertical flip (p = 0.5), rotation by k·90°,
  k ∈ {0, 1, 2, 3} uniform. Applied identically to image and mask.
- Intensity (`--intensity-aug`), each with p = 0.5: gamma x^γ with γ ~ U(0.7, 1.5);
  contrast (x − mean)·c + mean with c ~ U(0.75, 1.25); brightness shift ~ U(−0.1, 0.1);
  then clipped to [0, 1].

## 2.5 Evaluation

- **Primary metric: mean per-image Dice** on the test set, Dice = (2|P∩G| + ε)/(|P| + |G| + ε),
  ε = 1e-6, prediction thresholded at 0.5. Every test slice contains a tumor, so no
  empty-mask cases occur.
- Reported prediction: **MC-Dropout mean** of K = 20 passes (p = 0.2), thresholded at 0.5
  (“MC” columns). The deterministic (dropout-off) test Dice is also logged and agrees to
  within 0.0006 for every run (`tables/all_runs.csv`, `test_dice` vs `test_dice_mc`).
- With TTA: 4 flips cycled through the K = 20 passes (“+ TTA” columns).
- Also reported: median Dice, number of failures (Dice < 0.3), Dice per tumor class, and
  Dice per lesion-size quartile (quartiles of ground-truth area on the test set at 512 px;
  119–120 images each).
- **Statistics:** paired comparisons on the same 477 images. Wilcoxon signed-rank test
  on per-image Dice, Holm correction across the 8 main comparisons, 95% bootstrap CI
  of the mean difference (10,000 resamples over images). Configurations with 3 seeds
  are also compared with a paired t-test on per-seed mean Dice. When the two
  configurations do not share all seeds, only the shared seeds are used (only the
  hybrid-decoder-only ablation has a single seed, so it is compared against the seed-42
  runs). Subgroup tests are exploratory and not corrected.
- **Seeds:** most configurations were trained with seeds 42, 1, 2 (training randomness
  only; same split). Reported as mean ± sample SD over seeds.

## 2.6 Implementation and hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12 GB (driver 617.14), WSL2 Ubuntu on Windows 11 |
| Software | Python 3.12.14, PyTorch 2.14.0 (CUDA 12.6), torchvision 0.29.0, NumPy 2.5.2, SciPy 1.18.1 |
| Mixed precision | fp16 autocast for the network, loss in fp32 |
| Final model training time | 2.2 ± 0.2 h per seed (52–60 epochs, ~2.3 min/epoch) |
| Peak GPU memory (final model, batch 4) | ~2.6 GB |
| Inference (final model) | MC-Dropout K = 20: encoder once, decoder 20× |
