# 6. Limitations and notes for writing

## 6.1 Limitations to state in the paper

1. **The final model has one seed.** Configurations A–C have 3 seeds (SD 0.003–0.006).
   The final model's gain over C (+0.017 on seed 42, p < 0.0001 per-image) is about 3–5×
   that SD, so it is unlikely to be seed noise, but it has no seed-level error bar.
   Either say so explicitly or train seeds 1 and 2 (~5 h) before submission.
2. **The architecture components were not separated.** The hybrid decoder alone (ViT-T)
   did not help (−0.004); the full combination (ViT-S + hybrid + ViT skips + deep
   supervision) did (+0.017). Whether the gain comes mostly from ViT-S, from the ViT
   skips + deep supervision, or from their combination is not known. Write it as one
   combined change.
3. **The final model was still improving** (best epoch 55 of 60). A longer schedule may
   give more; the reported number is not a converged ceiling.
4. **Single dataset, single split.** One public dataset, one patient-level split (no
   cross-validation, no external test set). The test set has 34 patients; patient-level
   variance is not estimated.
5. **2-D slices only.** The dataset provides individual slices, not volumes; no 3-D context.
6. **Uncertainty does not detect misses** (Results 3.5). MC-Dropout sits only in the
   decoder; failures are confidently wrong in the true-boundary band (AUROC 0.27). Claim
   boundary localisation of uncertainty, not failure detection.
7. **Size threshold.** With T = 400 px at 512×512 only 8/477 test lesions are “small”
   under the method's own definition; use the area-quartile analysis for size results.
8. **Intensity augmentation** is a modest, borderline gain (Holm p = 0.059), and does not
   show on validation. Present it as such.
9. **Hyperparameters were not tuned on the test set**, but several configurations were
   compared on it during development. All selection inside a run (checkpoint, early
   stopping) used the validation set only.

## 6.2 Metric definitions to keep consistent

- “Dice” everywhere = **mean of per-image Dice** over the 477 test slices. Do not mix it
  with pooled (all-pixel) Dice, which would be higher because large tumors dominate.
- “Test Dice” = MC-Dropout mean prediction (K = 20) unless “+ TTA” is stated. The
  deterministic prediction differs by ≤ 0.0006.
- Lesion-size quartiles are quartiles of ground-truth area on the test set at 512 px.

## 6.3 Check before submission

- Exact references for the dataset (Cheng et al. 2015, 2016), EfficientSAM (Xiong et al.,
  CVPR 2024), LoRA (Hu et al., 2022), CBAM (Woo et al., 2018), U-Net, TransUNet, UNETR,
  MC-Dropout (Gal & Ghahramani, 2016), Tversky loss, boundary loss (Kervadec et al.).
  These are named from memory here; verify titles, years and venues.
- Any comparison with other papers on this dataset: confirm each one's split
  (patient vs slice level) and Dice definition before putting numbers side by side.
- Software versions in `02_data_and_protocol.md` are the ones installed at run time.
- The EfficientSAM license (in its repository) for redistributing the fine-tuned checkpoint.
