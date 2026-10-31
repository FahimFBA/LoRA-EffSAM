# 6. Limitations and notes for writing

## 6.1 Limitations

1. **Three seeds only.** All main configurations (A–D) have 3 training seeds (SD
   0.003–0.006); seed-level tests have little power, so the per-image paired tests are the
   main evidence. (Resolved since the first version: the final model now has 3 seeds,
   0.819 ± 0.003.)
2. **The architecture components were not separated.** The hybrid decoder alone (ViT-T,
   1 seed) did not help (−0.004); the full combination (ViT-S + hybrid + ViT skips + deep
   supervision) did (+0.014 over 3 seeds). Whether the gain comes mostly from ViT-S, from
   the ViT skips + deep supervision, or from their combination is not known. Write it as
   one combined change.
3. **Training length.** Seeds 1 and 2 of the final model early-stopped (best epochs 41, 37);
   seed 42 peaked at epoch 55 of 60. A longer schedule is unlikely to change the result
   much, but was not tested.
4. **Single dataset, single split.** One public dataset, one patient-level split (no
   cross-validation, no external test set). The test set has 34 patients; patient-level
   variance is not estimated.
5. **2-D slices only.** The dataset provides individual slices, not volumes; no 3-D context.
6. **Uncertainty does not detect misses** (Results 3.5). MC-Dropout sits only in the
   decoder; failures are confidently wrong in the true-boundary band (AUROC 0.14–0.28 over
   3 seeds). Claim boundary localisation of uncertainty, not failure detection. The
   small-vs-large lesion uncertainty difference is significant on 2 of 3 seeds only.
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
