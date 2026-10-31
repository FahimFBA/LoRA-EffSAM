# MC-Dropout uncertainty analysis (K=20, p=0.2, no TTA)

U = per-pixel variance of the K sigmoid outputs. Boundary band = 3 px either side of the ground-truth contour. Small/large = smallest/largest quartile of lesion area. AUROC: U_boundary as a score for detecting failed segmentations (Dice < 0.5).

| metric | img512_intaug_vits_hybrid_vsds | img512_intaug_vits_hybrid_vsds_s1 | img512_intaug_vits_hybrid_vsds_s2 | img512_intaug | enc512 |
|---|---|---|---|---|---|
| n | 477 | 477 | 477 | 477 | 477 |
| mean_dice | 0.8190 | 0.8215 | 0.8155 | 0.8027 | 0.7832 |
| median_dice | 0.9034 | 0.8990 | 0.9015 | 0.8921 | 0.8825 |
| U_boundary | 0.0026 | 0.0019 | 0.0017 | 0.0068 | 0.0039 |
| U_interior | 9.79e-04 | 3.88e-04 | 5.68e-04 | 0.0024 | 0.0016 |
| U_background | 3.56e-05 | 1.40e-05 | 1.88e-05 | 6.54e-05 | 5.83e-05 |
| boundary_gt_interior | 429/477 | 430/477 | 432/477 | 426/477 | 422/473 |
| boundary_vs_interior_p | 7.26e-62 | 4.02e-64 | 6.06e-63 | 1.45e-52 | 3.77e-51 |
| U_boundary_smallQ1 | 0.0038 | 0.0022 | 0.0022 | 0.0104 | 0.0052 |
| U_boundary_largeQ4 | 0.0021 | 0.0017 | 0.0016 | 0.0046 | 0.0029 |
| small_vs_large_p | 0.0478 | 4.57e-05 | 0.2003 | 3.22e-17 | 1.71e-14 |
| spearman_Uboundary_dice | 0.2248 | 0.4242 | 0.2411 | 0.0415 | 0.0638 |
| spearman_p | 7.06e-07 | 2.94e-22 | 9.84e-08 | 0.3659 | 0.1661 |
| n_fail_lt0_5 | 40 | 36 | 42 | 50 | 57 |
| auroc_Uboundary_flags_failure | 0.2684 | 0.1401 | 0.2835 | 0.4382 | 0.3137 |

Size stratification with the training threshold T (as in the method's Eq. 10-11):

- img512_intaug_vits_hybrid_vsds: {'U_small': 0.00883738232369069, 'U_large': 0.002443753130095881, 'n_small': 8, 'n_large': 469}
- img512_intaug_vits_hybrid_vsds_s1: {'U_small': 0.0018699908570802373, 'U_large': 0.0019220240765354046, 'n_small': 8, 'n_large': 469}
- img512_intaug_vits_hybrid_vsds_s2: {'U_small': 0.002791529186777111, 'U_large': 0.0016548469642472428, 'n_small': 8, 'n_large': 469}
- img512_intaug: {'U_small': 0.013581446488406392, 'U_large': 0.00664374563491178, 'n_small': 8, 'n_large': 469}
- enc512: {'U_small': 0.0058340399993172575, 'U_large': 0.0038168161999710165, 'n_small': 8, 'n_large': 469}
