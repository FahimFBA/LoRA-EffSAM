# Explainability statistics (final model)

## Learned parameters of the attention components (init: gamma 0, gate scale 8, gate bias -1, out-conv weights 0)

| run | CBAM gamma | refine gate scale | refine gate bias | refine out-conv ‖w‖ |
|---|---|---|---|---|
| img512_intaug_vits_hybrid_vsds | 0.0743 | 8.272 | -0.736 | 0.7626 |
| img512_intaug_vits_hybrid_vsds_s1 | 0.0758 | 8.320 | -0.674 | 0.7374 |
| img512_intaug_vits_hybrid_vsds_s2 | 0.0624 | 8.304 | -0.714 | 0.7458 |

## CBAM spatial attention, test set (seed 42)

- images: 477
- mean attention inside the ground-truth tumor: 0.6720; outside: 0.6733
- median per-image ratio inside/outside: 0.996
- fraction of images with higher attention inside the tumor: 0.457

## Cases in fig_explainability_maps

| case | id | Dice | attention inside | attention outside |
|---|---|---|---|---|
| Meningioma, median case | 552 | 0.944 | 0.6717 | 0.6691 |
| Glioma, median case | 2038 | 0.849 | 0.6632 | 0.6750 |
| Pituitary tumor, median case | 1732 | 0.906 | 0.6569 | 0.6675 |
| Smallest-size quartile, median case | 1789 | 0.867 | 0.6950 | 0.6708 |
| Worst test case | 2078 | 0.000 | 0.6877 | 0.6749 |
