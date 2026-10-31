# Test-time knock-out of components (final model, 3 seeds)

Deterministic test Dice (dropout off) of the TRAINED model with one component disabled; mean ± SD over seeds 42, 1, 2. Δ = change vs the full model. Not a training ablation.

| Knock-out | All: Dice | Δ | Glioma: Δ | Size Q1: Δ |
|---|---|---|---|---|
| Full model | 0.8187 ± 0.0029 | +0.0000 | +0.0000 | +0.0000 |
| CBAM attention off | 0.8186 ± 0.0028 | -0.0001 | +0.0000 | +0.0008 |
| Refine head off | 0.8181 ± 0.0030 | -0.0006 | -0.0005 | -0.0015 |
| LoRA off | 0.5372 ± 0.0318 | -0.2815 | -0.4410 | -0.2091 |
| ViT skips zeroed | 0.7487 ± 0.0032 | -0.0700 | -0.0978 | -0.1651 |
| CNN skips zeroed | 0.7931 ± 0.0126 | -0.0256 | -0.0123 | -0.0680 |
