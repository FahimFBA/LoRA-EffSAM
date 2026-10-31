# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 3: run inference-time MC-Dropout uncertainty analysis on a trained checkpoint.

    python evaluate_uncertainty.py --data-root /path/to/data \
        --checkpoint runs/best.pt --mc-passes 20

Model settings (encoder size, LoRA rank, dropout rate, attention toggles, ...) and the
train/val/test split settings are read from the checkpoint itself, so they always
match training. The model-shape flags below are only used for old checkpoints that
were saved as a bare state_dict.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json

import torch
from torch.utils.data import DataLoader

from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.uncertainty.mc_dropout import full_dataset_uncertainty_report
from lora_effsam.utils import set_seed, resolve_device
from lora_effsam.paths import data_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(data_dir()))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--mc-passes", type=int, default=20, help="K")
    p.add_argument("--small-lesion-threshold", type=float, default=None,
                    help="T (defaults to the value used in training)")
    p.add_argument("--boundary-band-px", type=int, default=3)
    p.add_argument("--tta", action="store_true",
                   help="flip test-time augmentation inside the MC passes (see mc_dropout_predict)")
    p.add_argument("--max-samples", type=int, default=0,
                    help="Evaluate a random subset of N test images (0 = all)")
    p.add_argument("--out", type=str, default="uncertainty_report.json")
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | mps | cpu")
    # Legacy (bare state_dict checkpoints only):
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--encoder-size", type=int, default=1024)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--lora-also-proj", action="store_true")
    p.add_argument("--variant", type=str, default="vitt", choices=["vitt", "vits"])
    p.add_argument("--checkpoint-path", type=str, default=None,
                    help="EfficientSAM weights path (NOT the trained --checkpoint above)")
    p.add_argument("--no-feature-attention", action="store_true")
    p.add_argument("--no-refine-head", action="store_true")
    p.add_argument("--mc-dropout-p", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)

    model, ckpt = build_model_from_checkpoint(
        args.checkpoint, device,
        image_size=args.image_size, variant=args.variant,
        checkpoint_path=args.checkpoint_path,
        lora_r=args.rank, lora_alpha=args.alpha or args.rank, lora_also_proj=args.lora_also_proj,
        use_feature_attention=not args.no_feature_attention,
        use_refine_head=not args.no_refine_head,
        mc_dropout_p=args.mc_dropout_p, encoder_size=args.encoder_size,
    )
    split = ckpt.get("split", {})
    seed = split.get("seed", args.seed)
    T = args.small_lesion_threshold or split.get("small_lesion_threshold", 100.0)
    set_seed(seed)
    print(f"Device: {device} | model: { {k: v for k, v in ckpt.get('model_args', {}).items() if k != 'checkpoint_path'} }")

    full_ds = BrainTumorSegDataset(args.data_root, image_size=model.image_size, to_rgb=True)
    _, _, test_ds = full_ds.split(val_frac=split.get("val_split", 0.15),
                                  test_frac=split.get("test_split", 0.15), seed=seed)
    test_ds = test_ds.subsample(args.max_samples, seed=seed)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test images: {len(test_ds)} | MC passes K={args.mc_passes}")

    report = full_dataset_uncertainty_report(
        model, loader, K=args.mc_passes, T=T,
        band_px=args.boundary_band_px, device=device, tta=args.tta,
    )

    # shuffle=False, so records are in test_ds.samples order: tag each with its image
    # id so per-image results can be paired across runs.
    assert len(report["per_image"]) == len(test_ds.samples)
    for rec, s in zip(report["per_image"], test_ds.samples):
        assert rec["label"] == s[2], "per-image records out of step with test samples"
        rec["id"] = test_ds._sample_id(s)
        rec["class"] = test_ds.classes[s[2]]

    strat = report["size_stratified"]
    print(f"Test Dice (MC-mean prediction): {report['mc_mean_dice']:.4f}")
    print("Size-stratified boundary uncertainty:")
    print(f"  Small lesions (A_GT < {T:.0f}): "
          f"U_boundary = {strat['U_small']:.4f}  (n={strat['n_small']})")
    print(f"  Large lesions (A_GT >= {T:.0f}): "
          f"U_boundary = {strat['U_large']:.4f}  (n={strat['n_large']})")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full per-image report written to {args.out}")


if __name__ == "__main__":
    main()
