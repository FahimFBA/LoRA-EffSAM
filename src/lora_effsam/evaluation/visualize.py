# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Save a figure of input / ground truth / prediction / MC-Dropout uncertainty for a few
test images (local replacement for the Colab notebook's visualization cell).

    python visualize.py --data-root data --checkpoint runs/best.pt --num 4 --out runs/samples.png
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

import matplotlib
matplotlib.use("Agg")  # file output only; no GUI backend needed
import matplotlib.pyplot as plt
import torch

from lora_effsam.data.dataset import BrainTumorSegDataset, TUMOR_CLASSES
from lora_effsam.models.segmenter import build_model_from_checkpoint
from lora_effsam.uncertainty.mc_dropout import mc_dropout_predict
from lora_effsam.utils import resolve_device, dice_score
from lora_effsam.paths import data_dir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(data_dir()))
    p.add_argument("--checkpoint", required=True, help="Checkpoint written by train.py")
    p.add_argument("--num", type=int, default=4, help="Number of test images to show")
    p.add_argument("--start", type=int, default=0, help="Index of the first test image")
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--out", default="samples.png")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    model, ckpt = build_model_from_checkpoint(args.checkpoint, device)
    split = ckpt.get("split", {})

    full_ds = BrainTumorSegDataset(args.data_root, image_size=model.image_size, to_rgb=True)
    _, _, test_ds = full_ds.split(split.get("val_split", 0.15), split.get("test_split", 0.15),
                                  seed=split.get("seed", 42))
    idxs = range(args.start, min(args.start + args.num, len(test_ds)))

    fig, axes = plt.subplots(len(idxs), 4, figsize=(16, 4 * len(idxs)), squeeze=False)
    for row, i in enumerate(idxs):
        img, mask, label = test_ds[i]
        res = mc_dropout_predict(model, img.unsqueeze(0).to(device), K=args.mc_passes)
        d = dice_score(res.pred_mask.cpu(), mask.unsqueeze(0)).item()

        ax = axes[row]
        ax[0].imshow(img[0], cmap="gray"); ax[0].set_title(f"Input ({TUMOR_CLASSES[label]})")
        ax[1].imshow(mask[0], cmap="gray"); ax[1].set_title("Ground truth")
        ax[2].imshow(res.pred_mask[0, 0].cpu(), cmap="gray"); ax[2].set_title(f"Prediction (Dice {d:.3f})")
        im = ax[3].imshow(res.uncertainty[0, 0].cpu(), cmap="inferno"); ax[3].set_title("Uncertainty (U_i)")
        fig.colorbar(im, ax=ax[3], fraction=0.046)
        for a in ax:
            a.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
