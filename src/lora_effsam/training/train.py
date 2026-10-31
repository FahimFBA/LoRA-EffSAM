# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 1 + Step 2 (+ attention additions): train the LoRA-adapted, real-EfficientSAM
encoder + attention-augmented decoder with the size-adaptive boundary-aware loss.

    lora-effsam train --checkpoint-dir runs/<name> --rank 4 --epochs 50

Runs on CUDA, Apple Silicon (MPS) or CPU; `--device auto` (default) picks the best
available. See README "Running locally on an Apple-Silicon Mac" for M1/M2 presets.

Requires third_party/EfficientSAM (see README) — this script does not fall back to
a toy backbone; if the real encoder can't be loaded, it errors out with a clear
message rather than silently training something else.
"""
import os

# Must be set before torch initializes MPS: any op without an MPS kernel falls back
# to CPU instead of crashing the run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
import sys
import time

import torch
from torch.utils.data import DataLoader

from lora_effsam.config import TrainConfig
from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.models.segmenter import LoRAMedSegmenter, build_model_from_checkpoint
from lora_effsam.losses.size_adaptive_loss import SizeAdaptiveBoundaryLoss
from lora_effsam.utils import (set_seed, dice_score, resolve_device, empty_device_cache,
                         batch_boundary_distance_maps)
from lora_effsam.paths import data_dir, runs_dir


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(data_dir()),
                    help="Folder containing Meningioma/, Glioma/, Pituitary tumor/")
    p.add_argument("--image-size", type=int, default=512,
                    help="Target resolution for masks/loss/metrics. The encoder "
                         "resizes to --encoder-size internally regardless of this.")
    p.add_argument("--encoder-size", type=int, default=1024,
                    help="Resolution the EfficientSAM ViT runs at (multiple of 16). "
                         "1024 = its pretraining size (4096 tokens). 512 (1024 tokens) "
                         "needs ~16x less attention memory and runs ~4x faster -- "
                         "recommended on 8GB Apple-Silicon machines.")
    p.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=None,
                    help="Recompute encoder activations during backward to save memory. "
                         "Defaults ON for mps/cpu, OFF for cuda.")
    p.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    p.add_argument("--batch-size", type=int, default=2,
                    help="PHYSICAL batch size actually run through the device at once. "
                         "Use --grad-accum-steps to reach a larger EFFECTIVE batch size "
                         "without raising memory.")
    p.add_argument("--grad-accum-steps", type=int, default=4,
                    help="Accumulate gradients over this many physical batches before "
                         "each optimizer step. Effective batch size = batch_size * "
                         "grad_accum_steps.")
    p.add_argument("--amp", dest="amp", action="store_true", default=None,
                    help="Mixed precision (fp16 autocast + GradScaler). CUDA only; "
                         "defaults on for cuda. Ignored on mps/cpu.")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                    help="Disable mixed precision even on cuda.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    p.add_argument("--early-stop-patience", type=int, default=10,
                    help="Stop if val Dice hasn't improved in this many epochs (0 disables)")
    p.add_argument("--rank", type=int, default=4, help="LoRA rank r (4 or 8)")
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha (defaults to rank)")
    p.add_argument("--lora-also-proj", action="store_true",
                    help="Ablation: also LoRA-adapt the attention output projection")
    p.add_argument("--variant", type=str, default="vitt", choices=["vitt", "vits"])
    p.add_argument("--decoder", type=str, default="simple", choices=["simple", "hybrid"],
                   help="simple: conv upsampling from the 1/16 ViT map; hybrid: U-Net style "
                        "decoder with CNN skip connections (models/hybrid_decoder.py)")
    p.add_argument("--vit-skips", action="store_true",
                   help="hybrid decoder: also feed ViT blocks at 1/4, 1/2, 3/4 depth into the decoder")
    p.add_argument("--deep-supervision", action="store_true",
                   help="hybrid decoder: extra loss terms on the 1/2, 1/4, 1/8 decoder levels")
    p.add_argument("--ds-weights", type=str, default="0.5,0.25,0.125",
                   help="deep-supervision loss weights for the 1/2, 1/4, 1/8 levels")
    p.add_argument("--lora-mlp", action="store_true",
                   help="ablation: also put LoRA on each block's MLP (fc1, fc2)")
    p.add_argument("--checkpoint-path", type=str, default=None,
                    help="Path to efficient_sam_<variant>.pt (defaults to "
                         "third_party/EfficientSAM/weights/efficient_sam_<variant>.pt)")
    p.add_argument("--no-feature-attention", action="store_true",
                    help="Disable the CBAM-style feature attention (ablation)")
    p.add_argument("--no-refine-head", action="store_true",
                    help="Disable the boundary-attention refinement head (ablation)")
    p.add_argument("--small-lesion-threshold", type=float, default=100.0,
                    help="T: pixel-area threshold separating small vs large lesions")
    p.add_argument("--lambda-t", type=float, default=1.0)
    p.add_argument("--lambda-b", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=20.0,
                   help="size-gate temperature in pixels (scale with area, i.e. x4 at 2x resolution)")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="max boundary-loss multiplier for small lesions")
    p.add_argument("--checkpoint-dir", type=str, default=str(runs_dir() / "run"))
    p.add_argument("--resume", action="store_true",
                    help="Continue from <checkpoint-dir>/last.pt (model, optimizer, "
                         "scheduler, epoch, early-stopping state)")
    p.add_argument("--max-train-samples", type=int, default=0,
                    help="Use a random subset of N training images (0 = all). For smoke tests.")
    p.add_argument("--max-eval-samples", type=int, default=0,
                    help="Use a random subset of N val/test images (0 = all).")
    p.add_argument("--intensity-aug", action="store_true",
                   help="random gamma/contrast/brightness on training images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=None,
                   help="seed for the patient-level train/val/test split (defaults to --seed); "
                        "fix it to repeat training with new seeds on the same test set")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto",
                    help="auto | cuda | mps | cpu  (auto: cuda > mps > cpu)")
    return p.parse_args(argv)


def run_epoch(model, loader, loss_fn, optimizer, device, train: bool,
              grad_accum_steps: int = 1, use_amp: bool = False, scaler=None,
              log_every: int = 50, aux_weights=()):
    """aux_weights: deep-supervision loss weights for the model's aux_logits (training
    only). The returned/logged loss is the main loss alone, so train and val losses
    stay comparable; the auxiliary terms only enter the backward pass."""
    if len(loader.dataset) == 0:
        # Can happen with a very small dataset + a class that has too few
        # patients for the requested val/test fraction to round up to >=1.
        import warnings
        warnings.warn(
            f"{'Train' if train else 'Eval'} loader is empty this epoch "
            "(0 samples) — check --val-split/--test-split against your patient "
            "count per class. Returning NaN for this split.",
            stacklevel=2,
        )
        return float("nan"), float("nan")

    model.train(mode=train)
    total_loss, total_dice, n = 0.0, 0.0, 0
    t0 = time.time()

    if train:
        optimizer.zero_grad()

    for step, (images, masks, _labels) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                logits = model(images)
            logits = logits.float()  # loss math (sums, divisions) in fp32 for stability
            aux = getattr(model, "aux_logits", None) if train and aux_weights else None
            dist_map = batch_boundary_distance_maps(masks) if aux else None  # shared
            loss, parts = loss_fn(logits, masks, dist_map)

            if train:
                full_loss = loss
                if aux:
                    for w, a in zip(aux_weights, aux):
                        full_loss = full_loss + w * loss_fn(a.float(), masks, dist_map)[0]
                scaled_loss = full_loss / grad_accum_steps
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_last_batch = (step + 1) == len(loader)
                if (step + 1) % grad_accum_steps == 0 or is_last_batch:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

        with torch.no_grad():
            prob = torch.sigmoid(logits)
            d = dice_score((prob >= 0.5).float(), masks).mean().item()

        bs = images.shape[0]
        total_loss += loss.item() * bs
        total_dice += d * bs
        n += bs

        if log_every and (step + 1) % log_every == 0:
            rate = (time.time() - t0) / (step + 1)
            eta = rate * (len(loader) - step - 1)
            print(f"    {'train' if train else 'eval '} {step + 1}/{len(loader)} "
                  f"loss={total_loss / n:.4f} dice={total_dice / n:.4f} "
                  f"({rate:.2f}s/batch, ~{eta / 60:.1f} min left)", flush=True)

    return total_loss / n, total_dice / n


def _atomic_save(obj, path):
    # Write-then-rename so a crash / Ctrl-C mid-save never corrupts the checkpoint.
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


NEW_MODEL_DEFAULTS = dict(decoder="simple", vit_skips=False, deep_supervision=False,
                          lora_mlp=False)

METRICS_FIELDS = ["epoch", "lr", "train_loss", "train_dice", "val_loss", "val_dice",
                  "epoch_minutes", "is_best", "timestamp"]


def _append_metrics(path, row):
    # One row per epoch, flushed immediately, so the full learning curve survives
    # crashes and --resume (rows are appended, never rewritten).
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = resolve_device(args.device)

    cfg = TrainConfig(
        data_root=args.data_root, image_size=args.image_size,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, seed=args.seed, num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
    )
    cfg.lora.rank = args.rank
    cfg.lora.alpha = args.alpha if args.alpha is not None else args.rank
    cfg.lora.also_proj = args.lora_also_proj
    cfg.model.variant = args.variant
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.use_feature_attention = not args.no_feature_attention
    cfg.model.use_refine_head = not args.no_refine_head
    cfg.model.encoder_size = args.encoder_size
    cfg.model.grad_checkpoint = (args.grad_checkpoint if args.grad_checkpoint is not None
                                 else not device.startswith("cuda"))
    cfg.loss.T = args.small_lesion_threshold
    cfg.loss.lambda_T = args.lambda_t
    cfg.loss.lambda_B = args.lambda_b
    cfg.loss.tau = args.tau
    cfg.loss.gamma = args.gamma
    cfg.uncertainty.size_threshold = args.small_lesion_threshold

    full_ds = BrainTumorSegDataset(cfg.data_root, image_size=cfg.image_size, to_rgb=True)
    print(f"Patient-level metadata found: {full_ds.has_patient_ids} "
          f"({'leakage-free split' if full_ds.has_patient_ids else 'WARNING: per-image split, see dataset.py'})")
    split_seed = args.split_seed if args.split_seed is not None else cfg.seed
    train_ds, val_ds, test_ds = full_ds.split(cfg.val_split, cfg.test_split, seed=split_seed)
    train_ds = train_ds.subsample(args.max_train_samples, seed=split_seed)
    val_ds = val_ds.subsample(args.max_eval_samples, seed=split_seed)
    test_ds = test_ds.subsample(args.max_eval_samples, seed=split_seed)
    train_ds.augment = True
    train_ds.intensity_aug = args.intensity_aug
    print(f"Split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    loader_kw = dict(num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
                     pin_memory=device.startswith("cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               drop_last=len(train_ds) >= cfg.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    # Everything needed to rebuild this exact model later (evaluation, visualization)
    # is stored inside each checkpoint, so those scripts can't drift out of sync.
    model_args = dict(
        image_size=cfg.image_size, variant=cfg.model.variant,
        checkpoint_path=cfg.model.checkpoint_path,
        lora_r=cfg.lora.rank, lora_alpha=cfg.lora.alpha, lora_also_proj=cfg.lora.also_proj,
        use_feature_attention=cfg.model.use_feature_attention,
        use_refine_head=cfg.model.use_refine_head,
        mc_dropout_p=cfg.uncertainty.dropout_p,
        encoder_size=cfg.model.encoder_size,
    )
    # Architecture options added later: stored only when non-default, so model_args
    # of earlier runs (and --resume of them) are unchanged.
    for k, v in dict(decoder=args.decoder, vit_skips=args.vit_skips,
                     deep_supervision=args.deep_supervision, lora_mlp=args.lora_mlp).items():
        if v != NEW_MODEL_DEFAULTS[k]:
            model_args[k] = v
    aux_weights = tuple(float(w) for w in args.ds_weights.split(",")) if args.deep_supervision else ()
    split_info = dict(data_root=cfg.data_root, val_split=cfg.val_split,
                      test_split=cfg.test_split, seed=split_seed,
                      small_lesion_threshold=cfg.loss.T)
    model = LoRAMedSegmenter(**model_args, grad_checkpoint=cfg.model.grad_checkpoint).to(device)

    print(f"Device: {device} | encoder size: {cfg.model.encoder_size} | "
          f"grad checkpointing: {cfg.model.grad_checkpoint}")
    print(f"Trainable parameters (LoRA + decoder + attention): {model.trainable_parameter_count():,}")

    use_amp = bool(args.amp if args.amp is not None else True) and device.startswith("cuda")
    if args.amp and not device.startswith("cuda"):
        print("Note: --amp is CUDA-only here; training in fp32 on", device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"Mixed precision (AMP): {use_amp} | physical batch: {args.batch_size} "
          f"x grad_accum: {args.grad_accum_steps} = effective batch: {effective_batch}")

    loss_fn = SizeAdaptiveBoundaryLoss(cfg.loss)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    last_path = os.path.join(cfg.checkpoint_dir, "last.pt")
    metrics_path = os.path.join(cfg.checkpoint_dir, "metrics.csv")
    config_path = os.path.join(cfg.checkpoint_dir, "config.json")

    # Each run directory is one experiment record: never start a fresh run on top of
    # an existing one (that would mix two configs' learning curves).
    if not args.resume and os.path.isfile(metrics_path):
        raise RuntimeError(
            f"{metrics_path} already exists. Use a new --checkpoint-dir for a new "
            "experiment, or --resume to continue this one.")
    if not os.path.isfile(config_path):
        _write_json({
            "args": vars(args), "model_args": model_args, "split": split_info,
            "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
            "trainable_parameters": model.trainable_parameter_count(),
            "effective_batch": effective_batch, "amp": use_amp, "device": device,
            "torch_version": torch.__version__, "command": " ".join(sys.argv),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, config_path)
    best_val_dice = -1.0
    epochs_since_improve = 0
    start_epoch = 1

    if args.resume:
        if not os.path.isfile(last_path):
            raise FileNotFoundError(f"--resume given but {last_path} does not exist.")
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt["model_args"] != model_args:
            raise RuntimeError(
                "--resume: model settings differ from the checkpoint's.\n"
                f"  checkpoint: {ckpt['model_args']}\n  now:        {model_args}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        # The saved lr reflects the OLD schedule (e.g. 0 at the end of a finished
        # cosine run); reset it so the schedule below is rebuilt for --epochs.
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"] = cfg.lr
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        best_val_dice = ckpt["best_val_dice"]
        epochs_since_improve = ckpt["epochs_since_improve"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {last_path}: continuing at epoch {start_epoch} "
              f"(best val Dice so far {best_val_dice:.4f})")

    # Built after resuming and fast-forwarded, so resuming with a different --epochs
    # gives the correct schedule for the new length.
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.3)
    if scheduler is not None and start_epoch > 1:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()"
            for _ in range(start_epoch - 1):
                scheduler.step()

    stopped_early = False
    best_epoch, epoch = None, start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]  # lr used for this epoch
        train_loss, train_dice = run_epoch(model, train_loader, loss_fn, optimizer,
                                            device, train=True,
                                            grad_accum_steps=args.grad_accum_steps,
                                            use_amp=use_amp, scaler=scaler,
                                            aux_weights=aux_weights)
        val_loss, val_dice = run_epoch(model, val_loader, loss_fn, optimizer,
                                        device, train=False, use_amp=use_amp)
        if scheduler is not None:
            scheduler.step()
        empty_device_cache(device)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{cfg.epochs}] lr={cur_lr:.2e} "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

        # `best_val_dice < 0` also saves when val Dice is NaN (empty val split), so
        # best.pt always exists for evaluation.
        is_best = val_dice > best_val_dice or best_val_dice < 0
        if is_best:
            if val_dice == val_dice:  # not NaN
                best_val_dice = val_dice
            best_epoch = epoch
            epochs_since_improve = 0
            _atomic_save({"model": model.state_dict(), "model_args": model_args,
                          "split": split_info, "epoch": epoch, "val_dice": val_dice},
                         best_path)
        else:
            epochs_since_improve += 1
            if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                stopped_early = True

        _append_metrics(metrics_path, {
            "epoch": epoch, "lr": epoch_lr,
            "train_loss": train_loss, "train_dice": train_dice,
            "val_loss": val_loss, "val_dice": val_dice,
            "epoch_minutes": round((time.time() - t0) / 60, 3), "is_best": int(is_best),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Saved EVERY epoch, so an interrupted run loses at most one epoch (--resume).
        _atomic_save({
            "model": model.state_dict(), "model_args": model_args, "split": split_info,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "best_val_dice": best_val_dice,
            "epochs_since_improve": epochs_since_improve,
        }, last_path)

        if stopped_early:
            print(f"Early stopping: no val Dice improvement in {args.early_stop_patience} epochs.")
            break

    print(f"Done training. Best val Dice: {best_val_dice:.4f}. "
          f"Checkpoints saved to {cfg.checkpoint_dir}/")

    # Held-out test Dice with the best (by val) checkpoint, deterministic (no dropout).
    if len(test_ds) > 0 and os.path.isfile(best_path):
        best_model, _ = build_model_from_checkpoint(best_path, device)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)
        test_loss, test_dice = run_epoch(best_model, test_loader, loss_fn, None, device,
                                          train=False, use_amp=use_amp)
        print(f"TEST (best.pt): loss={test_loss:.4f} dice={test_dice:.4f} (n={len(test_ds)})")
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        _write_json({
            "test_loss": test_loss, "test_dice": test_dice, "n_test": len(test_ds),
            "best_epoch": best_ckpt.get("epoch", best_epoch),
            "best_val_dice": best_val_dice, "stopped_early": stopped_early,
            "last_epoch": epoch, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, os.path.join(cfg.checkpoint_dir, "test.json"))


if __name__ == "__main__":
    main()
