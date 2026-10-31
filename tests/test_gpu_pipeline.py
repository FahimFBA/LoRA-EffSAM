# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""GPU tests with the real EfficientSAM encoder (skip without CUDA / weights / data / runs).

* every architecture option builds, trains one AMP step at batch 4 and supports MC + TTA
* encoder taps do not change the encoder output
* a 1-epoch training run end to end: records, resume, overwrite guard, evaluation
* the packaged final model (seed 42) reproduces its recorded test Dice
"""
import json

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import FINAL, needs_cuda, needs_data, needs_efficientsam, needs_final_runs

from lora_effsam import paths
from lora_effsam.config import LossConfig
from lora_effsam.data.dataset import BrainTumorSegDataset
from lora_effsam.losses.size_adaptive_loss import SizeAdaptiveBoundaryLoss
from lora_effsam.models.image_encoder import run_encoder
from lora_effsam.models.segmenter import LoRAMedSegmenter, build_model_from_checkpoint
from lora_effsam.uncertainty.mc_dropout import mc_dropout_predict
from lora_effsam.utils import batch_boundary_distance_maps, dice_score

BASE = dict(image_size=512, encoder_size=512, mc_dropout_p=0.2)
CONFIGS = {
    "simple": dict(),
    "hybrid": dict(decoder="hybrid"),
    "hybrid+skips+DS": dict(decoder="hybrid", vit_skips=True, deep_supervision=True),
    "final (ViT-S)": dict(variant="vits", decoder="hybrid", vit_skips=True, deep_supervision=True),
    "final + LoRA-MLP": dict(variant="vits", decoder="hybrid", vit_skips=True, deep_supervision=True,
                             lora_mlp=True),
}


@needs_cuda
@needs_efficientsam
@pytest.mark.parametrize("name", list(CONFIGS))
def test_architecture_trains_and_predicts(name):
    kw = CONFIGS[name]
    m = LoRAMedSegmenter(**BASE, **kw).cuda()
    x = torch.rand(4, 3, 512, 512, device="cuda")
    y = torch.zeros(4, 1, 512, 512, device="cuda"); y[:, :, 200:260, 220:300] = 1
    loss_fn = SizeAdaptiveBoundaryLoss(LossConfig(T=400, tau=20))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    m.train()
    with torch.autocast("cuda", dtype=torch.float16):
        lo = m(x)
    dm = batch_boundary_distance_maps(y)
    loss = loss_fn(lo.float(), y, dm)[0]
    if kw.get("deep_supervision"):
        assert m.aux_logits is not None and len(m.aux_logits) == 3
        loss = loss + sum(w * loss_fn(a.float(), y, dm)[0] for w, a in zip((0.5, 0.25, 0.125), m.aux_logits))
    else:
        assert m.aux_logits is None
    scaler.scale(loss).backward(); scaler.step(opt)
    assert torch.isfinite(loss)
    m.eval()
    with torch.no_grad():
        m(x[:1])
    assert m.aux_logits is None
    r = mc_dropout_predict(m, x[:1], K=8, tta=True)
    assert r.mean_prob.shape == (1, 1, 512, 512) and r.uncertainty.max() > 0


@needs_cuda
@needs_efficientsam
def test_encoder_taps_do_not_change_output():
    m = LoRAMedSegmenter(**BASE, decoder="hybrid", vit_skips=True).cuda().eval()
    xe = (torch.rand(2, 3, 512, 512, device="cuda") - m.pixel_mean) / m.pixel_std
    with torch.no_grad():
        plain = run_encoder(m.encoder, xe)
        neck, taps = run_encoder(m.encoder, xe, return_layers=m.vit_skip_layers)
    assert torch.allclose(plain, neck, atol=1e-5)
    assert m.vit_skip_layers == (8, 5, 2) and len(taps) == 3


@needs_cuda
@needs_efficientsam
@needs_data
def test_training_run_end_to_end(tmp_path):
    from lora_effsam.evaluation import evaluate_uncertainty
    from lora_effsam.training import train

    run = tmp_path / "smoke"
    args = ["--checkpoint-dir", str(run), "--image-size", "512", "--encoder-size", "512",
            "--batch-size", "4", "--grad-accum-steps", "2", "--variant", "vits", "--decoder", "hybrid",
            "--vit-skips", "--deep-supervision", "--intensity-aug", "--small-lesion-threshold", "400",
            "--max-train-samples", "16", "--max-eval-samples", "8", "--num-workers", "0"]
    train.main(args + ["--epochs", "1"])
    for f in ("config.json", "metrics.csv", "test.json", "best.pt", "last.pt"):
        assert (run / f).is_file(), f
    train.main(args + ["--epochs", "2", "--resume"])
    assert len((run / "metrics.csv").read_text().strip().splitlines()) == 3   # header + 2 epochs
    with pytest.raises(RuntimeError, match="already exists"):
        train.main(args + ["--epochs", "1"])                                  # overwrite guard
    evaluate_uncertainty.main(["--checkpoint", str(run / "best.pt"), "--max-samples", "4",
                               "--mc-passes", "4", "--tta", "--out", str(run / "u.json")])
    rep = json.loads((run / "u.json").read_text())
    assert len(rep["per_image"]) == 4 and "id" in rep["per_image"][0]


@needs_cuda
@needs_efficientsam
@needs_data
@needs_final_runs
def test_final_model_reproduces_recorded_test_dice():
    run = paths.runs_dir() / FINAL
    model, ck = build_model_from_checkpoint(str(run / "best.pt"), "cuda")
    model.eval()
    s = ck["split"]
    ds = BrainTumorSegDataset(str(paths.data_dir()), image_size=model.image_size, to_rgb=True)
    test = ds.split(s["val_split"], s["test_split"], seed=s["seed"])[2]
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y, _ in DataLoader(test, batch_size=4):
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x)
            d = dice_score((torch.sigmoid(lo.float()) >= 0.5).float(), y)
            total += d.sum().item(); n += len(d)
    recorded = json.loads((run / "test.json").read_text())["test_dice"]
    assert n == 477 and abs(total / n - recorded) < 1e-3
