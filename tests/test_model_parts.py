# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""CPU tests of LoRA, the hybrid decoder and MC-Dropout / TTA (no pretrained weights needed)."""
import torch
import torch.nn as nn

from lora_effsam.models.hybrid_decoder import HybridDecoder
from lora_effsam.models.lora import LoRALinear, inject_lora_into_vit
from lora_effsam.uncertainty.mc_dropout import mc_dropout_predict


class _TinyViT(nn.Module):
    """Mimics EfficientSAM's module names: blocks.<i>.attn.qkv and blocks.<i>.mlp.fc1/fc2."""

    def __init__(self, dim=8, n=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n):
            b = nn.Module()
            b.attn = nn.Module(); b.attn.qkv = nn.Linear(dim, 3 * dim)
            b.mlp = nn.Module(); b.mlp.fc1 = nn.Linear(dim, 2 * dim); b.mlp.fc2 = nn.Linear(2 * dim, dim)
            self.blocks.append(b)


def test_lora_starts_as_identity_and_freezes_base():
    base = nn.Linear(8, 8)
    lora = LoRALinear(base, r=4, alpha=4)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))                       # B = 0 at init
    assert not base.weight.requires_grad and lora.lora_A.weight.requires_grad


def test_inject_lora_qkv_only_and_mlp_ablation():
    for mlp, want in ((False, 2), (True, 6)):
        vit = inject_lora_into_vit(_TinyViT(), r=4, alpha=4, also_mlp=mlp)
        assert sum(isinstance(m, LoRALinear) for m in vit.modules()) == want
        trainable = {n for n, p in vit.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in n for n in trainable)


def _decoder(ds=True, vit=True):
    return HybridDecoder(neck_channels=32, image_size=64, vit_channels=16 if vit else 0,
                         mc_dropout_p=0.2, deep_supervision=ds)


def test_hybrid_decoder_shapes_and_deep_supervision():
    dec = _decoder()
    img, neck = torch.rand(2, 3, 64, 64), torch.randn(2, 32, 4, 4)
    taps = [torch.randn(2, 16, 4, 4) for _ in range(3)]
    dec.train()
    logits, aux = dec(img, neck, taps)
    assert logits.shape == (2, 1, 64, 64)
    assert aux is not None and len(aux) == 3 and all(a.shape == (2, 1, 64, 64) for a in aux)
    dec.eval()
    _, aux = dec(img, neck, taps)
    assert aux is None                                            # never at prediction time


class _FakeSegmenter(nn.Module):
    """Minimal model with the interface mc_dropout_predict expects."""

    def __init__(self):
        super().__init__()
        self.dec = _decoder(ds=False, vit=False)
        self.encoder_is_stochastic = False

    def enable_mc_dropout(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def encode(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x.mean(1, keepdim=True).repeat(1, 32, 1, 1), 4)

    def decode(self, x, feat):
        return self.dec(x, feat, [])[0]


def test_mc_dropout_and_tta():
    model = _FakeSegmenter()
    x = torch.rand(1, 3, 64, 64)
    r = mc_dropout_predict(model, x, K=8)
    assert r.mean_prob.shape == r.uncertainty.shape == (1, 1, 64, 64)
    assert r.uncertainty.max() > 0                               # dropout is active
    r_tta = mc_dropout_predict(model, x, K=8, tta=True)
    assert r_tta.mean_prob.shape == (1, 1, 64, 64)
    assert not model.training and model.dec.dropout.training      # only dropout in train mode
