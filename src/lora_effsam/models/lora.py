# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Step 1: Low-Rank Adaptation.

Implements, for a frozen linear layer W:

    h = Wx + (alpha / r) * B A x           (Eq. 2)

applied to the Q, K, V projections of each Transformer block (Eq. 3-5),
with W frozen and only A, B trainable.
"""
import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear `base` and adds a trainable low-rank branch.
    Forward: base(x) + (alpha / r) * B(A(x))
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 4, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        in_features, out_features = base.in_features, base.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        # Standard LoRA init: A ~ N(0, small), B = 0 so the adapter starts as a no-op.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:.4f}"


def _wrap_qkv_module(module: nn.Module, name: str, r: int, alpha: int, dropout: float,
                     also_proj: bool = False) -> bool:
    """
    Handles two common ViT attention layouts:
      (a) a single fused `qkv` Linear (out_features = 3 * dim)  -> EfficientSAM/SAM/timm style
      (b) separate `q_proj` / `k_proj` / `v_proj` (or `q`/`k`/`v`) Linears
    Returns True if it patched something on `module`.
    """
    patched = False

    if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
        module.qkv = LoRALinear(module.qkv, r=r, alpha=alpha, dropout=dropout)
        patched = True

    for attr_group in (("q_proj", "k_proj", "v_proj"), ("q", "k", "v"), ("query", "key", "value")):
        for attr in attr_group:
            sub = getattr(module, attr, None)
            if isinstance(sub, nn.Linear):
                setattr(module, attr, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout))
                patched = True

    # Optional ablation: also adapt the attention OUTPUT projection. Step 1 keeps this
    # frozen by default ("initially kept frozen"), so this is off unless requested.
    if patched and also_proj and isinstance(getattr(module, "proj", None), nn.Linear):
        module.proj = LoRALinear(module.proj, r=r, alpha=alpha, dropout=dropout)

    return patched


def _wrap_mlp_module(module: nn.Module, r: int, alpha: int, dropout: float) -> bool:
    """LoRA on a transformer MLP's fc1/fc2 Linears (ablation beyond Step 1's Q/K/V)."""
    patched = False
    for attr in ("fc1", "fc2"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Linear):
            setattr(module, attr, LoRALinear(sub, r=r, alpha=alpha, dropout=dropout))
            patched = True
    return patched


def inject_lora_into_vit(
    model: nn.Module,
    r: int = 4,
    alpha: int = 4,
    dropout: float = 0.0,
    target_modules: Iterable[str] = ("q", "k", "v"),
    also_proj: bool = False,
    also_mlp: bool = False,
) -> nn.Module:
    """
    Walks every submodule that looks like an attention block and injects LoRA into
    its Q/K/V projections (Eq. 3-5), leaving the attention output projection and the
    feed-forward network frozen, as specified in Step 1.

    Also freezes every other parameter in `model` (the original ViT weights and the
    prompt encoder, per the paper: "Frozen Parameters = Original ViT + Prompt Encoder").
    Call this AFTER loading pretrained weights and BEFORE building the optimizer.
    """
    # Freeze everything first; LoRALinear re-enables grad only on its A/B params.
    for p in model.parameters():
        p.requires_grad_(False)

    n_patched = 0
    # list() so we don't mutate the module tree while iterating it
    for name, module in list(model.named_modules()):
        lname = name.lower()
        if isinstance(module, LoRALinear):
            continue
        if "attn" in lname or "attention" in lname:
            if _wrap_qkv_module(module, name, r=r, alpha=alpha, dropout=dropout,
                                also_proj=also_proj):
                n_patched += 1
        elif also_mlp and lname.endswith("mlp"):
            # Optional ablation: Step 1 keeps the feed-forward network frozen.
            _wrap_mlp_module(module, r=r, alpha=alpha, dropout=dropout)

    if n_patched == 0:
        raise RuntimeError(
            "inject_lora_into_vit found no attention module with a `qkv` or "
            "`q/k/v`-style projection. Check the encoder's attribute names and "
            "extend `_wrap_qkv_module` accordingly."
        )
    return model


def trainable_parameters(model: nn.Module):
    """Θ_train = {Θ_LoRA, Θ_decoder} — everything with requires_grad=True."""
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_parameters(model))
