# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""
Central configuration. Every symbol here maps directly to a symbol in the
three methodology documents (Step 1: LoRA, Step 2: loss, Step 3: uncertainty).
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class LoRAConfig:
    # Step 1
    rank: int = 4                 # r in {4, 8}
    alpha: int = 4                # scaling factor alpha (paired with rank, e.g. alpha=rank)
    target_modules: Tuple[str, ...] = ("q", "k", "v")  # LoRA on Q,K,V projections only
    also_proj: bool = False       # ablation only; doc keeps attn output proj frozen
    freeze_backbone: bool = True  # original ViT weights + prompt encoder frozen


@dataclass
class ModelConfig:
    variant: str = "vitt"              # EfficientSAM variant: "vitt" (6M enc) or "vits" (23M enc)
    checkpoint_path: str = None        # defaults to third_party/EfficientSAM/weights/efficient_sam_<variant>.pt
    use_feature_attention: bool = True  # CBAM-style attention on encoder features
    use_refine_head: bool = True        # boundary-attention-gated residual refinement
    encoder_size: int = 1024            # ViT input size (1024 = pretraining; 512 for 8GB machines)
    grad_checkpoint: bool = False       # recompute encoder activations in backward (saves memory)


@dataclass
class LossConfig:
    # Step 2
    alpha0: float = 0.3       # base false-negative weight alpha_0
    alpha_max: float = 0.7    # max false-negative weight for tiny lesions
    beta0: float = 0.3        # fixed false-positive weight beta_0
    gamma: float = 2.0        # max multiplier for boundary gate on tiny structures
    T: int = 100               # small-tumor pixel-area threshold (dataset dependent!)
    tau: float = 20.0         # sigmoid temperature (smoothness of the size gate)
    eps: float = 1e-6         # smoothing constant epsilon
    lambda_T: float = 1.0     # weight on size-adaptive Tversky loss
    lambda_B: float = 1.0     # weight on size-weighted boundary loss


@dataclass
class UncertaintyConfig:
    # Step 3
    mc_passes: int = 20        # K stochastic forward passes
    dropout_p: float = 0.2     # dropout probability used ONLY for MC-Dropout layers
    boundary_band_px: int = 3  # half-width (in pixels) of the boundary band B
    size_threshold: int = 100  # reuse LossConfig.T for small/large stratification


@dataclass
class TrainConfig:
    data_root: str = "./data"
    image_size: int = 512
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
    num_workers: int = 2
    checkpoint_dir: str = "./runs"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


TUMOR_CLASSES = ("Meningioma", "Glioma", "Pituitary tumor")
