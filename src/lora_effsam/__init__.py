# Authors: Md. Fahim Bin Amin <fahimbinamin@gmail.com>, Israt Jahan Khan <isratjahankhanijk@gmail.com>
"""lora_effsam: parameter-efficient brain-tumor segmentation with LoRA-adapted EfficientSAM.

Encoder: EfficientSAM ViT with LoRA on Q/K/V (frozen otherwise). Decoder: hybrid CNN-ViT
U-Net-style decoder with CNN and transformer skips and deep supervision. Loss:
size-adaptive Tversky + size-weighted boundary loss. Uncertainty: MC-Dropout (+ optional
flip TTA). Analysis tools for statistics, figures and explainability live in
``lora_effsam.analysis``; every step is reachable through the ``lora_effsam`` CLI.
"""

__version__ = "0.2.0"
