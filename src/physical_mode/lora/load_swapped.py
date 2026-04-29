"""Loader for the M-PSwap Idefics2-MLP-pool variant.

Single function used by §4.6 layer sweep, M5b SAE intervention, M5a steering,
and any other downstream script that wants to evaluate the swapped + LoRA-tuned
Idefics2 variant in inference mode.

The merge ordering matters:
    1. Load base Idefics2
    2. ``swap_perceiver_to_mlp_pool`` (replaces with random-init MLPPoolResampler)
    3. Load ``mlp_pool_resampler.pt`` state_dict
    4. Apply LoRA adapters via ``PeftModel.from_pretrained``
    5. ``merge_and_unload`` so downstream code sees a plain ``nn.Module``
       with the same Idefics2 attribute paths (``model.model.text_model.layers``
       etc.) as the base model
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .idefics2_mlp_resampler import swap_perceiver_to_mlp_pool


DEFAULT_BASE_MODEL = "HuggingFaceM4/idefics2-8b"


def load_idefics2_mlp_pool(
    ckpt_dir: Path | str,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "cuda:0",
    merge_lora: bool = True,
):
    """Load the M-PSwap Idefics2-MLP variant from a training checkpoint.

    Args:
        ckpt_dir: directory containing ``mlp_pool_resampler.pt`` and PEFT LoRA adapters
            (``adapter_model.safetensors`` + ``adapter_config.json``).
        base_model_id: HuggingFace model id of base Idefics2 (must match training).
        device: target device.
        merge_lora: if True, merge LoRA into base LM weights and unload PEFT
            wrappers (recommended for inference). If False, return PeftModel.

    Returns:
        (model, processor): ready for inference.
    """
    ckpt_dir = Path(ckpt_dir)
    if not (ckpt_dir / "mlp_pool_resampler.pt").exists():
        raise FileNotFoundError(f"{ckpt_dir}/mlp_pool_resampler.pt not found — did training save it?")

    processor = AutoProcessor.from_pretrained(base_model_id, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map=device
    )

    new = swap_perceiver_to_mlp_pool(model, n_heads=8)
    state = torch.load(ckpt_dir / "mlp_pool_resampler.pt", map_location=device, weights_only=True)
    new.load_state_dict(state)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, ckpt_dir)
    if merge_lora:
        model = model.merge_and_unload()

    model.eval()
    return model, processor
