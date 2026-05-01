"""Loader for the M-LMSwap CLIP+Vicuna / CLIP+Mistral variants.

Reconstructs a trained variant from a Stage 2 checkpoint directory:
    - CLIP-ViT-L-336 (frozen, from base HF model)
    - 2-layer MLP projector (state from `multi_modal_projector.pt`)
    - Vicuna-7B-v1.5 (variant A) or Mistral-7B-Instruct-v0.2 (variant B), with
      LoRA adapters loaded via PeftModel.from_pretrained.

Companion to `m_lmswap_train.py` — reuses build_variant_model verbatim.
Used by regression eval + downstream M5a/M5b experiments on trained variants.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import LlavaProcessor


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_train_module():
    """Dynamic-import scripts/m_lmswap_train.py to reuse its build helpers."""
    train_path = _REPO_ROOT / "scripts" / "m_lmswap_train.py"
    spec = importlib.util.spec_from_file_location("_m_lmswap_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {train_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m_lmswap_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_lmswap_variant(
    ckpt_dir: str | Path,
    variant: str,
    device: str = "cuda:0",
    merge_lora: bool = True,
):
    """Reconstruct a trained M-LMSwap variant from a Stage 2 ckpt directory.

    Parameters
    ----------
    ckpt_dir : str | Path
        Stage 2 checkpoint directory (e.g. ``outputs/lmswap_run_a_stage2_*/step21000``).
        Must contain ``multi_modal_projector.pt``, ``adapter_config.json``,
        ``adapter_model.safetensors``, and processor files.
    variant : str
        ``"A"`` (CLIP+Vicuna) or ``"B"`` (CLIP+Mistral).
    device : str
        Torch device.
    merge_lora : bool
        If True, calls ``merge_and_unload()`` after loading LoRA adapters —
        flattens LoRA into the base LM weights for inference speed. Set False
        if downstream code needs to inspect / disable LoRA.

    Returns
    -------
    (model, processor)
        Model in eval mode on ``device`` in bf16; processor reloaded from
        ``ckpt_dir`` (preserves ``num_additional_image_tokens=1``).
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir missing: {ckpt_dir}")
    for required in ("multi_modal_projector.pt", "adapter_config.json", "adapter_model.safetensors"):
        if not (ckpt_dir / required).is_file():
            raise FileNotFoundError(f"missing {required} in {ckpt_dir}")

    train_mod = _load_train_module()

    model, _ = train_mod.build_variant_model(variant, device)

    # Load MLP weights BEFORE PEFT wrap — PEFT renames submodule paths.
    mlp_state = torch.load(
        ckpt_dir / "multi_modal_projector.pt", map_location=device, weights_only=True,
    )
    model.model.multi_modal_projector.load_state_dict(mlp_state)
    print(f"  ✓ loaded MLP weights from {ckpt_dir}/multi_modal_projector.pt")

    model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=False)
    print(f"  ✓ loaded LoRA adapters from {ckpt_dir}")

    if merge_lora:
        model = model.merge_and_unload()
        print("  ✓ merged LoRA into base LM weights (eval mode)")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    # Processor: reload from ckpt_dir to inherit num_additional_image_tokens=1.
    processor = LlavaProcessor.from_pretrained(str(ckpt_dir))

    return model, processor


__all__ = ["load_lmswap_variant"]
