"""M5b post-projection SAE — capture LLaVA-1.5 multi_modal_projector output.

Cross-model extension of `m5b_capture_post_projection.py` (Qwen-only).
Hooks `model.multi_modal_projector` forward output → (1, 576, 4096) per stim.

LLaVA-1.5's M5b is NULL on the pre-projection encoder (CHANGELOG 2026-04-28
evening) — top-k SAE ablation at any k ≤ 160 fails to break PMR. Hypothesis
to test: post-projection features may carry the physics-mode commitment that
encoder-side SAE missed (i.e., the projector itself is where the LLaVA family
constructs the kinetic feature, not the encoder).

If post-projection top-k breaks PMR cleanly → projector is the locus.
If still NULL → commitment routes through LM mid-layers (matches M5a positive
on LLaVA-Next L20-25 + M4 LM probe AUC 0.76).

This script also works for the trained M-LMSwap variants (same projector path
`model.multi_modal_projector`). Use `--lmswap-ckpt <step>` + `--lmswap-variant`
to capture from a trained checkpoint instead of HF model.

Output: `outputs/post_projection_<tag>/<sample_id>.safetensors` with key
`post_projection_visual` (576, 4096) per stim.

Usage
-----
    # LLaVA-1.5 baseline
    uv run python scripts/m5b_capture_post_projection_llava.py \\
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \\
        --output-dir outputs/post_projection_llava_1_5 \\
        --model-id llava-hf/llava-1.5-7b-hf

    # Trained M-LMSwap variant (same projector hook works)
    uv run python scripts/m5b_capture_post_projection_llava.py \\
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \\
        --output-dir outputs/post_projection_lmswap_a \\
        --lmswap-ckpt outputs/lmswap_run_a_stage2_*/step21000 \\
        --lmswap-variant A
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import safetensors.torch as st
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _hook_projector(model, projector_path: str = "model.multi_modal_projector"):
    """Register a forward hook on the multi_modal_projector. Returns (handle, captures)."""
    captures: dict = {"out": None}

    def hook(_module, _inputs, output):
        # output: (B=1, n_visual_tokens=576, hidden=4096) for LLaVA-1.5
        t = output[0] if isinstance(output, tuple) else output
        captures["out"] = t.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    # Walk the dotted path
    obj = model
    for part in projector_path.split("."):
        obj = getattr(obj, part)
    handle = obj.register_forward_hook(hook)
    return handle, captures


def _build_messages_llava15(label: str) -> tuple[str, str]:
    """Return (system_text, user_text) for LLaVA-1.5 / LMSwap-A inference."""
    sys_text = (
        "You are a careful observer of images. When asked what will happen "
        "next, describe the most likely next state or motion in one short sentence."
    )
    user_text = (
        f"The image shows a {label}. Describe what will happen to the {label} "
        "in the next moment, in one short sentence."
    )
    return sys_text, user_text


def _format_prompt_vicuna(sys_text: str, user_text: str) -> str:
    """LLaVA-1.5 / Vicuna chat format with literal <image>."""
    vicuna_sys = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    full_user = f"<image>\n{user_text}"
    return f"{vicuna_sys} {sys_text} USER: {full_user} ASSISTANT:"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stimulus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf",
                   help="HF model id (used if --lmswap-ckpt not given)")
    p.add_argument("--projector-path", default="model.multi_modal_projector",
                   help="dotted path to projector module to hook. Defaults to "
                        "LLaVA-style. Use 'model.connector' for Idefics2.")
    p.add_argument("--lmswap-ckpt", type=Path, default=None,
                   help="If set, load a trained M-LMSwap variant via load_lmswap_variant")
    p.add_argument("--lmswap-variant", choices=["A", "B"], default=None,
                   help="Required with --lmswap-ckpt")
    p.add_argument("--label", default="ball")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.stimulus_dir / "manifest.parquet")
    if args.limit:
        manifest = manifest.head(args.limit).reset_index(drop=True)
    print(f"Loaded manifest: {len(manifest)} stim")

    t0 = time.time()
    if args.lmswap_ckpt is not None:
        if args.lmswap_variant is None:
            raise SystemExit("--lmswap-variant required with --lmswap-ckpt")
        from physical_mode.lora.load_lmswap import load_lmswap_variant
        print(f"Loading M-LMSwap variant {args.lmswap_variant} from {args.lmswap_ckpt} on {args.device} ...")
        model, processor = load_lmswap_variant(
            ckpt_dir=args.lmswap_ckpt, variant=args.lmswap_variant,
            device=args.device, merge_lora=True,
        )
    else:
        print(f"Loading {args.model_id} on {args.device} ...")
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id, dtype=torch.bfloat16, device_map=args.device,
        )
        model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    handle, captures = _hook_projector(model, projector_path=args.projector_path)

    sys_text, user_text = _build_messages_llava15(args.label)

    try:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Capturing"):
            sid = row["sample_id"]
            img_path = args.stimulus_dir / row["image_path"]
            pil = Image.open(img_path).convert("RGB")

            if args.lmswap_ckpt is not None:
                # LMSwap variant — manual prompt assembly (no chat_template)
                prompt = _format_prompt_vicuna(sys_text, user_text)
                inputs = processor(images=pil, text=prompt, return_tensors="pt")
                inputs = {k: v.to(args.device) for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            else:
                # LLaVA-1.5 — use processor's chat template
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": sys_text}]},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ]},
                ]
                chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(images=pil, text=chat, return_tensors="pt").to(args.device)

            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=False)

            t = captures["out"]
            assert t is not None, "projector hook didn't fire"
            # squeeze batch dim if present: (1, 576, 4096) -> (576, 4096)
            if t.dim() == 3 and t.size(0) == 1:
                t = t.squeeze(0)
            st.save_file({"post_projection_visual": t}, args.output_dir / f"{sid}.safetensors")
            captures["out"] = None
    finally:
        handle.remove()

    elapsed_min = (time.time() - t0) / 60
    print(f"Wrote post-projection activations for {len(manifest)} stim to "
          f"{args.output_dir} ({elapsed_min:.1f} min total)")


if __name__ == "__main__":
    main()
