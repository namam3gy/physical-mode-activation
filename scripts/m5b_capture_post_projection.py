"""M5b post-projection SAE — capture Qwen2.5-VL merger output activations.

The existing M5b SAE was trained on `vision_hidden_31` (last SigLIP block,
pre-projection, 1280-dim). This script captures the *post-projection*
activation: output of `model.model.visual.merger`, which maps 4-patch
groups from 1280-dim to 3584-dim LM embedding space.

Hypothesis: physics-cue features may be more or less localized post-
projection vs pre-projection. If post-projection SAE breaks PMR with
similar k≈20 features, the projector preserves the physics-mode
commitment cleanly. If k>>20 or NULL, the projector distributes the
commitment across many features.

Output: `outputs/post_projection_qwen/<sample_id>.safetensors` with
key `post_projection_visual` per stim (n_visual_tokens × 3584).

Usage:
    uv run python scripts/m5b_capture_post_projection.py \\
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \\
        --output-dir outputs/post_projection_qwen
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import safetensors.torch as st
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def _hook_merger(model):
    captures: dict = {"out": None}

    def hook(_module, _inputs, output):
        # output: (n_merger_groups, 3584) for batch-1 forward
        t = output[0] if isinstance(output, tuple) else output
        captures["out"] = t.detach().to("cpu", dtype=torch.bfloat16).contiguous()

    handle = model.model.visual.merger.register_forward_hook(hook)
    return handle, captures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stimulus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="ball")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.stimulus_dir / "manifest.parquet")
    if args.limit:
        manifest = manifest.head(args.limit).reset_index(drop=True)
    print(f"Loaded manifest: {len(manifest)} stim")

    print(f"Loading {args.model_id} on {args.device} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s")

    handle, captures = _hook_merger(model)

    prompt_text = (
        "The image shows a {label}. Describe what will happen to the {label} "
        "in the next moment, in one short sentence."
    ).format(label=args.label)
    sys_prompt = (
        "You are a careful observer of images. When asked what will happen "
        "next, describe the most likely next state or motion in one short sentence."
    )

    try:
        for i, row in enumerate(tqdm(manifest.iterrows(), total=len(manifest), desc="Capturing")):
            _, row = row
            sid = row["sample_id"]
            img_path = args.stimulus_dir / row["image_path"]
            pil = Image.open(img_path).convert("RGB")

            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ]},
            ]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(images=pil, text=chat, return_tensors="pt").to(args.device)

            with torch.no_grad():
                _ = model(**inputs, output_hidden_states=False)

            t = captures["out"]
            assert t is not None, "merger hook didn't fire"
            st.save_file({"post_projection_visual": t}, args.output_dir / f"{sid}.safetensors")
            captures["out"] = None  # reset for next stim
    finally:
        handle.remove()

    elapsed = (time.time() - t0) / 60
    print(f"Wrote post-projection activations for {len(manifest)} stim to {args.output_dir} ({elapsed:.1f} min total)")


if __name__ == "__main__":
    main()
