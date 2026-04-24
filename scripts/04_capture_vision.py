"""Capture vision-encoder activations for an existing stimulus set.

Runs forward-only passes (no generate) through Qwen2.5-VL with hooks on
`model.model.visual.blocks[i]`. Reuses the existing manifest from an M2
run so we can probe the same stimuli that produced the behavioral labels.

Usage:
    uv run python scripts/04_capture_vision.py \
        --stimulus-dir inputs/mvp_full_<ts>_<hash> \
        --output-dir   outputs/mvp_full_<ts>_<hash>/vision_activations \
        --layers 3,7,11,15,19,23,27,31
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from physical_mode.inference.prompts import render as render_prompt
from physical_mode.models.vlm_runner import PhysModeVLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stimulus-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--layers",
        type=str,
        default="3,7,11,15,19,23,27,31",
        help="Comma-separated vision-encoder layer indices to capture.",
    )
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--label", default="ball", help="Prompt label (image content is invariant).")
    p.add_argument("--prompt-variant", default="open")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    layers = tuple(int(x) for x in args.layers.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.stimulus_dir / "manifest.parquet")
    if args.limit is not None:
        manifest = manifest.head(args.limit).reset_index(drop=True)

    print(f"Loading {args.model_id} with vision capture at layers {layers} ...")
    t0 = time.time()
    vlm = PhysModeVLM(
        model_id=args.model_id,
        torch_dtype=args.torch_dtype,
        device="cuda",
        capture_vision_layers=layers,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")

    rp = render_prompt(args.prompt_variant, args.label)

    # First-pass shape check so we fail fast if anything is off.
    first_row = manifest.iloc[0]
    first_img = args.stimulus_dir / first_row["image_path"]
    cap = vlm.capture(image=first_img, prompt=rp.user, system_prompt=rp.system)
    print("Sample capture shapes:")
    for li, t in cap.get("vision_hidden", {}).items():
        print(f"  vision_hidden_{li}: {tuple(t.shape)} {t.dtype}")
    vlm.save_capture(cap, args.output_dir / f"{first_row['sample_id']}.safetensors")

    for _, row in tqdm(manifest.iloc[1:].iterrows(), total=len(manifest) - 1, desc="Capturing"):
        img_path = args.stimulus_dir / row["image_path"]
        cap = vlm.capture(image=img_path, prompt=rp.user, system_prompt=rp.system)
        vlm.save_capture(cap, args.output_dir / f"{row['sample_id']}.safetensors")

    print(f"Wrote vision activations for {len(manifest)} stimuli to {args.output_dir}")


if __name__ == "__main__":
    main()
