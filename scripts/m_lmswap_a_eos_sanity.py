"""Quick sanity check: does Variant A spontaneously emit `</s>` (EOS) given enough tokens?

Background: regression_eval used max_new_tokens=80 and A's responses ran to the
limit mid-sentence. That looks superficially like the EOS-as-PAD bug we fixed
in B, but A's tokenizer defaults (pad=<unk>, padding_side=right) escaped that
bug at training time. This script tests the alternative hypothesis: A is just
verbose, hits the 80-token cap, and would terminate cleanly with more headroom.

Test: 5 M2 stim, max_new_tokens=512, decode WITHOUT skip_special_tokens, log
last 5 tokens + whether eos_token_id appears in generation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physical_mode.lora.load_lmswap import load_lmswap_variant  # noqa: E402

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions."
)
M2_OPEN_SYSTEM = "Predict what will happen next."
USER_TEXT = "What will happen next?"


def fmt_a(user_text: str) -> str:
    full_user = f"<image>\n{user_text}"
    return f"{VICUNA_SYSTEM} {M2_OPEN_SYSTEM} USER: {full_user} ASSISTANT:"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--stim-dir", type=Path,
                   default=ROOT / "inputs/mvp_full_20260424-093926_e9d79da3")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--n-stim", type=int, default=5)
    args = p.parse_args()

    print(f"[eos-sanity] loading A from {args.ckpt} on {args.device}...")
    model, processor = load_lmswap_variant(
        ckpt_dir=args.ckpt, variant="A", device=args.device, merge_lora=True,
    )
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id or eos_id
    print(f"[eos-sanity] eos_id={eos_id}, pad_id={pad_id}, max_new_tokens={args.max_new_tokens}")
    print("=" * 80)

    manifest = pd.read_parquet(args.stim_dir / "manifest.parquet")
    # Pick first N stim (any cell)
    picked = manifest.head(args.n_stim).to_dict("records")

    eos_count = 0
    prompt_str = fmt_a(USER_TEXT)
    for i, row in enumerate(picked):
        img_path = args.stim_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, text=prompt_str, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id,
            )

        gen_ids = out[0, prompt_len:].tolist()
        gen_len = len(gen_ids)
        # Strip trailing pad tokens (when eos fires, generate pads to longest)
        # but here batch_size=1 so trailing pads are after eos
        last_5_ids = gen_ids[-5:]
        last_5_str = processor.tokenizer.convert_ids_to_tokens(last_5_ids)
        has_eos = eos_id in gen_ids
        if has_eos:
            eos_pos = gen_ids.index(eos_id) + 1  # 1-indexed length up to and including eos
            eos_count += 1
        else:
            eos_pos = None

        decoded = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
        decoded_clean = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(f"\n[stim {i+1}/{args.n_stim}] {row['sample_id']}")
        print(f"  gen_len={gen_len} (cap={args.max_new_tokens})")
        print(f"  has_eos={has_eos}, eos_pos={eos_pos}")
        print(f"  last 5 tokens: {last_5_str}")
        print(f"  response (clean): {decoded_clean[:300]}{'...' if len(decoded_clean) > 300 else ''}")

    print("\n" + "=" * 80)
    print(f"[eos-sanity] EOS rate: {eos_count}/{args.n_stim} = {eos_count/args.n_stim:.0%}")
    if eos_count == args.n_stim:
        print("[eos-sanity] VERDICT: A learned EOS — verbose response is just verbose, NOT EOS-bug.")
        print("[eos-sanity] -> A retrain NOT needed.")
    elif eos_count == 0:
        print("[eos-sanity] VERDICT: A never emits EOS — looks like EOS-bug after all.")
        print("[eos-sanity] -> A retrain RECOMMENDED.")
    else:
        print(f"[eos-sanity] VERDICT: partial EOS ({eos_count}/{args.n_stim}). Inconclusive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
