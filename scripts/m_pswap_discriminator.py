"""Discriminator test: does the bad batch also NaN on base Idefics2 (no swap)?

If YES: pathology is in upstream the_cauldron data — fix is filter/cap.
If NO: pathology is interaction between our perceiver swap + LoRA + that input.
       Need deeper fix (or pivot to M-LMSwap fallback).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.lora.idefics2_mlp_resampler import swap_perceiver_to_mlp_pool


BAD_BATCH_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "outputs/mpswap_fp32_20260429-053240/nan_repro/bad_batch.pkl"
)
SWAPPED_CKPT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
    "outputs/mpswap_fp32_20260429-053240/step1000"
)


def load_batch():
    with BAD_BATCH_PATH.open("rb") as f:
        payload = pickle.load(f)
    print(f"Bad batch from training step {payload['step']}.{payload['accum']}, loss={payload['loss']}")
    print("Batch composition:")
    for j, m in enumerate(payload["meta"]):
        print(f"  [{j}] n_imgs={m['n_imgs']} img_sizes={m['img_sizes']} "
              f"user_len={m['user_len']} asst_len={m['asst_len']}")
        print(f"        USER: {m['user_preview']!r}")
        print(f"        ASST: {m['asst_preview']!r}")
    return payload


def run_forward(model, batch, label: str):
    ins = {
        "input_ids": batch["input_ids"].to("cuda:0"),
        "attention_mask": batch["attention_mask"].to("cuda:0"),
        "pixel_values": batch["pixel_values"].to("cuda:0"),
        "labels": batch["labels"].to("cuda:0"),
    }
    if batch.get("pixel_attention_mask") is not None:
        ins["pixel_attention_mask"] = batch["pixel_attention_mask"].to("cuda:0")

    model.eval()
    with torch.no_grad():
        out = model(**ins)
    finite = bool(torch.isfinite(out.loss).item())
    print(f"\n=== {label} ===")
    print(f"  loss: {out.loss.item()}")
    print(f"  finite: {finite}")
    return finite, out.loss.item()


def main():
    batch = load_batch()

    # ---- Test 1: BASE Idefics2 (vanilla, no swap, no LoRA) ----
    print("\nLoading base Idefics2 (no modifications)...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceM4/idefics2-8b", dtype=torch.bfloat16, device_map="cuda:0"
    )
    base_finite, base_loss = run_forward(base_model, batch, "BASE Idefics2 (no swap, no LoRA)")
    del base_model
    torch.cuda.empty_cache()

    # ---- Test 2: SWAPPED + LoRA-trained Idefics2 (our M-PSwap variant) ----
    print("\nLoading M-PSwap variant from step1000...")
    from peft import PeftModel
    base = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceM4/idefics2-8b", dtype=torch.bfloat16, device_map="cuda:0"
    )
    new = swap_perceiver_to_mlp_pool(base)
    state = torch.load(SWAPPED_CKPT / "mlp_pool_resampler.pt", map_location="cuda:0", weights_only=True)
    new.load_state_dict(state)
    swapped_model = PeftModel.from_pretrained(base, SWAPPED_CKPT, is_trainable=False)
    swapped_finite, swapped_loss = run_forward(swapped_model, batch, "SWAPPED + LoRA (our variant)")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("DISCRIMINATOR VERDICT")
    print("=" * 60)
    print(f"  BASE Idefics2:    finite={base_finite}  loss={base_loss}")
    print(f"  SWAPPED+LoRA:     finite={swapped_finite}  loss={swapped_loss}")
    print()
    if not base_finite and not swapped_finite:
        print("➜ DATA-LEVEL PATHOLOGY: both base and swapped fail.")
        print("  the_cauldron contains a sample whose forward pass overflows bf16 in")
        print("  the standard Idefics2 pipeline. Fix: filter the offending sample type")
        print("  (likely OCR/long-text or extreme-aspect image). Continue with M-PSwap.")
    elif base_finite and not swapped_finite:
        print("➜ INTERACTION PATHOLOGY: base OK, swapped fails.")
        print("  Our perceiver swap + LoRA introduces a numerical edge case for this")
        print("  specific input pattern. Fix path is non-trivial — recommend pivot to")
        print("  M-LMSwap (paper_gaps.md G3 fallback) per submission_plan §6 pruning.")
    elif not base_finite and swapped_finite:
        print("➜ UNEXPECTED: base fails but swapped succeeds. Worth investigating but")
        print("  this isn't the regime we're hitting in training.")
    else:
        print("➜ NO REPRO: neither variant fails on this batch. The NaN at training")
        print("  step 1461 may be non-deterministic (cudnn kernel selection, etc.) —")
        print("  recommend forcing attn_implementation='eager' and retrying.")


if __name__ == "__main__":
    main()
