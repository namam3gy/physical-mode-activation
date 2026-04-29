"""Deterministically reproduce the NaN-triggering batch.

Loads Run 3's step1000 ckpt (last clean fp32 weights), streams the_cauldron
with seed=42 (same as training), iterates batches matching the training collate,
runs forward + backward, and stops at the first non-finite loss/grad. Reports:
    - Which step (effective optimizer step) triggers it
    - Which Cauldron subset / sample produced the pathological batch
    - Image dims, text length, n_imgs of the offending batch

If the offending batch is the same across two runs of this repro, we have proof
of data-level cause and can move to filtering.
"""

from __future__ import annotations

import sys
import math
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from physical_mode.lora.idefics2_mlp_resampler import swap_perceiver_to_mlp_pool


CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs/mpswap_fp32_20260429-053240/step1000"
SEED = 42
BATCH_SIZE = 4
GRAD_ACCUM = 4
START_FROM = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1465  # past Run 3's 1461 NaN


def load_model():
    print(f"Loading {CKPT}")
    base = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceM4/idefics2-8b", dtype=torch.bfloat16, device_map="cuda:0"
    )
    new = swap_perceiver_to_mlp_pool(base)
    state = torch.load(f"{CKPT}/mlp_pool_resampler.pt", map_location="cuda:0", weights_only=True)
    new.load_state_dict(state)
    model = PeftModel.from_pretrained(base, CKPT, is_trainable=True)
    pr = model.base_model.model.model.connector.perceiver_resampler
    for p in pr.parameters():
        p.requires_grad_(True)
    model.train()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)
    return model, processor


def main():
    # Reuse the exact training collate + data stream by importing from scripts/
    # (importlib.util breaks dataclass module resolution; sys.path.insert + import works).
    import sys as _sys
    _sys.path.insert(0, "scripts")
    from m_pswap_train import TheCauldronStream, make_collate

    model, processor = load_model()
    collate = make_collate(processor, "cuda:0")
    stream = TheCauldronStream(
        ["aokvqa", "vqav2", "ocrvqa", "tallyqa", "iconqa"], 2000, seed=SEED
    )
    batch_iter = iter(stream)
    batch: list = []

    step = 0
    accum = 0
    n_skipped = 0
    print(f"Iterating from step 0 to ~{MAX_STEPS}, looking for NaN onset...")
    while step < MAX_STEPS:
        while len(batch) < BATCH_SIZE:
            try:
                batch.append(next(batch_iter))
            except StopIteration:
                stream = TheCauldronStream(
                    ["aokvqa", "vqav2", "ocrvqa", "tallyqa", "iconqa"], 2000, seed=SEED + step
                )
                batch_iter = iter(stream)

        # Gather sample metadata before tokenizing
        meta = []
        for ex in batch:
            text = ex["messages"][1]["content"][0]["text"]
            user = ex["messages"][0]["content"][-1]["text"]
            n_imgs = sum(1 for c in ex["messages"][0]["content"] if c.get("type") == "image")
            img_sizes = [im.size for im in ex["images"]]
            meta.append({
                "n_imgs": n_imgs, "img_sizes": img_sizes,
                "user_len": len(user), "asst_len": len(text),
                "user_preview": user[:80], "asst_preview": text[:80],
            })

        try:
            ins = collate(batch)
        except Exception as e:
            n_skipped += 1
            batch = []
            continue
        batch = []

        if (ins["labels"] != -100).sum() == 0:
            continue

        # Forward-only repro per advisor: backward is 70% of step time,
        # and `not torch.isfinite(loss)` already catches the trigger condition.
        # Skip backward unless loss is non-finite (we'll re-run with backward
        # at that point if we want to distinguish forward-NaN from backward-NaN).
        with torch.no_grad():
            out = model(**ins)
        loss_finite = bool(torch.isfinite(out.loss).item())
        if not loss_finite:
            print(f"\n💥 NON-FINITE LOSS at step {step}, accum {accum}!")
            print(f"  loss value: {out.loss.item()}")
            print(f"  Batch metadata:")
            for j, m in enumerate(meta):
                print(f"    [{j}] n_imgs={m['n_imgs']} img_sizes={m['img_sizes']} "
                      f"user_len={m['user_len']} asst_len={m['asst_len']}")
                print(f"        USER: {m['user_preview']!r}")
                print(f"        ASST: {m['asst_preview']!r}")
            # Probe inputs for NaN/inf
            for k, v in ins.items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    n_nan = int(v.isnan().sum())
                    n_inf = int(v.isinf().sum())
                    if n_nan or n_inf:
                        print(f"  ⚠ {k}: nan={n_nan} inf={n_inf}")
            # Save the offending batch for follow-up discriminator test
            import pickle
            from pathlib import Path
            out_dir = Path(CKPT).parent / "nan_repro"
            out_dir.mkdir(exist_ok=True)
            payload = {
                "step": step,
                "accum": accum,
                "loss": out.loss.item(),
                "meta": meta,
                "input_ids": ins["input_ids"].cpu(),
                "attention_mask": ins["attention_mask"].cpu(),
                "pixel_values": ins["pixel_values"].cpu(),
                "pixel_attention_mask": ins.get("pixel_attention_mask").cpu() if "pixel_attention_mask" in ins else None,
                "labels": ins["labels"].cpu(),
            }
            with (out_dir / "bad_batch.pkl").open("wb") as f:
                pickle.dump(payload, f)
            print(f"\n  → bad batch saved to {out_dir}/bad_batch.pkl for discriminator test")
            return

        accum += 1
        if accum >= GRAD_ACCUM:
            step += 1
            accum = 0
            if step % 100 == 0:
                print(f"  step {step}: loss {out.loss.item():.4f} (clean)", flush=True)
            if step >= START_FROM and step % 5 == 0:
                # dense logging in danger zone
                print(f"  [danger zone] step {step}: loss {out.loss.item():.4f}", flush=True)

    print(f"\nReached step {step} without NaN. n_skipped={n_skipped}.")


if __name__ == "__main__":
    main()
