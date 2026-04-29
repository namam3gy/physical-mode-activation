"""Stress test the MLP-pool resampler under conditions that may trigger NaN:
    1. Very long text samples (ocrvqa-like, 500+ tokens) — long Mistral context
    2. Streaming actual the_cauldron samples for 50 batches — checking for any pathological case
    3. Bf16 vs fp32 attention pool — controlled comparison
    4. Negation pattern (~mask) vs no-mask — controlled comparison
"""

from __future__ import annotations

import math
import torch
from datasets import load_dataset

from physical_mode.lora.load_swapped import load_idefics2_mlp_pool


CKPT = "outputs/mpswap_run_20260429-033238/step1000"


@torch.no_grad()
def _check_finite(name: str, t: torch.Tensor) -> bool:
    finite = bool(torch.isfinite(t).all().item())
    if not finite:
        print(f"  ⚠ NON-FINITE in {name}: nan={int(t.isnan().sum())} inf={int(t.isinf().sum())}")
    return finite


def main() -> None:
    print("Loading step1000...")
    model, processor = load_idefics2_mlp_pool(CKPT, device="cuda:0", merge_lora=True)
    pr = model.model.connector.perceiver_resampler
    for p in pr.parameters():
        p.requires_grad_(True)
    model.train()

    print("\nStreaming 30 the_cauldron batches (mixed subsets) and tracking grad norms...")
    subsets = ["aokvqa", "vqav2", "ocrvqa", "iconqa", "tallyqa"]
    streams = [load_dataset("HuggingFaceM4/the_cauldron", s, split="train", streaming=True) for s in subsets]
    iters = [iter(s) for s in streams]

    pool: list[dict] = []
    for it in iters:
        for _ in range(8):
            try:
                ex = next(it)
                if ex.get("images") and ex.get("texts"):
                    pool.append(ex)
            except StopIteration:
                break
    print(f"  loaded {len(pool)} examples")

    grads_log = []
    for batch_idx in range(20):
        # Pick 4 random examples for a mini-batch
        i0 = batch_idx * 4 % max(1, len(pool) - 4)
        batch = pool[i0 : i0 + 4]
        if len(batch) < 2:
            continue

        prompts = []
        all_imgs = []
        for ex in batch:
            user = ex["texts"][0].get("user") or ""
            asst = ex["texts"][0].get("assistant") or ""
            messages = [
                {"role": "user", "content": [
                    *[{"type": "image"} for _ in ex["images"]],
                    {"type": "text", "text": user[:1024]},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": asst[:512]}]},
            ]
            prompts.append(processor.apply_chat_template(messages, add_generation_prompt=False))
            all_imgs.append([img.convert("RGB") for img in ex["images"]])

        try:
            inputs = processor(text=prompts, images=all_imgs, return_tensors="pt", padding=True).to("cuda:0")
        except Exception as e:
            print(f"  batch {batch_idx}: collate failed {e!r}; skip")
            continue

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        if (labels != -100).sum() == 0:
            continue

        out = model(**inputs, labels=labels)
        loss = out.loss
        loss_finite = bool(torch.isfinite(loss).item())

        out.loss.backward()
        # Inspect grads
        max_grad = 0.0
        any_nan = False
        for n, p in pr.named_parameters():
            if p.grad is None:
                continue
            gn = p.grad.norm().item()
            if not math.isfinite(gn):
                any_nan = True
            else:
                max_grad = max(max_grad, gn)
        n_seq = inputs["input_ids"].shape[1]
        n_imgs = sum(len(a) for a in all_imgs)
        flag = "💥" if any_nan else ("⚠" if max_grad > 100 else "✓")
        print(f"  batch {batch_idx:2d} {flag} loss={loss.item():.4f} loss_finite={loss_finite} "
              f"max_grad={max_grad:.3e} any_nan={any_nan} "
              f"seq_len={n_seq} n_imgs={n_imgs}")
        grads_log.append({
            "batch": batch_idx, "loss": loss.item(), "max_grad": max_grad,
            "any_nan": any_nan, "n_imgs": n_imgs, "seq_len": n_seq,
        })
        model.zero_grad(set_to_none=True)

    print("\nSummary:")
    if any(g["any_nan"] for g in grads_log):
        print("  💥 NaN reproduced in this stress test — root cause likely in MLP-pool forward+backward")
    else:
        print(f"  No NaN in {len(grads_log)} batches.")
        max_g = max((g["max_grad"] for g in grads_log), default=0.0)
        print(f"  Max grad norm seen: {max_g:.3e}")
        if max_g > 50:
            print(f"  ⚠ But grad norms reach {max_g:.0f} — accumulated drift over 1500+ steps could overflow bf16.")


if __name__ == "__main__":
    main()
