"""Diagnose what attention_mask actually flows through MLPPoolResampler.

Loads step1000 checkpoint (last good weights), then probes:
    1. What attention_mask values does Idefics2's connector pass to perceiver_resampler?
    2. Are there ever rows where all keys are masked (the documented NaN trigger)?
    3. Forward + backward gradient with bf16 — do gradients become NaN?
    4. Does swapping out the ~mask negation for an additive -65504 bias change anything?
"""

from __future__ import annotations

import torch
from PIL import Image

from physical_mode.lora.load_swapped import load_idefics2_mlp_pool


CKPT = "outputs/mpswap_run_20260429-033238/step1000"


def main() -> None:
    print("Loading step1000 (last clean weights)...")
    model, processor = load_idefics2_mlp_pool(CKPT, device="cuda:0", merge_lora=True)
    # Set perceiver to require_grad so the backward pass exercises the relevant path.
    pr = model.model.connector.perceiver_resampler
    for p in pr.parameters():
        p.requires_grad_(True)
    model.train()

    captured: dict = {}

    def hook(module, args, kwargs):
        ctx = kwargs.get("context") if "context" in kwargs else (args[0] if args else None)
        am = kwargs.get("attention_mask") if "attention_mask" in kwargs else (args[1] if len(args) > 1 else None)
        captured["context_shape"] = tuple(ctx.shape) if ctx is not None else None
        captured["context_dtype"] = ctx.dtype if ctx is not None else None
        captured["context_finite"] = bool(torch.isfinite(ctx).all().item()) if ctx is not None else None
        if am is not None:
            captured["am_shape"] = tuple(am.shape)
            captured["am_dtype"] = am.dtype
            captured["am_unique"] = torch.unique(am).cpu().tolist()
            captured["am_min"] = am.min().item()
            captured["am_max"] = am.max().item()
            captured["am_zero_count_per_row"] = (am == 0).sum(dim=-1).cpu().tolist()
            captured["am_full_padded_rows"] = ((am == 0).all(dim=-1)).sum().item()
        return None

    handle = pr.register_forward_pre_hook(hook, with_kwargs=True)

    # Build a few different batch configurations and probe
    img1 = Image.new("RGB", (980, 980), color=(127, 127, 127))
    img2 = Image.new("RGB", (980, 980), color=(64, 64, 200))

    def run(messages_batch, images_batch, label: str) -> None:
        prompts = [processor.apply_chat_template(m, add_generation_prompt=False) for m in messages_batch]
        inputs = processor(text=prompts, images=images_batch, return_tensors="pt", padding=True).to("cuda:0")
        labels = inputs["input_ids"].clone()
        # Make all tokens labeled (not realistic for training but exposes the gradient path)
        out = model(**inputs, labels=labels)
        print(f"\n=== {label} ===")
        print(f"  loss: {out.loss.item():.4f}  finite: {torch.isfinite(out.loss).item()}")
        for k, v in captured.items():
            sv = str(v)
            if len(sv) > 100:
                sv = sv[:100] + "..."
            print(f"  {k}: {sv}")
        # Backward to inspect gradients
        out.loss.backward()
        for n, p in pr.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            print(f"  grad {n[:60]:60s}: norm={g.norm().item():.4e} finite={torch.isfinite(g).all().item()}")
        model.zero_grad(set_to_none=True)

    msgs1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    msgs1_full = [
        msgs1[0],
        {"role": "assistant", "content": [{"type": "text", "text": "A picture."}]},
    ]

    # Config 1: all batches single image
    run([msgs1_full] * 4, [[img1]] * 4, "single-image batch (uniform)")

    # Config 2: mixed 1- and 2-image batch
    msgs_2img = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare."}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "Different."}]}]
    run([msgs1_full, msgs_2img], [[img1], [img1, img2]], "mixed 1/2 image batch (potential padding)")

    handle.remove()
    print("\nDONE")


if __name__ == "__main__":
    main()
