"""Smoke test — load Idefics2, swap perceiver→MLPPoolResampler, run a forward pass.

Verifies:
    1. Module swap completes without shape errors.
    2. Untrained module produces non-NaN output of expected shape (B, 64, 4096).
    3. Generation runs end-to-end (output is bound to be garbage at this point —
       this only confirms the wiring is correct, not the convergence).

Run: ``uv run python scripts/m_pswap_smoke.py``.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.lora.idefics2_mlp_resampler import (
    MLPPoolResampler,
    count_params,
    swap_perceiver_to_mlp_pool,
)


MODEL_ID = "HuggingFaceM4/idefics2-8b"


def main() -> None:
    print(f"loading {MODEL_ID} on cuda:0 (bf16)...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0"
    )

    pr = model.model.connector.perceiver_resampler
    pr_params = count_params(pr)
    print(f"original perceiver_resampler params: {pr_params:,}")

    new = swap_perceiver_to_mlp_pool(model, n_heads=8, seed=42)
    new_params = count_params(new)
    print(f"new MLPPoolResampler params: {new_params:,}  ({new_params / pr_params * 100:.1f}% of perceiver)")
    assert isinstance(model.model.connector.perceiver_resampler, MLPPoolResampler)

    # 1. Direct forward through the new module — context shape is (B, S, 4096) post-modality_projection.
    print("---- direct module forward shape check ----")
    context = torch.randn(2, 200, 4096, dtype=torch.bfloat16, device="cuda:0")
    attn_mask = torch.ones(2, 200, dtype=torch.long, device="cuda:0")
    out = new(context=context, attention_mask=attn_mask)
    print(f"  out.shape: {tuple(out.shape)}  out.dtype: {out.dtype}")
    assert out.shape == (2, 64, 4096), f"unexpected output shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite values in output"
    print("  ✓ shape + finite check passes")

    # 2. End-to-end generation on a 1×1 white-noise image — confirms wiring is intact.
    print("---- end-to-end generation (untrained — expect garbage) ----")
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[[img]], return_tensors="pt").to("cuda:0")

    model.eval()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    print(f"  output (untrained, expect garbage): {text!r}")
    print("  ✓ generation pipeline runs end-to-end without errors")

    print("\nSMOKE OK — module swap is correctly wired. Ready for training.")


if __name__ == "__main__":
    main()
