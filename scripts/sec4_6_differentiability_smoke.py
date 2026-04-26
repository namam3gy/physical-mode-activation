"""§4.6 iter-1 gate: verify gradient flows from pixel_values through Qwen2.5-VL
vision tower + projector + LM 0..10 to the loss `−⟨mean(h_L10[visual]), v_L10⟩`.

If this passes, Approach A (pixel-space gradient ascent) is feasible. If it
fails (e.g., grad is None, or NaN, or the vision tower is not differentiable),
fall back to Approach C (activation-space optimization).

Usage:
    uv run python scripts/sec4_6_differentiability_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
STEERING_NPZ = PROJECT_ROOT / "outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz"
BASELINE_IMG = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_000.png"
PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."
LAYER = 10


def main() -> None:
    print("Loading processor + model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    v_unit = torch.from_numpy(np.load(STEERING_NPZ)["v_unit_10"]).to("cuda:0", torch.float32)
    print(f"v_L10 shape={tuple(v_unit.shape)} norm={v_unit.norm().item():.6f}")

    pil = Image.open(BASELINE_IMG).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil], text=[text], return_tensors="pt")
    raw = {k: v.to("cuda:0") for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)

    print(f"pixel_values shape={tuple(pv_leaf.shape)} dtype={pv_leaf.dtype} requires_grad={pv_leaf.requires_grad}")

    with torch.enable_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    h = out.hidden_states[LAYER + 1][0]  # +1 to skip embedding
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_index", None)
    print(f"image_token_id={image_token_id}")
    mask = (inputs["input_ids"][0] == image_token_id)
    n_visual = int(mask.sum().item())
    print(f"n_visual_tokens={n_visual}")
    h_visual = h[mask]

    h_mean = h_visual.mean(dim=0).to(torch.float32)
    projection = (h_mean * v_unit).sum()
    print(f"baseline projection={projection.item():.6f}")

    loss = -projection
    loss.backward()

    grad = pv_leaf.grad
    if grad is None:
        print("FAIL: pixel_values.grad is None — backward did not reach the leaf.")
        raise SystemExit(1)
    print(
        f"pixel_values.grad shape={tuple(grad.shape)} max_abs={grad.abs().max().item():.4e} "
        f"has_nan={torch.isnan(grad).any().item()}"
    )
    if torch.isnan(grad).any():
        print("FAIL: gradient contains NaN.")
        raise SystemExit(1)
    if grad.abs().max().item() < 1e-12:
        print("FAIL: gradient is zero — graph is detached somewhere.")
        raise SystemExit(1)

    print("PASS: gradient reaches pixel_values; magnitudes finite and non-trivial.")


if __name__ == "__main__":
    main()
