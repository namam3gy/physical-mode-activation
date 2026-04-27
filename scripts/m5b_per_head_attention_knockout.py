"""M5b — Per-head attention knockout (necessity test) on Qwen2.5-VL.

Refines `m5b_attention_mlp_knockout.py`'s layer-level finding (single-layer
attention had IE=0 everywhere) by zeroing one (layer, head) at a time. If
attention is truly redundant at the head level too, IE stays at 0. If a
small set of heads carry the visual-token → text decision attention, those
heads will show non-zero IE.

Method:
- Restrict to L8-L14 (the MLP-necessity zone identified by the layer-level
  knockout: L9 IE=+1.0, L8/L10/L11/L14 partial, L12/L13 0).
- For each clean SIP stim (n=20), each (layer L, head h):
  - Register forward_pre_hook on `layers[L].self_attn.o_proj`.
  - Hook zeros the input slice `x[:, :, h*head_dim:(h+1)*head_dim]` at
    prefill (seq_len > 1). This zeros head h's contribution before the
    output projection — cleanest per-head ablation.
- IE_necessity[L, h] = baseline_phys_rate − ablated_phys_rate.

Outputs:
- outputs/m5b_per_head/per_pair_results.csv (per stim × (L,h))
- outputs/m5b_per_head/per_head_ie.csv (aggregated)
- docs/figures/m5b_per_head_attention_ie.png (heatmap)

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_per_head_attention_knockout.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "outputs" / "m5b_per_head"
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


PROMPT_FC_TEMPLATE = (
    "The image shows a {label}. Which option best describes what will happen next?\n"
    "A) It falls down.\n"
    "B) It stays still.\n"
    "C) It moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)


def build_clean_pool(n_pairs: int) -> list[str]:
    manifest = pd.read_csv(PROJECT_ROOT / "outputs/m5b_sip/manifest.csv")
    return manifest["clean_sample_id"].head(n_pairs).tolist()


def make_zero_head_pre_hook(head_idx: int, head_dim: int):
    """Pre-hook on o_proj: zero one head's slice in the input tensor at prefill."""
    fired = [False]
    s = head_idx * head_dim
    e = s + head_dim
    def hook(module, inputs):
        if fired[0]:
            return inputs
        x = inputs[0]
        if x.shape[1] == 1:  # decode pass
            return inputs
        fired[0] = True
        x_new = x.clone()
        x_new[..., s:e] = 0.0
        return (x_new,) + inputs[1:]
    return hook


@torch.no_grad()
def run_with_head_ablation(model, processor, pil: Image.Image, prompt: str,
                           o_proj_module, head_idx: int, head_dim: int,
                           device: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = o_proj_module.register_forward_pre_hook(
        make_zero_head_pre_hook(head_idx, head_dim)
    )
    try:
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    finally:
        handle.remove()
    return text


@torch.no_grad()
def run_baseline(model, processor, pil: Image.Image, prompt: str, device: str) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _is_physics_letter(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return -1
    c = t[0]
    if c in ("A", "B", "C"):
        return 1
    if c == "D":
        return 0
    return -1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="circle")
    p.add_argument("--layers", default="8,9,10,11,12,13,14",
                   help="comma-separated LM layers to sweep")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_layers = [int(x) for x in args.layers.split(",")]
    clean_sids = build_clean_pool(args.n_pairs)
    print(f"Using {len(clean_sids)} clean stim from M5b SIP manifest.")
    print(f"Target layers: {target_layers}")

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    layers = model.model.language_model.layers

    # Resolve per-head shape from the o_proj weight: in_features = num_heads * head_dim.
    sample_attn = layers[0].self_attn
    o_proj = sample_attn.o_proj
    in_features = o_proj.in_features
    # Qwen2.5 LM config keeps num_attention_heads on text config / model config.
    num_heads = getattr(sample_attn, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(sample_attn.config, "num_attention_heads", None) if hasattr(sample_attn, "config") else None
    if num_heads is None:
        num_heads = model.config.text_config.num_attention_heads
    head_dim = in_features // num_heads
    print(f"  LM config: num_heads={num_heads}, head_dim={head_dim} (in_features={in_features}).")

    prompt = PROMPT_FC_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, sid in enumerate(clean_sids):
        path = M2_STIM_DIR / f"{sid}.png"
        if not path.exists():
            print(f"  [SKIP] missing {sid}")
            continue
        pil = Image.open(path).convert("RGB")

        bl = run_baseline(model, processor, pil, prompt, args.device)
        bl_phys = _is_physics_letter(bl)

        per_head = {}
        for L in target_layers:
            for h in range(num_heads):
                txt = run_with_head_ablation(model, processor, pil, prompt,
                                             layers[L].self_attn.o_proj,
                                             h, head_dim, args.device)
                per_head[(L, h)] = {"text": txt, "phys": _is_physics_letter(txt)}

        elapsed = (time.time() - t_start) / 60
        n_break = sum(1 for v in per_head.values() if v["phys"] == 0)
        print(f"  [{i+1}/{len(clean_sids)}] {sid} bl={bl_phys} broken={n_break}/{len(per_head)} "
              f"elapsed={elapsed:.1f}min")

        row = {"sample_id": sid, "baseline_text": bl, "baseline_phys": bl_phys}
        for (L, h), v in per_head.items():
            row[f"L{L}_h{h}_phys"] = v["phys"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_pair_results.csv", index=False)

    bl_phys_rate = float((df["baseline_phys"] == 1).sum() /
                         max(1, (df["baseline_phys"] >= 0).sum()))

    agg_rows = []
    for L in target_layers:
        for h in range(num_heads):
            col = f"L{L}_h{h}_phys"
            valid = df[col] >= 0
            if valid.sum() == 0:
                continue
            ablated_rate = int(df.loc[valid, col].sum()) / int(valid.sum())
            agg_rows.append({
                "layer": L, "head": h, "n": int(valid.sum()),
                "baseline_phys_rate": bl_phys_rate,
                "ablated_phys_rate": ablated_rate,
                "ie_necessity": bl_phys_rate - ablated_rate,
            })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(OUT_DIR / "per_head_ie.csv", index=False)

    # Heatmap (layers × heads)
    grid = np.zeros((len(target_layers), num_heads))
    for r in agg_rows:
        i_l = target_layers.index(r["layer"])
        grid[i_l, r["head"]] = r["ie_necessity"]

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * len(target_layers) + 1)))
    im = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels(range(num_heads), fontsize=8)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([f"L{L}" for L in target_layers])
    ax.set_xlabel("attention head")
    ax.set_ylabel("LM layer")
    ax.set_title(f"M5b — per-head attention knockout IE_necessity (n={len(df)})")
    plt.colorbar(im, ax=ax, label="IE_necessity = Δ P(physics)")

    nz_top = agg_df.sort_values("ie_necessity", ascending=False).head(10)
    print("\n=== Top 10 (layer, head) by IE_necessity ===")
    print(nz_top.round(3).to_string(index=False))

    out_png = FIG_DIR / "m5b_per_head_attention_ie.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
