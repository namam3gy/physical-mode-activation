"""M5b — Attention + MLP knockout (necessity test) on Qwen2.5-VL.

Complement to `m5b_sip_activation_patching.py`:
- SIP patching tests SUFFICIENCY of clean's hidden state at L for
  physics-mode (corrupted → patched → physics?).
- This script tests NECESSITY of L's attention or MLP for physics-
  mode (clean → ablated → still physics?).

Method:
- Take the clean SIP pair stim (cue=both seeds, baseline PMR=1).
- For each layer L:
  (a) Hook on `layers[L].self_attn` to zero the attention output.
      → tests if L's attention is necessary for physics response.
  (b) Hook on `layers[L].mlp` to zero the MLP output.
      → tests if L's MLP is necessary for physics response.
- IE_necessity = baseline PMR − ablated PMR. High = L is necessary.

Outputs:
- outputs/m5b_knockout/per_pair_results.csv
- outputs/m5b_knockout/per_layer_ie_attention.csv
- outputs/m5b_knockout/per_layer_ie_mlp.csv
- docs/figures/m5b_knockout_per_layer_ie.png

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_attention_mlp_knockout.py
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

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "outputs" / "m5b_knockout"
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
    """Reuse SIP manifest's clean_sample_id list."""
    manifest = pd.read_csv(PROJECT_ROOT / "outputs/m5b_sip/manifest.csv")
    return manifest["clean_sample_id"].head(n_pairs).tolist()


def make_zero_output_hook():
    """Zero the *first* tensor of the output (attention or MLP output)
    while keeping the rest of the tuple intact."""
    fired = [False]
    def hook(module, inputs, output):
        if fired[0]:
            return  # only apply at prefill (first forward)
        # We only want to ablate at the prefill pass. The check below
        # uses the input shape: prefill has seq_len > 1.
        h_in = inputs[0] if len(inputs) > 0 else None
        if h_in is not None and h_in.shape[1] == 1:
            return  # decode pass; skip
        fired[0] = True
        if isinstance(output, tuple):
            zero = torch.zeros_like(output[0])
            return (zero,) + output[1:]
        return torch.zeros_like(output)
    return hook


@torch.no_grad()
def run_with_ablation(model, processor, pil: Image.Image, prompt: str,
                      module_to_hook, device: str) -> str:
    """Run generate(pil) with a hook that zeros `module_to_hook`'s output
    on prefill only."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = module_to_hook.register_forward_hook(make_zero_output_hook())
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
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_sids = build_clean_pool(args.n_pairs)
    print(f"Using {len(clean_sids)} clean stim from M5b SIP manifest.")

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    layers = model.model.language_model.layers
    n_layers = len(layers)
    print(f"  {n_layers} LM layers.")

    prompt = PROMPT_FC_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, sid in enumerate(clean_sids):
        path = M2_STIM_DIR / f"{sid}.png"
        if not path.exists():
            print(f"  [SKIP] missing {sid}")
            continue
        pil = Image.open(path).convert("RGB")

        # Baseline (no ablation)
        bl = run_baseline(model, processor, pil, prompt, args.device)
        bl_phys = _is_physics_letter(bl)

        per_layer_attn = {}
        per_layer_mlp = {}
        for L in range(n_layers):
            # Attention knockout
            attn_text = run_with_ablation(model, processor, pil, prompt,
                                          layers[L].self_attn, args.device)
            per_layer_attn[L] = {"text": attn_text, "phys": _is_physics_letter(attn_text)}
            # MLP knockout
            mlp_text = run_with_ablation(model, processor, pil, prompt,
                                         layers[L].mlp, args.device)
            per_layer_mlp[L] = {"text": mlp_text, "phys": _is_physics_letter(mlp_text)}

        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{len(clean_sids)}] {sid} baseline={bl_phys} ({bl[:30]!r}) elapsed={elapsed:.1f}min")

        rows.append({
            "sample_id": sid, "baseline_text": bl, "baseline_phys": bl_phys,
            **{f"L{L}_attn_phys": per_layer_attn[L]["phys"] for L in range(n_layers)},
            **{f"L{L}_mlp_phys": per_layer_mlp[L]["phys"] for L in range(n_layers)},
            **{f"L{L}_attn_text": per_layer_attn[L]["text"] for L in range(n_layers)},
            **{f"L{L}_mlp_text": per_layer_mlp[L]["text"] for L in range(n_layers)},
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_pair_results.csv", index=False)

    # Aggregate
    bl_phys_rate = float((df["baseline_phys"] == 1).sum() /
                         max(1, (df["baseline_phys"] >= 0).sum()))

    attn_rows = []
    mlp_rows = []
    for L in range(n_layers):
        ac = f"L{L}_attn_phys"
        mc = f"L{L}_mlp_phys"
        v_a = df[ac] >= 0
        v_m = df[mc] >= 0
        if v_a.sum() > 0:
            patched = int(df.loc[v_a, ac].sum()) / int(v_a.sum())
            attn_rows.append({"layer": L, "n": int(v_a.sum()),
                              "baseline_phys_rate": bl_phys_rate,
                              "ablated_phys_rate": patched,
                              "ie_necessity": bl_phys_rate - patched})
        if v_m.sum() > 0:
            patched = int(df.loc[v_m, mc].sum()) / int(v_m.sum())
            mlp_rows.append({"layer": L, "n": int(v_m.sum()),
                             "baseline_phys_rate": bl_phys_rate,
                             "ablated_phys_rate": patched,
                             "ie_necessity": bl_phys_rate - patched})
    attn_df = pd.DataFrame(attn_rows)
    mlp_df = pd.DataFrame(mlp_rows)
    attn_df.to_csv(OUT_DIR / "per_layer_ie_attention.csv", index=False)
    mlp_df.to_csv(OUT_DIR / "per_layer_ie_mlp.csv", index=False)
    print("\n=== Attention knockout ===")
    print(attn_df.round(3).to_string(index=False))
    print("\n=== MLP knockout ===")
    print(mlp_df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if not attn_df.empty:
        axes[0].plot(attn_df["layer"], attn_df["ie_necessity"], "o-",
                     color="#d62728", linewidth=2, markersize=8)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("LM layer (target ablated)")
    axes[0].set_ylabel("IE_necessity = Δ P(physics)")
    axes[0].set_title("Attention knockout (necessity)")
    axes[0].grid(alpha=0.3); axes[0].set_ylim(-0.1, 1.1)
    if not mlp_df.empty:
        axes[1].plot(mlp_df["layer"], mlp_df["ie_necessity"], "s-",
                     color="#2ca02c", linewidth=2, markersize=8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("LM layer (target ablated)")
    axes[1].set_ylabel("IE_necessity = Δ P(physics)")
    axes[1].set_title("MLP knockout (necessity)")
    axes[1].grid(alpha=0.3); axes[1].set_ylim(-0.1, 1.1)
    fig.suptitle(f"M5b — Qwen2.5-VL knockout necessity test (n={len(df)} clean stim)")
    fig.tight_layout()
    out_png = FIG_DIR / "m5b_knockout_per_layer_ie.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
