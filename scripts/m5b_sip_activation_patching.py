"""M5b — Semantic Image Pairs + LM activation patching on Qwen2.5-VL.

Method:
  - Build SIP from M2 forced-choice predictions: for each (object, bg,
    event), pair cue=both seed_i (clean: physics-mode response) with
    cue=none seed_i (corrupted: abstract response). Filter to pairs
    where clean abs_rate=0 AND corrupted abs_rate=1.
  - For each pair, run LM activation patching across all 28 Qwen2-7B
    layers: replace corrupted's hidden state at layer L (visual-token
    positions) with cached clean's hidden state at layer L; measure
    output change.
  - Indirect effect (IE) per layer = P(physics-mode response | patched
    corrupted at L) − P(physics-mode response | corrupted baseline).

Outputs:
  - outputs/m5b_sip/manifest.csv
  - outputs/m5b_sip/per_layer_ie.csv
  - docs/figures/m5b_sip_per_layer_ie.png
  - docs/insights/m5b_sip_activation_patching.md (+ ko)

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_activation_patching.py
"""

from __future__ import annotations

import argparse
import json
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
OUT_DIR = PROJECT_ROOT / "outputs" / "m5b_sip"
M2_RUN = PROJECT_ROOT / "outputs/mvp_full_20260424-094103_8ae1fa3d"
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


# Mirrors src/physical_mode/inference/prompts.py FORCED_CHOICE_TEMPLATE
# (M2 FC prompt — produces clean letter-first responses).
PROMPT_FC_TEMPLATE = (
    "The image shows a {label}. Which option best describes what will happen next?\n"
    "A) It falls down.\n"
    "B) It stays still.\n"
    "C) It moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)


def build_sip_manifest() -> pd.DataFrame:
    """Build SIP manifest from M2 forced-choice predictions.

    Pair structure:
      - clean = cue_level=both, abs_rate=0 (physics)
      - corrupted = cue_level=none, abs_rate=1 (abstract)
      - same (object_level, bg_level, event_template), seed-by-index match.
    """
    df = pd.read_parquet(M2_RUN / "predictions.parquet")
    df_fc = df[df.prompt_variant == "forced_choice"].copy()
    df_fc["letter"] = df_fc["raw_text"].str.strip().str[0]
    df_fc["abstract"] = (df_fc["letter"] == "D").astype(int)
    cell = df_fc.groupby(
        ["object_level", "bg_level", "cue_level", "event_template", "seed"]
    ).agg(abs_rate=("abstract", "mean"), n=("letter", "count"),
          sample_id=("sample_id", "first")).reset_index()

    # Map seed within cue_level to a 0..9 index for matching
    cell["seed_idx"] = cell.groupby(
        ["object_level", "bg_level", "cue_level", "event_template"]
    )["seed"].rank(method="first").astype(int) - 1

    pairs = []
    for (obj, bg, evt), grp in cell.groupby(["object_level", "bg_level", "event_template"]):
        clean_grp = grp[grp.cue_level == "both"]
        corr_grp = grp[grp.cue_level == "none"]
        for idx in range(10):
            cs = clean_grp[clean_grp.seed_idx == idx]
            cr = corr_grp[corr_grp.seed_idx == idx]
            if len(cs) != 1 or len(cr) != 1:
                continue
            if cs["abs_rate"].iloc[0] > 0.0 or cr["abs_rate"].iloc[0] < 1.0:
                continue
            pairs.append({
                "obj": obj, "bg": bg, "event": evt, "pair_idx": idx,
                "clean_sample_id": cs["sample_id"].iloc[0],
                "corr_sample_id": cr["sample_id"].iloc[0],
                "clean_abs_rate": cs["abs_rate"].iloc[0],
                "corr_abs_rate": cr["abs_rate"].iloc[0],
            })
    return pd.DataFrame(pairs)


def _resolve_lm_layers(model) -> list:
    """Return the list of LM transformer blocks for Qwen2.5-VL."""
    return list(model.model.language_model.layers)


def _resolve_image_token_id(model) -> int:
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise RuntimeError("could not resolve image_token_id")


@torch.no_grad()
def cache_clean_hidden(model, processor, pil: Image.Image, prompt: str,
                       n_layers: int, device: str) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Forward(clean) and cache per-layer hidden state at visual-token positions.

    Returns (cache: dict[L → (n_visual, dim)], visual_mask: bool tensor (1, seq_len))
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_token_id = _resolve_image_token_id(model)
    visual_mask = (inputs["input_ids"][0] == image_token_id)

    out = model(**inputs, output_hidden_states=True, return_dict=True)
    cache = {}
    # hidden_states[0] = embedding; hidden_states[L+1] = output of layer L.
    for L in range(n_layers):
        h = out.hidden_states[L + 1][0]  # (seq_len, dim)
        cache[L] = h[visual_mask].detach().clone()
    return cache, visual_mask


def make_patch_hook(cached_visual: torch.Tensor, prompt_seq_len: int):
    """Hook for layers[L].forward: replace visual-token positions of the
    output hidden state with cached clean values. Fires only on prefill
    (when seq_len == prompt_seq_len)."""
    fired = [False]
    def hook(module, inputs, output):
        if fired[0]:
            return  # only patch on prefill
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != prompt_seq_len:
            return  # decode pass; skip
        # Patch at visual-token positions.
        # Caller must ensure cached_visual matches the corrupted run's
        # visual-token slot count. We assert this in run_patched().
        fired[0] = True
        h_new = h.clone()
        h_new[0, :cached_visual.shape[0], :] = cached_visual.to(h.dtype).to(h.device)
        # NB: positions are not strictly first n_visual; we use the corrupted
        # run's own visual_mask captured via the inputs. To be exact, we
        # apply via mask in run_patched (see below).
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook


@torch.no_grad()
def run_patched(model, processor, pil: Image.Image, prompt: str,
                cached_visual: torch.Tensor, target_L: int, device: str) -> str:
    """Forward + generate(corrupted) with patching on layers[target_L].
    Returns generated text."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_token_id = _resolve_image_token_id(model)
    visual_mask_corr = (inputs["input_ids"][0] == image_token_id)
    n_visual_corr = int(visual_mask_corr.sum().item())
    n_visual_clean = cached_visual.shape[0]
    if n_visual_corr != n_visual_clean:
        return f"[SKIP visual_mismatch corr={n_visual_corr} clean={n_visual_clean}]"

    prompt_seq_len = inputs["input_ids"].shape[1]

    # Hook factory using the corrupted run's visual mask
    fired = [False]
    def hook(module, inputs_, output):
        if fired[0]:
            return
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] != prompt_seq_len:
            return
        fired[0] = True
        h_new = h.clone()
        h_new[0, visual_mask_corr, :] = cached_visual.to(h.dtype).to(h.device)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new

    handle = model.model.language_model.layers[target_L].register_forward_hook(hook)
    try:
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out[:, prompt_seq_len:]
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
    """1 if response starts with A/B/C, 0 if D, NaN otherwise."""
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
    p.add_argument("--n-pairs", type=int, default=20, help="# SIP pairs to use")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="circle")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building SIP manifest...")
    manifest = build_sip_manifest()
    print(f"  {len(manifest)} clean SIP pairs.")
    manifest = manifest.head(args.n_pairs).copy()
    print(f"  Using {len(manifest)} pairs for patching.")
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    n_layers = len(model.model.language_model.layers)
    print(f"  {n_layers} LM layers.")

    prompt = PROMPT_FC_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, row in manifest.iterrows():
        clean_path = M2_STIM_DIR / f"{row.clean_sample_id}.png"
        corr_path = M2_STIM_DIR / f"{row.corr_sample_id}.png"
        if not (clean_path.exists() and corr_path.exists()):
            print(f"  [SKIP] missing stim {row.clean_sample_id} or {row.corr_sample_id}")
            continue
        clean_pil = Image.open(clean_path).convert("RGB")
        corr_pil = Image.open(corr_path).convert("RGB")

        # 1. Cache clean hidden states at all layers
        cache, _ = cache_clean_hidden(model, processor, clean_pil, prompt, n_layers, args.device)

        # 2. Baseline: corrupted no-patch
        bl = run_baseline(model, processor, corr_pil, prompt, args.device)
        bl_phys = _is_physics_letter(bl)

        # 3. Per-layer patched corrupted
        per_layer = {}
        for target_L in range(n_layers):
            patched_text = run_patched(model, processor, corr_pil, prompt,
                                       cache[target_L], target_L, args.device)
            phys = _is_physics_letter(patched_text)
            per_layer[target_L] = {"text": patched_text, "phys": phys}

        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{len(manifest)}] {row.obj}/{row.bg}/{row.event}/idx={row.pair_idx} "
              f"baseline={bl_phys} ({bl[:30]!r}) elapsed={elapsed:.1f}min")

        rows.append({
            "pair_idx": int(row.pair_idx), "obj": row.obj, "bg": row.bg,
            "event": row.event,
            "clean_sample_id": row.clean_sample_id,
            "corr_sample_id": row.corr_sample_id,
            "baseline_corr_text": bl,
            "baseline_corr_phys": bl_phys,
            **{f"L{L}_phys": per_layer[L]["phys"] for L in range(n_layers)},
            **{f"L{L}_text": per_layer[L]["text"] for L in range(n_layers)},
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_pair_results.csv", index=False)

    # Aggregate IE per layer
    ie_rows = []
    for L in range(n_layers):
        col = f"L{L}_phys"
        valid = df[col] >= 0
        if valid.sum() == 0:
            continue
        n = int(valid.sum())
        n_phys = int(df.loc[valid, col].sum())
        bl_phys_rate = float((df["baseline_corr_phys"] == 1).sum() / max(1, (df["baseline_corr_phys"] >= 0).sum()))
        patched_phys_rate = n_phys / n
        ie = patched_phys_rate - bl_phys_rate
        ie_rows.append({"layer": L, "n_pairs": n,
                        "baseline_phys_rate": bl_phys_rate,
                        "patched_phys_rate": patched_phys_rate,
                        "ie": ie})
    ie_df = pd.DataFrame(ie_rows)
    ie_df.to_csv(OUT_DIR / "per_layer_ie.csv", index=False)
    print("\n=== Per-layer IE summary ===")
    print(ie_df.round(3).to_string(index=False))

    # Plot
    if ie_df.empty:
        print("[WARN] no valid IE rows; skipping figure")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ie_df["layer"], ie_df["ie"], "o-", color="#2e4a7f", linewidth=2, markersize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("LM layer (target patched)")
    ax.set_ylabel("Indirect Effect (Δ P(physics-mode))")
    ax.set_title(f"M5b — SIP activation patching IE per layer (Qwen2.5-VL, n={len(df)} pairs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = FIG_DIR / "m5b_sip_per_layer_ie.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
