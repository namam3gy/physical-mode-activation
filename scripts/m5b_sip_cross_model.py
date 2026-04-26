"""M5b cross-model — SIP + activation patching across 4 non-Qwen VLMs.

Adapts `m5b_sip_activation_patching.py` (Qwen-only) to each non-Qwen
architecture: LLaVA-1.5 (LLaVA + LLaMA 32 layers), LLaVA-Next (LLaVA-
Next + Mistral 32 layers), Idefics2 (Mistral 32 layers), InternVL3
(Qwen2 28 layers, but InternViT vision).

Differences from Qwen version:
- SIP built from OPEN-prompt PMR (since cross-model M2 captures are
  open-only), not FC abstract rate.
- Layer-block resolver per model architecture.
- LLaVA-1.5 has clean class balance (n_neg=105); others very skewed
  (n_neg = 5, 1, 9). Most cross-model SIP construction fails for
  saturated models.

Output: outputs/m5b_sip_cross_model/<model>_per_layer_ie.csv,
        outputs/m5b_sip_cross_model/<model>_per_pair_results.csv,
        docs/figures/m5b_sip_cross_model_<model>_per_layer_ie.png

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_cross_model.py \
        --model-id llava-hf/llava-1.5-7b-hf \
        --capture-pattern "cross_model_llava_capture_*" \
        --label ball
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

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "outputs" / "m5b_sip_cross_model"
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


# OPEN prompt — same as M2 cross-model captures.
PROMPT_OPEN_TEMPLATE = (
    "The image shows a {label}. Describe what will happen to the "
    "{label} in the next moment, in one short sentence."
)


def get_lm_layers(model, model_id: str = None):
    """Return list of LM transformer blocks. Modern transformers
    unifies VLMs at `model.model.language_model.layers` (Qwen2.5-VL,
    LLaVA-1.5/Next, InternVL3); Idefics2 uses `model.model.text_model`."""
    paths = [
        ["model", "language_model", "layers"],
        ["model", "text_model", "layers"],
        ["language_model", "model", "layers"],
    ]
    for path in paths:
        try:
            obj = model
            for attr in path:
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList) and len(obj) > 0:
                return obj
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(f"Could not resolve LM layers for {model_id}")


def _resolve_image_token_id(model) -> int:
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise RuntimeError("could not resolve image_token_id")


def build_sip_open_prompt(capture_pattern: str, label: str = None) -> pd.DataFrame:
    """Build SIP from cross-model open-prompt M2 captures.

    Pairs cue=both seeds (clean: PMR=1) with cue=none seeds (corrupted: PMR=0)
    by index, sharing (object_level, bg_level, event_template).
    """
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{capture_pattern}/predictions.jsonl"))
    cands = [c for c in cands if c.stat().st_size > 0]
    if not cands:
        raise FileNotFoundError(capture_pattern)
    run_dir = cands[-1].parent
    df = pd.read_json(run_dir / "predictions.jsonl", lines=True)
    if "prompt_variant" in df.columns:
        df = df[df["prompt_variant"] == "open"]
    df = df.copy()
    df["pmr"] = df["raw_text"].apply(score_pmr)
    if label is not None:
        df = df[df["label"] == label]
    cell = df.groupby(
        ["object_level", "bg_level", "cue_level", "event_template", "seed"]
    ).agg(pmr=("pmr", "mean"), sample_id=("sample_id", "first")).reset_index()
    cell["seed_idx"] = cell.groupby(
        ["object_level", "bg_level", "cue_level", "event_template"]
    )["seed"].rank(method="first").astype(int) - 1
    pairs = []
    for (obj, bg, evt), grp in cell.groupby(
        ["object_level", "bg_level", "event_template"]
    ):
        clean_grp = grp[grp.cue_level == "both"]
        corr_grp = grp[grp.cue_level == "none"]
        for idx in range(10):
            cs = clean_grp[clean_grp.seed_idx == idx]
            cr = corr_grp[corr_grp.seed_idx == idx]
            if len(cs) != 1 or len(cr) != 1:
                continue
            if cs["pmr"].iloc[0] < 1.0 or cr["pmr"].iloc[0] > 0.0:
                continue
            pairs.append({
                "obj": obj, "bg": bg, "event": evt, "pair_idx": idx,
                "clean_sample_id": cs["sample_id"].iloc[0],
                "corr_sample_id": cr["sample_id"].iloc[0],
                "clean_pmr": cs["pmr"].iloc[0],
                "corr_pmr": cr["pmr"].iloc[0],
            })
    return pd.DataFrame(pairs), run_dir


@torch.no_grad()
def cache_clean_hidden(model, processor, pil: Image.Image, prompt: str,
                       n_layers: int, device: str) -> tuple[dict, torch.Tensor]:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    image_token_id = _resolve_image_token_id(model)
    visual_mask = (inputs["input_ids"][0] == image_token_id)
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    cache = {}
    for L in range(n_layers):
        h = out.hidden_states[L + 1][0]
        cache[L] = h[visual_mask].detach().clone()
    return cache, visual_mask


@torch.no_grad()
def run_patched(model, processor, pil: Image.Image, prompt: str,
                cached_visual: torch.Tensor, target_L: int, device: str,
                lm_layers) -> str:
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
    handle = lm_layers[target_L].register_forward_hook(hook)
    try:
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
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
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--capture-pattern", required=True)
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--label", default="ball",
                   help="Stim label to use; ball gives most physics-mode signal in OPEN")
    p.add_argument("--model-tag", default=None,
                   help="Short tag for output filenames; defaults to model-id stem")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = args.model_tag or args.model_id.split("/")[-1].replace(".", "_")

    print(f"Building SIP from {args.capture_pattern} (label={args.label})...")
    manifest, run_dir = build_sip_open_prompt(args.capture_pattern, args.label)
    print(f"  {len(manifest)} clean SIP pairs available.")
    if len(manifest) == 0:
        print(f"[ABORT] No SIP candidates for {args.model_id} on label={args.label}")
        return
    manifest = manifest.head(args.n_pairs).copy()
    print(f"  Using {len(manifest)} pairs.")
    manifest.to_csv(OUT_DIR / f"{model_tag}_manifest.csv", index=False)

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    lm_layers = get_lm_layers(model, args.model_id)
    n_layers = len(lm_layers)
    print(f"  {n_layers} LM layers via {type(lm_layers).__name__}.")

    prompt = PROMPT_OPEN_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, row in manifest.iterrows():
        clean_path = M2_STIM_DIR / f"{row.clean_sample_id}.png"
        corr_path = M2_STIM_DIR / f"{row.corr_sample_id}.png"
        if not (clean_path.exists() and corr_path.exists()):
            continue
        clean_pil = Image.open(clean_path).convert("RGB")
        corr_pil = Image.open(corr_path).convert("RGB")
        cache, _ = cache_clean_hidden(model, processor, clean_pil, prompt, n_layers, args.device)
        bl = run_baseline(model, processor, corr_pil, prompt, args.device)
        bl_pmr = score_pmr(bl)
        per_layer = {}
        for target_L in range(n_layers):
            patched_text = run_patched(model, processor, corr_pil, prompt,
                                       cache[target_L], target_L, args.device, lm_layers)
            patched_pmr = score_pmr(patched_text) if not patched_text.startswith("[SKIP") else -1
            per_layer[target_L] = {"text": patched_text, "pmr": patched_pmr}
        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{len(manifest)}] {row.obj}/{row.bg}/{row.event}/idx={row.pair_idx} "
              f"baseline_pmr={bl_pmr} ({bl[:40]!r}) elapsed={elapsed:.1f}min")
        rows.append({
            "pair_idx": int(row.pair_idx), "obj": row.obj, "bg": row.bg, "event": row.event,
            "clean_sample_id": row.clean_sample_id, "corr_sample_id": row.corr_sample_id,
            "baseline_corr_text": bl, "baseline_corr_pmr": bl_pmr,
            **{f"L{L}_pmr": per_layer[L]["pmr"] for L in range(n_layers)},
            **{f"L{L}_text": per_layer[L]["text"] for L in range(n_layers)},
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{model_tag}_per_pair_results.csv", index=False)

    ie_rows = []
    for L in range(n_layers):
        col = f"L{L}_pmr"
        valid = df[col] >= 0
        n = int(valid.sum())
        if n == 0:
            continue
        n_phys = int(df.loc[valid, col].sum())
        bl_phys_rate = float((df["baseline_corr_pmr"] == 1).sum() /
                             max(1, (df["baseline_corr_pmr"] >= 0).sum()))
        patched_phys_rate = n_phys / n
        ie = patched_phys_rate - bl_phys_rate
        ie_rows.append({"layer": L, "n_pairs": n,
                        "baseline_phys_rate": bl_phys_rate,
                        "patched_phys_rate": patched_phys_rate, "ie": ie})
    ie_df = pd.DataFrame(ie_rows)
    ie_df.to_csv(OUT_DIR / f"{model_tag}_per_layer_ie.csv", index=False)
    print("\n=== Per-layer IE ===")
    print(ie_df.round(3).to_string(index=False))

    if not ie_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(ie_df["layer"], ie_df["ie"], "o-", color="#2e4a7f", linewidth=2, markersize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("LM layer (target patched)")
        ax.set_ylabel("Indirect Effect (Δ P(physics-mode))")
        ax.set_title(f"M5b — SIP activation patching IE per layer ({args.model_id}, n={len(df)} pairs)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = FIG_DIR / f"m5b_sip_cross_model_{model_tag}_per_layer_ie.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
