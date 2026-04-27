"""Causal SAE-feature intervention on Qwen2.5-VL.

Test whether ablating (zeroing) the top-k physics-cue SAE features in the
vision-encoder output causes physics-mode commitment to drop. Symmetric to
the activation-patching / knockout experiments at the LM side, but at the
encoder.

Pipeline:
1. Load trained SAE + feature ranking.
2. Pick top-k features (largest physics − abstract activation delta).
3. For each evaluation stim: hook on `model.visual` last vision-encoder
   layer's output. The hook reads the visual-token activations, encodes via
   SAE, zeros the chosen features in z, decodes, and writes back.
4. Run inference; compare letter response (PMR) to baseline (no hook).

Outputs:
- outputs/sae_intervention/<tag>/results.csv — per (stim, k_features_zeroed)
  baseline vs intervention prediction.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_intervention.py \
        --sae-dir outputs/sae/qwen_vis31_5120 \
        --layer-key vision_hidden_31 \
        --top-k-list 1,5,10,20 --n-stim 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.sae.train import load_sae


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


PROMPT_FC_TEMPLATE = (
    "The image shows a {label}. Which option best describes what will happen next?\n"
    "A) It falls down.\n"
    "B) It stays still.\n"
    "C) It moves sideways.\n"
    "D) This is an abstract shape — nothing physical happens.\n"
    "Answer with a single letter (A, B, C, or D), then briefly justify."
)


def make_sae_intervention_hook(sae, feature_indices_to_zero: torch.Tensor):
    """Forward hook on the vision encoder's last layer output that subtracts
    *only* the target SAE features' raw-scale contributions, leaving
    everything else (other features + reconstruction residual) bit-identical.

    Bricken et al. 2023 trick. The SAE's `feature_contribution` handles
    input z-score normalization round-trip internally.
    """
    sae_device = next(sae.parameters()).device

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
            rest = output[1:]
        else:
            x = output
            rest = None
        original_dtype = x.dtype
        x_in = x.to(sae_device, dtype=torch.float32)
        flat_shape = x_in.shape
        x_flat = x_in.reshape(-1, flat_shape[-1])
        contribution = sae.feature_contribution(x_flat, feature_indices_to_zero)
        x_new_flat = x_flat - contribution
        x_new = x_new_flat.reshape(flat_shape).to(x.device, dtype=original_dtype)
        if rest is not None:
            return (x_new,) + rest
        return x_new

    return hook


def get_last_vision_layer(model):
    """Resolve the module whose output we want to intercept (= last vision encoder layer)."""
    # Qwen2.5-VL: model.visual is the vision tower; .blocks is a ModuleList of vision encoder layers.
    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is None:
        raise RuntimeError("Could not find model.visual.")
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        raise RuntimeError("Could not find model.visual.blocks.")
    return blocks[-1], len(blocks)


@torch.no_grad()
def run_with_hook(model, processor, pil, prompt, hook_module, hook_fn, device):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = hook_module.register_forward_hook(hook_fn) if hook_fn else None
    try:
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    finally:
        if handle is not None:
            handle.remove()


def _is_physics_letter(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return -1
    c = t[0]
    return 1 if c in ("A", "B", "C") else (0 if c == "D" else -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", type=Path, required=True)
    p.add_argument("--layer-key", default="vision_hidden_31",
                   help="captured-activation key (informational; the script hooks the actual model)")
    p.add_argument("--top-k-list", default="1,5,10,20",
                   help="comma-separated feature counts to zero")
    p.add_argument("--random-controls", type=int, default=0,
                   help="for the largest k in --top-k-list, also test N random "
                        "feature sets of the same size (drawn from features ranked "
                        "below the top-50 to avoid overlap)")
    p.add_argument("--n-stim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="circle")
    p.add_argument("--manifest",
                   default="outputs/m5b_sip/manifest.csv",
                   help="CSV with clean_sample_id column listing stim to use")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "sae_intervention" / args.sae_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = load_sae(args.sae_dir / "sae.pt", device=args.device)
    rank_df = pd.read_csv(args.sae_dir / "feature_ranking.csv")
    top_features = rank_df["feature_idx"].head(max(int(k) for k in args.top_k_list.split(","))).tolist()
    print(f"Top features (by physics − abstract delta): {top_features[:10]}...")

    # Build magnitude-matched random control sets via mass-weighted sampling:
    #   - draw the highest-mass non-top-20 features (rank 21..) until cumulative mass
    #     hits the target window. Avoids the L1-killed tail (rank > ~500) where
    #     features have ~zero mass.
    # This controls for "ablating ~same total activation magnitude" (the advisor's blocker:
    # the previous bottom-of-ranking sample had only ~1% of top-20's mass).
    max_k = max(int(k) for k in args.top_k_list.split(","))
    rank_df["mass"] = rank_df["mean_phys"] + rank_df["mean_abs"]
    top_mass = rank_df.head(max_k)["mass"].sum()
    # Use rank 21+; sort by mass descending so high-mass features are picked first.
    active_pool_df = rank_df.iloc[max_k:].sort_values("mass", ascending=False).reset_index(drop=True)
    target_mass_low = 0.7 * top_mass
    target_mass_high = 2.0 * top_mass
    rng = torch.Generator().manual_seed(args.seed)
    random_feature_sets = []
    attempts = 0
    while len(random_feature_sets) < args.random_controls and attempts < 20000:
        attempts += 1
        # Restrict to the top-300 highest-mass non-top-20 features.
        candidate_size = min(300, len(active_pool_df))
        idx = torch.randperm(candidate_size, generator=rng)[:max_k].tolist()
        chosen = active_pool_df.iloc[idx]
        chosen_mass = chosen["mass"].sum()
        if target_mass_low <= chosen_mass <= target_mass_high:
            random_feature_sets.append((chosen["feature_idx"].tolist(), chosen_mass))
    if not random_feature_sets and args.random_controls > 0:
        # Could not match target range; fall back to the highest-mass non-top-20 sample
        idx = torch.arange(min(max_k, len(active_pool_df)))
        chosen = active_pool_df.iloc[idx]
        random_feature_sets = [(chosen["feature_idx"].tolist(), chosen["mass"].sum())]
        print(f"WARNING: could not match top-{max_k} mass; using fallback set with mass {chosen['mass'].sum():.2f} ({chosen['mass'].sum()/top_mass*100:.0f}% of top-{max_k})")
    if random_feature_sets:
        print(f"Magnitude-matched random controls: {len(random_feature_sets)} sets (target [{target_mass_low:.2f}, {target_mass_high:.2f}])")
        print(f"  Top-{max_k} total mass: {top_mass:.2f}")
        for r, (feats, mass) in enumerate(random_feature_sets):
            print(f"  random_{r}: {len(feats)} features, mass={mass:.2f} ({mass/top_mass*100:.0f}% of top-{max_k})")

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    sample_ids = manifest["clean_sample_id"].head(args.n_stim).tolist()

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    last_vis_layer, n_vis_layers = get_last_vision_layer(model)
    print(f"  Hooking last vision encoder layer ({n_vis_layers} total).")

    prompt = PROMPT_FC_TEMPLATE.format(label=args.label)

    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids):
        path = M2_STIM_DIR / f"{sid}.png"
        if not path.exists():
            print(f"  [SKIP] missing {sid}")
            continue
        pil = Image.open(path).convert("RGB")

        # Baseline (no hook).
        bl = run_with_hook(model, processor, pil, prompt, last_vis_layer, None, args.device)
        bl_phys = _is_physics_letter(bl)

        for k_str in args.top_k_list.split(","):
            k = int(k_str)
            feature_idx = torch.tensor(top_features[:k], device=args.device, dtype=torch.long)
            hook_fn = make_sae_intervention_hook(sae, feature_idx)
            txt = run_with_hook(model, processor, pil, prompt, last_vis_layer, hook_fn, args.device)
            rows.append({
                "sample_id": sid, "condition": f"top_k={k}", "k_zeroed": k,
                "intervention_text": txt,
                "intervention_phys": _is_physics_letter(txt),
                "baseline_text": bl, "baseline_phys": bl_phys,
            })

        for r, (rand_features, rand_mass) in enumerate(random_feature_sets):
            feature_idx = torch.tensor(rand_features, device=args.device, dtype=torch.long)
            hook_fn = make_sae_intervention_hook(sae, feature_idx)
            txt = run_with_hook(model, processor, pil, prompt, last_vis_layer, hook_fn, args.device)
            rows.append({
                "sample_id": sid, "condition": f"random_{r}", "k_zeroed": len(rand_features),
                "control_mass": rand_mass,
                "intervention_text": txt,
                "intervention_phys": _is_physics_letter(txt),
                "baseline_text": bl, "baseline_phys": bl_phys,
            })

        elapsed = (time.time() - t_start) / 60
        print(f"  [{i+1}/{len(sample_ids)}] {sid} bl={bl_phys} elapsed={elapsed:.1f}min "
              f"(rows={len(rows)} including {sum(1 for r in rows if r.get('condition','').startswith('random_'))} random)")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)
    print("\n=== Aggregate (intervention phys rate by condition) ===")
    agg = df.groupby("condition")["intervention_phys"].agg(
        n=lambda x: int((x >= 0).sum()),
        intervention_phys_rate=lambda x: float((x == 1).sum() / max(1, (x >= 0).sum())),
    ).reset_index()
    print(agg.round(3).to_string(index=False))
    print(f"Wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
