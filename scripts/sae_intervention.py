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
from physical_mode.models.vlm_runner import _resolve_vision_blocks
from physical_mode.metrics.pmr import score_for_variant
from physical_mode.inference.prompts import render as render_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M2_STIM_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"


# Map --prompt-mode CLI value → physical_mode.inference.prompts variant name.
# Backward compatible: 'fc' and 'open' keep their original meaning.
PROMPT_MODE_TO_VARIANT = {
    "fc": "forced_choice",
    "open": "open",
    "describe_scene": "describe_scene",
    "meta_phys_yesno": "meta_phys_yesno",
}


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


def get_vision_layer(model, block_idx: int = -1):
    """Resolve the vision-encoder block to hook on. Cross-model via _resolve_vision_blocks.

    block_idx=-1 → last block (default; matches Qwen-original behavior). For Idefics2
    SigLIP-SO400M with 27 blocks, the cross-model SAE was trained on `vision_hidden_23`
    (block index 23, NOT last) — pass --vision-block-idx 23.
    """
    blocks = _resolve_vision_blocks(model)
    if blocks is None:
        raise RuntimeError("Could not resolve vision blocks for this architecture.")
    n = len(blocks)
    idx = block_idx if block_idx >= 0 else n + block_idx
    if not (0 <= idx < n):
        raise RuntimeError(f"vision-block-idx {block_idx} out of range for {n} blocks.")
    return blocks[idx], n


def get_post_projection_layer(model):
    """Resolve the post-projection (merger / projector) module to hook on.

    Qwen2.5-VL: model.model.visual.merger (Qwen2_5_VLPatchMerger). Output dim
    = LM embedding dim (3584 for 7B). Other architectures may need different
    paths; for now Qwen-only is supported.
    """
    if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "merger"):
        return model.model.visual.merger, 1
    raise RuntimeError(
        "Could not resolve post-projection module — this script's --hook-target merger "
        "currently supports Qwen2.5-VL (model.model.visual.merger) only."
    )


@torch.no_grad()
def run_with_hook(model, processor, pil, prompt, hook_module, hook_fn, device,
                  system_prompt: str | None = None, max_new_tokens: int = 32):
    content = [{"type": "image"}, {"type": "text", "text": prompt}]
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    msgs.append({"role": "user", "content": content})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    handle = hook_module.register_forward_hook(hook_fn) if hook_fn else None
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    finally:
        if handle is not None:
            handle.remove()


def _is_physics_letter(text: str) -> int:
    """Find the first standalone A/B/C/D letter in the response.

    Cross-model: Qwen returns bare "A\\n\\n...", LLaVA-1.5 returns just "A",
    Idefics2 returns "Answer: A". A simple substring scan that picks the
    first letter character occurring at the start of a token (after stripping
    common prefixes) handles all three.
    """
    if not text:
        return -1
    s = text.strip()
    # Strip leading "Answer:" / "Answer: " prefixes (Idefics2 / Mistral pattern).
    for pre in ("Answer:", "answer:", "Choice:", "choice:"):
        if s.startswith(pre):
            s = s[len(pre):].strip()
            break
    if not s:
        return -1
    c = s[0]
    return 1 if c in ("A", "B", "C") else (0 if c == "D" else -1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", type=Path, required=True)
    p.add_argument("--layer-key", default="vision_hidden_31",
                   help="captured-activation key (informational; the script hooks the actual model)")
    p.add_argument("--top-k-list", default="1,5,10,20",
                   help="comma-separated feature counts to zero")
    p.add_argument("--rank-by", default="delta", choices=["delta", "cohens_d"],
                   help="which feature ranking to use for the top-k subset "
                        "(delta = mean_phys − mean_abs; cohens_d filters high-baseline "
                        "outliers by dividing delta by pooled std)")
    p.add_argument("--random-controls", type=int, default=0,
                   help="number of magnitude-matched random feature sets to draw "
                        "(at the largest k in --top-k-list). Multi-seed: tries seeds "
                        "in order until --random-controls hits are accepted.")
    p.add_argument("--mass-window-low", type=float, default=0.7,
                   help="lower bound of mass-match window as a multiple of top-k mass")
    p.add_argument("--mass-window-high", type=float, default=2.0,
                   help="upper bound of mass-match window as a multiple of top-k mass")
    p.add_argument("--mass-window-low-fallback", type=float, default=0.5,
                   help="fallback mass-window low if main window yields too few sets")
    p.add_argument("--mass-window-high-fallback", type=float, default=2.5,
                   help="fallback mass-window high if main window yields too few sets")
    p.add_argument("--candidate-pool-size", type=int, default=300,
                   help="restrict random draws to the top-N highest-mass non-top-k features")
    p.add_argument("--n-stim", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--label", default="circle")
    p.add_argument("--manifest",
                   default="outputs/m5b_sip/manifest.csv",
                   help="FC mode: CSV with clean_sample_id column listing stim to use.")
    p.add_argument("--prompt-mode",
                   choices=["fc", "open", "describe_scene", "meta_phys_yesno"],
                   default="fc",
                   help="fc=force-choice letter scoring (Qwen-original protocol); "
                   "open=free-text kinetic prediction + score_pmr; "
                   "describe_scene=free-form description + score_describe (Phase 3 cross-prompt); "
                   "meta_phys_yesno=binary yes/no probe + score_meta_yesno (Phase 3 cross-prompt).")
    p.add_argument("--stimulus-dir", type=Path, default=None,
                   help="OPEN mode: stimulus directory containing manifest.parquet + images/.")
    p.add_argument("--test-subset", default=None,
                   help="OPEN mode: obj/bg/cue triple selecting stim cells "
                   "(e.g. 'filled/blank/both'). Pick a cell where baseline PMR≈1.")
    p.add_argument("--vision-block-idx", type=int, default=-1,
                   help="vision encoder block index to hook (default -1 = last). "
                   "Idefics2 (SigLIP-SO400M, 27 blocks) trained vision_hidden_23 "
                   "→ pass 23 (NOT last). Qwen / LLaVA-* / InternVL3 trained on "
                   "the last block.")
    p.add_argument("--hook-target", choices=["block", "merger"], default="block",
                   help="block=hook a vision-encoder block (pre-projection, default; "
                   "matches the original M5b SAE protocol). merger=hook the "
                   "post-projection projector (Qwen2.5-VL only). The SAE must be "
                   "trained on the matching activations (post-projection SAE = "
                   "merger output activations of dim 3584 for Qwen).")
    p.add_argument("--tag", default=None,
                   help="output subdirectory name (default: <sae-dir>.name + ranking + ts)")
    args = p.parse_args()

    if args.tag is None:
        args.tag = f"{args.sae_dir.name}_{args.rank_by}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = PROJECT_ROOT / "outputs" / "sae_intervention" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = load_sae(args.sae_dir / "sae.pt", device=args.device)
    rank_df = pd.read_csv(args.sae_dir / "feature_ranking.csv")
    if args.rank_by not in rank_df.columns:
        raise SystemExit(
            f"--rank-by={args.rank_by} but column missing from {args.sae_dir / 'feature_ranking.csv'}; "
            "rerun scripts/sae_rerank_features.py first."
        )
    rank_df_sorted = rank_df.sort_values(args.rank_by, ascending=False).reset_index(drop=True)
    max_k = max(int(k) for k in args.top_k_list.split(","))
    top_features = rank_df_sorted["feature_idx"].head(max_k).tolist()
    print(f"Top {max_k} features (ranked by {args.rank_by}): {top_features[:10]}...")

    # Magnitude-matched random control sets. We draw uniformly from the
    # high-mass non-top-k pool (excludes the L1-killed tail where features
    # have ~zero mass), then accept the draw if its summed mass falls inside
    # [mass-window-low × top_mass, mass-window-high × top_mass].
    #
    # Multi-seed loop: the prior single-seed approach got 1/3 hits because the
    # heavy-tailed mass distribution makes acceptance probabilistic. Loop over
    # incrementing seeds until args.random_controls hits accumulate. If still
    # insufficient after 50 seeds, widen to the fallback window.
    rank_df_sorted["mass"] = rank_df_sorted["mean_phys"] + rank_df_sorted["mean_abs"]
    top_mass = rank_df_sorted.head(max_k)["mass"].sum()
    active_pool_df = rank_df_sorted.iloc[max_k:].sort_values("mass", ascending=False).reset_index(drop=True)
    candidate_size = min(args.candidate_pool_size, len(active_pool_df))

    def _try_window(low_mult: float, high_mult: float, n_target: int, seeds_to_try: int = 50) -> list[tuple[list, float, int]]:
        target_low = low_mult * top_mass
        target_high = high_mult * top_mass
        out: list[tuple[list, float, int]] = []
        seen_feature_sets: set[tuple] = set()
        for s in range(args.seed, args.seed + seeds_to_try):
            if len(out) >= n_target:
                break
            rng_local = torch.Generator().manual_seed(s)
            # Try a few permutations per seed before bumping seed.
            for _ in range(40):
                if len(out) >= n_target:
                    break
                idx = torch.randperm(candidate_size, generator=rng_local)[:max_k].tolist()
                chosen = active_pool_df.iloc[idx]
                m = float(chosen["mass"].sum())
                if not (target_low <= m <= target_high):
                    continue
                feats = tuple(sorted(chosen["feature_idx"].tolist()))
                if feats in seen_feature_sets:
                    continue
                seen_feature_sets.add(feats)
                out.append((list(feats), m, s))
        return out

    random_feature_sets: list[tuple[list, float, int]] = []
    if args.random_controls > 0:
        random_feature_sets = _try_window(args.mass_window_low, args.mass_window_high, args.random_controls)
        if len(random_feature_sets) < args.random_controls:
            print(f"NOTE: only got {len(random_feature_sets)} hits in window "
                  f"[{args.mass_window_low}, {args.mass_window_high}]; widening to "
                  f"[{args.mass_window_low_fallback}, {args.mass_window_high_fallback}].")
            extras = _try_window(
                args.mass_window_low_fallback,
                args.mass_window_high_fallback,
                args.random_controls - len(random_feature_sets),
            )
            random_feature_sets.extend(extras)
        if not random_feature_sets:
            idx = torch.arange(min(max_k, len(active_pool_df)))
            chosen = active_pool_df.iloc[idx]
            random_feature_sets = [(chosen["feature_idx"].tolist(), float(chosen["mass"].sum()), -1)]
            print(f"WARNING: could not match top-{max_k} mass at any window; using fallback "
                  f"highest-mass set (mass {chosen['mass'].sum():.2f} = {chosen['mass'].sum()/top_mass*100:.0f}% of top-{max_k}).")
    if random_feature_sets:
        print(f"Magnitude-matched random controls: {len(random_feature_sets)} sets")
        print(f"  Top-{max_k} total mass: {top_mass:.2f}")
        for r, (feats, mass, used_seed) in enumerate(random_feature_sets):
            print(f"  random_{r}: {len(feats)} features, mass={mass:.2f} "
                  f"({mass/top_mass*100:.0f}% of top-{max_k}, seed={used_seed})")

    variant = PROMPT_MODE_TO_VARIANT[args.prompt_mode]
    rp = render_prompt(variant, args.label)
    is_fc_mode = (args.prompt_mode == "fc")

    if not is_fc_mode:
        if args.stimulus_dir is None or args.test_subset is None:
            raise SystemExit(
                f"--prompt-mode {args.prompt_mode} requires --stimulus-dir and --test-subset"
            )
        stim_manifest = pd.read_parquet(args.stimulus_dir / "manifest.parquet")
        obj, bg, cue = args.test_subset.split("/")
        sub = stim_manifest[
            (stim_manifest["object_level"] == obj)
            & (stim_manifest["bg_level"] == bg)
            & (stim_manifest["cue_level"] == cue)
        ].reset_index(drop=True)
        sample_ids = sub["sample_id"].head(args.n_stim).tolist()
        stim_image_dir = args.stimulus_dir / "images"
        prompt = rp.user
        system_prompt = rp.system
        max_new_tokens = 64
        print(f"{args.prompt_mode.upper()} mode: stim cell {obj}/{bg}/{cue}, n={len(sample_ids)}")
    else:
        manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
        sample_ids = manifest["clean_sample_id"].head(args.n_stim).tolist()
        stim_image_dir = M2_STIM_DIR
        prompt = rp.user
        system_prompt = rp.system
        max_new_tokens = 32

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    if args.hook_target == "merger":
        last_vis_layer, _ = get_post_projection_layer(model)
        print(f"  Hooking post-projection module (Qwen merger output, LM-space).")
    else:
        last_vis_layer, n_vis_layers = get_vision_layer(model, args.vision_block_idx)
        eff_idx = args.vision_block_idx if args.vision_block_idx >= 0 else n_vis_layers + args.vision_block_idx
        print(f"  Hooking vision encoder block {eff_idx} of {n_vis_layers}.")

    rows = []
    t_start = time.time()
    for i, sid in enumerate(sample_ids):
        path = stim_image_dir / f"{sid}.png"
        if not path.exists():
            print(f"  [SKIP] missing {sid}")
            continue
        pil = Image.open(path).convert("RGB")

        # Baseline (no hook).
        bl = run_with_hook(model, processor, pil, prompt, last_vis_layer, None, args.device,
                           system_prompt=system_prompt, max_new_tokens=max_new_tokens)
        bl_phys = _is_physics_letter(bl) if is_fc_mode else -1  # PMR scored later

        for k_str in args.top_k_list.split(","):
            k = int(k_str)
            feature_idx = torch.tensor(top_features[:k], device=args.device, dtype=torch.long)
            hook_fn = make_sae_intervention_hook(sae, feature_idx)
            txt = run_with_hook(model, processor, pil, prompt, last_vis_layer, hook_fn, args.device,
                                system_prompt=system_prompt, max_new_tokens=max_new_tokens)
            rows.append({
                "sample_id": sid, "condition": f"top_k={k}", "k_zeroed": k,
                "intervention_text": txt,
                "intervention_phys": _is_physics_letter(txt) if is_fc_mode else -1,
                "baseline_text": bl, "baseline_phys": bl_phys,
            })

        for r, (rand_features, rand_mass, rand_seed) in enumerate(random_feature_sets):
            feature_idx = torch.tensor(rand_features, device=args.device, dtype=torch.long)
            hook_fn = make_sae_intervention_hook(sae, feature_idx)
            txt = run_with_hook(model, processor, pil, prompt, last_vis_layer, hook_fn, args.device,
                                system_prompt=system_prompt, max_new_tokens=max_new_tokens)
            rows.append({
                "sample_id": sid, "condition": f"random_{r}", "k_zeroed": len(rand_features),
                "control_mass": rand_mass, "control_seed": rand_seed,
                "intervention_text": txt,
                "intervention_phys": _is_physics_letter(txt) if is_fc_mode else -1,
                "baseline_text": bl, "baseline_phys": bl_phys,
            })

        elapsed = (time.time() - t_start) / 60
        marker = bl_phys if is_fc_mode else (bl[:30].replace(chr(10), ' / '))
        print(f"  [{i+1}/{len(sample_ids)}] {sid} bl={marker!r} elapsed={elapsed:.1f}min "
              f"(rows={len(rows)} including {sum(1 for r in rows if r.get('condition','').startswith('random_'))} random)")

    df = pd.DataFrame(rows)

    if not is_fc_mode:
        df["intervention_pmr"] = df["intervention_text"].astype(str).map(
            lambda t: score_for_variant(t, variant)
        )
        df["baseline_pmr"] = df["baseline_text"].astype(str).map(
            lambda t: score_for_variant(t, variant)
        )

    df.to_csv(out_dir / "results.csv", index=False)
    print("\n=== Aggregate (rate by condition) ===")
    if not is_fc_mode:
        agg = df.groupby("condition").agg(
            n=("intervention_pmr", "count"),
            bl_pmr=("baseline_pmr", "mean"),
            int_pmr=("intervention_pmr", "mean"),
        ).reset_index()
    else:
        agg = df.groupby("condition")["intervention_phys"].agg(
            n=lambda x: int((x >= 0).sum()),
            intervention_phys_rate=lambda x: float((x == 1).sum() / max(1, (x >= 0).sum())),
        ).reset_index()
    print(agg.round(3).to_string(index=False))
    print(f"Wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
