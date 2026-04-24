"""Run Sub-task 3 (M4) analyses on M2-captured LM hidden states.

Two deliverables:
  1. Layer-wise logit-lens trajectories for a curated token set (physics
     verbs vs geometry nouns vs label words).
  2. Layer-wise linear probes for behavioral forced-choice PMR.

Usage:
    uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_<ts>_<hash>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from physical_mode.probing.lm import (
    load_lm_probing_dataset,
    resolve_token_ids,
    run_lm_layer_sweep,
    run_logit_lens_trajectories,
    switching_layer_per_sample,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--layers", default="5,10,15,20,25",
                   help="LM layers captured in M2")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--limit", type=int, default=None,
                   help="limit stimuli for a smoke run")
    args = p.parse_args()

    layers = tuple(int(x) for x in args.layers.split(","))
    activations = args.run_dir / "activations"
    preds_path = args.run_dir / "predictions_scored.parquet"
    out = args.run_dir / "probing_lm"
    out.mkdir(exist_ok=True)

    # ---------- 1. Per-layer PMR probes (no model needed) ----------
    print("=" * 72)
    print("(1) Per-layer PMR probe on LM hidden states")
    print("=" * 72)
    X_per_layer, y_fc, meta = load_lm_probing_dataset(
        activations, preds_path, layers=layers, pmr_source="forced_choice"
    )
    sweep_fc = run_lm_layer_sweep(X_per_layer, y_fc)
    print("Forced-choice PMR probe:")
    print(sweep_fc.to_string(index=False))
    sweep_fc.to_csv(out / "layer_sweep_forced_choice.csv", index=False)

    _, y_open, _ = load_lm_probing_dataset(
        activations, preds_path, layers=layers, pmr_source="open"
    )
    sweep_open = run_lm_layer_sweep(X_per_layer, y_open)
    print("\nOpen-ended PMR probe (contaminated by label prior):")
    print(sweep_open.to_string(index=False))
    sweep_open.to_csv(out / "layer_sweep_open.csv", index=False)

    # ---------- 2. Logit lens — needs lm_head + tokenizer ----------
    print()
    print("=" * 72)
    print("(2) Logit lens trajectories")
    print("=" * 72)
    print(f"Loading {args.model_id} (lm_head + tokenizer only)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    lm_head: torch.nn.Linear = model.lm_head

    token_ids = resolve_token_ids(tokenizer)
    print("Resolved tokens:")
    for cat in ("physics", "geometry", "label"):
        d = getattr(token_ids, cat)
        print(f"  {cat:>8}: {len(d)} tokens — {list(d.keys())}")

    sample_ids = meta["sample_id"].tolist()
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]

    traj = run_logit_lens_trajectories(
        activations_dir=activations,
        sample_ids=sample_ids,
        lm_head=lm_head,
        token_ids=token_ids,
        layers=layers,
        pool="mean",
    )
    traj.to_parquet(out / "logit_lens_trajectories.parquet", index=False)

    # Aggregate: mean logit per (layer, category)
    agg = (
        traj.groupby(["layer", "category"])["logit"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    print("\nMean logit by (layer, category):")
    print(agg.to_string(index=False))
    agg.to_csv(out / "logit_by_layer_category.csv", index=False)

    # Aggregate per object_level for "encoder-decoder boomerang at the LM side":
    # for each (layer, object_level), compute mean physics-logit minus mean
    # geometry-logit ("physics margin").
    meta_small = meta[["sample_id", "object_level", "bg_level", "cue_level"]]
    traj_with_meta = traj.merge(meta_small, on="sample_id")
    margin = (
        traj_with_meta.groupby(["layer", "category", "object_level"])["logit"]
        .mean().reset_index()
        .pivot_table(index=["layer", "object_level"], columns="category", values="logit")
        .reset_index()
    )
    margin["physics_margin"] = margin["physics"] - margin["geometry"]
    print("\nPhysics margin (mean physics logit − mean geometry logit) by layer × object_level:")
    print(margin.pivot(index="layer", columns="object_level", values="physics_margin").round(2).to_string())
    margin.to_csv(out / "physics_margin_by_layer_object.csv", index=False)

    # Switching layer per sample (earliest layer where physics ≥ geometry).
    switch = switching_layer_per_sample(traj)
    switch = switch.merge(meta_small, on="sample_id")
    print("\nSwitching layer distribution overall:")
    print(switch["switching_layer"].describe().to_string())
    print("\nMedian switching layer by object_level:")
    print(switch.groupby("object_level")["switching_layer"].median().to_string())
    switch.to_csv(out / "switching_layer_per_sample.csv", index=False)

    print(f"\nWrote all M4 outputs to {out}/")


if __name__ == "__main__":
    main()
