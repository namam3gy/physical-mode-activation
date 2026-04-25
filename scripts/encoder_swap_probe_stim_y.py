"""§4.5 ext — Stim-defined y sensitivity check.

The default `encoder_swap_probe.py` uses each model's own behavioral
PMR as the y target. Cross-model AUC comparison conflates encoder
discriminability with y-class-balance differences (LLaVA n_neg=191 vs
InternVL3 n_neg=21).

This script holds y constant across models by defining it from stim
properties:
  y=1 if obj_level ∈ {filled, shaded, textured}  (rendered as physical)
  y=0 if obj_level == "line"                     (abstract drawing)

The probe then asks: does the encoder representation linearly separate
the rendered-physical class from the line-drawing class? This is a
purer "encoder discriminability" test than the behavioral-y probe.

If AUC ranking matches the behavioral-y probe → encoder discriminability
*is* the driver. If it shifts → behavioral-y AUC was confounded by
y-balance.

Usage:
    uv run python scripts/encoder_swap_probe_stim_y.py \
        --vision-dir outputs/encoder_swap_<model>_vision_activations \
        --stim-dir inputs/m8a_qwen_<ts> \
        --layers <comma-list> \
        --out-dir outputs/encoder_swap_<model>_probe_stim_y
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physical_mode.probing.vision import (
    _load_layer_activations,
    run_layer_sweep,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vision-dir", type=Path, required=True)
    p.add_argument("--stim-dir", type=Path, required=True)
    p.add_argument("--layers", type=str, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--target", default="rendered_vs_line",
                   choices=["rendered_vs_line",
                            "physics_cell_vs_abstract_cell",
                            "within_line_context",
                            "within_textured_context",
                            "physical_shape_vs_abstract_shape"],
                   help="rendered_vs_line: y=(obj!=line). physics_cell_vs_abstract_cell: "
                        "y=1 (textured+ground+both); y=0 (line+blank+none). "
                        "within_line_context: hold obj=line; y=1 if (ground AND both), "
                        "y=0 if (blank AND none) — tests whether encoder picks up gravity-cue "
                        "context on otherwise-identical line drawings. "
                        "within_textured_context: same idea on obj=textured. "
                        "physical_shape_vs_abstract_shape (M8c): y=1 if shape ∈ "
                        "{ball, car, person, bird}; y=0 if shape == abstract.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]

    manifest = pd.read_parquet(args.stim_dir / "manifest.parquet")
    # Filter to samples whose captures exist on disk.
    manifest = manifest[
        manifest["sample_id"].apply(lambda sid: (args.vision_dir / f"{sid}.safetensors").exists())
    ].reset_index(drop=True)

    if args.target == "rendered_vs_line":
        y = (manifest["object_level"] != "line").astype(np.int64).to_numpy()
        print(f"Target rendered_vs_line: y=1 (rendered) {int(y.sum())} / y=0 (line) {int((y == 0).sum())}")
    elif args.target == "physics_cell_vs_abstract_cell":
        is_physics = (
            (manifest["object_level"] == "textured")
            & (manifest["bg_level"] == "ground")
            & (manifest["cue_level"] == "both")
        )
        is_abstract = (
            (manifest["object_level"] == "line")
            & (manifest["bg_level"] == "blank")
            & (manifest["cue_level"] == "none")
        )
        keep = is_physics | is_abstract
        manifest = manifest[keep].reset_index(drop=True)
        y = is_physics[keep].reset_index(drop=True).astype(np.int64).to_numpy()
        print(f"Target physics_cell_vs_abstract_cell: y=1 (physics cell) {int(y.sum())} / "
              f"y=0 (abstract cell) {int((y == 0).sum())}")
    elif args.target in ("within_line_context", "within_textured_context"):
        obj = "line" if args.target == "within_line_context" else "textured"
        is_obj = manifest["object_level"] == obj
        is_pos = is_obj & (manifest["bg_level"] == "ground") & (manifest["cue_level"] == "both")
        is_neg = is_obj & (manifest["bg_level"] == "blank") & (manifest["cue_level"] == "none")
        keep = is_pos | is_neg
        manifest = manifest[keep].reset_index(drop=True)
        y = is_pos[keep].reset_index(drop=True).astype(np.int64).to_numpy()
        print(f"Target {args.target} (obj={obj}, ground+both vs blank+none): "
              f"y=1 {int(y.sum())} / y=0 {int((y == 0).sum())}")
    elif args.target == "physical_shape_vs_abstract_shape":
        physical_shapes = {"ball", "car", "person", "bird"}
        is_phys = manifest["shape"].isin(physical_shapes)
        is_abs = manifest["shape"] == "abstract"
        keep = is_phys | is_abs
        manifest = manifest[keep].reset_index(drop=True)
        y = is_phys[keep].reset_index(drop=True).astype(np.int64).to_numpy()
        print(f"Target physical_shape_vs_abstract_shape (M8c): "
              f"y=1 (physical objects) {int(y.sum())} / y=0 (abstract) {int((y == 0).sum())}")

    sample_ids = manifest["sample_id"].tolist()
    X_per_layer = {}
    for li in layers:
        X_per_layer[li] = _load_layer_activations(args.vision_dir, sample_ids, li)
        print(f"  layer {li:>2}: X shape {X_per_layer[li].shape}")

    sweep = run_layer_sweep(X_per_layer, y)
    print(f"\n=== Stim-defined y layer sweep AUC ({args.target}) ===")
    print(sweep.round(3).to_string(index=False))
    out_name = f"layer_sweep_stim_y_{args.target}.csv"
    sweep.to_csv(args.out_dir / out_name, index=False)
    print(f"\nWrote {args.out_dir / out_name}")


if __name__ == "__main__":
    main()
