"""Extract square-only v_L for InternVL3 §4.6 alt-baseline sweep.

Splits the existing M8a InternVL3 captures by `shape == "square"`, recomputes
the mean-difference direction per layer using `compute_steering_vectors`, and
saves a `steering_vectors_square.npz` alongside the global one.

Why: global v_L was extracted from all 5 shapes (n_pos=382 / n_neg=18) — only
18 abstract examples is noise-dominated. Square-only gives a more balanced
split and directly answers "square abstract → square physics" instead of
"global physics → square cell."

CPU only — reads existing .safetensors, no GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_mode.metrics.pmr import score_pmr  # noqa: E402
from physical_mode.probing.steering import compute_steering_vectors, save_steering_vectors  # noqa: E402

LAYERS = (5, 10, 15, 20, 25)
RUN_DIR = ROOT / "outputs/encoder_swap_internvl3_m8a_capture_20260426-161917_aafca162"


def main() -> int:
    activations_dir = RUN_DIR / "activations"
    out_dir = RUN_DIR / "probing_steering"
    out_path = out_dir / "steering_vectors_square.npz"

    if not activations_dir.exists():
        print(f"ERROR: activations dir missing: {activations_dir}")
        return 1

    # Read predictions.jsonl, score PMR, filter to square only
    jsonl = RUN_DIR / "predictions.jsonl"
    df = pd.read_json(jsonl, lines=True)
    print(f"Total predictions: {len(df)} rows")
    print(f"Shapes: {df['shape'].value_counts().to_dict()}")
    print(f"Prompt variants: {df['prompt_variant'].value_counts().to_dict()}")

    if "pmr" not in df.columns:
        df["pmr"] = df["raw_text"].fillna("").apply(score_pmr)

    # Filter to square only — keep all factorial cells, all seeds
    sub = df[df["shape"] == "square"].copy()
    print(f"\nSquare-only subset: {len(sub)} rows")
    print(f"Square PMR distribution: pos={int(sub['pmr'].sum())}/{len(sub)} = {sub['pmr'].mean():.3f}")

    # Save filtered predictions for compute_steering_vectors to use
    tmp_parquet = out_dir / "_predictions_square_filtered.parquet"
    out_dir.mkdir(exist_ok=True)
    sub.to_parquet(tmp_parquet, index=False)
    print(f"  wrote filtered predictions → {tmp_parquet.name}")

    # Compute steering vectors with the filtered predictions
    print(f"\nComputing v_L for layers {LAYERS}...")
    vectors = compute_steering_vectors(
        activations_dir=activations_dir,
        predictions_path=tmp_parquet,
        layers=LAYERS,
        pmr_source="open",
    )

    save_steering_vectors(vectors, out_path)
    print()
    print("=" * 70)
    print(f"Saved square-only v_L → {out_path}")
    print("=" * 70)
    for li in LAYERS:
        v = vectors[li]
        print(f"  L{li}: dim={v.v.shape[0]}, ||v||={v.norm:.3f}, "
              f"n_pos={v.n_pos}, n_neg={v.n_neg}")

    # Cleanup the temp parquet
    tmp_parquet.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
