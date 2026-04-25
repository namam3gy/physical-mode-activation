"""M8a — per-shape PMR / GAR / paired-delta summary.

Computes:
  1. PMR by (shape, object_level) — does the line→filled→shaded→textured
     ramp replicate per shape?
  2. PMR by (shape, label_role) — does H7 (label-induced regime selection)
     replicate per shape? Labels are normalized to roles
     (physical / abstract / exotic) using LABELS_BY_SHAPE.
  3. GAR by (shape, label_role) at bg=ground — does GAR-by-label
     ordering replicate?

Usage:
    uv run python scripts/m8a_analyze.py --run-dir outputs/<m8a_run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import score_rows


# Map literal label → role within its shape's triplet.
# circle's triplet (ball, circle, planet) → roles (physical, abstract, exotic).
def _label_to_role(shape: str, label: str) -> str:
    physical, abstract, exotic = LABELS_BY_SHAPE[shape]
    if label == physical:
        return "physical"
    if label == abstract:
        return "abstract"
    if label == exotic:
        return "exotic"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()

    src = args.run_dir / "predictions.jsonl"
    if not src.exists():
        src = args.run_dir / "predictions.parquet"
    df = (
        pd.read_json(src, orient="records", lines=True)
        if src.suffix == ".jsonl"
        else pd.read_parquet(src)
    )
    print(f"Loaded {len(df)} predictions from {src}")
    df = score_rows(df)

    df["label_role"] = [
        _label_to_role(s, l) for s, l in zip(df["shape"], df["label"])
    ]

    # 1. PMR by (shape, object_level).
    pmr_obj = (
        df.groupby(["shape", "object_level"])["pmr"].mean().unstack().round(3)
    )
    pmr_obj = pmr_obj[["line", "filled", "shaded", "textured"]]  # canonical order
    print("\n=== PMR by (shape, object_level) ===")
    print(pmr_obj.to_string())

    # 2. PMR by (shape, label_role).
    pmr_lab = (
        df.groupby(["shape", "label_role"])["pmr"].mean().unstack().round(3)
    )
    pmr_lab = pmr_lab[["physical", "abstract", "exotic"]]
    print("\n=== PMR by (shape, label_role) ===")
    print(pmr_lab.to_string())

    # 3. GAR by (shape, label_role) — only where bg has ground.
    g = df[df["bg_level"].isin(["ground", "scene"])].copy()
    gar_lab = (
        g.groupby(["shape", "label_role"])["gar"].mean().unstack().round(3)
    )
    if not gar_lab.empty:
        gar_lab = gar_lab[["physical", "abstract", "exotic"]]
        print("\n=== GAR by (shape, label_role) | bg in {ground, scene} ===")
        print(gar_lab.to_string())

    # 4. Per-shape ramp monotonicity — diff between textured and line.
    ramp = (pmr_obj["textured"] - pmr_obj["line"]).round(3)
    print("\n=== PMR ramp (textured − line) per shape ===")
    print(ramp.to_string())
    print(f"\nMean ramp across shapes: {ramp.mean():.3f}")

    # Save summaries.
    out = args.run_dir
    pmr_obj.to_csv(out / "m8a_pmr_by_shape_obj.csv")
    pmr_lab.to_csv(out / "m8a_pmr_by_shape_label.csv")
    if not gar_lab.empty:
        gar_lab.to_csv(out / "m8a_gar_by_shape_label.csv")
    ramp.rename("ramp").to_csv(out / "m8a_ramp_per_shape.csv")
    print(f"\nWrote summaries to {out}")


if __name__ == "__main__":
    main()
