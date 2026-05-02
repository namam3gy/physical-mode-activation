"""§4.6 InternVL3 alt-baseline search.

Goal: find a (stim_source, shape/category, factorial cell) where InternVL3's
label-free PMR is low enough (≤ 0.3 ideally) that we can use it as the abstract
baseline for §4.6 layer sweep instead of the saturated `line_blank_none_fall`
mvp_full default.

Surveys all InternVL3 label-free prediction files, groups by available factorial
columns (object_level / bg_level / cue_level / shape / category / prompt_variant),
prints the lowest-PMR cells per stim source.

CPU only — no GPU, no inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.physical_mode.metrics.pmr import score_pmr  # noqa: E402

# Stim sources to survey, with (label, parquet path, expected factorial cols)
SOURCES = [
    ("M8a (5 shapes)",
     ROOT / "outputs/encoder_swap_internvl3_m8a_label_free_20260425-175257_41278d4b/predictions.parquet"),
    ("M8c (photos)",
     ROOT / "outputs/encoder_swap_internvl3_m8c_label_free_20260425-183351_f85e8c4c/predictions.parquet"),
    ("M8d (3 object categories)",
     ROOT / "outputs/encoder_swap_internvl3_m8d_label_free_20260426-002403_311212d6/predictions.parquet"),
    ("M8d-subtype",
     ROOT / "outputs/m8d_subtype_internvl3_20260501-205546_8f102d68/predictions.parquet"),
    ("multi_prompt (4 prompts)",
     ROOT / "outputs/multi_prompt_internvl3_20260428-131440_2f3f953b/predictions.parquet"),
]


def survey_one(label: str, path: Path) -> None:
    if not path.exists():
        print(f"\n=== {label} — MISSING ===")
        return
    df = pd.read_parquet(path)
    print(f"\n=== {label} — {len(df)} rows ===")
    print(f"columns: {list(df.columns)[:20]}")

    # Find response text col
    resp_col = None
    for c in ("response", "raw_text", "raw_response", "label_free_text", "text"):
        if c in df.columns:
            resp_col = c
            break
    if resp_col is None:
        print("  (no recognized response column)")
        return
    print(f"  response col: {resp_col!r}")

    # Compute PMR per row if not already there
    if "pmr" not in df.columns:
        df["pmr"] = df[resp_col].fillna("").map(score_pmr)

    # Identify factorial columns
    factorial_candidates = ("object_level", "bg_level", "cue_level", "shape",
                            "category", "event_template", "label",
                            "prompt_variant", "subtype", "scene_id")
    factorial_cols = [c for c in factorial_candidates if c in df.columns]
    print(f"  factorial cols: {factorial_cols}")
    print(f"  overall PMR: {df['pmr'].mean():.3f}  (n={len(df)})")

    if not factorial_cols:
        return

    # Group by factorial cols, find lowest-PMR cells (n ≥ 5 to be useful)
    grp = df.groupby(factorial_cols, dropna=False, observed=False)
    cell_summary = grp.agg(n=("pmr", "size"), pmr=("pmr", "mean")).reset_index()
    # Filter: at least 5 samples, sort by ascending PMR
    cell_summary = cell_summary[cell_summary["n"] >= 5].sort_values("pmr")
    print(f"  Lowest-PMR cells (n ≥ 5):")
    for _, row in cell_summary.head(8).iterrows():
        cell_str = "/".join(f"{c}={row[c]}" for c in factorial_cols)
        marker = " ★" if row["pmr"] <= 0.3 else (" ✓" if row["pmr"] <= 0.5 else "")
        print(f"    n={int(row['n']):3d}  pmr={row['pmr']:.2f}  {cell_str}{marker}")


def main() -> int:
    print("InternVL3 alt-baseline search for §4.6 layer sweep")
    print(f"Scoring threshold: ★ pmr ≤ 0.3 (ideal), ✓ pmr ≤ 0.5 (usable)")
    print("=" * 80)

    for label, path in SOURCES:
        survey_one(label, path)

    print()
    print("=" * 80)
    print("DECISION GUIDE:")
    print("  - Pick a ★ cell with n ≥ 10 → use as new baseline for §4.6 sweep")
    print("  - Stim must come with a per-class v_L (from M2 capture or fresh extract)")
    print("  - If only ✓ cells exist, sweep is harder to interpret (less headroom)")
    print("  - If no ★/✓ cells exist, InternVL3 protocol-untestable → close-out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
