"""M-MP Phase 2 — describe_scene hand-label gate helper.

Per `references/paper_gaps.md` G1 / `docs/m_mp_multi_prompt_design.md`,
the `describe_scene` scorer must clear a scorer-vs-hand-agreement gate
of ≥ 0.85 (Cohen's κ ≥ 0.70) on N=50 hand-labeled outputs per model
*before* Phase 2 full-scale runs.

This script:
  1. Loads the Phase 1 smoke outputs from all 5 models.
  2. Filters to `describe_scene` prompt (~144 outputs/model).
  3. Stratified-samples N=50 per model across (object_level × bg_level)
     cells so the hand-label set spans the saturation gradient.
  4. Writes a CSV with the scorer's automatic prediction + an empty
     `hand_label` column for the human rater to fill in.
  5. After hand-labels are filled in, compute agreement +
     Cohen's kappa per model.

Usage:
  Generate the labeling sheet:
    uv run python scripts/m_mp_describe_label_helper.py prepare \\
        --output describe_label_sheet.csv

  Compute agreement after labeling:
    uv run python scripts/m_mp_describe_label_helper.py score \\
        --input describe_label_sheet.csv

The labeling rubric (in the CSV header):
  - hand_label = 1: the description expresses physics-mode commitment
    (suspended, falling, hovering, gravity, momentum, ball-as-physical, etc.).
  - hand_label = 0: the description is geometric / drawing-style
    (outline, circle, sketch, line drawing, geometric shape, etc.).
  - hand_label = -1: ambiguous / cannot decide.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_mode.metrics.pmr import score_describe  # noqa: E402


PHASE1_OUTPUTS = {
    "Qwen": "outputs/multi_prompt_qwen_20260428-125946_717081e4",
    "LLaVA-1.5": "outputs/multi_prompt_llava_20260428-130700_f4489f0d",
    "LLaVA-Next": "outputs/multi_prompt_llava_next_20260428-130900_0f78fbe8",
    "Idefics2": "outputs/multi_prompt_idefics2_20260428-130444_299dcad8",
    "InternVL3": "outputs/multi_prompt_internvl3_20260428-131440_2f3f953b",
}


def prepare(output_csv: Path, n_per_model: int = 50, seed: int = 42) -> None:
    rows = []
    for model, path in PHASE1_OUTPUTS.items():
        df = pd.read_parquet(Path(path) / "predictions.parquet")
        df = df[df.prompt_variant == "describe_scene"].copy()

        # Stratify by object_level × bg_level (12 cells × 3 labels = 36 strata).
        df["stratum"] = df["object_level"].astype(str) + "/" + df["bg_level"].astype(str)
        df["scorer_pred"] = df["raw_text"].apply(score_describe)

        # Sample roughly evenly across strata; if a stratum has < ceil(n/12) rows, take all.
        per_stratum = max(1, n_per_model // 12)
        sampled = (
            df.groupby("stratum", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_stratum), random_state=seed))
              .reset_index(drop=True)
        )
        # If we got fewer than n_per_model rows, top up randomly.
        if len(sampled) < n_per_model:
            extra = df.drop(sampled.index, errors="ignore").sample(
                min(len(df) - len(sampled), n_per_model - len(sampled)),
                random_state=seed,
            )
            sampled = pd.concat([sampled, extra]).reset_index(drop=True)

        sampled = sampled.head(n_per_model).copy()
        sampled["model"] = model
        sampled["hand_label"] = ""
        rows.append(sampled[["model", "sample_id", "label", "object_level",
                             "bg_level", "cue_level", "raw_text",
                             "scorer_pred", "hand_label"]])

    out = pd.concat(rows).reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    # Header note for the human rater.
    header = (
        "# Describe-scene hand-label sheet (M-MP Phase 2 gate)\n"
        "# Rubric:\n"
        "#   hand_label = 1 → description expresses physics-mode commitment\n"
        "#                    (suspended, falling, hovering, gravity, momentum,\n"
        "#                     ball-as-physical, real-world-object framing).\n"
        "#   hand_label = 0 → description is geometric / drawing-style\n"
        "#                    (outline, sketch, geometric shape, simple line).\n"
        "#   hand_label = -1 → ambiguous / cannot decide.\n"
        "# Gate: scorer-vs-hand agreement >= 0.85 AND Cohen's kappa >= 0.70\n"
        "# per model. If fail, refine the lexicon in scripts/.../pmr.py.\n"
        "#\n"
    )
    text = output_csv.read_text()
    output_csv.write_text(header + text)
    print(f"Wrote {len(out)} rows to {output_csv}")
    print("Models in sheet:", out["model"].value_counts().to_dict())


def score(input_csv: Path) -> None:
    # Skip header lines starting with '#'
    df = pd.read_csv(input_csv, comment="#")
    df["hand_label"] = pd.to_numeric(df["hand_label"], errors="coerce")
    df["scorer_pred"] = df["scorer_pred"].astype(int)

    print(f"Loaded {len(df)} rows; hand-labeled {df['hand_label'].notna().sum()}")
    if df["hand_label"].isna().any():
        unfilled = df["hand_label"].isna().sum()
        print(f"WARNING: {unfilled} rows unlabeled — only computing on labeled subset.")
        df = df.dropna(subset=["hand_label"])

    # Drop ambiguous (-1) per rubric.
    n_amb = (df["hand_label"] == -1).sum()
    if n_amb:
        print(f"Dropping {n_amb} ambiguous (-1) rows.")
        df = df[df["hand_label"] != -1]

    df["hand_label"] = df["hand_label"].astype(int)

    print()
    print(f"{'Model':<12} | {'n':>4} | {'agreement':>10} | {'kappa':>8} | {'gate (≥0.85 / κ ≥ 0.70)'}")
    print("-" * 70)
    for model in PHASE1_OUTPUTS.keys():
        sub = df[df["model"] == model]
        if len(sub) == 0:
            continue
        agreement = (sub["scorer_pred"] == sub["hand_label"]).mean()
        kappa = _cohens_kappa(sub["scorer_pred"].values, sub["hand_label"].values)
        passes = "PASS" if agreement >= 0.85 and kappa >= 0.70 else "FAIL"
        print(f"{model:<12} | {len(sub):>4d} | {agreement:>10.3f} | {kappa:>8.3f} | {passes}")


def _cohens_kappa(a, b) -> float:
    """Compute Cohen's kappa for two binary rater arrays."""
    from collections import Counter

    if len(a) != len(b) or len(a) == 0:
        return float("nan")
    n = len(a)
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    pe = sum((counts_a[v] / n) * (counts_b[v] / n) for v in {0, 1})
    if pe == 1.0:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare", help="Generate the labeling sheet.")
    p_prep.add_argument("--output", type=Path, default=Path("describe_label_sheet.csv"))
    p_prep.add_argument("--n-per-model", type=int, default=50)
    p_prep.add_argument("--seed", type=int, default=42)
    p_score = sub.add_parser("score", help="Compute scorer-vs-hand agreement.")
    p_score.add_argument("--input", type=Path, required=True)
    args = p.parse_args()

    cwd = Path.cwd()
    if (cwd / "configs").is_dir() is False:
        # Allow running from anywhere.
        os.chdir(Path(__file__).resolve().parents[1])

    if args.cmd == "prepare":
        prepare(args.output, args.n_per_model, args.seed)
    elif args.cmd == "score":
        score(args.input)


if __name__ == "__main__":
    main()
