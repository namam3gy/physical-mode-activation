"""Score predictions with PMR/GAR/RC and write per-axis summaries.

Usage:
    uv run python scripts/03_score_and_summarize.py --run-dir outputs/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physical_mode.metrics.pmr import response_consistency, score_rows, summarize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()

    preds_path = args.run_dir / "predictions.jsonl"
    if not preds_path.exists():
        preds_path = args.run_dir / "predictions.parquet"
    df = (
        pd.read_json(preds_path, orient="records", lines=True)
        if preds_path.suffix == ".jsonl"
        else pd.read_parquet(preds_path)
    )
    print(f"Loaded {len(df)} predictions from {preds_path}")

    scored = score_rows(df)
    scored.to_parquet(args.run_dir / "predictions_scored.parquet", index=False)
    scored.to_csv(args.run_dir / "predictions_scored.csv", index=False)

    sums = summarize(scored)
    for name, s in sums.items():
        s.to_csv(args.run_dir / f"summary_{name}.csv", index=False)

    rc = response_consistency(
        scored, group_cols=["object_level", "bg_level", "cue_level", "event_template", "label", "prompt_variant"]
    )
    rc.to_csv(args.run_dir / "response_consistency.csv", index=False)

    print("\n=== Overall ===")
    print(sums["overall"].to_string(index=False))
    print("\n=== By object_level ===")
    print(sums["by_object_level"].to_string(index=False))
    print("\n=== By cue_level ===")
    print(sums["by_cue_level"].to_string(index=False))


if __name__ == "__main__":
    main()
