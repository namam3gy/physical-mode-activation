"""M8d — 50-stim stratified sampler + classifier-validation report.

Generates a CSV scaffold for hand-annotation of model responses to validate
the classify_regime keyword classifier. Sampling is stratified across
(model × category × label_role) cells, ~2 responses per cell. Ties
broken by seed.

Two-stage workflow:
  1. `--mode sample`: emit a stratified-sample CSV with columns
     {sample_id, model, shape, label, label_role, raw_text, predicted_regime,
     hand_regime}. The user fills the `hand_regime` column with one of
     {kinetic, static, abstract, ambiguous}.
  2. `--mode score`: re-read the filled CSV; compute false-positive +
     false-negative rates of `predicted_regime` vs `hand_regime`. Print
     a confusion matrix + per-regime precision/recall + overall combined
     error rate. Threshold for paper-ready signal: combined error < 15 %.

Usage:
    # Stage 1 — sample
    uv run python scripts/m8d_hand_annotate.py --mode sample \\
        --qwen-labeled outputs/m8d_qwen_<ts>/predictions.jsonl \\
        --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \\
        --out docs/experiments/m8d_hand_annotate.csv

    # ... fill the hand_regime column ...

    # Stage 2 — score
    uv run python scripts/m8d_hand_annotate.py --mode score \\
        --csv docs/experiments/m8d_hand_annotate.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import classify_regime


CATEGORIES = ("car", "person", "bird")
ROLES = ("physical", "abstract", "exotic")
REGIMES = ("kinetic", "static", "abstract", "ambiguous")


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    return pd.read_parquet(path)


def _label_to_role(shape: str, label: str) -> str:
    if label == "_nolabel":
        return "_nolabel"
    p, a, e = LABELS_BY_SHAPE[shape]
    if label == p:
        return "physical"
    if label == a:
        return "abstract"
    if label == e:
        return "exotic"
    return "other"


def sample_stratified(qwen: pd.DataFrame, llava: pd.DataFrame, n_per_cell: int = 2, seed: int = 42) -> pd.DataFrame:
    """Stratified random sample: model × category × role × n_per_cell rows."""
    rng = random.Random(seed)
    rows = []
    for model_name, df in [("qwen", qwen), ("llava", llava)]:
        df = df.copy()
        df["label_role"] = [_label_to_role(s, l) for s, l in zip(df["shape"], df["label"])]
        for cat in CATEGORIES:
            for role in ROLES:
                cell = df[(df["shape"] == cat) & (df["label_role"] == role)]
                if cell.empty:
                    continue
                k = min(n_per_cell, len(cell))
                idxs = rng.sample(list(cell.index), k=k)
                for i in idxs:
                    r = cell.loc[i]
                    pred = classify_regime(r["shape"], r["raw_text"])
                    rows.append(
                        {
                            "model": model_name,
                            "sample_id": r["sample_id"],
                            "shape": r["shape"],
                            "object_level": r["object_level"],
                            "bg_level": r["bg_level"],
                            "cue_level": r["cue_level"],
                            "event_template": r["event_template"],
                            "label": r["label"],
                            "label_role": r["label_role"],
                            "raw_text": r["raw_text"],
                            "predicted_regime": pred,
                            "hand_regime": "",
                        }
                    )
    return pd.DataFrame(rows)


def score_csv(csv: Path) -> dict:
    df = pd.read_csv(csv)
    df = df[df["hand_regime"].notna() & (df["hand_regime"] != "")]
    if df.empty:
        raise ValueError(f"No hand_regime annotations in {csv}")

    df["hand_regime"] = df["hand_regime"].str.strip().str.lower()
    valid = set(REGIMES)
    bad = df[~df["hand_regime"].isin(valid)]
    if not bad.empty:
        print(f"WARNING: {len(bad)} rows have invalid hand_regime values:")
        print(bad[["sample_id", "hand_regime"]].head(10).to_string())
        df = df[df["hand_regime"].isin(valid)]

    df["correct"] = (df["predicted_regime"] == df["hand_regime"]).astype(int)
    n = len(df)
    n_correct = int(df["correct"].sum())
    err = (n - n_correct) / n if n else float("nan")

    print(f"\nHand-annotated rows: {n}; correct: {n_correct}; error rate: {err:.3f}")
    print(f"Threshold for paper-ready signal: combined error rate < 0.150")
    print(f"Result: {'PASS' if err < 0.15 else 'FAIL'}")

    print("\nConfusion matrix (rows = predicted, cols = hand):")
    cm = pd.crosstab(df["predicted_regime"], df["hand_regime"], dropna=False).reindex(
        index=list(REGIMES), columns=list(REGIMES), fill_value=0
    )
    print(cm.to_string())

    print("\nPer-regime precision / recall:")
    pr_rows = []
    for r in REGIMES:
        tp = int(((df["predicted_regime"] == r) & (df["hand_regime"] == r)).sum())
        fp = int(((df["predicted_regime"] == r) & (df["hand_regime"] != r)).sum())
        fn = int(((df["predicted_regime"] != r) & (df["hand_regime"] == r)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        pr_rows.append({"regime": r, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec})
    pr = pd.DataFrame(pr_rows).round(3)
    print(pr.to_string(index=False))

    return {"n": n, "n_correct": n_correct, "error_rate": err, "confusion": cm, "per_regime": pr}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("sample", "score"), required=True)
    p.add_argument("--qwen-labeled", type=Path)
    p.add_argument("--llava-labeled", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--csv", type=Path)
    p.add_argument("--n-per-cell", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.mode == "sample":
        if args.qwen_labeled is None or args.llava_labeled is None or args.out is None:
            raise SystemExit("--qwen-labeled, --llava-labeled, --out are required in sample mode")
        q = _load(args.qwen_labeled)
        l = _load(args.llava_labeled)
        df = sample_stratified(q, l, n_per_cell=args.n_per_cell, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df)} stratified-sample rows to {args.out}")
        print("Fill the `hand_regime` column with one of:", REGIMES)
        return

    if args.mode == "score":
        if args.csv is None:
            raise SystemExit("--csv is required in score mode")
        score_csv(args.csv)


if __name__ == "__main__":
    main()
