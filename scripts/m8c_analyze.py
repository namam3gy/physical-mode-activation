"""M8c — per-category PMR / paired-delta on real photographs.

Computes:
  1. PMR by (category, label_role) per model — does the H7 pattern hold
     on real photos?
  2. PMR(_nolabel) baseline per (category, model) — is the encoder
     more saturated on photos than on synthetic-textured stim?
  3. Paired-delta (label_role − _nolabel) per (category, model).
  4. Comparison vs synthetic counterparts:
     - photo ball vs M8a circle textured
     - photo car/person/bird vs M8d corresponding synthetic textured
     - photo abstract vs M8a circle line (synthetic abstract baseline)
  5. classify_regime regime distribution where applicable (car/person/bird
     photo categories), parallel to M8d.

Usage:
    uv run python scripts/m8c_analyze.py \\
        --qwen-labeled outputs/m8c_qwen_<ts>/predictions.jsonl \\
        --qwen-nolabel outputs/m8c_qwen_label_free_<ts>/predictions.jsonl \\
        --llava-labeled outputs/m8c_llava_<ts>/predictions.jsonl \\
        --llava-nolabel outputs/m8c_llava_label_free_<ts>/predictions.jsonl \\
        --out-dir outputs/m8c_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import classify_regime, score_rows


CATEGORIES = ("ball", "car", "person", "bird", "abstract")
ROLES = ("physical", "abstract", "exotic")
REGIMES = ("kinetic", "static", "abstract", "ambiguous")


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    return pd.read_parquet(path)


def _label_to_role(shape: str, label: str) -> str:
    if label == "_nolabel":
        return "_nolabel"
    triplet = LABELS_BY_SHAPE.get(shape)
    if triplet is None:
        return label
    physical, abstract, exotic = triplet
    if label == physical:
        return "physical"
    if label == abstract:
        return "abstract"
    if label == exotic:
        return "exotic"
    return label


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = score_rows(df)  # adds pmr, gar, hold_still, abstract_reject
    out["label_role"] = [_label_to_role(s, l) for s, l in zip(out["shape"], out["label"])]
    # classify_regime is only defined for car/person/bird; mark others
    # as 'n/a'.
    def _regime(s, t):
        try:
            return classify_regime(s, t)
        except ValueError:
            return "n/a"
    out["regime"] = [_regime(s, t) for s, t in zip(out["shape"], out["raw_text"])]
    out["pmr_regime"] = (out["regime"].isin(["kinetic", "static"])).astype(int)
    return out


def pmr_by_role(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["shape", "label_role"])["pmr"].mean().unstack().round(3)
    return g.reindex(index=list(CATEGORIES), columns=list(ROLES))


def paired_delta(lbl: pd.DataFrame, nl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        for role in ROLES:
            l_sub = lbl[(lbl["shape"] == cat) & (lbl["label_role"] == role)]
            n_sub = nl[nl["shape"] == cat]
            if l_sub.empty or n_sub.empty:
                continue
            l_agg = l_sub.groupby("sample_id")["pmr"].mean().rename(role)
            n_agg = n_sub.groupby("sample_id")["pmr"].mean().rename("_nolabel")
            joined = pd.concat([l_agg, n_agg], axis=1).dropna()
            rows.append({
                "category": cat,
                "label_role": role,
                "n_pairs": int(len(joined)),
                "delta": float((joined[role] - joined["_nolabel"]).mean()),
                "labeled_pmr": float(joined[role].mean()),
                "nolabel_pmr": float(joined["_nolabel"].mean()),
            })
    return pd.DataFrame(rows).round(3)


def regime_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(category, label_role) regime fractions (only for car/person/bird)."""
    rows = []
    for cat in CATEGORIES:
        if cat not in ("car", "person", "bird"):
            continue
        for role in (*ROLES, "_nolabel"):
            sub = df[(df["shape"] == cat) & (df["label_role"] == role)]
            if sub.empty:
                continue
            n = len(sub)
            row = {"category": cat, "label_role": role, "n": n}
            for r in REGIMES:
                row[r] = (sub["regime"] == r).mean()
            rows.append(row)
    return pd.DataFrame(rows).round(3)


def model_block(name: str, lbl: pd.DataFrame, nl: pd.DataFrame, out_dir: Path) -> dict:
    pmr_role = pmr_by_role(lbl)
    pd_role = paired_delta(lbl, nl)
    nolabel_pmr = nl.groupby("shape")["pmr"].mean().reindex(list(CATEGORIES)).round(3)
    rd = regime_distribution(pd.concat([lbl, nl], ignore_index=True))

    pmr_role.to_csv(out_dir / f"m8c_{name}_pmr_by_role.csv")
    pd_role.to_csv(out_dir / f"m8c_{name}_paired_delta.csv", index=False)
    nolabel_pmr.to_csv(out_dir / f"m8c_{name}_nolabel_pmr.csv")
    rd.to_csv(out_dir / f"m8c_{name}_regime_distribution.csv", index=False)

    print(f"\n=== {name}: PMR(_nolabel) baseline by category ===")
    print(nolabel_pmr.to_string())
    print(f"\n=== {name}: PMR by (category, label_role) ===")
    print(pmr_role.to_string())
    print(f"\n=== {name}: paired-delta vs _nolabel ===")
    print(pd_role.to_string(index=False))
    if not rd.empty:
        print(f"\n=== {name}: regime distribution (car/person/bird only) ===")
        print(rd.to_string(index=False))

    return {
        "pmr_role": pmr_role,
        "paired_delta": pd_role,
        "nolabel_pmr": nolabel_pmr,
        "regime_distribution": rd,
    }


def synthetic_comparison(out_dir: Path) -> None:
    """Compare M8c photo PMR(_nolabel) to the M8a/M8d synthetic-textured PMR.

    Reads the latest M8a + M8d label-free outputs from disk and emits a
    combined CSV `m8c_synthetic_vs_photo.csv` per (model × category).
    """
    project_root = Path(__file__).resolve().parents[1]
    out_root = project_root / "outputs"

    pairs = []
    for model_name, lf_glob in [
        ("qwen", "m8a_qwen_label_free_*"),
        ("llava", "m8a_llava_label_free_*"),
    ]:
        cands = sorted(out_root.glob(f"{lf_glob}/predictions.jsonl"))
        if not cands:
            continue
        df = pd.read_json(cands[-1], lines=True)
        df = score_rows(df)
        # M8a: shape == 'circle' is the synthetic ball.
        sub = df[(df["shape"] == "circle") & (df["object_level"] == "textured")]
        pairs.append({
            "model": model_name,
            "source": "synthetic-textured-circle",
            "category": "ball",
            "n": int(len(sub)),
            "pmr_nolabel": float(sub["pmr"].mean()),
        })
        # M8a abstract baseline: circle / line
        sub = df[(df["shape"] == "circle") & (df["object_level"] == "line")]
        pairs.append({
            "model": model_name,
            "source": "synthetic-line-circle",
            "category": "abstract-baseline",
            "n": int(len(sub)),
            "pmr_nolabel": float(sub["pmr"].mean()),
        })

    for model_name, lf_glob in [
        ("qwen", "m8d_qwen_label_free_*"),
        ("llava", "m8d_llava_label_free_*"),
    ]:
        cands = sorted(out_root.glob(f"{lf_glob}/predictions.jsonl"))
        if not cands:
            continue
        df = pd.read_json(cands[-1], lines=True)
        df = score_rows(df)
        for cat in ("car", "person", "bird"):
            sub = df[(df["shape"] == cat) & (df["object_level"] == "textured")]
            pairs.append({
                "model": model_name,
                "source": "synthetic-textured-m8d",
                "category": cat,
                "n": int(len(sub)),
                "pmr_nolabel": float(sub["pmr"].mean()),
            })

    df = pd.DataFrame(pairs).round(3)
    df.to_csv(out_dir / "m8c_synthetic_baseline.csv", index=False)
    print("\n=== Synthetic baseline PMR(_nolabel) for comparison ===")
    print(df.to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--qwen-labeled", type=Path, required=True)
    p.add_argument("--qwen-nolabel", type=Path, required=True)
    p.add_argument("--llava-labeled", type=Path, required=True)
    p.add_argument("--llava-nolabel", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading + annotating predictions ...")
    q_lbl = annotate(_load(args.qwen_labeled))
    q_nl = annotate(_load(args.qwen_nolabel))
    l_lbl = annotate(_load(args.llava_labeled))
    l_nl = annotate(_load(args.llava_nolabel))
    print(f"  Qwen labeled: {len(q_lbl)} rows; nolabel: {len(q_nl)} rows")
    print(f"  LLaVA labeled: {len(l_lbl)} rows; nolabel: {len(l_nl)} rows")

    model_block("qwen", q_lbl, q_nl, args.out_dir)
    model_block("llava", l_lbl, l_nl, args.out_dir)

    synthetic_comparison(args.out_dir)

    # Save annotated parquets for figures.
    pd.concat([q_lbl, q_nl], ignore_index=True).to_parquet(args.out_dir / "m8c_qwen_annotated.parquet", index=False)
    pd.concat([l_lbl, l_nl], ignore_index=True).to_parquet(args.out_dir / "m8c_llava_annotated.parquet", index=False)

    print(f"\nWrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
