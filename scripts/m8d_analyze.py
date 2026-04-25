"""M8d — per-category PMR / paired-delta / regime distribution roll-up.

Computes:
  1. PMR_regime by (category, object_level) — H1 ramp test, computed on
     the union of {fall, horizontal}.
  2. PMR_regime by (category, label_role) — H7 paired-delta test, on the
     `horizontal` subset (natural-event cell where regime selection is
     cleanest), with the `fall` subset reported separately.
  3. Regime distribution by (category, label_role, model) — per-cell
     fraction of {kinetic, static, abstract, ambiguous}. The H7 signal in
     M8d is the *categorical* split rather than a binary PMR; this is the
     primary figure-ready table.
  4. Paired-delta vs label-free baseline per (category, label_role).
     PMR_regime(role) − PMR_regime(_nolabel) on matched sample_ids.
  5. Pre-registered scoring (Qwen X/3, LLaVA Y/3) parallel to M8a.

PMR_regime convention:
  PMR_regime = 1 iff classify_regime(text, category) ∈ {kinetic, static}.
  This captures "the model treated the object as physical" regardless of
  which regime fired. The original PMR is gravity-verb-biased and
  systematically undercounts kinetic responses for car (drives) /
  person (walks); classify_regime is category-aware.

Usage:
    uv run python scripts/m8d_analyze.py \\
        --qwen-labeled outputs/m8d_qwen_<ts>/predictions.jsonl \\
        --qwen-nolabel outputs/m8d_qwen_label_free_<ts>/predictions.jsonl \\
        --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \\
        --llava-nolabel outputs/m8d_llava_label_free_<ts>/predictions.jsonl \\
        --out-dir outputs/m8d_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import classify_regime, score_rows


CATEGORIES = ("car", "person", "bird")
OBJ_LEVELS = ("line", "filled", "shaded", "textured")
ROLES = ("physical", "abstract", "exotic")
REGIMES = ("kinetic", "static", "abstract", "ambiguous")


def _load(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    return pd.read_parquet(path)


def _label_to_role(shape: str, label: str) -> str:
    if label == "_nolabel":
        return "_nolabel"
    physical, abstract, exotic = LABELS_BY_SHAPE[shape]
    if label == physical:
        return "physical"
    if label == abstract:
        return "abstract"
    if label == exotic:
        return "exotic"
    return "other"


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add: regime, pmr_regime, pmr (legacy), label_role columns."""
    out = score_rows(df)  # adds legacy pmr / gar / hold_still / abstract_reject
    out["regime"] = [classify_regime(s, t) for s, t in zip(out["shape"], out["raw_text"])]
    out["pmr_regime"] = (out["regime"].isin(["kinetic", "static"])).astype(int)
    out["label_role"] = [_label_to_role(s, l) for s, l in zip(out["shape"], out["label"])]
    return out


def pmr_by_obj(df: pd.DataFrame) -> pd.DataFrame:
    """PMR_regime by (category, object_level), event-union."""
    g = (
        df.groupby(["shape", "object_level"])["pmr_regime"]
        .mean()
        .unstack()
        .round(3)
    )
    return g.reindex(index=list(CATEGORIES), columns=list(OBJ_LEVELS))


def pmr_by_role(df: pd.DataFrame, event_subset: tuple[str, ...] | None = None) -> pd.DataFrame:
    """PMR_regime by (category, label_role)."""
    sub = df if event_subset is None else df[df["event_template"].isin(event_subset)]
    g = (
        sub.groupby(["shape", "label_role"])["pmr_regime"]
        .mean()
        .unstack()
        .round(3)
    )
    return g.reindex(index=list(CATEGORIES), columns=list(ROLES))


def regime_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(category, label_role) regime fractions."""
    rows = []
    for cat in CATEGORIES:
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


def paired_delta(
    lbl: pd.DataFrame, nl: pd.DataFrame, event_subset: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """PMR_regime(role) − PMR_regime(_nolabel) per (category, role) on matched sample_ids."""
    if event_subset is not None:
        lbl = lbl[lbl["event_template"].isin(event_subset)]
        nl = nl[nl["event_template"].isin(event_subset)]
    rows = []
    for cat in CATEGORIES:
        for role in ROLES:
            l_sub = lbl[(lbl["shape"] == cat) & (lbl["label_role"] == role)]
            n_sub = nl[nl["shape"] == cat]
            if l_sub.empty or n_sub.empty:
                continue
            # Aggregate by sample_id (each sample_id has 1 _nolabel response and
            # 1 labeled response per role × seed; T=0.7 so paired by sample_id is fine).
            l_agg = l_sub.groupby("sample_id")["pmr_regime"].mean().rename(role)
            n_agg = n_sub.groupby("sample_id")["pmr_regime"].mean().rename("_nolabel")
            joined = pd.concat([l_agg, n_agg], axis=1).dropna()
            rows.append(
                {
                    "category": cat,
                    "label_role": role,
                    "n_pairs": int(len(joined)),
                    "delta": float((joined[role] - joined["_nolabel"]).mean()),
                    "labeled_pmr": float(joined[role].mean()),
                    "nolabel_pmr": float(joined["_nolabel"].mean()),
                }
            )
    return pd.DataFrame(rows).round(3)


def preregistered_scoring(
    pmr_obj: pd.DataFrame, pmr_role_horizontal: pd.DataFrame, pd_horizontal: pd.DataFrame
) -> dict[str, dict]:
    """Strict M8a-style pre-registration scoring per (category × criterion).

    Criteria (parallel to M8a):
      A. H1 ramp:    PMR(textured) − PMR(line) ≥ 0.05 AND no inversion >0.05
                     between adjacent levels.
      B. H7 (PMR):   PMR(physical) − PMR(abstract) ≥ 0.05 on horizontal subset.
      C. Visual-saturation Δ: paired-delta(physical) ≥ 0.05 on horizontal subset.

    Returns per-criterion summary {criterion: {category: bool, "passes": int}}.
    """

    def h1_ramp_pass(row: pd.Series) -> bool:
        try:
            ramp = row["textured"] - row["line"]
            adj = [row[OBJ_LEVELS[i + 1]] - row[OBJ_LEVELS[i]] for i in range(len(OBJ_LEVELS) - 1)]
            min_step = min(adj)
            return (ramp >= 0.05) and (min_step >= -0.05)
        except KeyError:
            return False

    def h7_pmr_pass(row: pd.Series) -> bool:
        try:
            return (row["physical"] - row["abstract"]) >= 0.05
        except KeyError:
            return False

    a = {cat: bool(h1_ramp_pass(pmr_obj.loc[cat])) for cat in CATEGORIES if cat in pmr_obj.index}
    b = {cat: bool(h7_pmr_pass(pmr_role_horizontal.loc[cat])) for cat in CATEGORIES if cat in pmr_role_horizontal.index}
    pd_phys = (
        pd_horizontal[pd_horizontal["label_role"] == "physical"]
        .set_index("category")["delta"]
    )
    c = {cat: bool(pd_phys.get(cat, 0.0) >= 0.05) for cat in CATEGORIES}

    return {
        "H1_ramp": {**a, "passes": sum(a.values()), "total": len(a)},
        "H7_pmr_horizontal": {**b, "passes": sum(b.values()), "total": len(b)},
        "saturation_delta_horizontal": {**c, "passes": sum(c.values()), "total": len(c)},
    }


def model_block(name: str, lbl: pd.DataFrame, nl: pd.DataFrame, out_dir: Path) -> dict:
    """Compute + write all per-model summaries; return preregistered scores."""
    pmr_obj = pmr_by_obj(lbl)
    pmr_role_all = pmr_by_role(lbl)
    pmr_role_h = pmr_by_role(lbl, event_subset=("horizontal",))
    pmr_role_f = pmr_by_role(lbl, event_subset=("fall",))
    rd = regime_distribution(pd.concat([lbl, nl], ignore_index=True))
    pd_all = paired_delta(lbl, nl)
    pd_h = paired_delta(lbl, nl, event_subset=("horizontal",))
    pd_f = paired_delta(lbl, nl, event_subset=("fall",))
    nolabel_baseline = (
        nl.groupby(["shape", "object_level"])["pmr_regime"].mean().unstack().round(3)
    )
    nolabel_baseline = nolabel_baseline.reindex(index=list(CATEGORIES), columns=list(OBJ_LEVELS))

    pmr_obj.to_csv(out_dir / f"m8d_{name}_pmr_by_obj.csv")
    pmr_role_all.to_csv(out_dir / f"m8d_{name}_pmr_by_role_all.csv")
    pmr_role_h.to_csv(out_dir / f"m8d_{name}_pmr_by_role_horizontal.csv")
    pmr_role_f.to_csv(out_dir / f"m8d_{name}_pmr_by_role_fall.csv")
    rd.to_csv(out_dir / f"m8d_{name}_regime_distribution.csv", index=False)
    pd_all.to_csv(out_dir / f"m8d_{name}_paired_delta_all.csv", index=False)
    pd_h.to_csv(out_dir / f"m8d_{name}_paired_delta_horizontal.csv", index=False)
    pd_f.to_csv(out_dir / f"m8d_{name}_paired_delta_fall.csv", index=False)
    nolabel_baseline.to_csv(out_dir / f"m8d_{name}_nolabel_pmr_by_obj.csv")

    print(f"\n=== {name}: PMR_regime by (category, object_level) — event union ===")
    print(pmr_obj.to_string())
    print(f"\n=== {name}: nolabel PMR_regime by (category, object_level) — event union ===")
    print(nolabel_baseline.to_string())
    print(f"\n=== {name}: PMR_regime by (category, label_role) — horizontal subset ===")
    print(pmr_role_h.to_string())
    print(f"\n=== {name}: regime distribution by (category, label_role) ===")
    print(rd.to_string(index=False))
    print(f"\n=== {name}: paired-delta vs _nolabel — horizontal subset ===")
    print(pd_h.to_string(index=False))

    scores = preregistered_scoring(pmr_obj, pmr_role_h, pd_h)
    print(f"\n=== {name}: pre-registered scoring (strict) ===")
    for crit, dct in scores.items():
        per_cat = {k: v for k, v in dct.items() if k in CATEGORIES}
        print(f"  {crit}: {dct['passes']}/{dct['total']}  per-category: {per_cat}")
    return scores


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

    qwen_scores = model_block("qwen", q_lbl, q_nl, args.out_dir)
    llava_scores = model_block("llava", l_lbl, l_nl, args.out_dir)

    print("\n\n========== HEADLINE TABLE (M8d strict pre-registered scoring) ==========")
    print(f"{'Criterion':<32}{'Qwen':<12}{'LLaVA':<12}")
    print("-" * 56)
    for crit in qwen_scores:
        q = qwen_scores[crit]
        l = llava_scores[crit]
        print(f"{crit:<32}{q['passes']}/{q['total']:<10}{l['passes']}/{l['total']:<10}")

    # Save concatenated annotated dataframes for figures.
    q_all = pd.concat([q_lbl, q_nl], ignore_index=True)
    l_all = pd.concat([l_lbl, l_nl], ignore_index=True)
    q_all.to_parquet(args.out_dir / "m8d_qwen_annotated.parquet", index=False)
    l_all.to_parquet(args.out_dir / "m8d_llava_annotated.parquet", index=False)

    print(f"\nWrote summaries to {args.out_dir}")


if __name__ == "__main__":
    main()
