"""Compare Qwen MCQ Phase 3 (M5a + M5b) against existing yesno + describe results.

Audit follow-up #8 from `docs/insights/review_audit_2026-04-28.md`:
- If MCQ M5a flips like describe (10/10 at α=40) and M5b breaks like describe →
  the boundary is the *yes/no format*; categorical task isn't the dissociator.
- If MCQ M5a stays like yesno (0/10 at α=40) and M5b doesn't break →
  the boundary is the *categorical task*; any non-generative prompt blocks.
- Mixed → ambiguous; both task and format matter.

Usage:
  uv run python scripts/m_mp_mcq_phase3_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physical_mode.metrics.pmr import score_for_variant  # noqa: E402


def load_steering(path: Path, variant: str) -> dict[float, float]:
    """Return {alpha: PMR} for a Phase 3 steering output."""
    pq = path / "intervention_predictions.parquet"
    if not pq.exists():
        return {}
    df = pd.read_parquet(pq)
    if "pmr" not in df.columns:
        df["pmr"] = df["raw_text"].apply(lambda t: score_for_variant(t, variant))
    return {float(a): float(df[df.alpha == a]["pmr"].mean()) for a in sorted(df.alpha.unique())}


def load_sae_results(path: Path) -> dict[str, float]:
    """Return {condition: PMR} for an SAE intervention output."""
    csv = path / "results.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    if "intervention_pmr" in df.columns:
        return df.groupby("condition")["intervention_pmr"].mean().to_dict()
    if "intervention_phys" in df.columns:
        return df.groupby("condition")["intervention_phys"].mean().to_dict()
    return {}


def main() -> None:
    project = Path(__file__).resolve().parents[1]

    m5a_yesno = project / "outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/phase3_yesno"
    m5a_describe = project / "outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/phase3_describe"
    m5a_mcq = project / "outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/mcq_audit_l10_a40"

    m5b_yesno = project / "outputs/sae_intervention/phase3_qwen_yesno_k20"
    m5b_describe = project / "outputs/sae_intervention/phase3_qwen_describe_k20_v2"
    m5b_mcq = project / "outputs/sae_intervention/qwen_vis31_5120_mcq_audit"

    print("=" * 75)
    print("Qwen Phase 3 — M5a runtime steering at L10 (line/blank/none × circle)")
    print("=" * 75)
    print(f"{'prompt':<22}{'α=0':>10}{'α=40':>10}")
    for label, path, variant in [
        ("describe_scene", m5a_describe, "describe_scene"),
        ("meta_phys_yesno", m5a_yesno, "meta_phys_yesno"),
        ("meta_phys_mcq (NEW)", m5a_mcq, "meta_phys_mcq"),
    ]:
        d = load_steering(path, variant)
        a0 = d.get(0.0, float("nan"))
        a40 = d.get(40.0, float("nan"))
        print(f"  {label:<20}{a0:>10.3f}{a40:>10.3f}")

    print()
    print("=" * 75)
    print("Qwen Phase 3 — M5b SAE intervention top-20 (shaded/ground/both × ball)")
    print("=" * 75)
    print(f"{'prompt':<22}{'baseline':>12}{'top_k=20':>12}{'random':>12}")
    for label, path in [
        ("describe_scene", m5b_describe),
        ("meta_phys_yesno", m5b_yesno),
        ("meta_phys_mcq (NEW)", m5b_mcq),
    ]:
        if not path.exists():
            print(f"  {label:<20} (path not found: {path.name})")
            continue
        # Re-load with raw_text for re-scoring.
        csv = path / "results.csv"
        if not csv.exists():
            print(f"  {label:<20} (results.csv missing)")
            continue
        df = pd.read_csv(csv)
        variant = (
            "meta_phys_mcq" if "mcq" in str(path) else
            "meta_phys_yesno" if "yesno" in str(path) else
            "describe_scene"
        )
        df["bl_pmr"] = df["baseline_text"].astype(str).map(
            lambda t: score_for_variant(t, variant)
        )
        df["iv_pmr"] = df["intervention_text"].astype(str).map(
            lambda t: score_for_variant(t, variant)
        )
        bl = df["bl_pmr"].mean()
        topk = df[df.condition.str.startswith("top_k=20")]["iv_pmr"].mean() if (df.condition.str.startswith("top_k=20")).any() else float("nan")
        rand = df[df.condition.str.startswith("random_")]["iv_pmr"].mean() if (df.condition.str.startswith("random_")).any() else float("nan")
        print(f"  {label:<20}{bl:>12.3f}{topk:>12.3f}{rand:>12.3f}")

    print()
    print("Interpretation: ")
    print("  - MCQ flips like describe (α=40 → 1.0) AND breaks like describe (top_k=20 → 0)")
    print("    → 'yes/no format' is the boundary; categorical task is OK.")
    print("  - MCQ stays like yesno (α=40 → 0.0) AND doesn't break (top_k=20 → 1.0)")
    print("    → categorical task is the boundary; format secondary.")
    print("  - Mixed → both task and format matter; need finer probe.")


if __name__ == "__main__":
    main()
