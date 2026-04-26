"""§4.6 cross-model follow-up — analyze M2 vs M8a v_L per model + LLaVA-1.5
layer sweep.

Two analysis pieces:

1. **Class balance + cosine similarity**: compare per-model M2 v_L vs M8a
   v_L. Same direction (high cosine) means class imbalance wasn't the
   issue — saturation reading robust. Different direction means M2 v_L
   was noise; M8a v_L is the real "physics-mode direction" for that
   model.

2. **LLaVA-1.5 §4.6 layer sweep result**: aggregate PMR flip counts at
   L5 / L10 / L15 / L20 / L25. If any layer flips PMR > 0, the original
   §4.6 LLaVA-1.5 null was a wrong-layer artifact. If 0 flips at every
   layer, the saturation-specific reading holds.

Outputs:
- outputs/sec4_6_followup/comparison.csv
- outputs/sec4_6_followup/llava_layer_sweep.csv
- docs/insights/sec4_6_cross_model_m8a_followup.md (draft)

Usage:
    uv run python scripts/sec4_6_followup_analyze.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "sec4_6_followup"

# (model, M2 capture pattern, M8a capture pattern)
MODELS = [
    ("LLaVA-Next", "cross_model_llava_next_capture_*", "encoder_swap_llava_next_m8a_capture_*"),
    ("Idefics2",   "cross_model_idefics2_capture_*",   "encoder_swap_idefics2_m8a_capture_*"),
    ("InternVL3",  "cross_model_internvl3_capture_*",  "encoder_swap_internvl3_m8a_capture_*"),
]
LAYERS = (5, 10, 15, 20, 25)


def _latest(pattern: str) -> Path | None:
    cands = sorted(PROJECT_ROOT.glob(
        f"outputs/{pattern}/probing_steering/steering_vectors.npz"))
    return cands[-1].parent.parent if cands else None


def compare_v_L() -> pd.DataFrame:
    rows = []
    for model, m2_pat, m8a_pat in MODELS:
        m2_dir = _latest(m2_pat)
        m8a_dir = _latest(m8a_pat)
        if m2_dir is None:
            print(f"[SKIP {model}] no M2 v_L")
            continue
        if m8a_dir is None:
            print(f"[SKIP {model}] no M8a v_L (capture not done?)")
            continue
        m2 = np.load(m2_dir / "probing_steering/steering_vectors.npz")
        m8a = np.load(m8a_dir / "probing_steering/steering_vectors.npz")
        for L in LAYERS:
            v_m2 = m2[f"v_unit_{L}"]
            v_m8a = m8a[f"v_unit_{L}"]
            cos = float(v_m2 @ v_m8a)
            rows.append({
                "model": model, "layer": L,
                "m2_norm": float(np.linalg.norm(m2[f"v_{L}"])),
                "m8a_norm": float(np.linalg.norm(m8a[f"v_{L}"])),
                "cos_unit": cos,
                "interpretation": (
                    "same direction" if cos > 0.7 else
                    "moderate alignment" if cos > 0.3 else
                    "weakly aligned / orthogonal" if cos > -0.3 else
                    "anti-aligned"
                ),
            })
    return pd.DataFrame(rows)


def aggregate_llava_layer_sweep() -> pd.DataFrame:
    """Aggregate PMR flip counts per layer from sec4_6_counterfactual_llava_L<X>_*."""
    rows = []
    # L10 from existing sweep dir
    l10_dirs = sorted(PROJECT_ROOT.glob("outputs/sec4_6_counterfactual_llava_2*"))
    if l10_dirs:
        l10 = l10_dirs[-1] / "results_aggregated.csv"
        if l10.exists():
            df = pd.read_csv(l10)
            for _, r in df.iterrows():
                rows.append({"layer": 10, **r.to_dict()})
    # New layers from this overnight chain
    for L in (5, 15, 20, 25):
        cands = sorted(PROJECT_ROOT.glob(f"outputs/sec4_6_counterfactual_llava_L{L}_*/results_aggregated.csv"))
        if not cands:
            continue
        df = pd.read_csv(cands[-1])
        for _, r in df.iterrows():
            rows.append({"layer": L, **r.to_dict()})
    return pd.DataFrame(rows)


def write_insight_doc(comp_df: pd.DataFrame, sweep_df: pd.DataFrame) -> Path:
    """Draft insight doc in English (Korean translation deferred)."""
    md = ["---",
          "section: §4.6 cross-model follow-up (M8a v_L + LLaVA layer sweep)",
          "date: 2026-04-26",
          "status: draft (auto-generated; needs human review)",
          "---",
          "",
          "# §4.6 cross-model follow-up — M8a v_L + LLaVA-1.5 layer sweep",
          "",
          "Two analyses:",
          "1. M2 vs M8a v_L per model (class balance was n_neg = 1/5/9 on M2; "
          "n_neg = 100-280 on M8a).",
          "2. LLaVA-1.5 §4.6 layer sweep at L5/L10/L15/L20/L25.",
          "",
          "## 1. M2 vs M8a v_L cosine similarity",
          "",
          comp_df.round(3).to_markdown(index=False) if not comp_df.empty else "*No data — captures not done.*",
          "",
          "**Interpretation**:",
          "- High cosine (> 0.7) at clean-class-balance models means M2 v_L and M8a v_L point at the same direction → class imbalance wasn't the issue.",
          "- Low cosine (< 0.3) means M2 v_L was noise → M8a is the real direction.",
          "",
          "## 2. LLaVA-1.5 §4.6 layer sweep (PMR flips per layer)",
          "",
          sweep_df.round(3).to_markdown(index=False) if not sweep_df.empty else "*No data — sweep not done.*",
          "",
          "**Headline**:",
          "- If any layer flips PMR > 0/5 for v_L10 configs → original §4.6 LLaVA null was a wrong-layer artifact.",
          "- If all layers 0/5 → saturation-specific reading holds across layers.",
          "",
          "**Decision**: see comp_df + sweep_df above and the user's review.",
          ""]
    out = PROJECT_ROOT / "docs/insights/sec4_6_cross_model_m8a_followup.md"
    out.write_text("\n".join(md))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp_df = compare_v_L()
    sweep_df = aggregate_llava_layer_sweep()

    if not comp_df.empty:
        comp_df.to_csv(OUT_DIR / "comparison.csv", index=False)
        print("=== M2 vs M8a v_L cosine similarity ===")
        print(comp_df.round(3).to_string(index=False))

    if not sweep_df.empty:
        sweep_df.to_csv(OUT_DIR / "llava_layer_sweep.csv", index=False)
        print("\n=== LLaVA-1.5 §4.6 layer sweep ===")
        print(sweep_df.round(3).to_string(index=False))

    out = write_insight_doc(comp_df, sweep_df)
    print(f"\nDraft insight doc: {out}")


if __name__ == "__main__":
    main()
