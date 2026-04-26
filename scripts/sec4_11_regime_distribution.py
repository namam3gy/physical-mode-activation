"""§4.11 — Categorical H7 follow-up: regime distribution per (model × category × label_role).

Extends the existing m8d_regime_distribution.png (Qwen vs LLaVA only) to the
4-model M8d lineup (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2).

InternVL3 not run on M8d, so it's omitted.

For each (model × category × label_role) cell, classify each response into one
of {kinetic, static, abstract, ambiguous} via classify_regime, then plot a
stacked-bar matrix.

Usage:
    uv run python scripts/sec4_11_regime_distribution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import classify_regime, score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

CATEGORIES = ("car", "person", "bird")
ROLES = ("physical", "abstract", "exotic")
REGIMES = ("kinetic", "static", "abstract", "ambiguous")
REGIME_COLORS = {
    "kinetic": "#2ca02c",
    "static":  "#1f77b4",
    "abstract": "#d62728",
    "ambiguous": "#808080",
}

# Patterns: labeled glob excludes "_label_free" so the labeled arm doesn't
# accidentally match the label-free dir.
MODEL_RUNS = {
    "Qwen2.5-VL":  ("m8d_qwen_2*",                       "m8d_qwen_label_free_*"),
    "LLaVA-1.5":   ("m8d_llava_2*",                      "m8d_llava_label_free_*"),
    "LLaVA-Next":  ("encoder_swap_llava_next_m8d_2*",    "encoder_swap_llava_next_m8d_label_free_*"),
    "Idefics2":    ("encoder_swap_idefics2_m8d_2*",      "encoder_swap_idefics2_m8d_label_free_*"),
    "InternVL3":   ("encoder_swap_internvl3_m8d_2*",     "encoder_swap_internvl3_m8d_label_free_*"),
}


def _latest(pattern: str) -> Path | None:
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{pattern}/predictions.jsonl"))
    cands = [c for c in cands if c.stat().st_size > 0]
    return cands[-1] if cands else None


def _label_to_role(shape: str, label: str) -> str:
    triplet = LABELS_BY_SHAPE[shape]
    if label == triplet[0]: return "physical"
    if label == triplet[1]: return "abstract"
    if label == triplet[2]: return "exotic"
    return "_nolabel"


def load_model_data(model: str) -> pd.DataFrame:
    lbl_pat, nl_pat = MODEL_RUNS[model]
    lbl_path = _latest(lbl_pat)
    nl_path = _latest(nl_pat)
    if lbl_path is None or nl_path is None:
        print(f"WARN: {model} missing data (lbl={lbl_path}, nl={nl_path})")
        return pd.DataFrame()
    lbl = pd.read_json(lbl_path, lines=True)
    nl = pd.read_json(nl_path, lines=True)
    df = pd.concat([lbl, nl], ignore_index=True)
    # Subset to horizontal events (M8d's contrasts are sharpest there per m8d insight).
    df = df[df["event_template"] == "horizontal"].copy()
    df["regime"] = [classify_regime(s, t) for s, t in zip(df["shape"], df["raw_text"])]
    df["label_role"] = [_label_to_role(s, l) for s, l in zip(df["shape"], df["label"])]
    df["model"] = model
    return df


def regime_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Per (category × label_role) regime fractions."""
    rows = []
    for cat in CATEGORIES:
        for role in (*ROLES, "_nolabel"):
            sub = df[(df["shape"] == cat) & (df["label_role"] == role)]
            if len(sub) == 0:
                continue
            counts = sub["regime"].value_counts(normalize=True).to_dict()
            for r in REGIMES:
                rows.append({"category": cat, "label_role": role, "regime": r,
                             "fraction": counts.get(r, 0.0), "n": len(sub)})
    return pd.DataFrame(rows)


def main() -> None:
    all_data = {}
    for model in MODEL_RUNS:
        df = load_model_data(model)
        if len(df) > 0:
            all_data[model] = df

    if not all_data:
        print("No data found.")
        return

    # 4 models × 3 categories × 4 label_roles (physical, abstract, exotic, _nolabel)
    n_models = len(all_data)
    n_cats = len(CATEGORIES)
    n_roles = 4  # physical, abstract, exotic, _nolabel

    fig, axes = plt.subplots(n_models, n_cats, figsize=(4.0 * n_cats, 2.6 * n_models),
                              sharex=True, sharey=True, squeeze=False)

    role_order = ["_nolabel", "physical", "abstract", "exotic"]
    role_label = {"_nolabel": "no label", "physical": "phys", "abstract": "abs", "exotic": "exotic"}

    for i, (model, df) in enumerate(all_data.items()):
        fracs = regime_fractions(df)
        for j, cat in enumerate(CATEGORIES):
            ax = axes[i][j]
            sub = fracs[fracs["category"] == cat]
            x = np.arange(len(role_order))
            cum = np.zeros(len(role_order))
            for regime in REGIMES:
                vals = []
                for role in role_order:
                    cell = sub[(sub["label_role"] == role) & (sub["regime"] == regime)]
                    vals.append(float(cell["fraction"].iloc[0]) if len(cell) else 0)
                vals = np.array(vals)
                ax.bar(x, vals, bottom=cum, color=REGIME_COLORS[regime],
                       edgecolor="white", linewidth=0.5,
                       label=regime if (i == 0 and j == 0) else None)
                cum += vals
            ax.set_xticks(x)
            ax.set_xticklabels([role_label[r] for r in role_order], fontsize=8)
            if i == 0:
                ax.set_title(cat, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(model, fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(REGIMES), fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("§4.11 — M8d regime distribution per (model × category × label role)\n"
                 "horizontal-event subset; n_per_cell ≈ 40 for label roles, 40 for _nolabel",
                 y=1.06, fontsize=11)
    fig.tight_layout()

    suffix = f"{n_models}model"
    out = FIG_DIR / f"sec4_11_regime_distribution_{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Also save the underlying table for reference
    rows_xm = []
    for model, df in all_data.items():
        fracs = regime_fractions(df)
        fracs["model"] = model
        rows_xm.append(fracs)
    table = pd.concat(rows_xm, ignore_index=True)
    table_out = PROJECT_ROOT / "outputs" / "sec4_11_regime_distribution.csv"
    table_out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_out, index=False)
    print(f"Wrote {table_out}")


if __name__ == "__main__":
    main()
