"""§4.7 — RC (response consistency) reinterpreted as per-axis decision stability.

The pilot couldn't measure RC at T=0 (all responses identical, RC=1).
Under T=0.7 (M8a settings), RC measures within-cell stability across the
5 seeds. §4.7 asks: **which input axis (object_level / bg_level /
cue_level) is the strongest stabilizer?**

Method:
- For each (model × shape × object_level × bg_level × cue_level) cell
  with n_seeds = 5, compute RC = max(count(pmr=v)) / 5 over v ∈ {0, 1}.
- Aggregate across cells: per-axis-level mean RC per model.
- Visualize: 5-model grid showing mean RC by each factor level.

Models: 5 (Qwen, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3) × M8a
label-free arm (n=400 each, 80 cells × 5 seeds).

Usage:
    uv run python scripts/sec4_7_rc_per_axis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import response_consistency, score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

MODELS = {
    "Qwen2.5-VL":  "m8a_qwen_label_free_*",
    "LLaVA-1.5":   "m8a_llava_label_free_*",
    "LLaVA-Next":  "encoder_swap_llava_next_m8a_label_free_*",
    "Idefics2":    "encoder_swap_idefics2_label_free_*",
    "InternVL3":   "encoder_swap_internvl3_m8a_label_free_*",
}

ENCODER_COLOR = {
    "Qwen2.5-VL":  "#1f77b4",
    "LLaVA-1.5":   "#d62728",
    "LLaVA-Next":  "#ff7f0e",
    "Idefics2":    "#5fa8d3",
    "InternVL3":   "#2ca02c",
}

OBJ_ORDER = ("line", "filled", "shaded", "textured")
BG_ORDER = ("blank", "ground")
CUE_ORDER = ("none", "both")


def _latest(pat: str) -> Path | None:
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{pat}/predictions.jsonl"))
    cands = [c for c in cands if c.stat().st_size > 0]
    return cands[-1] if cands else None


def load_and_score(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    return score_rows(df)


def main() -> None:
    rows = []
    for model, pat in MODELS.items():
        path = _latest(pat)
        if path is None:
            print(f"WARN: missing data for {model}")
            continue
        df = load_and_score(path)
        rc_df = response_consistency(df, ["shape", "object_level", "bg_level", "cue_level"])
        # RC summary: mean RC overall + per-axis-level mean.
        rc_df["model"] = model
        rows.append(rc_df)

    if not rows:
        return
    all_rc = pd.concat(rows, ignore_index=True)

    # Headline: per-(model × axis × level) mean RC
    summary_rows = []
    for model in MODELS:
        sub = all_rc[all_rc["model"] == model]
        if len(sub) == 0:
            continue
        for axis_name, levels in [("object_level", OBJ_ORDER),
                                   ("bg_level", BG_ORDER),
                                   ("cue_level", CUE_ORDER)]:
            for lvl in levels:
                cell = sub[sub[axis_name] == lvl]
                summary_rows.append({
                    "model": model,
                    "axis": axis_name,
                    "level": lvl,
                    "mean_rc": float(cell["rc"].mean()),
                    "std_rc": float(cell["rc"].std()),
                    "n_cells": int(len(cell)),
                })
    summary = pd.DataFrame(summary_rows)
    print(summary.round(3).to_string(index=False))

    # Save table
    out_csv = PROJECT_ROOT / "outputs" / "sec4_7_rc_per_axis.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Figure: 3-panel (one per axis), bars per (level × model)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    width = 0.16

    for ax, (axis_name, levels, axis_label) in zip(axes, [
        ("object_level", OBJ_ORDER, "object_level"),
        ("bg_level", BG_ORDER, "bg_level"),
        ("cue_level", CUE_ORDER, "cue_level"),
    ]):
        x = np.arange(len(levels))
        offset0 = -(len(MODELS) - 1) / 2 * width
        for k, model in enumerate(MODELS):
            sub = summary[(summary["model"] == model) & (summary["axis"] == axis_name)]
            vals = []
            errs = []
            for lvl in levels:
                cell = sub[sub["level"] == lvl]
                if len(cell):
                    vals.append(float(cell["mean_rc"].iloc[0]))
                    errs.append(float(cell["std_rc"].iloc[0]) if not pd.isna(cell["std_rc"].iloc[0]) else 0)
                else:
                    vals.append(np.nan); errs.append(0)
            bar_x = x + offset0 + k * width
            ax.bar(bar_x, vals, width, color=ENCODER_COLOR[model],
                   edgecolor="black", alpha=0.9, label=model if axis_name == "object_level" else None)
            ax.errorbar(bar_x, vals, yerr=errs, fmt="none", ecolor="black", capsize=2, linewidth=0.6)
            for bx, v in zip(bar_x, vals):
                if not np.isnan(v):
                    ax.text(bx, v + 0.015, f"{v:.2f}", ha="center", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_xlabel(axis_label)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(f"mean RC by {axis_label}")
        ax.grid(True, alpha=0.3, axis="y")
        if axis_name == "object_level":
            ax.set_ylabel("response consistency (RC) — fraction of majority pmr per cell (n=5)")

    fig.legend(loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("§4.7 — Per-axis response-consistency (RC) on M8a label-free, 5 models",
                 y=1.07, fontsize=12)
    fig.tight_layout()

    out_png = FIG_DIR / "sec4_7_rc_per_axis.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
