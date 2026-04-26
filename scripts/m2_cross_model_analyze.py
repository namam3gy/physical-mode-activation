"""M2 cross-model — aggregate PMR / H1 ramp / H2 paired-delta across 5 VLMs.

Pairs each model's M2-stim run with its label-free counterpart. Loads all
runs from `outputs/cross_model_*` (this milestone) plus the original Qwen
M2 (`mvp_full_*`). Scores PMR per response, computes per-(model × prompt
× label) summary with bootstrap CIs (5000 iters), and generates:

  - outputs/m2_cross_model_summary/per_label_pmr.csv
  - outputs/m2_cross_model_summary/per_object_level_pmr.csv (H1 ramp)
  - outputs/m2_cross_model_summary/h2_paired_delta.csv
  - docs/figures/m2_cross_model_pmr_ladder.png
  - docs/figures/m2_cross_model_h1_ramp.png
  - docs/figures/m2_cross_model_h2_paired_delta.png

Usage:
    uv run python scripts/m2_cross_model_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "outputs" / "m2_cross_model_summary"

# (English M2 labeled run pattern, M2 label-free run pattern). Patterns
# match the latest run of each kind by glob + sort.
MODEL_RUNS: dict[str, tuple[str, str]] = {
    "Qwen2.5-VL":  ("mvp_full_2*",                                "label_free_2*"),
    "LLaVA-1.5":   ("cross_model_llava_2*",                       "cross_model_llava_label_free_*"),
    "LLaVA-Next":  ("cross_model_llava_next_capture_2*",          "cross_model_llava_next_label_free_*"),
    "Idefics2":    ("cross_model_idefics2_capture_2*",            "cross_model_idefics2_label_free_*"),
    "InternVL3":   ("cross_model_internvl3_2*",                   "cross_model_internvl3_label_free_*"),
}

LABELS = ["ball", "circle", "planet"]
NO_LABEL = "_nolabel"
OBJECT_LEVELS = ["line", "filled", "shaded", "textured"]
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

MODEL_COLORS = {
    "Qwen2.5-VL":  "#1f77b4",
    "LLaVA-1.5":   "#ff7f0e",
    "LLaVA-Next":  "#d62728",
    "Idefics2":    "#2ca02c",
    "InternVL3":   "#9467bd",
}


def _latest(pattern: str) -> Path:
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{pattern}/predictions.jsonl"))
    # Filter out _capture sub-paths that match the bare pattern accidentally
    cands = [c for c in cands if c.stat().st_size > 0]
    if not cands:
        raise FileNotFoundError(f"No outputs match {pattern}")
    return cands[-1]


def _score_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pmr"] = df["raw_text"].apply(score_pmr)
    return df


def _bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(BOOTSTRAP_N, dtype=float)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_label = []
    per_object = []
    paired_delta = []

    for model, (lbl_pat, nl_pat) in MODEL_RUNS.items():
        try:
            lbl_path = _latest(lbl_pat)
            nl_path = _latest(nl_pat)
        except FileNotFoundError as e:
            print(f"[SKIP] {model}: {e}")
            continue
        print(f"[{model}] labeled={lbl_path.parent.name}  label-free={nl_path.parent.name}")

        lbl_df = _score_rows(pd.read_json(lbl_path, lines=True))
        nl_df = _score_rows(pd.read_json(nl_path, lines=True))

        # Filter to open prompt only (ignore FC if present)
        if "prompt_variant" in lbl_df.columns:
            lbl_df = lbl_df[lbl_df["prompt_variant"] == "open"]
        if "prompt_variant" in nl_df.columns:
            nl_df = nl_df[nl_df["prompt_variant"].isin(["open_no_label", "open"])]

        # Per-label aggregate
        for label in LABELS:
            sub = lbl_df[lbl_df["label"] == label]
            mean, lo, hi = _bootstrap_ci(sub["pmr"].to_numpy())
            per_label.append({
                "model": model, "prompt": "open", "label": label,
                "n": len(sub), "pmr_mean": mean, "ci_lo": lo, "ci_hi": hi,
            })
        # No-label
        mean, lo, hi = _bootstrap_ci(nl_df["pmr"].to_numpy())
        per_label.append({
            "model": model, "prompt": "open_no_label", "label": NO_LABEL,
            "n": len(nl_df), "pmr_mean": mean, "ci_lo": lo, "ci_hi": hi,
        })

        # H1 ramp: per object_level mean PMR (open, all labels pooled)
        for obj in OBJECT_LEVELS:
            sub = lbl_df[lbl_df["object_level"] == obj]
            mean, lo, hi = _bootstrap_ci(sub["pmr"].to_numpy())
            per_object.append({
                "model": model, "object_level": obj,
                "n": len(sub), "pmr_mean": mean, "ci_lo": lo, "ci_hi": hi,
            })

        # H2 paired delta: PMR(label) - PMR(_nolabel) on common (sample_id) cells
        # Match by stim sample_id (lbl_df has 3 labels per sample, nl_df has 1)
        for label in LABELS:
            lbl_sub = lbl_df[lbl_df["label"] == label][["sample_id", "pmr"]]
            paired = lbl_sub.merge(
                nl_df[["sample_id", "pmr"]],
                on="sample_id", suffixes=("_lbl", "_nl"))
            delta = paired["pmr_lbl"].to_numpy() - paired["pmr_nl"].to_numpy()
            mean, lo, hi = _bootstrap_ci(delta)
            paired_delta.append({
                "model": model, "label": label, "n": len(paired),
                "delta_mean": mean, "ci_lo": lo, "ci_hi": hi,
            })

    return (pd.DataFrame(per_label), pd.DataFrame(per_object),
            pd.DataFrame(paired_delta))


# ---- Figures ----

def plot_pmr_ladder(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    cats = LABELS + [NO_LABEL]
    x = np.arange(len(cats))
    width = 0.16
    models = list(MODEL_RUNS.keys())
    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        means = []; los = []; his = []
        for c in cats:
            row = sub[sub["label"] == c]
            if len(row):
                means.append(row["pmr_mean"].iloc[0])
                los.append(row["pmr_mean"].iloc[0] - row["ci_lo"].iloc[0])
                his.append(row["ci_hi"].iloc[0] - row["pmr_mean"].iloc[0])
            else:
                means.append(np.nan); los.append(0); his.append(0)
        ax.bar(x + (i - 2) * width, means, width=width,
               yerr=[los, his], capsize=2, label=model,
               color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("PMR (mean ± 95% bootstrap CI)")
    ax.set_title("§M2 cross-model — per-label PMR (5 VLMs on M2 stim, open prompt)")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h1_ramp(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(OBJECT_LEVELS))
    for model in MODEL_RUNS.keys():
        sub = df[df["model"] == model].set_index("object_level").reindex(OBJECT_LEVELS)
        means = sub["pmr_mean"].to_numpy()
        los = means - sub["ci_lo"].to_numpy()
        his = sub["ci_hi"].to_numpy() - means
        ax.errorbar(x, means, yerr=[los, his], capsize=3,
                    marker="o", linewidth=2, markersize=8,
                    label=model, color=MODEL_COLORS[model])
    ax.set_xticks(x)
    ax.set_xticklabels(OBJECT_LEVELS)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("object_level (abstraction axis)")
    ax.set_ylabel("PMR (mean ± 95% CI)")
    ax.set_title("§M2 cross-model — H1 abstraction ramp per model (5 VLMs)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_h2_paired_delta(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(LABELS))
    width = 0.16
    models = list(MODEL_RUNS.keys())
    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        means = []; los = []; his = []
        for label in LABELS:
            row = sub[sub["label"] == label]
            if len(row):
                means.append(row["delta_mean"].iloc[0])
                los.append(row["delta_mean"].iloc[0] - row["ci_lo"].iloc[0])
                his.append(row["ci_hi"].iloc[0] - row["delta_mean"].iloc[0])
            else:
                means.append(np.nan); los.append(0); his.append(0)
        ax.bar(x + (i - 2) * width, means, width=width,
               yerr=[los, his], capsize=2, label=model,
               color=MODEL_COLORS[model])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("PMR(label) − PMR(_nolabel) (mean ± 95% CI)")
    ax.set_title("§M2 cross-model — H2 paired delta (5 VLMs)")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_label, per_object, paired = aggregate()

    per_label.to_csv(OUT_DIR / "per_label_pmr.csv", index=False)
    per_object.to_csv(OUT_DIR / "per_object_level_pmr.csv", index=False)
    paired.to_csv(OUT_DIR / "h2_paired_delta.csv", index=False)
    print(f"\nCSVs written to {OUT_DIR}")
    print()
    print("=== Per-label PMR (5 VLMs × 4 categories) ===")
    print(per_label.round(3).to_string(index=False))
    print()
    print("=== H1 ramp (per-object_level) ===")
    print(per_object.round(3).to_string(index=False))
    print()
    print("=== H2 paired delta (PMR(label) − PMR(_nolabel)) ===")
    print(paired.round(3).to_string(index=False))

    plot_pmr_ladder(per_label, FIG_DIR / "m2_cross_model_pmr_ladder.png")
    plot_h1_ramp(per_object, FIG_DIR / "m2_cross_model_h1_ramp.png")
    plot_h2_paired_delta(paired, FIG_DIR / "m2_cross_model_h2_paired_delta.png")


if __name__ == "__main__":
    main()
