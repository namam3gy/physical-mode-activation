"""M8d figures — non-ball categories, classify_regime-driven analysis.

Figures (in addition to the pre-existing m8d_shape_grid.png):
  m8d_full_scene_samples.png    — 12 sample stimuli covering category ×
                                   abstraction × event for paper figures.
  m8d_pmr_ramp.png              — PMR_regime by (category × object_level),
                                   event-union, Qwen vs LLaVA.
  m8d_pmr_by_role.png           — PMR_regime by (category × label_role),
                                   horizontal subset, Qwen vs LLaVA.
  m8d_paired_delta.png          — paired-delta(role) − _nolabel heatmap,
                                   horizontal subset, Qwen vs LLaVA.
  m8d_regime_distribution.png   — stacked bar of regime distribution per
                                   (category × label_role × model).

Reads pre-annotated parquet from `m8d_analyze.py --out-dir`.

Usage:
    uv run python scripts/m8d_figures.py --summary-dir outputs/m8d_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import classify_regime, score_rows


CATEGORIES = ("car", "person", "bird")
OBJ_LEVELS = ("line", "filled", "shaded", "textured")
ROLES = ("physical", "abstract", "exotic")
REGIMES = ("kinetic", "static", "abstract", "ambiguous")
REGIME_COLORS = {
    "kinetic":   "#1f77b4",
    "static":    "#ff7f0e",
    "abstract":  "#7f7f7f",
    "ambiguous": "#d62728",
}
ROLE_COLORS = {
    "physical": "#1f77b4",
    "abstract": "#ff7f0e",
    "exotic":   "#2ca02c",
    "_nolabel": "#7f7f7f",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "docs" / "figures"


def role_of(shape: str, label: str) -> str:
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


def load_annotated(parq: Path) -> pd.DataFrame:
    df = pd.read_parquet(parq)
    needed = {"regime", "pmr_regime", "label_role"}
    missing = needed - set(df.columns)
    if missing:
        df = score_rows(df)
        df["regime"] = [classify_regime(s, t) for s, t in zip(df["shape"], df["raw_text"])]
        df["pmr_regime"] = (df["regime"].isin(["kinetic", "static"])).astype(int)
        df["label_role"] = [role_of(s, l) for s, l in zip(df["shape"], df["label"])]
    return df


# ---------------------------------------------------------------------------
# Figure: PMR ramp — PMR_regime by (category × object_level)
# ---------------------------------------------------------------------------


def fig_pmr_ramp(q_lbl: pd.DataFrame, l_lbl: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, df, title in [(axes[0], q_lbl, "Qwen2.5-VL-7B"), (axes[1], l_lbl, "LLaVA-1.5-7B")]:
        for cat in CATEGORIES:
            sub = df[df["shape"] == cat]
            ramp = [sub[sub["object_level"] == lv]["pmr_regime"].mean() for lv in OBJ_LEVELS]
            ax.plot(OBJ_LEVELS, ramp, marker="o", label=cat, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel("PMR_regime  (kinetic ∨ static)")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("object_level")
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="category", loc="lower right", fontsize=10)
    fig.suptitle("M8d — PMR_regime ramp by (category × object_level)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: PMR by role on horizontal subset
# ---------------------------------------------------------------------------


def fig_pmr_by_role(q_lbl: pd.DataFrame, l_lbl: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    width = 0.25
    x = np.arange(len(CATEGORIES))
    for ax, df, title in [(axes[0], q_lbl, "Qwen2.5-VL-7B"), (axes[1], l_lbl, "LLaVA-1.5-7B")]:
        sub = df[df["event_template"] == "horizontal"]
        for j, role in enumerate(ROLES):
            ys = [
                sub[(sub["shape"] == c) & (sub["label_role"] == role)]["pmr_regime"].mean()
                for c in CATEGORIES
            ]
            ax.bar(x + (j - 1) * width, ys, width, label=role, color=ROLE_COLORS[role])
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PMR_regime")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    axes[1].legend(title="label_role", fontsize=10)
    fig.suptitle("M8d — PMR_regime by (category × label_role) | horizontal subset", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: paired-delta heatmap on horizontal subset
# ---------------------------------------------------------------------------


def fig_paired_delta(q_lbl: pd.DataFrame, q_nl: pd.DataFrame, l_lbl: pd.DataFrame, l_nl: pd.DataFrame, out: Path) -> None:
    def matrix(lbl: pd.DataFrame, nl: pd.DataFrame) -> np.ndarray:
        lbl = lbl[lbl["event_template"] == "horizontal"]
        nl = nl[nl["event_template"] == "horizontal"]
        m = np.zeros((len(CATEGORIES), len(ROLES)))
        for i, cat in enumerate(CATEGORIES):
            n_agg = nl[nl["shape"] == cat].groupby("sample_id")["pmr_regime"].mean().rename("_nolabel")
            for j, role in enumerate(ROLES):
                l_agg = (
                    lbl[(lbl["shape"] == cat) & (lbl["label_role"] == role)]
                    .groupby("sample_id")["pmr_regime"]
                    .mean()
                    .rename(role)
                )
                joined = pd.concat([n_agg, l_agg], axis=1).dropna()
                m[i, j] = (joined[role] - joined["_nolabel"]).mean() if len(joined) else np.nan
        return m

    q_pd = matrix(q_lbl, q_nl)
    l_pd = matrix(l_lbl, l_nl)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, mat, title in [(axes[0], q_pd, "Qwen2.5-VL-7B"), (axes[1], l_pd, "LLaVA-1.5-7B")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        ax.set_xticks(range(len(ROLES)), ROLES)
        ax.set_yticks(range(len(CATEGORIES)), CATEGORIES)
        for i in range(len(CATEGORIES)):
            for j in range(len(ROLES)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(v) > 0.35 else "black",
                    fontsize=10,
                )
        ax.set_title(f"paired Δ — {title}\n(PMR_regime(role) − PMR_regime(_nolabel))")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("M8d — paired-delta vs label-free baseline | horizontal subset", y=1.05, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: regime distribution stacked bars
# ---------------------------------------------------------------------------


def fig_regime_distribution(q_all: pd.DataFrame, l_all: pd.DataFrame, out: Path) -> None:
    """Per-(category × label_role) stacked-bar of regime fractions, on horizontal subset."""

    def fractions(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
        sub = df[df["event_template"] == "horizontal"]
        result = {}
        for cat in CATEGORIES:
            for role in (*ROLES, "_nolabel"):
                cell = sub[(sub["shape"] == cat) & (sub["label_role"] == role)]
                if cell.empty:
                    continue
                fracs = {r: (cell["regime"] == r).mean() for r in REGIMES}
                result[(cat, role)] = fracs
        return result

    q_frac = fractions(q_all)
    l_frac = fractions(l_all)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

    for ax, frac, title in [(axes[0], q_frac, "Qwen2.5-VL-7B"), (axes[1], l_frac, "LLaVA-1.5-7B")]:
        cells = [(c, r) for c in CATEGORIES for r in (*ROLES, "_nolabel") if (c, r) in frac]
        x_labels = [f"{c}\n{r}" for c, r in cells]
        x_pos = np.arange(len(cells))
        bottom = np.zeros(len(cells))
        for regime in REGIMES:
            heights = np.array([frac[cell].get(regime, 0.0) for cell in cells])
            ax.bar(x_pos, heights, bottom=bottom, color=REGIME_COLORS[regime], label=regime, width=0.8)
            bottom += heights
        ax.set_xticks(x_pos, x_labels, rotation=70, fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("fraction of responses")
        ax.set_title(title)
        # Vertical separators between categories.
        for i in range(1, len(CATEGORIES)):
            ax.axvline((4 * i) - 0.5, color="black", lw=0.5, alpha=0.4)

    axes[1].legend(title="regime", fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("M8d — regime distribution by (category × label_role) | horizontal subset", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: full-scene samples (12-cell grid)
# ---------------------------------------------------------------------------


def fig_full_scene_samples(stimulus_dir: Path, out: Path) -> None:
    """Grid of (category × abstraction-level × event) representative stimuli."""

    rows = []
    for cat in CATEGORIES:
        for obj in OBJ_LEVELS:
            for ev in ("fall", "horizontal"):
                # Pick the seed-0 ground / both / cue version.
                # Filename pattern: <cat>_<obj>_<bg>_<cue>_<ev>_<seed>.png
                cand = stimulus_dir / "images" / f"{cat}_{obj}_ground_both_{ev}_000.png"
                if cand.exists():
                    rows.append((cat, obj, ev, cand))

    n_rows = len(CATEGORIES)
    n_cols = len(OBJ_LEVELS) * 2  # 4 obj × 2 events
    cell = 180
    grid = Image.new("RGB", (cell * n_cols, cell * n_rows + 28 * n_rows), (255, 255, 255))

    for cat, obj, ev, path in rows:
        r = CATEGORIES.index(cat)
        c = OBJ_LEVELS.index(obj) * 2 + (0 if ev == "fall" else 1)
        img = Image.open(path).resize((cell, cell), Image.LANCZOS)
        grid.paste(img, (c * cell, r * (cell + 28)))

    # Use matplotlib to draw labels on top.
    fig, ax = plt.subplots(figsize=(n_cols * 1.4, n_rows * 1.6))
    ax.imshow(grid)
    ax.set_xticks([(j * cell) + cell / 2 for j in range(n_cols)])
    ax.set_xticklabels(
        [f"{OBJ_LEVELS[i]}\n{ev}" for i in range(len(OBJ_LEVELS)) for ev in ("fall", "horiz")],
        rotation=0,
        fontsize=8,
    )
    ax.set_yticks([(r * (cell + 28)) + cell / 2 for r in range(n_rows)])
    ax.set_yticklabels(list(CATEGORIES), fontsize=10)
    ax.tick_params(length=0)
    ax.set_title("M8d — full-scene samples (bg=ground, cue=both, seed=0)", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, required=True, help="Output dir from m8d_analyze.py")
    p.add_argument(
        "--stimulus-dir",
        type=Path,
        default=None,
        help="Stimulus dir for the full-scene samples figure. If omitted, uses latest inputs/m8d_qwen_*.",
    )
    p.add_argument("--fig-dir", type=Path, default=FIG)
    args = p.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    q_all = load_annotated(args.summary_dir / "m8d_qwen_annotated.parquet")
    l_all = load_annotated(args.summary_dir / "m8d_llava_annotated.parquet")
    q_lbl = q_all[q_all["label"] != "_nolabel"]
    q_nl  = q_all[q_all["label"] == "_nolabel"]
    l_lbl = l_all[l_all["label"] != "_nolabel"]
    l_nl  = l_all[l_all["label"] == "_nolabel"]

    fig_pmr_ramp(q_lbl, l_lbl, args.fig_dir / "m8d_pmr_ramp.png")
    fig_pmr_by_role(q_lbl, l_lbl, args.fig_dir / "m8d_pmr_by_role.png")
    fig_paired_delta(q_lbl, q_nl, l_lbl, l_nl, args.fig_dir / "m8d_paired_delta.png")
    fig_regime_distribution(q_all, l_all, args.fig_dir / "m8d_regime_distribution.png")

    if args.stimulus_dir is None:
        candidates = sorted((PROJECT_ROOT / "inputs").glob("m8d_qwen_*"))
        stim_dir = candidates[-1] if candidates else None
    else:
        stim_dir = args.stimulus_dir
    if stim_dir is not None and stim_dir.exists():
        fig_full_scene_samples(stim_dir, args.fig_dir / "m8d_full_scene_samples.png")

    print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
