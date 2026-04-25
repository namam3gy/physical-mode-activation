"""M8c figures — real photographs vs synthetic baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CATEGORIES = ("ball", "car", "person", "bird", "abstract")
ROLES = ("physical", "abstract", "exotic")
ROLE_COLORS = {
    "physical": "#1f77b4",
    "abstract": "#ff7f0e",
    "exotic":   "#2ca02c",
    "_nolabel": "#7f7f7f",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG = PROJECT_ROOT / "docs" / "figures"


def fig_photo_grid(stim_dir: Path, out: Path) -> None:
    """5 categories × 4 sample photos per category — overview grid."""
    cell = 200
    n_per_cat = 4
    grid = Image.new("RGB", (cell * n_per_cat, cell * len(CATEGORIES) + 32 * len(CATEGORIES)), (255, 255, 255))
    for r, cat in enumerate(CATEGORIES):
        for c in range(n_per_cat):
            img_path = stim_dir / "images" / f"{cat}_photo_{c:03d}.png"
            if not img_path.exists():
                continue
            img = Image.open(img_path).resize((cell, cell), Image.LANCZOS)
            grid.paste(img, (c * cell, r * (cell + 32)))

    fig, ax = plt.subplots(figsize=(n_per_cat * 1.6, len(CATEGORIES) * 1.7))
    ax.imshow(grid)
    ax.set_yticks([(r * (cell + 32)) + cell / 2 for r in range(len(CATEGORIES))])
    ax.set_yticklabels(list(CATEGORIES), fontsize=11)
    ax.set_xticks([(c * cell) + cell / 2 for c in range(n_per_cat)])
    ax.set_xticklabels([f"sample {c}" for c in range(n_per_cat)], fontsize=9)
    ax.tick_params(length=0)
    ax.set_title("M8c — sample photos per category", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_pmr_by_category(summary_dir: Path, out: Path) -> None:
    q_lbl = pd.read_csv(summary_dir / "m8c_qwen_pmr_by_role.csv", index_col=0)
    l_lbl = pd.read_csv(summary_dir / "m8c_llava_pmr_by_role.csv", index_col=0)
    q_nl = pd.read_csv(summary_dir / "m8c_qwen_nolabel_pmr.csv", index_col=0).iloc[:, 0]
    l_nl = pd.read_csv(summary_dir / "m8c_llava_nolabel_pmr.csv", index_col=0).iloc[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    width = 0.2
    x = np.arange(len(CATEGORIES))
    for ax, lbl, nl, title in [(axes[0], q_lbl, q_nl, "Qwen2.5-VL-7B"), (axes[1], l_lbl, l_nl, "LLaVA-1.5-7B")]:
        for j, role in enumerate(ROLES):
            ys = [lbl.loc[c, role] if c in lbl.index else 0 for c in CATEGORIES]
            ax.bar(x + (j - 1.5) * width, ys, width, label=role, color=ROLE_COLORS[role])
        # _nolabel as fourth bar.
        ys_nl = [nl.get(c, 0) for c in CATEGORIES]
        ax.bar(x + 1.5 * width, ys_nl, width, label="_nolabel", color=ROLE_COLORS["_nolabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(CATEGORIES, rotation=20)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("PMR")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    axes[1].legend(title="label_role", fontsize=9, loc="upper right")
    fig.suptitle("M8c — PMR by (category × label_role) on real photos", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_paired_synthetic_vs_photo(summary_dir: Path, out: Path) -> None:
    """Compare PMR(_nolabel) for synthetic-textured vs photo, per (model × category)."""
    syn = pd.read_csv(summary_dir / "m8c_synthetic_baseline.csv")
    q_nl = pd.read_csv(summary_dir / "m8c_qwen_nolabel_pmr.csv", index_col=0).iloc[:, 0]
    l_nl = pd.read_csv(summary_dir / "m8c_llava_nolabel_pmr.csv", index_col=0).iloc[:, 0]

    rows = []
    for cat in ("ball", "car", "person", "bird"):
        for model_name, photo_pmr in [("qwen", q_nl.get(cat, np.nan)), ("llava", l_nl.get(cat, np.nan))]:
            # Find synthetic-textured matching this category.
            if cat == "ball":
                syn_sub = syn[(syn["model"] == model_name) & (syn["source"] == "synthetic-textured-circle")]
            else:
                syn_sub = syn[(syn["model"] == model_name) & (syn["source"] == "synthetic-textured-m8d") & (syn["category"] == cat)]
            syn_pmr = syn_sub["pmr_nolabel"].iloc[0] if not syn_sub.empty else np.nan
            rows.append({
                "category": cat,
                "model": model_name,
                "synthetic_textured": syn_pmr,
                "photo": photo_pmr,
                "delta": photo_pmr - syn_pmr,
            })
    delta_df = pd.DataFrame(rows).round(3)
    delta_df.to_csv(summary_dir / "m8c_synthetic_vs_photo.csv", index=False)
    print(delta_df.to_string(index=False))

    cats = ["ball", "car", "person", "bird"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    width = 0.18
    x = np.arange(len(cats))
    for j, (model_name, color) in enumerate([("qwen", "#d62728"), ("llava", "#1f77b4")]):
        sub = delta_df[delta_df["model"] == model_name].set_index("category").reindex(cats)
        ax.bar(x + (j * 2 - 1) * width, sub["synthetic_textured"].values, width,
               label=f"{model_name} synthetic-textured", color=color, alpha=0.5, edgecolor="black")
        ax.bar(x + (j * 2) * width, sub["photo"].values, width,
               label=f"{model_name} photo", color=color, alpha=1.0, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("PMR(_nolabel)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("M8c — paired PMR(_nolabel): synthetic-textured vs photo", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary-dir", type=Path, required=True)
    p.add_argument("--stimulus-dir", type=Path, default=None)
    p.add_argument("--fig-dir", type=Path, default=FIG)
    args = p.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.stimulus_dir is None:
        cands = sorted((PROJECT_ROOT / "inputs").glob("m8c_photos_*"))
        stim_dir = cands[-1] if cands else None
    else:
        stim_dir = args.stimulus_dir
    if stim_dir is not None and stim_dir.exists():
        fig_photo_grid(stim_dir, args.fig_dir / "m8c_photo_grid.png")
    fig_pmr_by_category(args.summary_dir, args.fig_dir / "m8c_pmr_by_category.png")
    fig_paired_synthetic_vs_photo(args.summary_dir, args.fig_dir / "m8c_paired_synthetic_vs_photo.png")

    print(f"Wrote figures to {args.fig_dir}")


if __name__ == "__main__":
    main()
