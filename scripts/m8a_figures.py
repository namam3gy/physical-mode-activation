"""M8a figures — per-shape PMR ramp, label-role bars, paired-delta heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"
FIG = PROJECT_ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def role_of(shape: str, label: str) -> str:
    p, a, e = LABELS_BY_SHAPE[shape]
    return {"physical": p, "abstract": a, "exotic": e}.get(label, label)


def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = score_rows(df)
    if "label" in df.columns:
        df["label_role"] = [
            "physical" if l == LABELS_BY_SHAPE[s][0] else
            "abstract" if l == LABELS_BY_SHAPE[s][1] else
            "exotic"   if l == LABELS_BY_SHAPE[s][2] else l
            for s, l in zip(df["shape"], df["label"])
        ]
    return df


def main():
    Q_LBL = OUT / "m8a_qwen_20260425-092423_bf03832e" / "predictions.jsonl"
    Q_NL  = OUT / "m8a_qwen_label_free_20260425-094239_26c66949" / "predictions.jsonl"
    L_LBL = OUT / "m8a_llava_20260425-095133_a2b5f318" / "predictions.jsonl"
    L_NL  = OUT / "m8a_llava_label_free_20260425-100253_99a20dd8" / "predictions.jsonl"

    q_lbl = load_scored(Q_LBL)
    q_nl  = load_scored(Q_NL)
    l_lbl = load_scored(L_LBL)
    l_nl  = load_scored(L_NL)

    shapes = ["circle", "square", "triangle", "hexagon", "polygon"]
    obj_levels = ["line", "filled", "shaded", "textured"]
    roles = ["physical", "abstract", "exotic"]

    # ---------- Figure 1: PMR ramp by shape × object_level (Qwen vs LLaVA) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, df, title in [(axes[0], q_lbl, "Qwen2.5-VL-7B"), (axes[1], l_lbl, "LLaVA-1.5-7B")]:
        for shp in shapes:
            sub = df[df["shape"] == shp]
            ramp = [sub[sub["object_level"] == lv]["pmr"].mean() for lv in obj_levels]
            ax.plot(obj_levels, ramp, marker="o", label=shp, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel("PMR")
        ax.set_ylim(0, 1)
        ax.set_xlabel("object_level")
        ax.grid(True, alpha=0.3)
    axes[1].legend(title="shape", loc="lower right", fontsize=9)
    fig.suptitle("M8a — PMR ramp by (shape × object_level)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "m8a_pmr_ramp.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: PMR by (shape, label_role) — Qwen vs LLaVA ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    width = 0.25
    x = np.arange(len(shapes))
    colors = {"physical": "#1f77b4", "abstract": "#ff7f0e", "exotic": "#2ca02c"}
    for ax, df, title in [(axes[0], q_lbl, "Qwen2.5-VL-7B"), (axes[1], l_lbl, "LLaVA-1.5-7B")]:
        for j, role in enumerate(roles):
            ys = [df[(df["shape"] == s) & (df["label_role"] == role)]["pmr"].mean() for s in shapes]
            ax.bar(x + (j - 1) * width, ys, width, label=role, color=colors[role])
        ax.set_xticks(x)
        ax.set_xticklabels(shapes, rotation=30)
        ax.set_ylim(0, 1)
        ax.set_ylabel("PMR")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    axes[1].legend(title="label_role", fontsize=9)
    fig.suptitle("M8a — PMR by (shape × label_role)", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "m8a_pmr_by_role.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 3: paired-delta heatmaps ------------------------------
    def paired_delta(lbl, nl):
        out = np.zeros((len(shapes), len(roles)))
        for i, shp in enumerate(shapes):
            s_nl = nl[nl["shape"] == shp].groupby("sample_id")["pmr"].mean().rename("_nolabel")
            for j, r in enumerate(roles):
                s_l = lbl[(lbl["shape"] == shp) & (lbl["label_role"] == r)].groupby("sample_id")["pmr"].mean().rename(r)
                joined = pd.concat([s_nl, s_l], axis=1).dropna()
                out[i, j] = (joined[r] - joined["_nolabel"]).mean()
        return out

    q_pd = paired_delta(q_lbl, q_nl)
    l_pd = paired_delta(l_lbl, l_nl)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mat, title in [(axes[0], q_pd, "Qwen2.5-VL-7B"), (axes[1], l_pd, "LLaVA-1.5-7B")]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
        ax.set_xticks(range(len(roles)), roles)
        ax.set_yticks(range(len(shapes)), shapes)
        for i in range(len(shapes)):
            for j in range(len(roles)):
                v = mat[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.35 else "black", fontsize=9)
        ax.set_title(f"paired Δ — {title}\n(PMR(role) − PMR(_nolabel))")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG / "m8a_paired_delta.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote 3 figures to {FIG}")


if __name__ == "__main__":
    main()
