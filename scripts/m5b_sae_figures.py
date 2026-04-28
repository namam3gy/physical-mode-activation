"""Figures for M5b SAE intervention (Cohen's d revision, 2026-04-27 evening).

Two panels:
  (a) Dense k-sweep with 95 % Wilson CIs for both delta-rank (morning run)
      and Cohen's-d-rank (evening run), plus 3 mass-matched random controls.
  (b) Per-cluster phys rate matrix on the Cohen's-d run (5 clusters × 9
      top-k + 3 random conditions).

Usage:
    uv run python scripts/m5b_sae_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "docs" / "figures"


def _wilson(n_phys: int, n: int) -> tuple[float, float, float]:
    rate = n_phys / n if n else float("nan")
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = binomtest(int(n_phys), n).proportion_ci(confidence_level=0.95, method="wilson")
    return rate, ci.low, ci.high


def _per_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        n = len(g)
        n_phys = int((g["intervention_phys"] == 1).sum())
        rate, lo, hi = _wilson(n_phys, n)
        rows.append((cond, n, n_phys, rate, lo, hi))
    return pd.DataFrame(rows, columns=["condition", "n", "n_phys", "rate", "wilson_lo", "wilson_hi"])


def _cluster(sid: str) -> str:
    if sid.startswith("line_blank"):
        return "line_blank"
    if sid.startswith("line_ground"):
        return "line_ground"
    if sid.startswith("filled_blank"):
        return "filled_blank"
    if sid.startswith("filled_ground"):
        return "filled_ground"
    if sid.startswith("filled_scene"):
        return "filled_scene"
    return sid.rsplit("_", 1)[0]


def _topk_int(s: str) -> int | None:
    m = re.match(r"top_k=(\d+)", s)
    return int(m.group(1)) if m else None


def main() -> None:
    morning_csv = PROJECT_ROOT / "outputs/sae_intervention/qwen_vis31_5120/results.csv"
    evening_csv = PROJECT_ROOT / "outputs/sae_intervention/qwen_vis31_5120_cohens_d_v2/results.csv"
    morning = pd.read_csv(morning_csv)
    evening = pd.read_csv(evening_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ---- Panel (a): dense k-sweep with Wilson CIs ----
    ax = axes[0]
    morn_agg = _per_condition(morning)
    even_agg = _per_condition(evening)

    morn_top = morn_agg[morn_agg["condition"].str.startswith("top_k=")].copy()
    morn_top["k"] = morn_top["condition"].map(_topk_int)
    morn_top = morn_top.sort_values("k")

    even_top = even_agg[even_agg["condition"].str.startswith("top_k=")].copy()
    even_top["k"] = even_top["condition"].map(_topk_int)
    even_top = even_top.sort_values("k")

    ax.errorbar(
        morn_top["k"], morn_top["rate"],
        yerr=[morn_top["rate"] - morn_top["wilson_lo"], morn_top["wilson_hi"] - morn_top["rate"]],
        marker="s", color="#888888", linewidth=1.5, capsize=3,
        label="delta-rank (morning, k ∈ {1, 5, 10, 20})",
    )
    ax.errorbar(
        even_top["k"], even_top["rate"],
        yerr=[even_top["rate"] - even_top["wilson_lo"], even_top["wilson_hi"] - even_top["rate"]],
        marker="o", color="#1f77b4", linewidth=2.0, capsize=3,
        label="Cohen's-d-rank (evening, dense)",
    )

    # Random controls — plot as horizontal scatter at k=top_k_max with jitter.
    rand = even_agg[even_agg["condition"].str.startswith("random_")].copy()
    if not rand.empty:
        # Place random points at k=35 (just to the right of k=30).
        x_pts = np.linspace(33, 37, len(rand))
        ax.errorbar(
            x_pts, rand["rate"],
            yerr=[rand["rate"] - rand["wilson_lo"], rand["wilson_hi"] - rand["rate"]],
            marker="^", color="#2ca02c", linewidth=0, capsize=3,
            label="3 mass-matched random (k=30, 72-102 % mass)",
        )

    ax.axhline(1.0, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.axhline(0.5, color="#eeeeee", linewidth=0.5)
    ax.axhline(0.0, color="#cccccc", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, 39)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("k (number of top SAE features ablated)")
    ax.set_ylabel("Phys rate (n=20 stim, 95 % Wilson CI)")
    ax.set_title(
        "(a) k-sweep: top-N ablation breaks physics-mode\nrandom k=30 controls retain physics-mode at all 3 mass-matched sets",
        fontsize=10,
    )
    ax.set_xticks([1, 5, 10, 15, 20, 30, 35])
    ax.set_xticklabels(["1", "5", "10", "15", "20", "30", "rand×3"])
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ---- Panel (b): per-cluster matrix ----
    ax = axes[1]
    df_e = evening.copy()
    df_e["cluster"] = df_e["sample_id"].apply(_cluster)

    cond_order = [
        "top_k=1", "top_k=2", "top_k=3", "top_k=5", "top_k=7", "top_k=10",
        "top_k=15", "top_k=20", "top_k=30", "random_0", "random_1", "random_2",
    ]
    cluster_order = ["line_blank", "filled_blank", "line_ground", "filled_ground", "filled_scene"]
    cluster_n = df_e.groupby("cluster")["sample_id"].nunique().to_dict()

    pivot = df_e.pivot_table(
        index="cluster", columns="condition", values="intervention_phys", aggfunc="mean"
    )
    pivot = pivot.reindex(index=cluster_order, columns=cond_order)

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([c.replace("top_k=", "k=").replace("random_", "rand_") for c in cond_order],
                       rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cluster_order)))
    ax.set_yticklabels([f"{c} (n={cluster_n.get(c, 0)})" for c in cluster_order], fontsize=9)
    ax.set_title("(b) Per-cluster phys rate (Cohen's-d run)\n"
                 "line_blank breaks at k=5-10 + recovers at k=15;\n"
                 "others hold to k=15-20", fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            txt = f"{v:.2f}" if pd.notna(v) else ""
            color = "white" if v < 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Phys rate", fontsize=9)

    fig.suptitle(
        "M5b SAE intervention on Qwen vision encoder (last layer, pre-projection):\n"
        "Cohen's-d ranking + dense k-sweep + 3 mass-matched random controls + per-cluster pivot",
        fontsize=11, y=1.005,
    )
    fig.tight_layout()

    out_path = OUT_DIR / "m5b_sae_intervention_revised.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
