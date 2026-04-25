"""M9 — generalization audit (paper Table 1).

Consolidates M8a + M8d + M8c findings across the 3 encoders (Qwen2.5-VL-7B,
LLaVA-1.5-7B, Idefics2-8b) into a single paper-ready table + figure.

What this audit answers (one shot):
  1. Does PMR(_nolabel) ceiling hold across stim sources?
     → mean PMR(_nolabel) per (model × stim) bar chart.
  2. Does H7 measurability tracks ceiling-saturation?
     → H7 PASS rate per (model × stim) bar chart.
  3. Per-category breakdown for any audit reader who wants the grain.
     → unified per-(model × stim × category) row CSV.

Reuses `encoder_swap_analyze.collect_for_stim` so stim-source loading stays
DRY — this script is a pure consumer/aggregator on top of those CSVs.

Usage:
    uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from encoder_swap_analyze import (  # type: ignore
    M8A_SHAPES, M8D_SHAPES, M8C_SHAPES, ENCODER_TABLE, PREFIXES, _latest, _load_with_role,
    collect_for_stim,
)


BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_table1(all_pmr: pd.DataFrame, all_h7: pd.DataFrame) -> pd.DataFrame:
    """One row per (model × stim × shape): pmr_nolabel + role-PMRs + h7_delta + h7_pass."""
    if len(all_pmr) == 0 or len(all_h7) == 0:
        return pd.DataFrame()
    merge_keys = ["stim", "model", "encoder", "lm", "shape"]
    table = all_pmr.merge(
        all_h7[merge_keys + ["physical_pmr", "abstract_pmr", "exotic_pmr", "h7_phys_minus_abs"]],
        on=merge_keys, how="outer",
    )
    table["h7_pass_strict"] = table["h7_phys_minus_abs"].fillna(-9.0).ge(0.05).astype(int)
    table = table[merge_keys + [
        "n", "pmr_nolabel", "physical_pmr", "abstract_pmr", "exotic_pmr",
        "h7_phys_minus_abs", "h7_pass_strict",
    ]]
    return table.sort_values(merge_keys).reset_index(drop=True)


def _h7_bootstrap_ci(predictions: pd.DataFrame, shapes: tuple[str, ...]) -> tuple[float, float, float]:
    """Per-prediction bootstrap of mean-across-shapes (mean phys PMR - mean abs PMR).

    Resampling unit is *predictions within (shape, role)* — accounts for the
    sampling noise inside each cell. Across-shape variability enters because
    each bootstrap iter recomputes the per-shape delta from resampled rows.
    Returns (mean_h7, ci_low, ci_high) at 95%.
    """
    if len(predictions) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    cells: list[tuple[str, str, np.ndarray]] = []
    for shp in shapes:
        for role in ("physical", "abstract"):
            sub = predictions[(predictions["shape"] == shp) & (predictions["label_role"] == role)]
            if len(sub) > 0:
                cells.append((shp, role, sub["pmr"].to_numpy()))
    if not cells:
        return float("nan"), float("nan"), float("nan")

    deltas = np.empty(BOOTSTRAP_N, dtype=float)
    for b in range(BOOTSTRAP_N):
        per_shape_phys: dict[str, float] = {}
        per_shape_abs: dict[str, float] = {}
        for shp, role, arr in cells:
            sample = rng.choice(arr, size=len(arr), replace=True)
            mean_val = float(sample.mean())
            if role == "physical":
                per_shape_phys[shp] = mean_val
            else:
                per_shape_abs[shp] = mean_val
        per_shape_deltas = [
            per_shape_phys[s] - per_shape_abs[s]
            for s in shapes
            if s in per_shape_phys and s in per_shape_abs
        ]
        deltas[b] = float(np.mean(per_shape_deltas)) if per_shape_deltas else float("nan")

    mean_h7 = float(np.nanmean(deltas))
    ci_low = float(np.nanpercentile(deltas, 2.5))
    ci_high = float(np.nanpercentile(deltas, 97.5))
    return mean_h7, ci_low, ci_high


def _pmr_nolabel_bootstrap_ci(predictions: pd.DataFrame, shapes: tuple[str, ...]) -> tuple[float, float, float]:
    """Bootstrap CI on mean-across-shapes PMR(_nolabel).

    Resampling unit: predictions within each shape (label is uniformly _nolabel,
    so role does not enter). Returns (mean_pmr, ci_low, ci_high) at 95%.
    """
    if len(predictions) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)

    arrs = []
    for shp in shapes:
        sub = predictions[predictions["shape"] == shp]
        if len(sub) > 0:
            arrs.append(sub["pmr"].to_numpy())
    if not arrs:
        return float("nan"), float("nan"), float("nan")

    means = np.empty(BOOTSTRAP_N, dtype=float)
    for b in range(BOOTSTRAP_N):
        per_shape_means = [float(rng.choice(a, size=len(a), replace=True).mean()) for a in arrs]
        means[b] = float(np.mean(per_shape_means))

    return float(np.nanmean(means)), float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def build_summary(table: pd.DataFrame, *, with_bootstrap: bool = True) -> pd.DataFrame:
    """Per (model × stim) summary: mean PMR(_nolabel), mean H7, H7 PASS rate.

    When `with_bootstrap=True`, also computes 95% bootstrap CI on mean H7
    delta (resampling unit: predictions within each shape × role cell).
    """
    if len(table) == 0:
        return pd.DataFrame()

    stim_to_shapes = {"m8a": M8A_SHAPES, "m8d": M8D_SHAPES, "m8c": M8C_SHAPES}
    rows = []
    for (stim, model), grp in table.groupby(["stim", "model"], sort=False):
        row: dict = {
            "stim": stim,
            "model": model,
            "encoder": grp["encoder"].iloc[0],
            "lm": grp["lm"].iloc[0],
            "n_shapes": int(len(grp)),
            "mean_pmr_nolabel": float(grp["pmr_nolabel"].mean()),
            "mean_h7_delta": float(grp["h7_phys_minus_abs"].mean()),
            "h7_pass_rate": float(grp["h7_pass_strict"].mean()),
        }
        if with_bootstrap:
            base = PREFIXES.get(stim, {}).get(model)
            h7_ci_low = h7_ci_high = float("nan")
            pmr_ci_low = pmr_ci_high = float("nan")
            if base:
                lbl_path = _latest(base)
                if lbl_path is not None:
                    df = _load_with_role(lbl_path)
                    _, h7_ci_low, h7_ci_high = _h7_bootstrap_ci(df, stim_to_shapes[stim])
                nl_path = _latest(f"{base}_label_free")
                if nl_path is not None:
                    from physical_mode.metrics.pmr import score_rows as _score_rows
                    nl_df = _score_rows(pd.read_json(nl_path, lines=True))
                    _, pmr_ci_low, pmr_ci_high = _pmr_nolabel_bootstrap_ci(nl_df, stim_to_shapes[stim])
            row["pmr_ci_low"] = pmr_ci_low
            row["pmr_ci_high"] = pmr_ci_high
            row["h7_ci_low"] = h7_ci_low
            row["h7_ci_high"] = h7_ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def fig_summary(summary: pd.DataFrame, out: Path) -> None:
    """Two-panel: mean PMR(_nolabel) and H7 PASS rate per (model × stim)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if len(summary) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return

    stim_order = ["m8a", "m8d", "m8c"]
    model_order = [m for m, _, _ in ENCODER_TABLE]
    color_by_encoder = {
        "SigLIP": "#1f77b4",
        "CLIP-ViT-L": "#d62728",
        "SigLIP-SO400M": "#5fa8d3",
        "InternViT": "#2ca02c",
    }
    width = 0.8 / max(len(model_order), 1)
    x = np.arange(len(stim_order))
    offset0 = -(len(model_order) - 1) / 2 * width

    for ax, col, title, ylim, err_cols in [
        (axes[0], "mean_pmr_nolabel", "mean PMR(_nolabel) (95% bootstrap CI)", (0, 1.1),
         ("pmr_ci_low", "pmr_ci_high")),
        (axes[1], "mean_h7_delta", "mean H7 delta (95% bootstrap CI)", (-0.4, 0.7),
         ("h7_ci_low", "h7_ci_high")),
    ]:
        for k, model in enumerate(model_order):
            vals: list[float] = []
            encs: list[str] = []
            errs_low: list[float] = []
            errs_high: list[float] = []
            for stim in stim_order:
                row = summary[(summary["stim"] == stim) & (summary["model"] == model)]
                vals.append(float(row[col].iloc[0]) if len(row) else float("nan"))
                encs.append(row["encoder"].iloc[0] if len(row) else "")
                if err_cols and len(row):
                    errs_low.append(float(row[err_cols[0]].iloc[0]))
                    errs_high.append(float(row[err_cols[1]].iloc[0]))
                else:
                    errs_low.append(float("nan"))
                    errs_high.append(float("nan"))
            color = color_by_encoder.get(encs[0], "gray") if encs[0] else "gray"
            bar_x = x + offset0 + k * width
            bars = ax.bar(bar_x, vals, width,
                          label=f"{model} ({encs[0]})" if encs[0] else model,
                          color=color, edgecolor="black")
            if err_cols:
                yerr_lo = [v - lo for v, lo in zip(vals, errs_low)]
                yerr_hi = [hi - v for v, hi in zip(vals, errs_high)]
                ax.errorbar(bar_x, vals, yerr=[yerr_lo, yerr_hi],
                            fmt="none", ecolor="black", capsize=3, linewidth=0.8)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.2f}",
                            ha="center", fontsize=7)

        ax.set_xticks(x, ["M8a (synth shapes)", "M8d (synth categories)", "M8c (real photos)"])
        ax.set_ylim(*ylim)
        if err_cols:
            ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("M9 — generalization audit (3 models × 3 stim sources)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_table1_heatmap(table: pd.DataFrame, out: Path) -> None:
    """Per-(model × stim × shape) PMR(_nolabel) and H7 heatmaps stacked vertically."""
    if len(table) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    stim_specs = [
        ("m8a", M8A_SHAPES, "M8a — 5 synth geometric shapes"),
        ("m8d", M8D_SHAPES, "M8d — 3 synth categories"),
        ("m8c", M8C_SHAPES, "M8c — 5 real-photo categories"),
    ]
    model_order = [m for m, _, _ in ENCODER_TABLE]
    model_labels = []
    for m, enc, lm in ENCODER_TABLE:
        model_labels.append(f"{m}\n({enc.split('-')[0]}+{lm.split('-')[0]})")

    for c, (stim, shapes, title) in enumerate(stim_specs):
        sub = table[table["stim"] == stim]
        if len(sub) == 0:
            for r in range(2):
                axes[r, c].text(0.5, 0.5, f"{stim} — no data", ha="center", va="center",
                                transform=axes[r, c].transAxes)
                axes[r, c].axis("off")
            continue

        p_pivot = sub.pivot_table(index="model", columns="shape", values="pmr_nolabel").reindex(
            index=model_order, columns=list(shapes))
        h_pivot = sub.pivot_table(index="model", columns="shape", values="h7_phys_minus_abs").reindex(
            index=model_order, columns=list(shapes))

        for r, (mat, cmap, vmin, vmax, fmt, sub_title) in enumerate([
            (p_pivot, "Blues", 0.0, 1.0, ".2f", "PMR(_nolabel)"),
            (h_pivot, "RdBu_r", -0.7, 0.7, "+.2f", "H7 (physical − abstract)"),
        ]):
            ax = axes[r, c]
            im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=20)
            ax.set_yticks(range(mat.shape[0]), model_labels, fontsize=9)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    if not np.isnan(v):
                        if cmap == "Blues":
                            color = "white" if v > 0.55 else "black"
                        else:
                            color = "white" if abs(v) > 0.35 else "black"
                        ax.text(j, i, format(v, fmt), ha="center", va="center",
                                color=color, fontsize=9)
            if r == 0:
                ax.set_title(f"{title}\n{sub_title}", fontsize=10)
            else:
                ax.set_title(sub_title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle("M9 — Table 1 (3 models × 3 stim × per-shape PMR + H7)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parts_pmr = []
    parts_h7 = []
    for stim, shapes in (("m8a", M8A_SHAPES), ("m8d", M8D_SHAPES), ("m8c", M8C_SHAPES)):
        pmr, h7 = collect_for_stim(stim, shapes)
        parts_pmr.append(pmr)
        parts_h7.append(h7)

    all_pmr = pd.concat(parts_pmr, ignore_index=True)
    all_h7 = pd.concat(parts_h7, ignore_index=True)
    table = build_table1(all_pmr, all_h7)
    summary = build_summary(table)

    print("\n=== M9 Table 1: per-(model × stim × shape) ===")
    print(table.round(3).to_string(index=False))
    print("\n=== M9 summary: per-(model × stim) ===")
    print(summary.round(3).to_string(index=False))

    table.to_csv(args.out_dir / "m9_table1.csv", index=False)
    summary.to_csv(args.out_dir / "m9_summary.csv", index=False)

    fig_summary(summary, PROJECT_ROOT / "docs" / "figures" / "m9_summary.png")
    fig_table1_heatmap(table, PROJECT_ROOT / "docs" / "figures" / "m9_table1_heatmap.png")
    print(f"\nWrote {args.out_dir / 'm9_table1.csv'}")
    print(f"Wrote {args.out_dir / 'm9_summary.csv'}")
    print(f"Wrote {PROJECT_ROOT / 'docs' / 'figures' / 'm9_summary.png'}")
    print(f"Wrote {PROJECT_ROOT / 'docs' / 'figures' / 'm9_table1_heatmap.png'}")


if __name__ == "__main__":
    main()
