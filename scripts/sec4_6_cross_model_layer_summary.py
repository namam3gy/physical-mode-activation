"""§4.6 cross-model — aggregate Qwen + LLaVA-1.5 + LLaVA-Next layer sweeps.

Reads per-model `results_aggregated.csv` files and produces:
  - outputs/sec4_6_cross_model_layer_summary/cross_model_layer_table.csv
  - docs/figures/sec4_6_cross_model_layer_sweep.png (3-panel matrix +
    annotated heatmap)

The table shape: (model × layer) → (n_v_unit_flipped/n, n_random_flipped/n).

Usage:
    uv run python scripts/sec4_6_cross_model_layer_summary.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "sec4_6_cross_model_layer_summary"
FIG_OUT = PROJECT_ROOT / "docs" / "figures" / "sec4_6_cross_model_layer_sweep.png"


def _load_unified(run_dir: Path) -> pd.DataFrame:
    """Load unified-layer-sweep results from results_aggregated.csv. Schema:
    config_name = 'L<X>_v_unit_eps<E>' or 'L<X>_random_eps<E>'.
    """
    df = pd.read_csv(run_dir / "results_aggregated.csv")
    rows = []
    for _, r in df.iterrows():
        m = re.match(r"L(\d+)_(v_unit|random)_eps([0-9.]+)", r["config_name"])
        if not m:
            continue
        layer = int(m.group(1))
        kind = m.group(2)
        eps = float(m.group(3))
        rows.append({
            "layer": layer, "kind": kind, "eps": eps,
            "n": r["n"], "synth_pmr_mean": r["synth_pmr_mean"],
            "n_flipped": r["n_flipped"],
        })
    return pd.DataFrame(rows)


def _load_llava15_separate(layer_dirs: dict[int, Path]) -> pd.DataFrame:
    """LLaVA-1.5 was run per-layer in separate dirs with bounded_eps0.1 + control_v_random_*."""
    rows = []
    for layer, run_dir in layer_dirs.items():
        agg = run_dir / "results_aggregated.csv"
        if not agg.exists():
            continue
        df = pd.read_csv(agg)
        # eps=0.1 v_unit: config_name == "bounded_eps0.1"
        v_row = df[df["config_name"] == "bounded_eps0.1"]
        if not v_row.empty:
            rows.append({
                "layer": layer, "kind": "v_unit", "eps": 0.1,
                "n": int(v_row.iloc[0]["n"]),
                "synth_pmr_mean": float(v_row.iloc[0]["synth_pmr_mean"]),
                "n_flipped": int(v_row.iloc[0]["n_flipped"]),
            })
        # Average across 3 random seeds.
        rand_rows = df[df["config_name"].str.startswith("control_v_random_")]
        if not rand_rows.empty:
            n_total = int(rand_rows["n"].sum())
            n_flipped = int(rand_rows["n_flipped"].sum())
            rows.append({
                "layer": layer, "kind": "random", "eps": 0.1,
                "n": n_total, "synth_pmr_mean": n_flipped / n_total if n_total else 0.0,
                "n_flipped": n_flipped,
            })
    return pd.DataFrame(rows)


def _wilson_ci(n_phys: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rate = n_phys / n
    ci = binomtest(int(n_phys), n).proportion_ci(0.95, "wilson")
    return rate, ci.low, ci.high


def _latest_unified(prefix: str) -> Path | None:
    """Find the most recent <prefix>_<ts> dir with results_aggregated.csv."""
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{prefix}*"))
    cands = [c for c in cands if (c / "results_aggregated.csv").exists()]
    return cands[-1] if cands else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources: dict[str, pd.DataFrame] = {}

    # Qwen — prefer the latest dated `_n10_` sweep (which has all 5 layers in
    # one run); fall back to the original n=3 unified + the morning L10 and
    # smoke L25 extras only if no n=10 sweep is present.
    qwen_n10 = _latest_unified("sec4_6_qwen_layer_sweep_n10_")
    qwen_unified = PROJECT_ROOT / "outputs/sec4_6_qwen_layer_sweep_unified"
    qwen_use_extras = False
    if qwen_n10 is not None:
        sources["Qwen2.5-VL-7B"] = _load_unified(qwen_n10)
    elif (qwen_unified / "results_aggregated.csv").exists():
        sources["Qwen2.5-VL-7B"] = _load_unified(qwen_unified)
        qwen_use_extras = True

    # Add Qwen L25 (smoke run from earlier — uses bounded_eps0.1/0.2 + control_v_random_0).
    qwen_l25 = PROJECT_ROOT / "outputs/sec4_6_qwen_layer_sweep_L25_smoke"
    if qwen_use_extras and (qwen_l25 / "results_aggregated.csv").exists():
        df_l25 = pd.read_csv(qwen_l25 / "results_aggregated.csv")
        v_row = df_l25[df_l25["config_name"] == "bounded_eps0.1"]
        rand_row = df_l25[df_l25["config_name"] == "control_v_random_0"]
        extra_rows = []
        if not v_row.empty:
            extra_rows.append({
                "layer": 25, "kind": "v_unit", "eps": 0.1,
                "n": int(v_row.iloc[0]["n"]), "synth_pmr_mean": float(v_row.iloc[0]["synth_pmr_mean"]),
                "n_flipped": int(v_row.iloc[0]["n_flipped"]),
            })
        if not rand_row.empty:
            extra_rows.append({
                "layer": 25, "kind": "random", "eps": 0.1,
                "n": int(rand_row.iloc[0]["n"]), "synth_pmr_mean": float(rand_row.iloc[0]["synth_pmr_mean"]),
                "n_flipped": int(rand_row.iloc[0]["n_flipped"]),
            })
        sources["Qwen2.5-VL-7B"] = pd.concat(
            [sources.get("Qwen2.5-VL-7B", pd.DataFrame()), pd.DataFrame(extra_rows)],
            ignore_index=True,
        )

    # Add Qwen L10 from morning standard run (sec4_6_counterfactual_20260426-050343 has bounded_eps0.1).
    qwen_l10 = PROJECT_ROOT / "outputs/sec4_6_counterfactual_20260426-050343"
    if qwen_use_extras and (qwen_l10 / "results_aggregated.csv").exists():
        df = pd.read_csv(qwen_l10 / "results_aggregated.csv")
        v_row = df[df["config_name"] == "bounded_eps0.1"]
        rand_rows = df[df["config_name"].str.startswith("control_v_random_")]
        extra_rows = []
        if not v_row.empty:
            extra_rows.append({
                "layer": 10, "kind": "v_unit", "eps": 0.1,
                "n": int(v_row.iloc[0]["n"]), "synth_pmr_mean": float(v_row.iloc[0]["synth_pmr_mean"]),
                "n_flipped": int(v_row.iloc[0]["n_flipped"]),
            })
        if not rand_rows.empty:
            n_total = int(rand_rows["n"].sum())
            n_flipped = int(rand_rows["n_flipped"].sum())
            extra_rows.append({
                "layer": 10, "kind": "random", "eps": 0.1,
                "n": n_total, "synth_pmr_mean": n_flipped / n_total if n_total else 0.0,
                "n_flipped": n_flipped,
            })
        sources["Qwen2.5-VL-7B"] = pd.concat(
            [sources.get("Qwen2.5-VL-7B", pd.DataFrame()), pd.DataFrame(extra_rows)],
            ignore_index=True,
        )

    # LLaVA-1.5 — prefer the latest n=10 unified sweep; fall back to per-layer dirs.
    llava15_n10 = _latest_unified("sec4_6_llava15_layer_sweep_n10_")
    if llava15_n10 is not None:
        sources["LLaVA-1.5-7B"] = _load_unified(llava15_n10)
    else:
        llava15_dirs = {
            5: PROJECT_ROOT / "outputs/sec4_6_counterfactual_llava_L5_20260426-162623",
            10: PROJECT_ROOT / "outputs/sec4_6_counterfactual_llava_20260426-114111",
            15: PROJECT_ROOT / "outputs/sec4_6_counterfactual_llava_L15_20260426-162623",
            20: PROJECT_ROOT / "outputs/sec4_6_counterfactual_llava_L20_20260426-162623",
            25: PROJECT_ROOT / "outputs/sec4_6_counterfactual_llava_L25_20260426-162623",
        }
        sources["LLaVA-1.5-7B"] = _load_llava15_separate(llava15_dirs)

    # LLaVA-Next — prefer the latest n=10 unified sweep.
    llava_next_n10 = _latest_unified("sec4_6_llava_next_layer_sweep_n10_")
    llava_next_unified = PROJECT_ROOT / "outputs/sec4_6_llava_next_layer_sweep_unified"
    if llava_next_n10 is not None:
        sources["LLaVA-Next-Mistral-7B"] = _load_unified(llava_next_n10)
    elif (llava_next_unified / "results_aggregated.csv").exists():
        sources["LLaVA-Next-Mistral-7B"] = _load_unified(llava_next_unified)

    # Idefics2 — n=10 unified sweep at L5/10/15/20/25 + deeper-layer
    # extension at L26/28/30/31 (sec4_6_idefics2_layer_sweep_unified_<ts>).
    idefics2_n10 = _latest_unified("sec4_6_idefics2_layer_sweep_n10_")
    idefics2_l26_31 = _latest_unified("sec4_6_idefics2_layer_sweep_unified_")
    idefics2_dfs: list[pd.DataFrame] = []
    if idefics2_n10 is not None:
        idefics2_dfs.append(_load_unified(idefics2_n10))
    if idefics2_l26_31 is not None:
        idefics2_dfs.append(_load_unified(idefics2_l26_31))
    if idefics2_dfs:
        sources["Idefics2-8B"] = pd.concat(idefics2_dfs, ignore_index=True)

    # InternVL3 — n=10 unified sweep (note: baseline_pmr=1.0 saturates protocol).
    internvl3_n10 = _latest_unified("sec4_6_internvl3_layer_sweep_n10_")
    if internvl3_n10 is not None:
        sources["InternVL3-8B"] = _load_unified(internvl3_n10)

    print("=== Cross-model layer sweep summary ===")
    all_rows = []
    for model, df in sources.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            rate, lo, hi = _wilson_ci(int(r["n_flipped"]), int(r["n"]))
            all_rows.append({
                "model": model, "layer": int(r["layer"]), "kind": r["kind"],
                "eps": r["eps"], "n": int(r["n"]), "n_flipped": int(r["n_flipped"]),
                "phys_rate": rate, "wilson_lo": lo, "wilson_hi": hi,
            })
    out = pd.DataFrame(all_rows).sort_values(["model", "layer", "kind"])
    print(out.round(3).to_string(index=False))
    out.to_csv(OUT_DIR / "cross_model_layer_table.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'cross_model_layer_table.csv'}")

    # Wide pivot: model × layer for v_unit only.
    v_only = out[out["kind"] == "v_unit"].copy()
    pivot = v_only.pivot(index="model", columns="layer", values="phys_rate")
    print("\n=== v_unit phys_rate (model × layer) ===")
    print(pivot.round(3).to_string())

    # Figure: 1 panel per model showing v_unit vs random rates by layer with Wilson CIs.
    # 5 models → 2-row grid (3 + 2) to keep panels readable.
    models = list(sources.keys())
    n = len(models)
    if n <= 3:
        n_rows, n_cols = 1, n
    else:
        n_cols = 3
        n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), sharey=True)
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, model in zip(axes_flat, models):
        sub = out[out["model"] == model]
        for kind, color, marker in [("v_unit", "#1f77b4", "o"), ("random", "#888888", "s")]:
            kdf = sub[sub["kind"] == kind].sort_values("layer")
            if kdf.empty:
                continue
            ax.errorbar(
                kdf["layer"], kdf["phys_rate"],
                yerr=[kdf["phys_rate"] - kdf["wilson_lo"], kdf["wilson_hi"] - kdf["phys_rate"]],
                marker=marker, color=color, capsize=3,
                label=kind.replace("v_unit", "v_unit_<L>").replace("random", "random_dir"),
            )
        ax.axhline(1.0, color="#cccccc", linewidth=0.5, linestyle="--")
        ax.axhline(0.0, color="#cccccc", linewidth=0.5, linestyle="--")
        layer_max = int(sub["layer"].max()) if not sub.empty else 25
        ax.set_xlim(2, max(28, layer_max + 3))
        ax.set_ylim(-0.05, 1.05)
        if layer_max > 25:
            ax.set_xticks([5, 10, 15, 20, 25, 28, 31])
        else:
            ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlabel("LM layer (gradient ascent target)")
        ax.set_title(model, fontsize=11)
        if ax is axes_flat[0]:
            ax.set_ylabel("PMR flip rate (95 % Wilson CI)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
    # Hide unused axes.
    for ax in axes_flat[len(models):]:
        ax.set_visible(False)
    fig.suptitle("§4.6 cross-model — pixel-encodability per LM layer (n=10 stim, eps=0.1)\n"
                 "Qwen broad shortcuts; LLaVA-1.5/-Next concentrated late; Idefics2 falsifies "
                 "(9-layer L5-L31 evidence); InternVL3 protocol-saturated (baseline=1)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=160, bbox_inches="tight")
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
