"""Aggregate cross-model SAE intervention results.

Reads per-model results.csv and emits:
- A unified summary table (top_k drop curve + random control rate per model).
- A figure with per-model curves on a shared axis.
- A markdown table fragment for the insight doc.

Usage:
    uv run python scripts/m5b_sae_intervention_cross_model_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from sae_intervention import _is_physics_letter  # noqa: E402

MODELS = [
    # Use OPEN-mode Qwen for cross-model uniformity (FC original preserved separately).
    ("Qwen2.5-VL-7B", "outputs/sae_intervention/qwen_vis31_open_full/results.csv"),
    ("LLaVA-1.5-7B", "outputs/sae_intervention/llava15_vis22_4096_full/results.csv"),
    ("LLaVA-Next-7B", "outputs/sae_intervention/llava_next_vis22_4096_full/results.csv"),
    ("Idefics2-8B", "outputs/sae_intervention/idefics2_vis26_4608_full/results.csv"),
    ("InternVL3-8B", "outputs/sae_intervention/internvl3_vis23_4096_full/results.csv"),
]


def rescore(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "intervention_pmr" in df.columns and "baseline_pmr" in df.columns:
        # OPEN mode — already PMR scored
        df["int_rate_col"] = df["intervention_pmr"]
        df["bl_rate_col"] = df["baseline_pmr"]
    else:
        # FC mode (Qwen original): use first-letter scorer (handle "Answer:" prefix)
        df["int_rate_col"] = df["intervention_text"].astype(str).map(_is_physics_letter)
        df["bl_rate_col"] = df["baseline_text"].astype(str).map(_is_physics_letter)
        df["int_rate_col"] = (df["int_rate_col"] == 1).astype(int)
        df["bl_rate_col"] = (df["bl_rate_col"] == 1).astype(int)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        n_valid = len(g)
        rate = float(g["int_rate_col"].mean())
        bl_rate = float(g["bl_rate_col"].mean())
        rows.append({"condition": cond, "n": n_valid, "bl_rate": bl_rate, "int_rate": rate})
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def parse_topk(condition: str) -> int | None:
    if condition.startswith("top_k="):
        return int(condition.split("=")[1])
    return None


def main() -> None:
    all_summaries = {}
    for name, path in MODELS:
        p = PROJECT_ROOT / path
        if not p.exists():
            print(f"[SKIP] {name}: {p} not found")
            continue
        df = rescore(pd.read_csv(p))
        s = summarize(df)
        all_summaries[name] = s
        print(f"\n=== {name} ===")
        print(s.to_string(index=False))

    # Markdown table
    print("\n\n=== Markdown table ===\n")
    print("| Model | k=1 | k=5 | k=10 | k=20 | k=40 | k=80 | k=160 | random |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for name, s in all_summaries.items():
        s_top = s[s["condition"].str.startswith("top_k=")].copy()
        s_top["k"] = s_top["condition"].apply(parse_topk)
        s_top = s_top.sort_values("k").set_index("k")
        s_rand = s[s["condition"].str.startswith("random")]
        rand_rate = s_rand["int_rate"].mean() if len(s_rand) else float("nan")

        cells = []
        for k in [1, 5, 10, 20, 40, 80, 160]:
            if k in s_top.index:
                cells.append(f"{s_top.loc[k, 'int_rate']:.2f}")
            else:
                cells.append("—")
        cells.append(f"{rand_rate:.2f}" if rand_rate == rand_rate else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")

    # Figure: top-k drop curve per model (random as horizontal line)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Qwen2.5-VL-7B": "C0", "LLaVA-1.5-7B": "C3", "LLaVA-Next-7B": "C2",
              "Idefics2-8B": "C1", "InternVL3-8B": "C4"}
    for name, s in all_summaries.items():
        s_top = s[s["condition"].str.startswith("top_k=")].copy()
        s_top["k"] = s_top["condition"].apply(parse_topk)
        s_top = s_top.sort_values("k")
        ax.plot(s_top["k"], s_top["int_rate"], "o-", label=name,
                color=colors.get(name, None), linewidth=2, markersize=7)
        s_rand = s[s["condition"].str.startswith("random")]
        if len(s_rand):
            ax.axhline(s_rand["int_rate"].mean(), linestyle="--",
                       color=colors.get(name, None), alpha=0.4)
    ax.set_xlabel("# of top SAE features ablated (zeroed)")
    ax.set_ylabel("PMR rate (intervention)")
    ax.set_title("M5b SAE intervention — cross-model")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 40, 80, 160])
    ax.set_xticklabels(["1", "5", "10", "20", "40", "80", "160"])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_fig = PROJECT_ROOT / "docs/figures/m5b_sae_intervention_cross_model.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=140)
    print(f"\nWrote figure: {out_fig}")


if __name__ == "__main__":
    main()
