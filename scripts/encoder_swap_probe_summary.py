"""§4.5 ext — 4-model encoder probe summary (M6 r3 + r4 closing the chain).

Reads layer-sweep AUC CSVs for 2 captured models (Idefics2 + InternVL3)
plus the M6 r2 baselines for Qwen + LLaVA, and produces a single
paper-ready figure showing the encoder family → encoder probe AUC →
behavioral PMR(_nolabel) chain across all 4 model points.

Usage:
    uv run python scripts/encoder_swap_probe_summary.py \
        --idefics2 outputs/encoder_swap_idefics2_probe \
        --internvl3 outputs/encoder_swap_internvl3_probe \
        --out-dir outputs/encoder_swap_probe_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# (display name, encoder family, LM, color)
MODEL_TABLE = [
    ("Qwen2.5-VL",   "SigLIP",        "Qwen2-7B",      "#1f77b4"),
    ("LLaVA-1.5",    "CLIP-ViT-L",    "Vicuna-7B",     "#d62728"),
    ("Idefics2",     "SigLIP-SO400M", "Mistral-7B",    "#5fa8d3"),
    ("InternVL3",    "InternViT",     "InternLM2-7B",  "#2ca02c"),
]


# M6 r2 baselines — Qwen + LLaVA AUC at deepest captured layer (M2 stim,
# multiple layers all ~0.99 / 0.73 — see docs/insights/m6_r2_cross_model.md).
# M8a behavioral PMR(_nolabel) from §4.5 / M9 — same stim across all 4 models.
M8A_PMR = {
    "Qwen2.5-VL": 0.838,
    "LLaVA-1.5":  0.175,
    "Idefics2":   0.882,
    # InternVL3 filled in from the label-free run at runtime.
}

M6_R2_AUC = {
    "Qwen2.5-VL": 0.99,  # M3 / M6 r2b — SigLIP, layer-7 of 32, AUC ≥ 0.99 across layers
    "LLaVA-1.5":  0.73,  # M6 r2b — CLIP-ViT-L, layer 7 of 24, ~0.72-0.73 across layers
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--idefics2", type=Path, required=True,
                   help="outputs/encoder_swap_idefics2_probe — must contain layer_sweep.csv")
    p.add_argument("--internvl3", type=Path, required=True,
                   help="outputs/encoder_swap_internvl3_probe — must contain layer_sweep.csv")
    p.add_argument("--internvl3-pmr", type=float, required=False, default=None,
                   help="Override behavioral PMR(_nolabel) for InternVL3 if not auto-inferred.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    idefics2_sweep = pd.read_csv(args.idefics2 / "layer_sweep.csv")
    internvl3_sweep = pd.read_csv(args.internvl3 / "layer_sweep.csv")

    # Take the deepest-layer AUC for each model (paper convention).
    idefics2_auc = float(idefics2_sweep["auc_mean"].iloc[-1])
    internvl3_auc = float(internvl3_sweep["auc_mean"].iloc[-1])

    pmr = dict(M8A_PMR)
    if "InternVL3" not in pmr:
        if args.internvl3_pmr is not None:
            pmr["InternVL3"] = args.internvl3_pmr
        else:
            # Derive from the label-free run dir alongside the probe dir.
            cand = sorted(PROJECT_ROOT.glob("outputs/encoder_swap_internvl3_m8a_label_free_*/predictions.jsonl"))
            if cand:
                from physical_mode.metrics.pmr import score_rows
                df = score_rows(pd.read_json(cand[-1], lines=True))
                pmr["InternVL3"] = float(df["pmr"].mean())
            else:
                pmr["InternVL3"] = float("nan")

    auc = dict(M6_R2_AUC)
    auc["Idefics2"] = idefics2_auc
    auc["InternVL3"] = internvl3_auc

    # Build a flat table.
    rows = []
    for name, encoder, lm, color in MODEL_TABLE:
        rows.append({
            "model": name, "encoder": encoder, "lm": lm,
            "encoder_auc": auc.get(name),
            "behavioral_pmr_m8a": pmr.get(name),
            "color": color,
        })
    table = pd.DataFrame(rows)
    print("\n=== 4-model encoder-saturation chain (M8a PMR + encoder AUC) ===")
    print(table.drop(columns=["color"]).round(3).to_string(index=False))
    table.drop(columns=["color"]).to_csv(args.out_dir / "encoder_chain_table.csv", index=False)

    # ------------------------------------------------------------------
    # Figure: 2 panels.
    #   (a) layer sweep for the 2 captured models + horizontal lines for
    #       M6 r2 baselines.
    #   (b) AUC vs M8a PMR scatter (4 points).
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    # Panel a
    ax = axes[0]
    ax.plot(idefics2_sweep["layer"], idefics2_sweep["auc_mean"], "o-",
            color="#5fa8d3", label="Idefics2 (SigLIP-SO400M)", linewidth=2)
    ax.plot(internvl3_sweep["layer"], internvl3_sweep["auc_mean"], "s-",
            color="#2ca02c", label="InternVL3 (InternViT)", linewidth=2)
    ax.axhline(M6_R2_AUC["Qwen2.5-VL"], linestyle="--", color="#1f77b4",
               alpha=0.7, label="Qwen2.5-VL (SigLIP, M6 r2 baseline)")
    ax.axhline(M6_R2_AUC["LLaVA-1.5"], linestyle="--", color="#d62728",
               alpha=0.7, label="LLaVA-1.5 (CLIP, M6 r2 baseline)")
    ax.set_xlabel("layer")
    ax.set_ylabel("vision-encoder probe AUC (physics vs abstract)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Vision-encoder probe AUC by layer\n(2 captured + 2 M6 r2 baselines)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel b
    ax = axes[1]
    for _, row in table.iterrows():
        if pd.isna(row["encoder_auc"]) or pd.isna(row["behavioral_pmr_m8a"]):
            continue
        ax.scatter([row["encoder_auc"]], [row["behavioral_pmr_m8a"]], s=200,
                   color=row["color"], edgecolor="black", zorder=5)
        ax.annotate(f"{row['model']}\n({row['encoder']})",
                    (row["encoder_auc"], row["behavioral_pmr_m8a"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("vision-encoder probe AUC (deepest captured layer)")
    ax.set_ylabel("behavioral mean PMR(_nolabel) on M8a")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.plot([0.5, 1.05], [0.0, 1.05], "k--", alpha=0.3, linewidth=0.7,
            label="y = x reference")
    ax.set_title("AUC ↔ behavioral PMR — H-encoder-saturation chain\n(4 model points: 3 non-CLIP + 1 CLIP)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig_path = PROJECT_ROOT / "docs" / "figures" / "encoder_chain_4model.png"
    fig.suptitle("§4.5 ext — Encoder-saturation chain across 4 VLMs (M6 r3 + r4 close the loop)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out_dir / 'encoder_chain_table.csv'}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
