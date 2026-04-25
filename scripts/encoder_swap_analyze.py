"""§4.5 — Cross-encoder swap analysis.

Compares Idefics2 (SigLIP + Mistral-7B) PMR(_nolabel) and paired-delta
patterns to:
  - Qwen2.5-VL-7B  (SigLIP + Qwen2-7B)
  - LLaVA-1.5-7B   (CLIP + Vicuna-7B)

Hypothesis: encoder type drives behavioral PMR(_nolabel) saturation.
  - SigLIP-based models (Qwen + Idefics2) should show ceiling-like
    PMR(_nolabel) on synthetic-textured stim.
  - CLIP-based model (LLaVA) should show low PMR(_nolabel).

If Idefics2 patterns with Qwen → encoder hypothesis confirmed.
If Idefics2 patterns with LLaVA → LM family or some other factor matters.

Usage:
    uv run python scripts/encoder_swap_analyze.py \\
        --idefics2-labeled outputs/encoder_swap_idefics2_<ts>/predictions.jsonl \\
        --idefics2-nolabel outputs/encoder_swap_idefics2_label_free_<ts>/predictions.jsonl \\
        --out-dir outputs/encoder_swap_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import score_rows


SHAPES = ("circle", "square", "triangle", "hexagon", "polygon")
ROLES = ("physical", "abstract", "exotic")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_with_role(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = score_rows(df)
    def role(s, l):
        if l == "_nolabel":
            return "_nolabel"
        triplet = LABELS_BY_SHAPE.get(s)
        if triplet is None:
            return l
        p, a, e = triplet
        return "physical" if l == p else "abstract" if l == a else "exotic" if l == e else l
    df["label_role"] = [role(s, l) for s, l in zip(df["shape"], df["label"])]
    return df


def _latest(prefix: str) -> Path | None:
    out_root = PROJECT_ROOT / "outputs"
    if "label_free" in prefix:
        cands = sorted(out_root.glob(f"{prefix}_*/predictions.jsonl"))
    else:
        cands = sorted(out_root.glob(f"{prefix}_2*/predictions.jsonl"))
    return cands[-1] if cands else None


def collect_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect PMR(_nolabel) per (model, shape) on M8a stim across 3 encoders."""
    rows_pmr = []
    rows_h7 = []

    for model_name, lbl_glob, nl_glob, encoder, lm in [
        ("qwen", "m8a_qwen", "m8a_qwen_label_free", "SigLIP", "Qwen2-7B"),
        ("llava", "m8a_llava", "m8a_llava_label_free", "CLIP-ViT-L", "Vicuna-7B"),
        ("idefics2", "encoder_swap_idefics2", "encoder_swap_idefics2_label_free", "SigLIP-SO400M", "Mistral-7B"),
    ]:
        nl_path = _latest(nl_glob)
        lbl_path = _latest(lbl_glob)
        if nl_path is None or lbl_path is None:
            print(f"WARN: {model_name} missing data, skipping")
            continue

        nl = pd.read_json(nl_path, lines=True)
        nl = score_rows(nl)
        for shp in SHAPES:
            sub = nl[nl["shape"] == shp]
            rows_pmr.append({
                "model": model_name, "encoder": encoder, "lm": lm,
                "shape": shp, "pmr_nolabel": float(sub["pmr"].mean()), "n": int(len(sub)),
            })

        lbl = _load_with_role(lbl_path)
        for shp in SHAPES:
            sub = lbl[lbl["shape"] == shp]
            phys = sub[sub["label_role"] == "physical"]["pmr"].mean()
            absr = sub[sub["label_role"] == "abstract"]["pmr"].mean()
            exo  = sub[sub["label_role"] == "exotic"]["pmr"].mean()
            rows_h7.append({
                "model": model_name, "encoder": encoder, "lm": lm,
                "shape": shp,
                "physical_pmr": float(phys), "abstract_pmr": float(absr), "exotic_pmr": float(exo),
                "h7_phys_minus_abs": float(phys - absr),
            })

    return pd.DataFrame(rows_pmr), pd.DataFrame(rows_h7)


def fig_encoder_swap(pmr: pd.DataFrame, h7: pd.DataFrame, out: Path) -> None:
    """3-panel figure: PMR(_nolabel) heatmap, H7 heatmap, summary bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: PMR(_nolabel) heatmap (model × shape)
    p_pivot = pmr.pivot(index="model", columns="shape", values="pmr_nolabel").reindex(
        index=["qwen", "llava", "idefics2"], columns=list(SHAPES))
    ax = axes[0]
    im = ax.imshow(p_pivot.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SHAPES)), SHAPES, rotation=20)
    ax.set_yticks(range(3), ["Qwen\n(SigLIP+Qwen)", "LLaVA\n(CLIP+Vicuna)", "Idefics2\n(SigLIP+Mistral)"])
    for i in range(p_pivot.shape[0]):
        for j in range(p_pivot.shape[1]):
            v = p_pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "black", fontsize=10)
    ax.set_title("PMR(_nolabel) by (model × shape)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Panel 2: H7 paired-difference heatmap (model × shape)
    h_pivot = h7.pivot(index="model", columns="shape", values="h7_phys_minus_abs").reindex(
        index=["qwen", "llava", "idefics2"], columns=list(SHAPES))
    ax = axes[1]
    im = ax.imshow(h_pivot.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(SHAPES)), SHAPES, rotation=20)
    ax.set_yticks(range(3), ["Qwen", "LLaVA", "Idefics2"])
    for i in range(h_pivot.shape[0]):
        for j in range(h_pivot.shape[1]):
            v = h_pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.35 else "black", fontsize=10)
    ax.set_title("H7 (physical − abstract) by (model × shape)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Panel 3: Summary — mean PMR(_nolabel) per model with encoder annotation.
    ax = axes[2]
    summary = pmr.groupby(["model", "encoder"])["pmr_nolabel"].mean().reset_index()
    summary = summary.set_index("model").reindex(["qwen", "llava", "idefics2"]).reset_index()
    colors = ["#1f77b4" if "SigLIP" in e else "#d62728" for e in summary["encoder"]]
    bars = ax.bar(summary["model"], summary["pmr_nolabel"], color=colors, edgecolor="black")
    for bar, enc in zip(bars, summary["encoder"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                enc, ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean PMR(_nolabel) across 5 shapes")
    ax.set_title("encoder-swap summary")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("§4.5 — Cross-encoder swap (Idefics2 SigLIP+Mistral as third point)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pmr, h7 = collect_baselines()
    print("\n=== PMR(_nolabel) per (model × shape) ===")
    print(pmr.round(3).to_string(index=False))
    print("\n=== H7 paired-difference (physical − abstract) per (model × shape) ===")
    print(h7.round(3).to_string(index=False))
    print("\n=== Mean PMR(_nolabel) by encoder family ===")
    print(pmr.groupby(["model", "encoder", "lm"])["pmr_nolabel"].mean().round(3).to_string())
    print("\n=== Mean H7 by encoder family ===")
    print(h7.groupby(["model", "encoder", "lm"])["h7_phys_minus_abs"].mean().round(3).to_string())

    pmr.to_csv(args.out_dir / "encoder_swap_pmr_nolabel.csv", index=False)
    h7.to_csv(args.out_dir / "encoder_swap_h7.csv", index=False)

    fig_path = PROJECT_ROOT / "docs" / "figures" / "encoder_swap_heatmap.png"
    fig_encoder_swap(pmr, h7, fig_path)
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
