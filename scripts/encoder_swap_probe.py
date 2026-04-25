"""§4.5 ext — Idefics2 vision-encoder probe (M6 r3 follow-up).

Loads captured Idefics2 SigLIP-SO400M activations on M8a stim and trains
per-layer linear probes for "physics-vs-abstract" using behavioral PMR
labels from the M8a Qwen labeled run as the y-axis.

Closes the AUC ↔ PMR ↔ H7 chain at the third SigLIP point: M6 r2
established Qwen 0.99 / LLaVA 0.73 / InternVL3 not captured. Idefics2
expected ~0.99 if H-encoder-saturation is encoder-family-driven.

Usage:
    uv run python scripts/encoder_swap_idefics2_probe.py \
        --vision-dir outputs/encoder_swap_idefics2_vision_activations \
        --predictions outputs/encoder_swap_idefics2_<ts>/predictions.parquet \
        --out-dir outputs/encoder_swap_idefics2_probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.probing.vision import (
    load_probing_dataset,
    probe_per_object_level,
    run_layer_sweep,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# M6 r2 baseline — vision-encoder probe AUC at the topmost captured layer.
# (See `docs/insights/m6_r2_*` and `references/roadmap.md` H-encoder-saturation row.)
M6_R2_BASELINE = {
    "qwen":     {"encoder": "SigLIP",        "auc": 0.99, "behavioral_pmr": 0.95},
    "llava":    {"encoder": "CLIP-ViT-L",    "auc": 0.73, "behavioral_pmr": 0.38},
    "internvl3":{"encoder": "InternViT",     "auc": None, "behavioral_pmr": 0.99},
}


def load_full_activations(vision_dir: Path, sample_ids: list[str], layer: int) -> np.ndarray:
    """Load mean-pooled activations directly so the probe sees (n_samples, dim)."""
    from safetensors.torch import load_file
    rows = []
    key = f"vision_hidden_{layer}"
    for sid in sample_ids:
        f = vision_dir / f"{sid}.safetensors"
        data = load_file(str(f))
        t = data[key].to(dtype=__import__("torch").float32).numpy()
        # 2D (n_tokens, dim) or 3D (n_tiles, n_patches, dim) → (dim,)
        if t.ndim == 3:
            t = t.reshape(-1, t.shape[-1])
        rows.append(t.mean(axis=0))
    return np.stack(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vision-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--layers", type=str, default="3,9,18,24")
    p.add_argument("--pmr-source", default="open",
                   help="open / forced_choice / either / majority")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading {args.predictions} ...")
    X_per_layer, y, meta = load_probing_dataset(
        vision_dir=args.vision_dir,
        predictions_path=args.predictions,
        layers=layers,
        pmr_source=args.pmr_source,
    )

    print("\n=== Layer sweep AUC ===")
    sweep = run_layer_sweep(X_per_layer, y)
    print(sweep.round(3).to_string(index=False))
    sweep.to_csv(args.out_dir / "layer_sweep.csv", index=False)

    print("\n=== Per-(layer × object_level) AUC ===")
    by_obj = probe_per_object_level(X_per_layer, y, meta)
    print(by_obj.round(3).to_string(index=False))
    by_obj.to_csv(args.out_dir / "by_object_level.csv", index=False)

    # ------------------------------------------------------------------
    # Per-shape AUC (M8a-specific: 5 shapes; useful to see whether AUC
    # tracks per-shape PMR or is uniformly high across shapes).
    # ------------------------------------------------------------------
    rows = []
    for shape in sorted(meta["shape"].unique() if "shape" in meta.columns else []):
        sub_idx = meta.index[meta["shape"] == shape].to_numpy()
        if len(sub_idx) < 10 or len(set(y[sub_idx])) < 2:
            continue
        for li in layers:
            from physical_mode.probing.vision import train_layer_probe
            r = train_layer_probe(X_per_layer[li][sub_idx], y[sub_idx])
            rows.append({"shape": shape, "layer": li, "auc_mean": r.auc_mean,
                         "auc_std": r.auc_std, "n": len(sub_idx)})
    by_shape = pd.DataFrame(rows)
    if len(by_shape):
        print("\n=== Per-(layer × shape) AUC ===")
        print(by_shape.round(3).to_string(index=False))
        by_shape.to_csv(args.out_dir / "by_shape.csv", index=False)

    # ------------------------------------------------------------------
    # 4-model comparison figure (Qwen / LLaVA / InternVL3 baseline + Idefics2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: Idefics2 layer sweep with M6 r2 baseline lines.
    ax = axes[0]
    ax.plot(sweep["layer"], sweep["auc_mean"], "o-", color="#5fa8d3",
            label="Idefics2 (SigLIP-SO400M)", linewidth=2)
    for model, info in M6_R2_BASELINE.items():
        if info["auc"] is None:
            continue
        ax.axhline(info["auc"], linestyle="--", alpha=0.6,
                   color={"qwen": "#1f77b4", "llava": "#d62728"}[model],
                   label=f"{model} ({info['encoder']}, M6 r2 baseline)")
    ax.set_xlabel("layer")
    ax.set_ylabel("vision-encoder probe AUC (physics vs abstract)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Vision-encoder probe AUC by layer\n(Idefics2 + M6 r2 baselines)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: 4-model AUC vs behavioral PMR scatter (the headline H-encoder-
    # saturation chain plot).
    ax = axes[1]
    # Idefics2 from current run.
    idefics2_auc = float(sweep["auc_mean"].iloc[-1])  # last-layer
    idefics2_pmr = 0.882  # from §4.5 M8a — source of truth in encoder_swap_idefics2.md
    points = [
        ("Qwen",      M6_R2_BASELINE["qwen"]["auc"],     M6_R2_BASELINE["qwen"]["behavioral_pmr"],     "SigLIP",        "#1f77b4"),
        ("LLaVA-1.5", M6_R2_BASELINE["llava"]["auc"],    M6_R2_BASELINE["llava"]["behavioral_pmr"],    "CLIP-ViT-L",    "#d62728"),
        ("Idefics2",  idefics2_auc,                      idefics2_pmr,                                 "SigLIP-SO400M", "#5fa8d3"),
    ]
    for name, auc, pmr, encoder, color in points:
        ax.scatter([auc], [pmr], s=180, color=color, edgecolor="black", zorder=5)
        ax.annotate(f"{name}\n({encoder})", (auc, pmr), xytext=(8, 4),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("vision encoder probe AUC (deepest captured layer)")
    ax.set_ylabel("behavioral mean PMR(_nolabel) on M8a")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("AUC ↔ behavioral PMR — H-encoder-saturation chain\n(Idefics2 = 3rd point)")
    ax.grid(True, alpha=0.3)
    ax.plot([0.5, 1.05], [0.0, 1.05], "k--", alpha=0.3, linewidth=0.7,
            label="y = x reference")
    ax.legend(loc="lower right", fontsize=8)

    fig_path = PROJECT_ROOT / "docs" / "figures" / "encoder_swap_idefics2_probe.png"
    fig.suptitle("§4.5 — Idefics2 vision-encoder probe (M6 r3 closes the chain)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote {args.out_dir / 'layer_sweep.csv'}")
    print(f"Wrote {args.out_dir / 'by_object_level.csv'}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
