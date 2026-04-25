"""Generate session-summary figures (2026-04-25).

Produces three figures:
1. 5-model cross-stim PMR bar chart (M8a × M8d × M8c) with bootstrap CIs
2. §4.2 image-vs-label trade-off (M8d phys-minus-abs vs M8c phys-minus-abs)
3. §4.10 cross-model attention-on-visual-tokens by layer

Usage:
    uv run python scripts/make_session_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

ENCODER_COLOR = {
    "qwen": "#1f77b4",
    "llava": "#d62728",
    "llava_next": "#ff7f0e",
    "idefics2": "#5fa8d3",
    "internvl3": "#2ca02c",
}

MODEL_DISPLAY = {
    "qwen": "Qwen2.5-VL\n(SigLIP)",
    "llava": "LLaVA-1.5\n(CLIP)",
    "llava_next": "LLaVA-Next\n(CLIP+AnyRes)",
    "idefics2": "Idefics2\n(SigLIP-SO400M)",
    "internvl3": "InternVL3\n(InternViT)",
}

MODEL_ORDER = ["qwen", "llava", "llava_next", "idefics2", "internvl3"]


def fig1_cross_stim_pmr() -> None:
    """5-model × 3-stim PMR bar chart with bootstrap CIs."""
    summary = pd.read_csv(PROJECT_ROOT / "outputs/m9_audit/m9_summary.csv")

    stim_order = ["m8a", "m8d", "m8c"]
    stim_label = {"m8a": "M8a (synth shapes)",
                  "m8d": "M8d (synth categories)",
                  "m8c": "M8c (real photos)"}

    fig, ax = plt.subplots(figsize=(11, 5))
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models
    x = np.arange(len(stim_order))
    offset0 = -(n_models - 1) / 2 * width

    for k, model in enumerate(MODEL_ORDER):
        means, los, his = [], [], []
        for stim in stim_order:
            row = summary[(summary["model"] == model) & (summary["stim"] == stim)]
            if len(row) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            means.append(float(row["mean_pmr_nolabel"].iloc[0]))
            los.append(float(row["pmr_ci_low"].iloc[0]))
            his.append(float(row["pmr_ci_high"].iloc[0]))

        bar_x = x + offset0 + k * width
        bars = ax.bar(bar_x, means, width, label=MODEL_DISPLAY[model].replace("\n", " "),
                      color=ENCODER_COLOR[model], edgecolor="black", alpha=0.9)
        yerr_lo = [m - l if not np.isnan(l) else 0 for m, l in zip(means, los)]
        yerr_hi = [h - m if not np.isnan(h) else 0 for m, h in zip(means, his)]
        ax.errorbar(bar_x, means, yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="black", capsize=2.5, linewidth=0.8)
        for bx, m in zip(bar_x, means):
            if not np.isnan(m):
                ax.text(bx, m + 0.025, f"{m:.2f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([stim_label[s] for s in stim_order])
    ax.set_ylabel("mean PMR(_nolabel) (95% bootstrap CI)")
    ax.set_ylim(0, 1.05)
    ax.set_title("5-model × 3-stim PMR ladder (M8a M8d synthetic + M8c photos)\n"
                 "Encoder family separates on synthetic; photos compress.")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    out = FIG_DIR / "session_5model_cross_stim_pmr.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig2_image_vs_label() -> None:
    """§4.2 image vs label dominance: M8d phys-minus-abs vs M8c phys-minus-abs."""
    summary = pd.read_csv(PROJECT_ROOT / "outputs/m9_audit/m9_summary.csv")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    width = 0.35
    x = np.arange(len(MODEL_ORDER))

    m8d_vals, m8c_vals = [], []
    m8d_lo, m8d_hi, m8c_lo, m8c_hi = [], [], [], []
    for m in MODEL_ORDER:
        m8d = summary[(summary["stim"] == "m8d") & (summary["model"] == m)]
        m8c = summary[(summary["stim"] == "m8c") & (summary["model"] == m)]
        m8d_vals.append(float(m8d["mean_h7_delta"].iloc[0]) if len(m8d) else np.nan)
        m8c_vals.append(float(m8c["mean_h7_delta"].iloc[0]) if len(m8c) else np.nan)
        m8d_lo.append(float(m8d["h7_ci_low"].iloc[0]) if len(m8d) else np.nan)
        m8d_hi.append(float(m8d["h7_ci_high"].iloc[0]) if len(m8d) else np.nan)
        m8c_lo.append(float(m8c["h7_ci_low"].iloc[0]) if len(m8c) else np.nan)
        m8c_hi.append(float(m8c["h7_ci_high"].iloc[0]) if len(m8c) else np.nan)

    bars1 = ax.bar(x - width/2, m8d_vals, width, label="M8d (synthetic categories)",
                   color="#666666", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width/2, m8c_vals, width, label="M8c (real photos)",
                   color="#cccccc", edgecolor="black", alpha=0.85)

    yerr1_lo = [v - l if not np.isnan(l) else 0 for v, l in zip(m8d_vals, m8d_lo)]
    yerr1_hi = [h - v if not np.isnan(h) else 0 for v, h in zip(m8d_vals, m8d_hi)]
    yerr2_lo = [v - l if not np.isnan(l) else 0 for v, l in zip(m8c_vals, m8c_lo)]
    yerr2_hi = [h - v if not np.isnan(h) else 0 for v, h in zip(m8c_vals, m8c_hi)]
    ax.errorbar(x - width/2, m8d_vals, yerr=[yerr1_lo, yerr1_hi],
                fmt="none", ecolor="black", capsize=3, linewidth=0.8)
    ax.errorbar(x + width/2, m8c_vals, yerr=[yerr2_lo, yerr2_hi],
                fmt="none", ecolor="black", capsize=3, linewidth=0.8)

    for xi, v in zip(x - width/2, m8d_vals):
        if not np.isnan(v):
            ax.text(xi, v + (0.02 if v >= 0 else -0.04), f"{v:+.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + width/2, m8c_vals):
        if not np.isnan(v):
            ax.text(xi, v + (0.02 if v >= 0 else -0.04), f"{v:+.2f}", ha="center", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m].replace("\n", " ") for m in MODEL_ORDER], rotation=12)
    ax.set_ylabel("H7 (PMR_physical_label − PMR_abstract_label)")
    ax.set_ylim(-0.15, 0.45)
    ax.set_title("§4.2 — Label-driven H7 effect halves on real photos\n"
                 "Synthetic stim: label can shift PMR by up to +0.31. Photos: ≤ +0.15.")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out = FIG_DIR / "session_image_vs_label_h7.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig3_attention_cross_model() -> None:
    """§4.10 cross-model: fraction of last-token attention on visual tokens by layer."""
    from safetensors.torch import load_file

    LAYERS = (5, 15, 20, 25)
    runs = {
        "qwen":       sorted(PROJECT_ROOT.glob("outputs/attention_viz_qwen_*"))[-1],
        "llava":      sorted(PROJECT_ROOT.glob("outputs/attention_viz_llava_2*"))[-1],
        "llava_next": sorted(PROJECT_ROOT.glob("outputs/attention_viz_llava_next_*"))[-1],
        "idefics2":   sorted(PROJECT_ROOT.glob("outputs/attention_viz_idefics2_*"))[-1],
        "internvl3":  sorted(PROJECT_ROOT.glob("outputs/attention_viz_internvl3_*"))[-1],
    }

    # Common sample id across all 5 runs.
    common = None
    for rd in runs.values():
        ids = {p.stem for p in (rd / "activations").glob("*.safetensors")}
        common = ids if common is None else common & ids
    sample_id = sorted(common)[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model, rd in runs.items():
        d = load_file(str(rd / "activations" / f"{sample_id}.safetensors"))
        mask = d["visual_token_mask"].numpy().astype(bool)
        vis_pos = np.where(mask)[0]
        seq_len = mask.shape[0]
        n_vis = int(mask.sum())
        baseline = n_vis / seq_len
        fracs = []
        for li in LAYERS:
            attn = d[f"lm_attn_{li}"].float().numpy()       # (heads, q, k)
            a = attn[:, -1, :].mean(axis=0)                 # (k,)
            fracs.append(float(a[vis_pos].sum()))
        ax.plot(LAYERS, fracs, "o-", color=ENCODER_COLOR[model], linewidth=2.2,
                markersize=9, label=f"{MODEL_DISPLAY[model].replace(chr(10), ' ')} (n_vis/seq={baseline:.0%})")
        ax.axhline(baseline, color=ENCODER_COLOR[model], linestyle="--", alpha=0.25)

    ax.set_xticks(LAYERS)
    ax.set_xlabel("LM layer")
    ax.set_ylabel("fraction of last-token attention on visual tokens")
    ax.set_ylim(0, 1.0)
    ax.set_title("§4.10 — Cross-model attention to visual tokens (same M8a stim)\n"
                 f"sample: {sample_id}\n"
                 "Dashed = uniform-attention baseline (n_visual / seq_len). All 5 models attend ≪ baseline.")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "session_attention_cross_model.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    fig1_cross_stim_pmr()
    fig2_image_vs_label()
    fig3_attention_cross_model()
