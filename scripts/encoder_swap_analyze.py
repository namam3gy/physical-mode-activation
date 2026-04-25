"""§4.5 — Cross-encoder swap analysis (extended cross-stim version).

Compares Idefics2 (SigLIP-SO400M + Mistral-7B) PMR(_nolabel) and
H7 paired-delta to:
  - Qwen2.5-VL-7B  (SigLIP + Qwen2-7B)
  - LLaVA-1.5-7B   (CLIP-ViT-L/14 + Vicuna-7B)

across THREE stim sources:
  - M8a: 5 synthetic geometric shapes (circle/square/triangle/hexagon/polygon)
  - M8d: 3 synthetic categories (car/person/bird) × 4 abstraction levels
  - M8c: 5 photo categories (ball/car/person/bird/abstract) — real photographs

If Idefics2 (SigLIP) tracks Qwen across all three stim sources, encoder
family is causally confirmed cross-stim — paper claim moves from "5-shape
encoder-swap" to "cross-shape × cross-category × cross-source encoder-swap".

Usage:
    uv run python scripts/encoder_swap_analyze.py \\
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


M8A_SHAPES = ("circle", "square", "triangle", "hexagon", "polygon")
M8D_SHAPES = ("car", "person", "bird")
M8C_SHAPES = ("ball", "car", "person", "bird", "abstract")
ROLES = ("physical", "abstract", "exotic")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# (model_name, encoder, lm) — order matters: rows of figure
ENCODER_TABLE: tuple[tuple[str, str, str], ...] = (
    ("qwen", "SigLIP", "Qwen2-7B"),
    ("llava", "CLIP-ViT-L", "Vicuna-7B"),
    ("idefics2", "SigLIP-SO400M", "Mistral-7B"),
)


# Run-dir prefixes per (stim, model). label_free arms append `_label_free` after the model.
PREFIXES: dict[str, dict[str, str]] = {
    "m8a": {
        "qwen": "m8a_qwen",
        "llava": "m8a_llava",
        "idefics2": "encoder_swap_idefics2",
    },
    "m8d": {
        "qwen": "m8d_qwen",
        "llava": "m8d_llava",
        "idefics2": "encoder_swap_idefics2_m8d",
    },
    "m8c": {
        "qwen": "m8c_qwen",
        "llava": "m8c_llava",
        "idefics2": "encoder_swap_idefics2_m8c",
    },
}


def _load_with_role(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = score_rows(df)

    def role(s: str, l: str) -> str:
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
    """Return latest predictions.jsonl matching `<prefix>_<timestamp>`.

    Uses a digit-leading suffix glob so `m8a_qwen_*` does not match
    `m8a_qwen_label_free_*` (label_free starts with `l`, not a digit).
    Skips empty / aborted predictions.jsonl files.
    """
    out_root = PROJECT_ROOT / "outputs"
    if "label_free" in prefix:
        cands = sorted(out_root.glob(f"{prefix}_*/predictions.jsonl"))
    else:
        cands = sorted(out_root.glob(f"{prefix}_2*/predictions.jsonl"))
    cands = [p for p in cands if p.stat().st_size > 0]
    return cands[-1] if cands else None


def collect_for_stim(stim: str, shapes: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect (PMR_nolabel, H7) DataFrames for a given stim source across the 3 encoders."""
    rows_pmr: list[dict] = []
    rows_h7: list[dict] = []

    prefix_map = PREFIXES[stim]
    for model_name, encoder, lm in ENCODER_TABLE:
        base = prefix_map.get(model_name)
        if base is None:
            continue
        nl_path = _latest(f"{base}_label_free")
        lbl_path = _latest(base)
        if nl_path is None or lbl_path is None:
            print(f"WARN [{stim}] {model_name} missing data ("
                  f"label_free={nl_path is not None}, labeled={lbl_path is not None}), skipping")
            continue

        nl = score_rows(pd.read_json(nl_path, lines=True))
        for shp in shapes:
            sub = nl[nl["shape"] == shp]
            if len(sub) == 0:
                continue
            rows_pmr.append({
                "stim": stim, "model": model_name, "encoder": encoder, "lm": lm,
                "shape": shp, "pmr_nolabel": float(sub["pmr"].mean()), "n": int(len(sub)),
            })

        lbl = _load_with_role(lbl_path)
        for shp in shapes:
            sub = lbl[lbl["shape"] == shp]
            if len(sub) == 0:
                continue
            phys = sub[sub["label_role"] == "physical"]["pmr"].mean()
            absr = sub[sub["label_role"] == "abstract"]["pmr"].mean()
            exo = sub[sub["label_role"] == "exotic"]["pmr"].mean()
            rows_h7.append({
                "stim": stim, "model": model_name, "encoder": encoder, "lm": lm,
                "shape": shp,
                "physical_pmr": float(phys),
                "abstract_pmr": float(absr),
                "exotic_pmr": float(exo),
                "h7_phys_minus_abs": float(phys - absr),
            })

    return pd.DataFrame(rows_pmr), pd.DataFrame(rows_h7)


def _heatmap_panel(ax, pivot: pd.DataFrame, *, cmap: str, vmin: float, vmax: float,
                   fmt: str, title: str, model_labels: list[str]) -> None:
    im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=20)
    ax.set_yticks(range(pivot.shape[0]), model_labels)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                color = "white" if (cmap == "Blues" and v > 0.55) or (cmap == "RdBu_r" and abs(v) > 0.35) else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center", color=color, fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)


def fig_encoder_swap_m8a(pmr: pd.DataFrame, h7: pd.DataFrame, out: Path) -> None:
    """Backward-compatible 3-panel figure (M8a only)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    p_pivot = pmr.pivot(index="model", columns="shape", values="pmr_nolabel").reindex(
        index=["qwen", "llava", "idefics2"], columns=list(M8A_SHAPES))
    h_pivot = h7.pivot(index="model", columns="shape", values="h7_phys_minus_abs").reindex(
        index=["qwen", "llava", "idefics2"], columns=list(M8A_SHAPES))
    model_labels = ["Qwen\n(SigLIP+Qwen)", "LLaVA\n(CLIP+Vicuna)", "Idefics2\n(SigLIP+Mistral)"]
    _heatmap_panel(axes[0], p_pivot, cmap="Blues", vmin=0, vmax=1, fmt=".2f",
                   title="PMR(_nolabel) by (model × shape)", model_labels=model_labels)
    _heatmap_panel(axes[1], h_pivot, cmap="RdBu_r", vmin=-0.7, vmax=0.7, fmt="+.2f",
                   title="H7 (physical − abstract) by (model × shape)", model_labels=["Qwen", "LLaVA", "Idefics2"])

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
    ax.set_title("encoder-swap summary (M8a)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("§4.5 — Cross-encoder swap (Idefics2 SigLIP+Mistral, M8a 5 shapes)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_encoder_swap_extended(all_pmr: pd.DataFrame, all_h7: pd.DataFrame, out: Path) -> None:
    """Cross-stim figure: 3 stim sources × 3 models × shapes.

    Layout: 3 rows (m8a / m8d / m8c) × 2 columns (PMR_nolabel heatmap, H7 heatmap)
    + a final summary row showing mean PMR(_nolabel) per (stim, model).
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    stim_specs = [
        ("m8a", M8A_SHAPES, "M8a — synthetic geometric shapes"),
        ("m8d", M8D_SHAPES, "M8d — synthetic non-ball categories"),
        ("m8c", M8C_SHAPES, "M8c — real photographs"),
    ]
    model_labels = ["Qwen\n(SigLIP+Qwen)", "LLaVA\n(CLIP+Vicuna)", "Idefics2\n(SigLIP+Mistral)"]

    for r, (stim, shapes, title) in enumerate(stim_specs):
        sub_p = all_pmr[all_pmr["stim"] == stim]
        sub_h = all_h7[all_h7["stim"] == stim]
        if len(sub_p) == 0:
            for c in range(2):
                axes[r, c].text(0.5, 0.5, f"{stim} — no data", ha="center", va="center",
                                transform=axes[r, c].transAxes)
                axes[r, c].axis("off")
            continue
        p_pivot = sub_p.pivot(index="model", columns="shape", values="pmr_nolabel").reindex(
            index=["qwen", "llava", "idefics2"], columns=list(shapes))
        h_pivot = sub_h.pivot(index="model", columns="shape", values="h7_phys_minus_abs").reindex(
            index=["qwen", "llava", "idefics2"], columns=list(shapes))
        _heatmap_panel(axes[r, 0], p_pivot, cmap="Blues", vmin=0, vmax=1, fmt=".2f",
                       title=f"{title}\nPMR(_nolabel)", model_labels=model_labels)
        _heatmap_panel(axes[r, 1], h_pivot, cmap="RdBu_r", vmin=-0.7, vmax=0.7, fmt="+.2f",
                       title=f"{title}\nH7 (physical − abstract)", model_labels=model_labels)

    fig.suptitle("§4.5 ext — Cross-encoder swap × cross-stim (M8a / M8d / M8c)",
                 y=1.005, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_encoder_swap_summary_bar(all_pmr: pd.DataFrame, out: Path) -> None:
    """Single bar chart: mean PMR(_nolabel) per (stim, model). The headline view."""
    fig, ax = plt.subplots(figsize=(9, 5))
    summary = all_pmr.groupby(["stim", "model", "encoder"])["pmr_nolabel"].mean().reset_index()

    stim_order = ["m8a", "m8d", "m8c"]
    model_order = ["qwen", "llava", "idefics2"]
    width = 0.25
    x = np.arange(len(stim_order))
    color_by_encoder = {"SigLIP": "#1f77b4", "CLIP-ViT-L": "#d62728", "SigLIP-SO400M": "#5fa8d3"}

    for k, model in enumerate(model_order):
        vals: list[float] = []
        encs: list[str] = []
        for stim in stim_order:
            row = summary[(summary["stim"] == stim) & (summary["model"] == model)]
            vals.append(float(row["pmr_nolabel"].iloc[0]) if len(row) else float("nan"))
            encs.append(row["encoder"].iloc[0] if len(row) else "")
        color = color_by_encoder.get(encs[0], "gray") if encs[0] else "gray"
        bars = ax.bar(x + (k - 1) * width, vals, width, label=f"{model} ({encs[0]})",
                      color=color, edgecolor="black")
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{v:.2f}",
                        ha="center", fontsize=8)

    ax.set_xticks(x, ["M8a (synth shapes)", "M8d (synth categories)", "M8c (real photos)"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("mean PMR(_nolabel) (averaged across shapes)")
    ax.set_title("§4.5 ext — encoder-swap summary across stim sources")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_pmr_parts: list[pd.DataFrame] = []
    all_h7_parts: list[pd.DataFrame] = []

    for stim, shapes in (("m8a", M8A_SHAPES), ("m8d", M8D_SHAPES), ("m8c", M8C_SHAPES)):
        pmr, h7 = collect_for_stim(stim, shapes)
        all_pmr_parts.append(pmr)
        all_h7_parts.append(h7)
        print(f"\n=== {stim.upper()} PMR(_nolabel) per (model × shape) ===")
        print(pmr.round(3).to_string(index=False) if len(pmr) else "(no data)")
        print(f"\n=== {stim.upper()} H7 paired-difference per (model × shape) ===")
        print(h7.round(3).to_string(index=False) if len(h7) else "(no data)")

    all_pmr = pd.concat(all_pmr_parts, ignore_index=True) if all_pmr_parts else pd.DataFrame()
    all_h7 = pd.concat(all_h7_parts, ignore_index=True) if all_h7_parts else pd.DataFrame()

    print("\n=== Mean PMR(_nolabel) by (stim, model, encoder) ===")
    if len(all_pmr):
        print(all_pmr.groupby(["stim", "model", "encoder"])["pmr_nolabel"].mean().round(3).to_string())
    print("\n=== Mean H7 by (stim, model, encoder) ===")
    if len(all_h7):
        print(all_h7.groupby(["stim", "model", "encoder"])["h7_phys_minus_abs"].mean().round(3).to_string())

    # Backward-compat artifacts (M8a only, for legacy consumers).
    m8a_pmr = all_pmr[all_pmr["stim"] == "m8a"].copy() if len(all_pmr) else all_pmr
    m8a_h7 = all_h7[all_h7["stim"] == "m8a"].copy() if len(all_h7) else all_h7
    if len(m8a_pmr):
        m8a_pmr.drop(columns=["stim"]).to_csv(args.out_dir / "encoder_swap_pmr_nolabel.csv", index=False)
        m8a_h7.drop(columns=["stim"]).to_csv(args.out_dir / "encoder_swap_h7.csv", index=False)
        fig_encoder_swap_m8a(m8a_pmr, m8a_h7, PROJECT_ROOT / "docs" / "figures" / "encoder_swap_heatmap.png")

    # Extended cross-stim artifacts.
    if len(all_pmr):
        all_pmr.to_csv(args.out_dir / "encoder_swap_extended_pmr.csv", index=False)
        all_h7.to_csv(args.out_dir / "encoder_swap_extended_h7.csv", index=False)
        fig_encoder_swap_extended(all_pmr, all_h7, PROJECT_ROOT / "docs" / "figures" / "encoder_swap_extended_heatmap.png")
        fig_encoder_swap_summary_bar(all_pmr, PROJECT_ROOT / "docs" / "figures" / "encoder_swap_extended_summary_bar.png")
        print(f"\nWrote {args.out_dir / 'encoder_swap_extended_pmr.csv'}")
        print(f"Wrote {PROJECT_ROOT / 'docs' / 'figures' / 'encoder_swap_extended_heatmap.png'}")
        print(f"Wrote {PROJECT_ROOT / 'docs' / 'figures' / 'encoder_swap_extended_summary_bar.png'}")


if __name__ == "__main__":
    main()
