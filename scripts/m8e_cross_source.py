"""M8e — cross-source paired analysis (synthetic-textured vs photo).

Consolidates M8a (circle textured), M8d (car/person/bird textured), and
M8c (photo) into a single per-(model × category × source_type) view.

Output:
  - `m8e_cross_source.csv` — wide table of PMR(_nolabel) per
    (model, category, source_type ∈ {synthetic-textured, photo}).
  - `m8e_paired_delta.csv` — per-(model, category) delta = photo − synth.
  - `m8e_h7_cross_source.csv` — per-(model, category, source_type) H7
    paired-difference (physical − abstract).

Generates figure `m8e_cross_source_heatmap.png` summarizing the pattern.

Usage:
    uv run python scripts/m8e_cross_source.py --out-dir outputs/m8e_summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.inference.prompts import LABELS_BY_SHAPE
from physical_mode.metrics.pmr import score_rows


CATEGORIES = ("ball", "car", "person", "bird")
ROLES = ("physical", "abstract", "exotic")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "outputs"


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


def _latest(prefix: str) -> Path:
    """Return the latest run dir matching the prefix (excluding label_free
    sub-prefixes via a more specific glob)."""
    if "label_free" in prefix:
        cands = sorted(OUT_ROOT.glob(f"{prefix}_*/predictions.jsonl"))
    else:
        # e.g., 'm8c_qwen' — exclude 'm8c_qwen_label_free_*'
        cands = sorted(OUT_ROOT.glob(f"{prefix}_2*/predictions.jsonl"))
    return cands[-1] if cands else None


def collect_synthetic_baselines() -> pd.DataFrame:
    """Pull synthetic-textured PMR(_nolabel) baselines from M8a + M8d label-free runs."""
    rows = []
    for model_name, m8a_lf, m8d_lf in [
        ("qwen", "m8a_qwen_label_free", "m8d_qwen_label_free"),
        ("llava", "m8a_llava_label_free", "m8d_llava_label_free"),
    ]:
        # M8a: circle textured = ball.
        a = pd.read_json(_latest(m8a_lf), lines=True) if _latest(m8a_lf) else None
        if a is not None:
            a = score_rows(a)
            sub = a[(a["shape"] == "circle") & (a["object_level"] == "textured")]
            rows.append({"model": model_name, "category": "ball", "source_type": "synthetic-textured",
                         "pmr_nolabel": float(sub["pmr"].mean()), "n": int(len(sub))})

        # M8d: car/person/bird textured.
        d = pd.read_json(_latest(m8d_lf), lines=True) if _latest(m8d_lf) else None
        if d is not None:
            d = score_rows(d)
            for cat in ("car", "person", "bird"):
                sub = d[(d["shape"] == cat) & (d["object_level"] == "textured")]
                rows.append({"model": model_name, "category": cat, "source_type": "synthetic-textured",
                             "pmr_nolabel": float(sub["pmr"].mean()), "n": int(len(sub))})
    return pd.DataFrame(rows)


def collect_photo_baselines() -> pd.DataFrame:
    """Pull photo PMR(_nolabel) from M8c label-free runs."""
    rows = []
    for model_name, m8c_lf in [
        ("qwen", "m8c_qwen_label_free"),
        ("llava", "m8c_llava_label_free"),
    ]:
        c = pd.read_json(_latest(m8c_lf), lines=True) if _latest(m8c_lf) else None
        if c is not None:
            c = score_rows(c)
            for cat in CATEGORIES:
                sub = c[c["shape"] == cat]
                if not sub.empty:
                    rows.append({"model": model_name, "category": cat, "source_type": "photo",
                                 "pmr_nolabel": float(sub["pmr"].mean()), "n": int(len(sub))})
    return pd.DataFrame(rows)


def collect_h7_per_source() -> pd.DataFrame:
    """For each (model, category, source_type), compute H7 paired-difference
    PMR(physical) − PMR(abstract) per the labeled-arm runs."""
    rows = []

    # Synthetic side.
    for model_name, m8a_lbl, m8d_lbl in [
        ("qwen", "m8a_qwen", "m8d_qwen"),
        ("llava", "m8a_llava", "m8d_llava"),
    ]:
        a = _load_with_role(_latest(m8a_lbl)) if _latest(m8a_lbl) else None
        if a is not None:
            sub = a[(a["shape"] == "circle") & (a["object_level"] == "textured")]
            phys = sub[sub["label_role"] == "physical"]["pmr"].mean()
            absr = sub[sub["label_role"] == "abstract"]["pmr"].mean()
            rows.append({"model": model_name, "category": "ball",
                         "source_type": "synthetic-textured",
                         "physical_pmr": float(phys), "abstract_pmr": float(absr),
                         "h7_delta": float(phys - absr)})

        d = _load_with_role(_latest(m8d_lbl)) if _latest(m8d_lbl) else None
        if d is not None:
            for cat in ("car", "person", "bird"):
                sub = d[(d["shape"] == cat) & (d["object_level"] == "textured")]
                phys = sub[sub["label_role"] == "physical"]["pmr"].mean()
                absr = sub[sub["label_role"] == "abstract"]["pmr"].mean()
                rows.append({"model": model_name, "category": cat,
                             "source_type": "synthetic-textured",
                             "physical_pmr": float(phys), "abstract_pmr": float(absr),
                             "h7_delta": float(phys - absr)})

    # Photo side.
    for model_name, m8c_lbl in [
        ("qwen", "m8c_qwen"),
        ("llava", "m8c_llava"),
    ]:
        c = _load_with_role(_latest(m8c_lbl)) if _latest(m8c_lbl) else None
        if c is not None:
            for cat in CATEGORIES:
                sub = c[c["shape"] == cat]
                phys = sub[sub["label_role"] == "physical"]["pmr"].mean()
                absr = sub[sub["label_role"] == "abstract"]["pmr"].mean()
                rows.append({"model": model_name, "category": cat,
                             "source_type": "photo",
                             "physical_pmr": float(phys), "abstract_pmr": float(absr),
                             "h7_delta": float(phys - absr)})

    return pd.DataFrame(rows)


def fig_cross_source_heatmap(synth: pd.DataFrame, photo: pd.DataFrame, h7: pd.DataFrame, out: Path) -> None:
    """3-panel heatmap: PMR(_nolabel) synthetic / photo / H7 delta cross-source."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel 1: PMR(_nolabel) synthetic-textured (rows=model, cols=category)
    s_pivot = synth.pivot(index="model", columns="category", values="pmr_nolabel").reindex(
        index=["qwen", "llava"], columns=list(CATEGORIES))
    p_pivot = photo.pivot(index="model", columns="category", values="pmr_nolabel").reindex(
        index=["qwen", "llava"], columns=list(CATEGORIES))

    for ax, mat, title in [
        (axes[0], s_pivot, "synthetic-textured PMR(_nolabel)"),
        (axes[1], p_pivot, "photo PMR(_nolabel)"),
    ]:
        im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=20)
        ax.set_yticks(range(2), ["Qwen", "LLaVA"])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v > 0.55 else "black", fontsize=10)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    # Panel 3: H7 cross-source delta — photo H7 − synthetic H7.
    h7_synth = h7[h7["source_type"] == "synthetic-textured"].pivot(
        index="model", columns="category", values="h7_delta").reindex(
        index=["qwen", "llava"], columns=list(CATEGORIES))
    h7_photo = h7[h7["source_type"] == "photo"].pivot(
        index="model", columns="category", values="h7_delta").reindex(
        index=["qwen", "llava"], columns=list(CATEGORIES))
    h7_diff = h7_photo - h7_synth

    ax = axes[2]
    im = ax.imshow(h7_diff.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)), CATEGORIES, rotation=20)
    ax.set_yticks(range(2), ["Qwen", "LLaVA"])
    for i in range(h7_diff.shape[0]):
        for j in range(h7_diff.shape[1]):
            v = h7_diff.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.35 else "black", fontsize=10)
    ax.set_title("H7 photo − H7 synthetic\n(positive: photos amplify H7)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle("M8e — cross-source paired analysis (synthetic-textured vs photo)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting synthetic baselines (M8a circle + M8d textured) ...")
    syn = collect_synthetic_baselines()
    print(syn.to_string(index=False))
    syn.to_csv(args.out_dir / "m8e_synth_pmr_nolabel.csv", index=False)

    print("\nCollecting photo baselines (M8c) ...")
    photo = collect_photo_baselines()
    print(photo.to_string(index=False))
    photo.to_csv(args.out_dir / "m8e_photo_pmr_nolabel.csv", index=False)

    # Compute paired delta (photo − synthetic).
    print("\nComputing paired delta (photo − synthetic) ...")
    paired = []
    for cat in CATEGORIES:
        for model_name in ("qwen", "llava"):
            s = syn[(syn["model"] == model_name) & (syn["category"] == cat)]
            p = photo[(photo["model"] == model_name) & (photo["category"] == cat)]
            if not s.empty and not p.empty:
                paired.append({
                    "model": model_name, "category": cat,
                    "synthetic_textured_pmr": s["pmr_nolabel"].iloc[0],
                    "photo_pmr": p["pmr_nolabel"].iloc[0],
                    "delta_photo_minus_synth": p["pmr_nolabel"].iloc[0] - s["pmr_nolabel"].iloc[0],
                })
    paired_df = pd.DataFrame(paired).round(3)
    print(paired_df.to_string(index=False))
    paired_df.to_csv(args.out_dir / "m8e_paired_delta.csv", index=False)

    print("\nCollecting H7 paired-difference per source ...")
    h7 = collect_h7_per_source()
    h7_round = h7.round(3)
    print(h7_round.to_string(index=False))
    h7_round.to_csv(args.out_dir / "m8e_h7_cross_source.csv", index=False)

    fig_path = PROJECT_ROOT / "docs" / "figures" / "m8e_cross_source_heatmap.png"
    fig_cross_source_heatmap(syn, photo, h7, fig_path)
    print(f"\nWrote heatmap to {fig_path}")


if __name__ == "__main__":
    main()
