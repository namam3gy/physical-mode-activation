"""§4.3 ext — Korean vs English label prior across 5 VLMs on M8a circle stim.

Cross-model extension of `scripts/sec4_3_korean_vs_english.py` (Qwen-only).
Pairs each model's existing English M8a run (with labels ball/circle/planet)
with its dedicated `sec4_3_korean_labels_<model>` Korean run (공/원/행성),
filtered to the circle subset (n=80 per label per language). Bootstrap CIs
(5000 iters, prediction-level resampling).

Five models: Qwen2.5-VL, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3.

Headline: does the multilingual-label result (cross-label ordering preserved
between EN and KO; Korean magnitude ≈ English on common labels) generalize
beyond Qwen2.5-VL?

Usage:
    uv run python scripts/sec4_3_korean_vs_english_cross_model.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

LABEL_PAIRS = [("ball", "공"), ("circle", "원"), ("planet", "행성")]
ROLE_NAMES = {"ball": "physical", "circle": "abstract", "planet": "exotic"}
KO_ROMAN = {"공": "gong", "원": "won", "행성": "haengseong"}

# (English M8a run pattern, Korean §4.3 run pattern) per model.
MODEL_RUNS: dict[str, tuple[str, str]] = {
    "Qwen2.5-VL":  ("m8a_qwen_2*",                      "sec4_3_korean_labels_qwen_*"),
    "LLaVA-1.5":   ("m8a_llava_2*",                     "sec4_3_korean_labels_llava_2*"),
    "LLaVA-Next":  ("encoder_swap_llava_next_m8a_2*",   "sec4_3_korean_labels_llava_next_*"),
    "Idefics2":    ("encoder_swap_idefics2_2*",         "sec4_3_korean_labels_idefics2_*"),
    "InternVL3":   ("encoder_swap_internvl3_m8a_2*",    "sec4_3_korean_labels_internvl3_*"),
}

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42


def _latest(pattern: str) -> Path:
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{pattern}/predictions.jsonl"))
    cands = [c for c in cands if c.stat().st_size > 0 and "_label_free" not in c.parent.name]
    if not cands:
        raise FileNotFoundError(f"No outputs match {pattern}")
    return cands[-1]


def _bootstrap_pmr_ci(pmr_vec: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(pmr_vec)
    means = np.empty(BOOTSTRAP_N, dtype=float)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        means[b] = pmr_vec[idx].mean()
    return float(pmr_vec.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _aggregate_one_model(model: str, en_path: Path, ko_path: Path) -> pd.DataFrame:
    en = score_rows(pd.read_json(en_path, lines=True))
    en = en[en["shape"] == "circle"].copy()
    ko = score_rows(pd.read_json(ko_path, lines=True))
    ko = ko[ko["shape"] == "circle"].copy()

    rows = []
    for en_label, ko_label in LABEL_PAIRS:
        for lang, df, lbl in [("English", en, en_label), ("Korean", ko, ko_label)]:
            sub = df[df["label"] == lbl]
            if len(sub) == 0:
                raise RuntimeError(f"{model}: no rows for {lang} label {lbl} in {en_path if lang=='English' else ko_path}")
            mean, lo, hi = _bootstrap_pmr_ci(sub["pmr"].to_numpy())
            rows.append({
                "model": model,
                "language": lang,
                "role": ROLE_NAMES[en_label],
                "label": lbl,
                "n": len(sub),
                "pmr_mean": mean,
                "pmr_ci_low": lo,
                "pmr_ci_high": hi,
            })
    return pd.DataFrame(rows)


def main() -> None:
    all_rows: list[pd.DataFrame] = []
    en_paths: dict[str, Path] = {}
    ko_paths: dict[str, Path] = {}
    for model, (en_pat, ko_pat) in MODEL_RUNS.items():
        en_paths[model] = _latest(en_pat)
        ko_paths[model] = _latest(ko_pat)
        all_rows.append(_aggregate_one_model(model, en_paths[model], ko_paths[model]))
    table = pd.concat(all_rows, ignore_index=True)

    print("=== §4.3 Korean vs English — 5 VLMs on M8a circle (80 stim/label) ===")
    print("\nResolved runs:")
    for m in MODEL_RUNS:
        print(f"  {m}: EN={en_paths[m].parent.name} | KO={ko_paths[m].parent.name}")
    print("\n" + table.round(3).to_string(index=False))

    out_csv = PROJECT_ROOT / "outputs" / "sec4_3_korean_vs_english_cross_model.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Figure: 1 row × 5 model panels, paired EN vs KO bars per role.
    n_models = len(MODEL_RUNS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.0 * n_models, 5.2),
                             sharey=True)
    x = np.arange(len(LABEL_PAIRS))
    width = 0.35

    for ax, (model, _) in zip(axes, MODEL_RUNS.items()):
        sub = table[table["model"] == model]
        en_means, en_lo, en_hi = [], [], []
        ko_means, ko_lo, ko_hi = [], [], []
        en_labels_x, ko_labels_x = [], []
        for en_label, ko_label in LABEL_PAIRS:
            er = sub[(sub["language"] == "English") & (sub["label"] == en_label)].iloc[0]
            kr = sub[(sub["language"] == "Korean") & (sub["label"] == ko_label)].iloc[0]
            en_means.append(er["pmr_mean"]); en_lo.append(er["pmr_ci_low"]); en_hi.append(er["pmr_ci_high"])
            ko_means.append(kr["pmr_mean"]); ko_lo.append(kr["pmr_ci_low"]); ko_hi.append(kr["pmr_ci_high"])
            en_labels_x.append(en_label); ko_labels_x.append(ko_label)

        ax.bar(x - width / 2, en_means, width, label="English",
               color="#1f77b4", edgecolor="black", alpha=0.9)
        ax.bar(x + width / 2, ko_means, width, label="Korean (romanized)",
               color="#d62728", edgecolor="black", alpha=0.9)
        ax.errorbar(x - width / 2, en_means,
                    yerr=[[m - l for m, l in zip(en_means, en_lo)],
                          [h - m for m, h in zip(en_means, en_hi)]],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.errorbar(x + width / 2, ko_means,
                    yerr=[[m - l for m, l in zip(ko_means, ko_lo)],
                          [h - m for m, h in zip(ko_means, ko_hi)]],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)

        for xi, m in zip(x - width / 2, en_means):
            ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=8)
        for xi, m in zip(x + width / 2, ko_means):
            ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"physical\n(ball /\n{KO_ROMAN['공']})",
            f"abstract\n(circle /\n{KO_ROMAN['원']})",
            f"exotic\n(planet /\n{KO_ROMAN['행성']})",
        ], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(model, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("PMR (95% bootstrap CI), n=80 per label")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("§4.3 — Korean vs English label prior across 5 VLMs on M8a circle "
                 "(Korean labels inserted into English question template)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = FIG_DIR / "sec4_3_korean_vs_english_cross_model.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Per-model EN-vs-KO Δ table for headline interpretation.
    print("\n=== Per-model EN→KO Δ (KO − EN) ===")
    delta_rows = []
    for model in MODEL_RUNS:
        for en_label, ko_label in LABEL_PAIRS:
            sub = table[table["model"] == model]
            en_pmr = sub[(sub["language"] == "English") & (sub["label"] == en_label)].iloc[0]["pmr_mean"]
            ko_pmr = sub[(sub["language"] == "Korean") & (sub["label"] == ko_label)].iloc[0]["pmr_mean"]
            delta_rows.append({
                "model": model,
                "role": ROLE_NAMES[en_label],
                "en_pmr": en_pmr,
                "ko_pmr": ko_pmr,
                "delta": ko_pmr - en_pmr,
            })
    delta = pd.DataFrame(delta_rows)
    print(delta.round(3).to_string(index=False))
    delta_csv = PROJECT_ROOT / "outputs" / "sec4_3_korean_vs_english_cross_model_deltas.csv"
    delta.to_csv(delta_csv, index=False)
    print(f"Wrote {delta_csv}")


if __name__ == "__main__":
    main()
