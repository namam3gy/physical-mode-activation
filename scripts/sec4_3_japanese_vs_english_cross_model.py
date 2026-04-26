"""§4.3 ext — Japanese vs English label prior across 5 VLMs on M8a circle stim.

Mirror of `scripts/sec4_3_korean_vs_english_cross_model.py` for Japanese
labels (ボール / 円 / 惑星). Pairs each model's existing English M8a
circle subset (n=80 per label) with its dedicated
`sec4_3_japanese_labels_<model>` Japanese run. Bootstrap CIs (5000
iters, prediction-level resampling).

Tests whether the Korean finding (cross-label ordering preserved 4/5,
LLaVA-1.5 swing largest from Vicuna-LM Korean weakness, InternVL3
ceiling) generalizes to Japanese — a different language with different
LM SFT coverage profiles per model.

Headline-prediction (pre-registered before looking at the data):
- 4/5 ordering preserved (multilingual semantic representation).
- LLaVA-1.5 swing similar or larger (Vicuna's Japanese is comparable to
  its Korean — both weak; underlying LM is LLaMA-2 with little non-EN SFT).
- InternVL3 swing minimal (InternLM3 has stronger multilingual coverage).
- Qwen swing minimal-to-moderate (Qwen2.5 has strong Japanese coverage).
- Idefics2 may show token-frequency effects on 惑星 (compound noun,
  less common than ball/circle) similar to Korean 행성.

Usage:
    uv run python scripts/sec4_3_japanese_vs_english_cross_model.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

LABEL_PAIRS = [("ball", "ボール"), ("circle", "円"), ("planet", "惑星")]
ROLE_NAMES = {"ball": "physical", "circle": "abstract", "planet": "exotic"}
JA_ROMAN = {"ボール": "booru", "円": "en", "惑星": "wakusei"}

# (English M8a run pattern, Japanese §4.3 run pattern) per model.
MODEL_RUNS: dict[str, tuple[str, str]] = {
    "Qwen2.5-VL":  ("m8a_qwen_2*",                      "sec4_3_japanese_labels_qwen_*"),
    "LLaVA-1.5":   ("m8a_llava_2*",                     "sec4_3_japanese_labels_llava_2*"),
    "LLaVA-Next":  ("encoder_swap_llava_next_m8a_2*",   "sec4_3_japanese_labels_llava_next_*"),
    "Idefics2":    ("encoder_swap_idefics2_2*",         "sec4_3_japanese_labels_idefics2_*"),
    "InternVL3":   ("encoder_swap_internvl3_m8a_2*",    "sec4_3_japanese_labels_internvl3_*"),
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


def _aggregate_one_model(model: str, en_path: Path, ja_path: Path) -> pd.DataFrame:
    en = score_rows(pd.read_json(en_path, lines=True))
    en = en[en["shape"] == "circle"].copy()
    ja = score_rows(pd.read_json(ja_path, lines=True))
    ja = ja[ja["shape"] == "circle"].copy()

    rows = []
    for en_label, ja_label in LABEL_PAIRS:
        for lang, df, lbl in [("English", en, en_label), ("Japanese", ja, ja_label)]:
            sub = df[df["label"] == lbl]
            if len(sub) == 0:
                raise RuntimeError(f"{model}: no rows for {lang} label {lbl} in {en_path if lang=='English' else ja_path}")
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
    ja_paths: dict[str, Path] = {}
    for model, (en_pat, ja_pat) in MODEL_RUNS.items():
        en_paths[model] = _latest(en_pat)
        ja_paths[model] = _latest(ja_pat)
        all_rows.append(_aggregate_one_model(model, en_paths[model], ja_paths[model]))
    table = pd.concat(all_rows, ignore_index=True)

    print("=== §4.3 Japanese vs English — 5 VLMs on M8a circle (80 stim/label) ===")
    print("\nResolved runs:")
    for m in MODEL_RUNS:
        print(f"  {m}: EN={en_paths[m].parent.name} | JA={ja_paths[m].parent.name}")
    print("\n" + table.round(3).to_string(index=False))

    out_csv = PROJECT_ROOT / "outputs" / "sec4_3_japanese_vs_english_cross_model.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Figure: 1 row × 5 model panels, paired EN vs JA bars per role.
    n_models = len(MODEL_RUNS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.0 * n_models, 5.2),
                             sharey=True)
    x = np.arange(len(LABEL_PAIRS))
    width = 0.35

    for ax, (model, _) in zip(axes, MODEL_RUNS.items()):
        sub = table[table["model"] == model]
        en_means, en_lo, en_hi = [], [], []
        ja_means, ja_lo, ja_hi = [], [], []
        for en_label, ja_label in LABEL_PAIRS:
            er = sub[(sub["language"] == "English") & (sub["label"] == en_label)].iloc[0]
            jr = sub[(sub["language"] == "Japanese") & (sub["label"] == ja_label)].iloc[0]
            en_means.append(er["pmr_mean"]); en_lo.append(er["pmr_ci_low"]); en_hi.append(er["pmr_ci_high"])
            ja_means.append(jr["pmr_mean"]); ja_lo.append(jr["pmr_ci_low"]); ja_hi.append(jr["pmr_ci_high"])

        ax.bar(x - width / 2, en_means, width, label="English",
               color="#1f77b4", edgecolor="black", alpha=0.9)
        ax.bar(x + width / 2, ja_means, width, label="Japanese (romanized)",
               color="#2ca02c", edgecolor="black", alpha=0.9)
        ax.errorbar(x - width / 2, en_means,
                    yerr=[[m - l for m, l in zip(en_means, en_lo)],
                          [h - m for m, h in zip(en_means, en_hi)]],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)
        ax.errorbar(x + width / 2, ja_means,
                    yerr=[[m - l for m, l in zip(ja_means, ja_lo)],
                          [h - m for m, h in zip(ja_means, ja_hi)]],
                    fmt="none", ecolor="black", capsize=3, linewidth=0.8)

        for xi, m in zip(x - width / 2, en_means):
            ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=8)
        for xi, m in zip(x + width / 2, ja_means):
            ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([
            f"physical\n(ball /\n{JA_ROMAN['ボール']})",
            f"abstract\n(circle /\n{JA_ROMAN['円']})",
            f"exotic\n(planet /\n{JA_ROMAN['惑星']})",
        ], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(model, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("PMR (95% bootstrap CI), n=80 per label")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("§4.3 — Japanese vs English label prior across 5 VLMs on M8a circle "
                 "(Japanese labels inserted into English question template)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = FIG_DIR / "sec4_3_japanese_vs_english_cross_model.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Per-model EN-vs-JA Δ table for headline interpretation.
    print("\n=== Per-model EN→JA Δ (JA − EN) ===")
    delta_rows = []
    for model in MODEL_RUNS:
        for en_label, ja_label in LABEL_PAIRS:
            sub = table[table["model"] == model]
            en_pmr = sub[(sub["language"] == "English") & (sub["label"] == en_label)].iloc[0]["pmr_mean"]
            ja_pmr = sub[(sub["language"] == "Japanese") & (sub["label"] == ja_label)].iloc[0]["pmr_mean"]
            delta_rows.append({
                "model": model,
                "role": ROLE_NAMES[en_label],
                "en_pmr": en_pmr,
                "ja_pmr": ja_pmr,
                "delta": ja_pmr - en_pmr,
            })
    delta = pd.DataFrame(delta_rows)
    print(delta.round(3).to_string(index=False))
    delta_csv = PROJECT_ROOT / "outputs" / "sec4_3_japanese_vs_english_cross_model_deltas.csv"
    delta.to_csv(delta_csv, index=False)
    print(f"Wrote {delta_csv}")


if __name__ == "__main__":
    main()
