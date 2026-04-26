"""§4.3 — Korean vs English label prior on Qwen2.5-VL circle stim.

Compares per-label PMR between English (ball/circle/planet) and Korean
(공/원/행성) labels on the same M8a circle subset (80 stim per label,
n=240 per language). Bootstrap CIs (5000 iters, prediction-level
resampling).

Headline: does Korean label strength match English? Does the cross-
label ordering survive? Bonus: Qwen2.5-VL is multilingual; the prompt
template is English with the label inserted, so this isolates label-
prior strength independent of question language.

Usage:
    uv run python scripts/sec4_3_korean_vs_english.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"

KO_RUN = PROJECT_ROOT / "outputs/sec4_3_korean_labels_qwen_20260426-003750_0355e3de/predictions.jsonl"
EN_RUN = sorted(PROJECT_ROOT.glob("outputs/m8a_qwen_2*/predictions.jsonl"))[-1]

LABEL_PAIRS = [("ball", "공"), ("circle", "원"), ("planet", "행성")]
ROLE_NAMES = {"ball": "physical", "circle": "abstract", "planet": "exotic"}

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42


def _bootstrap_pmr_ci(pmr_vec: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(pmr_vec)
    means = np.empty(BOOTSTRAP_N, dtype=float)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        means[b] = pmr_vec[idx].mean()
    return float(pmr_vec.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    en = score_rows(pd.read_json(EN_RUN, lines=True))
    en = en[en["shape"] == "circle"].copy()
    ko = score_rows(pd.read_json(KO_RUN, lines=True))
    ko = ko[ko["shape"] == "circle"].copy()

    rows = []
    for en_label, ko_label in LABEL_PAIRS:
        for lang, df, lbl in [("English", en, en_label), ("Korean", ko, ko_label)]:
            sub = df[df["label"] == lbl]
            mean, lo, hi = _bootstrap_pmr_ci(sub["pmr"].to_numpy())
            rows.append({
                "language": lang, "role": ROLE_NAMES[en_label],
                "label": lbl, "n": len(sub),
                "pmr_mean": mean, "pmr_ci_low": lo, "pmr_ci_high": hi,
            })
    table = pd.DataFrame(rows)
    print("=== §4.3 Korean vs English — Qwen2.5-VL on M8a circle (80 stim/label) ===")
    print(table.round(3).to_string(index=False))

    out_csv = PROJECT_ROOT / "outputs" / "sec4_3_korean_vs_english.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Figure: paired bars (EN vs KO) per role
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(LABEL_PAIRS))
    width = 0.35
    en_means, en_lo, en_hi, ko_means, ko_lo, ko_hi = [], [], [], [], [], []
    en_labels_x, ko_labels_x = [], []
    for en_label, ko_label in LABEL_PAIRS:
        en_row = table[(table["language"] == "English") & (table["label"] == en_label)].iloc[0]
        ko_row = table[(table["language"] == "Korean") & (table["label"] == ko_label)].iloc[0]
        en_means.append(en_row["pmr_mean"])
        en_lo.append(en_row["pmr_ci_low"]); en_hi.append(en_row["pmr_ci_high"])
        ko_means.append(ko_row["pmr_mean"])
        ko_lo.append(ko_row["pmr_ci_low"]); ko_hi.append(ko_row["pmr_ci_high"])
        en_labels_x.append(en_label); ko_labels_x.append(ko_label)

    # Romanizations for the figure (no Korean font on this system).
    KO_ROMAN = {"공": "gong", "원": "won", "행성": "haengseong"}
    bars1 = ax.bar(x - width/2, en_means, width, label="English", color="#1f77b4",
                   edgecolor="black", alpha=0.9)
    bars2 = ax.bar(x + width/2, ko_means, width, label="Korean (Hangul, romanized)", color="#d62728",
                   edgecolor="black", alpha=0.9)
    yerr1_lo = [m - l for m, l in zip(en_means, en_lo)]
    yerr1_hi = [h - m for m, h in zip(en_means, en_hi)]
    yerr2_lo = [m - l for m, l in zip(ko_means, ko_lo)]
    yerr2_hi = [h - m for m, h in zip(ko_means, ko_hi)]
    ax.errorbar(x - width/2, en_means, yerr=[yerr1_lo, yerr1_hi],
                fmt="none", ecolor="black", capsize=4, linewidth=0.8)
    ax.errorbar(x + width/2, ko_means, yerr=[yerr2_lo, yerr2_hi],
                fmt="none", ecolor="black", capsize=4, linewidth=0.8)

    for xi, m, l in zip(x - width/2, en_means, en_labels_x):
        ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(xi, -0.05, l, ha="center", fontsize=9, color="#1f77b4")
    for xi, m, l in zip(x + width/2, ko_means, ko_labels_x):
        ax.text(xi, m + 0.025, f"{m:.2f}", ha="center", fontsize=9)
        ax.text(xi, -0.05, KO_ROMAN[l], ha="center", fontsize=9, color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels([
        "physical\n(ball / gong)",
        "abstract\n(circle / won)",
        "exotic\n(planet / haengseong)",
    ], fontsize=10)
    ax.set_ylabel("PMR (95% bootstrap CI), n=80 per label")
    ax.set_ylim(0, 1.1)
    ax.set_title("§4.3 — Qwen2.5-VL: Korean vs English label prior on M8a circle\n"
                 "Korean labels (gong/won/haengseong) are inserted into the English question "
                 "template.\nCross-label ordering preserved across languages.")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    out_png = FIG_DIR / "sec4_3_korean_vs_english.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Quick sanity: a few raw responses
    print("\n=== Sample raw responses ===")
    for en_label, ko_label in LABEL_PAIRS:
        for lang, df, lbl in [("EN", en, en_label), ("KO", ko, ko_label)]:
            samples = df[df["label"] == lbl].head(2)
            for _, row in samples.iterrows():
                print(f"  [{lang}/{lbl}] pmr={int(row['pmr'])}: {row['raw_text'][:80]}")


if __name__ == "__main__":
    main()
