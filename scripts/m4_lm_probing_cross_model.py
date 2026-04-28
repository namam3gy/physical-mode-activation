"""M4 LM logit-lens cross-model — 5-model × 5-layer probe AUC.

Reuses existing M2 cross-model captures (LLaVA-1.5 / LLaVA-Next /
Idefics2 / InternVL3) which stored `lm_hidden_{5,10,15,20,25}` at
visual-token positions. Adds Qwen2.5-VL from `mvp_full` for the
5th model point.

For each (model × layer):
- Load activations: per-stim mean visual-token hidden state.
- Load per-stim PMR (mean across labels) from predictions_with_pmr
  parquet. Binarize at 0.5.
- 5-fold cross-validated AUC via sklearn LogisticRegression.

Output: `outputs/m4_lm_probing_cross_model/probe_auc.csv` + figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors.torch as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "Qwen2.5-VL-7B": (
        "outputs/mvp_full_20260424-094103_8ae1fa3d/activations",
        "outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv",
    ),
    "LLaVA-1.5-7B": (
        "outputs/cross_model_llava_capture_20260425-054821_65214a5d/activations",
        "outputs/cross_model_llava_capture_20260425-054821_65214a5d/predictions_scored.csv",
    ),
    "LLaVA-Next-Mistral-7B": (
        "outputs/cross_model_llava_next_capture_20260426-110246_621a66ff/activations",
        "outputs/cross_model_llava_next_capture_20260426-110246_621a66ff/predictions_with_pmr.parquet",
    ),
    "Idefics2-8B": (
        "outputs/cross_model_idefics2_capture_20260426-111434_49ac35be/activations",
        "outputs/cross_model_idefics2_capture_20260426-111434_49ac35be/predictions_with_pmr.parquet",
    ),
    "InternVL3-8B": (
        "outputs/cross_model_internvl3_capture_20260426-112246_3569fe27/activations",
        "outputs/cross_model_internvl3_capture_20260426-112246_3569fe27/predictions_with_pmr.parquet",
    ),
}

LAYERS = (5, 10, 15, 20, 25)


def _load_pmr(path: Path) -> dict[str, float]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df.groupby("sample_id")["pmr"].mean().to_dict()


def _stim_features(act_dir: Path, layer: int, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-stim mean visual-token hidden state at `lm_hidden_{layer}`."""
    key = f"lm_hidden_{layer}"
    feats = []
    kept_ids = []
    for sid in sample_ids:
        f = act_dir / f"{sid}.safetensors"
        if not f.exists():
            continue
        d = st.load_file(f)
        if key not in d:
            continue
        a = d[key].float()  # (n_visual_tokens, d) for most; multi-tile flat
        if a.dim() == 3:
            a = a.reshape(-1, a.shape[-1])
        feats.append(a.mean(dim=0).numpy())
        kept_ids.append(sid)
    return np.stack(feats), kept_ids


def _probe_auc(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42) -> tuple[float, int]:
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0 or min(n_pos, n_neg) < 2:
        return float("nan"), len(y)
    skf = StratifiedKFold(n_splits=min(n_folds, min(n_pos, n_neg)),
                          shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=200, C=1.0).fit(X[tr], y[tr])
        score = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], score))
    return float(np.mean(aucs)), len(y)


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "m4_lm_probing_cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, (act_path, pred_path) in MODELS.items():
        act_dir = PROJECT_ROOT / act_path
        if not act_dir.exists():
            print(f"[SKIP] {model}: {act_dir} missing")
            continue
        pmr = _load_pmr(PROJECT_ROOT / pred_path)
        sample_ids = sorted(pmr.keys())
        print(f"[{model}] n_stim={len(sample_ids)}, n_phys={sum(p >= 0.5 for p in pmr.values())}")
        for layer in LAYERS:
            X, kept = _stim_features(act_dir, layer, sample_ids)
            if len(kept) == 0:
                print(f"  L{layer}: no activations")
                continue
            y = np.array([1 if pmr[s] >= 0.5 else 0 for s in kept])
            auc, n = _probe_auc(X, y)
            print(f"  L{layer}: AUC={auc:.3f} (n={n}, n_phys={int(y.sum())}, dim={X.shape[1]})")
            rows.append({
                "model": model, "layer": layer, "auc": auc,
                "n": n, "n_phys": int(y.sum()), "n_abs": n - int(y.sum()),
                "dim": X.shape[1],
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "probe_auc.csv", index=False)
    print(f"\nWrote {out_dir / 'probe_auc.csv'}")

    pivot = df.pivot(index="layer", columns="model", values="auc")
    print("\n=== AUC pivot (layer × model) ===")
    print(pivot.round(3).to_string())

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in pivot.columns:
        ax.plot(pivot.index, pivot[model], marker="o", label=model)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("LM layer (visual-token mean hidden state)")
    ax.set_ylabel("Probe AUC (5-fold, label-free PMR ≥ 0.5)")
    ax.set_title("M4 LM logit-lens cross-model — 5-model × 5-layer probe AUC")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = PROJECT_ROOT / "docs" / "figures" / "m4_lm_probing_cross_model.png"
    fig.savefig(fig_path, dpi=120)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
