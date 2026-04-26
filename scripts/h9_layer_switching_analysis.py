"""H9 explicit test — cross-model layer-wise emergence of physics-mode
representation in the LM backbone.

Hypothesis (research plan §2.4): "Qwen2-VL / InternVL2 switch to physics
mode at earlier layers than LLaVA-1.5 (thanks to a larger vision encoder
and a more sophisticated projector)."

Method (per model):
- Load 480 M2 stim × 5 captured LM layers (5, 10, 15, 20, 25).
- Per (stim, layer): mean-pool LM hidden state at visual-token positions
  (using the saved `visual_token_mask`).
- Per (model, layer): train logistic-regression probe on PMR label
  (per-stim PMR collapsed across labels), 5-fold CV AUC.
- Define "switching layer" = first layer where probe AUC ≥ 0.85 (or
  earliest layer reaching ≥ 90% of the model's max AUC).

Outputs:
- outputs/h9_switching/per_layer_auc.csv
- outputs/h9_switching/switching_layer.csv
- docs/figures/h9_layer_switching.png (5-model probe AUC × captured layer)
- docs/insights/h9_layer_switching.md (+ ko)

Usage:
    uv run python scripts/h9_layer_switching_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUT_DIR = PROJECT_ROOT / "outputs" / "h9_switching"
LAYERS = [5, 10, 15, 20, 25]

# Per-model captured M2 run + LM-backbone layer counts (for relative depth).
MODELS: list[tuple[str, str, int]] = [
    ("Qwen2.5-VL", "mvp_full_2*",                      28),  # Qwen2-7B
    ("LLaVA-1.5",  "cross_model_llava_capture_*",      32),  # LLaMA-2 / Vicuna-7B
    ("LLaVA-Next", "cross_model_llava_next_capture_*", 32),  # Mistral-7B
    ("Idefics2",   "cross_model_idefics2_capture_*",   32),  # Mistral-7B
    ("InternVL3",  "cross_model_internvl3_capture_*",  48),  # InternLM3-8B has 48 layers
]
MODEL_COLORS = {
    "Qwen2.5-VL":  "#1f77b4",
    "LLaVA-1.5":   "#ff7f0e",
    "LLaVA-Next":  "#d62728",
    "Idefics2":    "#2ca02c",
    "InternVL3":   "#9467bd",
}


def _latest(pattern: str) -> Path:
    cands = sorted(PROJECT_ROOT.glob(f"outputs/{pattern}/predictions.jsonl"))
    cands = [c for c in cands if c.stat().st_size > 0 and (c.parent / "activations").exists()]
    if not cands:
        raise FileNotFoundError(pattern)
    return cands[-1].parent


def _load_per_stim(run_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Return X = (n_stim, hidden_dim) mean-pooled LM hidden at visual tokens,
    and y = per-stim PMR label (collapsed across labels)."""
    preds = pd.read_json(run_dir / "predictions.jsonl", lines=True)
    if "prompt_variant" in preds.columns:
        preds = preds[preds["prompt_variant"] == "open"]
    preds = preds.copy()
    preds["pmr"] = preds["raw_text"].apply(score_pmr)
    # Per-stim PMR: average across labels then threshold > 0.5
    y_per_sid = preds.groupby("sample_id")["pmr"].mean()
    y_bin = (y_per_sid >= 0.5).astype(int)

    activations_dir = run_dir / "activations"
    X = []
    y = []
    for sid, yi in y_bin.items():
        f = activations_dir / f"{sid}.safetensors"
        if not f.exists():
            continue
        data = load_file(f)
        if f"lm_hidden_{layer}" not in data:
            continue
        # NB: lm_hidden_* tensors are already pre-filtered to visual-token
        # positions (the capture step did the mask). Just mean-pool.
        h_np = data[f"lm_hidden_{layer}"].float().numpy().astype(np.float32)
        if h_np.shape[0] == 0:
            continue
        pooled = h_np.mean(axis=0)
        X.append(pooled)
        y.append(int(yi))
    return np.stack(X), np.array(y)


def _probe_auc(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[float, str]:
    """Return (AUC, mode). Tries 5-fold stratified CV; falls back to
    full-data fit (training AUC) when min-class count < 5."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), "degenerate"
    min_class = int(min(y.sum(), len(y) - y.sum()))
    if min_class >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[train_idx], y[train_idx])
            scores = clf.predict_proba(X[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], scores))
        return float(np.mean(aucs)), "5fold_cv"
    # Fallback: train on all, evaluate training AUC (overfits but informative
    # when min_class is too small for CV).
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, scores)), f"fullfit_min_class={min_class}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, pat, n_layers in MODELS:
        try:
            run_dir = _latest(pat)
        except FileNotFoundError as e:
            print(f"[SKIP] {model}: {e}")
            continue
        print(f"[{model}] {run_dir.name} ({n_layers} LM layers total)")
        for L in LAYERS:
            try:
                X, y = _load_per_stim(run_dir, L)
                auc, mode = _probe_auc(X, y)
                rel_depth = L / n_layers
                rows.append({
                    "model": model, "layer": L, "n_layers_total": n_layers,
                    "rel_depth": rel_depth,
                    "n_stim": len(y), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()),
                    "auc": auc, "auc_mode": mode,
                })
                print(f"  L{L} (rel={rel_depth:.2f}): n={len(y)} n_pos={int(y.sum())} AUC={auc:.3f} ({mode})")
            except Exception as e:
                print(f"  L{L}: ERROR {e}")
                rows.append({"model": model, "layer": L, "n_layers_total": n_layers,
                             "rel_depth": L / n_layers, "n_stim": 0, "n_pos": 0,
                             "n_neg": 0, "auc": float("nan"), "auc_mode": "error"})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_layer_auc.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'per_layer_auc.csv'}")

    # Switching-layer summary: first layer with AUC >= 0.85 (or earliest reaching 90% of model max).
    sw_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("layer")
        max_auc = sub["auc"].max()
        if np.isnan(max_auc):
            continue
        threshold_abs = 0.85
        threshold_rel = 0.9 * max_auc
        first_abs = sub[sub["auc"] >= threshold_abs].head(1)
        first_rel = sub[sub["auc"] >= threshold_rel].head(1)
        sw_rows.append({
            "model": model,
            "n_layers_total": int(sub["n_layers_total"].iloc[0]),
            "max_auc": max_auc,
            "first_layer_auc>=0.85": int(first_abs["layer"].iloc[0]) if len(first_abs) else None,
            "first_layer_auc>=0.9*max": int(first_rel["layer"].iloc[0]) if len(first_rel) else None,
            "rel_depth_at_threshold": (
                first_rel["rel_depth"].iloc[0] if len(first_rel) else None),
        })
    sw = pd.DataFrame(sw_rows)
    sw.to_csv(OUT_DIR / "switching_layer.csv", index=False)
    print(f"\n=== Switching layer summary ===")
    print(sw.round(3).to_string(index=False))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("layer")
        axes[0].plot(sub["layer"], sub["auc"], "o-", label=model,
                     color=MODEL_COLORS.get(model, "gray"), linewidth=2, markersize=8)
        axes[1].plot(sub["rel_depth"], sub["auc"], "o-", label=model,
                     color=MODEL_COLORS.get(model, "gray"), linewidth=2, markersize=8)
    axes[0].axhline(0.85, color="black", linestyle=":", alpha=0.5, label="AUC=0.85")
    axes[0].set_xlabel("Captured LM layer (absolute index)")
    axes[0].set_ylabel("Probe AUC (PMR label)")
    axes[0].set_title("LM probe AUC vs absolute layer index")
    axes[0].legend(loc="lower right", fontsize=10); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0.45, 1.02)

    axes[1].axhline(0.85, color="black", linestyle=":", alpha=0.5)
    axes[1].set_xlabel("Relative depth (layer / n_layers_total)")
    axes[1].set_ylabel("Probe AUC")
    axes[1].set_title("LM probe AUC vs relative depth — H9 cross-model")
    axes[1].legend(loc="lower right", fontsize=10); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0.45, 1.02)
    fig.tight_layout()
    out_png = FIG_DIR / "h9_layer_switching.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
