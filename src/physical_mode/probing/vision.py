"""Sub-task 2 — vision encoder probing.

Trains layer-wise linear probes on captured Qwen2.5-VL vision-encoder
activations against behavioral PMR labels. Tests the "encoder-decoder
boomerang" claim: does the encoder linearly separate "physical-like" from
"geometric-like" stimuli even when the behavioral output doesn't?

Typical usage
-------------
>>> from physical_mode.probing.vision import (
...     load_probing_dataset, train_layer_probe, run_layer_sweep)
>>> X_per_layer, y, meta = load_probing_dataset(
...     vision_dir="outputs/<run>/vision_activations",
...     predictions_path="outputs/<run>/predictions_scored.parquet",
...     pmr_source="forced_choice",  # cleaner signal per pilot
... )
>>> results = run_layer_sweep(X_per_layer, y)

The probe target is a *per-stimulus* PMR label, averaged across the three
labels (circle/ball/planet) × both prompt variants. Default is the
forced-choice majority, which is less contaminated by language prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


@dataclass
class ProbeResult:
    layer: int
    auc_mean: float
    auc_std: float
    accuracy_mean: float
    n_pos: int
    n_neg: int


def _mean_pool(tensor: np.ndarray) -> np.ndarray:
    """(n_tokens, dim) -> (dim,); mean across the token axis."""
    if tensor.ndim != 2:
        raise ValueError(f"expected (tokens, dim), got shape {tensor.shape}")
    return tensor.mean(axis=0)


def _load_layer_activations(
    vision_dir: Path,
    sample_ids: Iterable[str],
    layer: int,
) -> np.ndarray:
    """Load mean-pooled vision activations for each sample id at one layer."""
    from safetensors.torch import load_file

    rows: list[np.ndarray] = []
    key = f"vision_hidden_{layer}"
    for sid in sample_ids:
        f = vision_dir / f"{sid}.safetensors"
        data = load_file(str(f))
        if key not in data:
            raise KeyError(f"{key} not in {f.name}; available: {list(data)}")
        t = data[key].to(dtype=__import__("torch").float32).numpy()
        rows.append(_mean_pool(t))
    return np.stack(rows)


def _aggregate_pmr(
    df: pd.DataFrame,
    pmr_source: Literal["open", "forced_choice", "either", "majority"],
) -> pd.DataFrame:
    """Reduce per-(sample × label × prompt) rows to a single PMR label per sample."""
    if pmr_source in ("open", "forced_choice"):
        sub = df[df["prompt_variant"] == pmr_source]
    else:
        sub = df
    # Average PMR across labels + (optionally) prompt variants; threshold at 0.5.
    per_sample = sub.groupby("sample_id")["pmr"].mean().rename("pmr_mean")
    per_sample_binary = (per_sample >= 0.5).astype(int).rename("y")
    return per_sample_binary.reset_index()


def load_probing_dataset(
    vision_dir: Path | str,
    predictions_path: Path | str,
    layers: Iterable[int],
    pmr_source: Literal["open", "forced_choice", "either", "majority"] = "forced_choice",
) -> tuple[dict[int, np.ndarray], np.ndarray, pd.DataFrame]:
    """Assemble {layer: X} features, y labels, and per-sample metadata.

    Returns
    -------
    X_per_layer : dict mapping layer index -> (n_samples, dim) float32 matrix.
    y : (n_samples,) int binary array (PMR label per the chosen pmr_source).
    meta : DataFrame with columns sample_id, object_level, bg_level, cue_level, label, y.
    """
    vision_dir = Path(vision_dir)
    preds = pd.read_parquet(predictions_path)
    agg = _aggregate_pmr(preds, pmr_source)

    # Attach factorial axes (they're constant across variants for a given sample_id).
    axes = (
        preds[["sample_id", "object_level", "bg_level", "cue_level", "event_template"]]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )
    meta = axes.merge(agg, on="sample_id", how="inner").reset_index(drop=True)
    # Keep only samples that have a capture file on disk.
    meta = meta[
        meta["sample_id"].apply(lambda sid: (vision_dir / f"{sid}.safetensors").exists())
    ].reset_index(drop=True)
    print(f"Probing dataset: {len(meta)} samples (pmr_source={pmr_source}).")
    print(f"  y=1 (physics): {int(meta['y'].sum())} / {len(meta)}")

    sample_ids = meta["sample_id"].tolist()
    X_per_layer: dict[int, np.ndarray] = {}
    for li in layers:
        X_per_layer[int(li)] = _load_layer_activations(vision_dir, sample_ids, int(li))
        print(f"  layer {li:>2}: X shape {X_per_layer[int(li)].shape}")
    y = meta["y"].to_numpy(dtype=np.int64)
    return X_per_layer, y, meta


def train_layer_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    C: float = 1.0,
) -> ProbeResult:
    """Stratified k-fold logistic-regression probe. Returns mean/std AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    aucs: list[float] = []
    accs: list[float] = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xva = scaler.transform(X[val_idx])
        clf = LogisticRegression(C=C, max_iter=1000, solver="liblinear")
        clf.fit(Xtr, y[train_idx])
        # Guard against degenerate folds (all same class).
        try:
            proba = clf.predict_proba(Xva)[:, 1]
            aucs.append(roc_auc_score(y[val_idx], proba))
        except ValueError:
            aucs.append(float("nan"))
        pred = clf.predict(Xva)
        accs.append(accuracy_score(y[val_idx], pred))

    return ProbeResult(
        layer=-1,  # overwritten by run_layer_sweep
        auc_mean=float(np.nanmean(aucs)),
        auc_std=float(np.nanstd(aucs)),
        accuracy_mean=float(np.mean(accs)),
        n_pos=int((y == 1).sum()),
        n_neg=int((y == 0).sum()),
    )


def run_layer_sweep(
    X_per_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Train a probe at each captured layer; return a sorted summary frame."""
    rows: list[dict] = []
    for li, X in sorted(X_per_layer.items()):
        r = train_layer_probe(X, y, n_splits=n_splits, seed=seed)
        rows.append(
            {
                "layer": li,
                "auc_mean": r.auc_mean,
                "auc_std": r.auc_std,
                "accuracy_mean": r.accuracy_mean,
                "n_pos": r.n_pos,
                "n_neg": r.n_neg,
            }
        )
    return pd.DataFrame(rows)


def probe_per_object_level(
    X_per_layer: dict[int, np.ndarray],
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Train probe per (layer × object_level). Useful for the 'boomerang' plot.

    For each object level, train a probe on the subset restricted to that level
    and report AUC. Compares encoder signal to behavioral PMR at the same cells.
    """
    rows: list[dict] = []
    for obj in sorted(meta["object_level"].unique()):
        sub_idx = meta.index[meta["object_level"] == obj].to_numpy()
        if len(sub_idx) < 2 * n_splits or len(set(y[sub_idx])) < 2:
            # not enough samples or only one class present
            continue
        y_sub = y[sub_idx]
        for li, X in sorted(X_per_layer.items()):
            X_sub = X[sub_idx]
            r = train_layer_probe(X_sub, y_sub, n_splits=n_splits, seed=seed)
            rows.append(
                {
                    "object_level": obj,
                    "layer": li,
                    "auc_mean": r.auc_mean,
                    "auc_std": r.auc_std,
                    "n_pos": r.n_pos,
                    "n_neg": r.n_neg,
                }
            )
    return pd.DataFrame(rows)
