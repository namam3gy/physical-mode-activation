"""M5 Phase 1 — derive "physics-mode direction" in LM residual stream.

For each captured LM layer we compute a VTI-style steering vector:

    v_L = mean_L(h | PMR=1) - mean_L(h | PMR=0)

over the M2 forced-choice PMR labels. The vector's projection distribution
across samples tells us (a) how separable the direction is, and (b) which
layer carries the strongest "physics-mode" signal — the preferred target
for Phase 2 injection experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


@dataclass
class SteeringVector:
    layer: int
    v: np.ndarray          # (dim,) — raw difference-of-means
    v_unit: np.ndarray     # (dim,) — L2-normalized
    mean_pos: np.ndarray   # (dim,)
    mean_neg: np.ndarray   # (dim,)
    n_pos: int
    n_neg: int
    norm: float


def _load_lm_hidden(activations_dir: Path, sample_id: str, layer: int) -> np.ndarray:
    from safetensors.torch import load_file
    import torch

    data = load_file(str(activations_dir / f"{sample_id}.safetensors"))
    key = f"lm_hidden_{layer}"
    if key not in data:
        raise KeyError(f"{key} not in {sample_id}.safetensors")
    return data[key].to(dtype=torch.float32).numpy()


def _pool(tensor: np.ndarray) -> np.ndarray:
    return tensor.mean(axis=0)


def compute_steering_vectors(
    activations_dir: Path | str,
    predictions_path: Path | str,
    layers: Iterable[int],
    pmr_source: Literal["open", "forced_choice"] = "forced_choice",
) -> dict[int, SteeringVector]:
    """Mean-pooled `h | PMR=1` minus mean-pooled `h | PMR=0`, per layer."""
    activations_dir = Path(activations_dir)
    preds = pd.read_parquet(predictions_path)
    sub = preds[preds["prompt_variant"] == pmr_source] if pmr_source in (
        "open", "forced_choice"
    ) else preds
    per_sample = sub.groupby("sample_id")["pmr"].mean()
    y_bin = (per_sample >= 0.5).astype(int)

    vectors: dict[int, SteeringVector] = {}
    for li in layers:
        pos_acc: list[np.ndarray] = []
        neg_acc: list[np.ndarray] = []
        for sid, yi in y_bin.items():
            f = activations_dir / f"{sid}.safetensors"
            if not f.exists():
                continue
            h = _load_lm_hidden(activations_dir, sid, int(li))
            pooled = _pool(h)
            (pos_acc if yi == 1 else neg_acc).append(pooled)
        mu_pos = np.stack(pos_acc).mean(axis=0)
        mu_neg = np.stack(neg_acc).mean(axis=0)
        v = mu_pos - mu_neg
        norm = float(np.linalg.norm(v))
        v_unit = v / (norm + 1e-8)
        vectors[int(li)] = SteeringVector(
            layer=int(li),
            v=v.astype(np.float32),
            v_unit=v_unit.astype(np.float32),
            mean_pos=mu_pos.astype(np.float32),
            mean_neg=mu_neg.astype(np.float32),
            n_pos=len(pos_acc),
            n_neg=len(neg_acc),
            norm=norm,
        )
    return vectors


def project_onto_direction(
    activations_dir: Path | str,
    predictions_path: Path | str,
    layer: int,
    v_unit: np.ndarray,
) -> pd.DataFrame:
    """Per-sample projection of mean-pooled hidden state onto v_unit.

    Returns a DataFrame with columns sample_id, projection, plus factorial axes
    from predictions_scored.parquet. Useful for plotting histograms / boxplots
    split by object_level / cue_level.
    """
    activations_dir = Path(activations_dir)
    preds = pd.read_parquet(predictions_path)
    axes = preds[
        ["sample_id", "object_level", "bg_level", "cue_level", "event_template"]
    ].drop_duplicates("sample_id").reset_index(drop=True)

    rows: list[dict] = []
    for _, r in axes.iterrows():
        f = activations_dir / f"{r['sample_id']}.safetensors"
        if not f.exists():
            continue
        h = _load_lm_hidden(activations_dir, r["sample_id"], int(layer))
        pooled = _pool(h)
        proj = float(np.dot(pooled, v_unit))
        rows.append({**r.to_dict(), "projection": proj})
    return pd.DataFrame(rows)


def save_steering_vectors(
    vectors: dict[int, SteeringVector],
    out_path: Path,
) -> None:
    """Persist vectors to .npz keyed by layer."""
    import numpy as _np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    meta_rows: list[dict] = []
    for li, sv in sorted(vectors.items()):
        payload[f"v_{li}"] = sv.v
        payload[f"v_unit_{li}"] = sv.v_unit
        payload[f"mean_pos_{li}"] = sv.mean_pos
        payload[f"mean_neg_{li}"] = sv.mean_neg
        meta_rows.append({
            "layer": li, "norm": sv.norm, "n_pos": sv.n_pos, "n_neg": sv.n_neg,
            "dim": int(sv.v.shape[0]),
        })
    _np.savez(out_path, **payload)
    pd.DataFrame(meta_rows).to_csv(out_path.with_suffix(".csv"), index=False)


def load_steering_vectors(npz_path: Path) -> dict[int, np.ndarray]:
    """Load only the unit vectors — for use at intervention time."""
    data = np.load(npz_path)
    vs: dict[int, np.ndarray] = {}
    for k in data.files:
        if k.startswith("v_unit_"):
            li = int(k.split("_")[-1])
            vs[li] = data[k]
    return vs
