"""Sub-task 3 — LM backbone logit lens and layer-wise probes.

Two analyses on M2-captured LM hidden states at visual-token positions:

1. **Logit lens** — apply `model.lm_head` to hidden states at each captured
   layer. Track how the logits for a curated set of "physics" vs "geometry"
   tokens evolve through the LM. The layer where physics-verb logits first
   dominate geometry-noun logits is the "switching layer" per Neo et al. 2024.

2. **Per-layer PMR probe** — sklearn LogisticRegression on mean-pooled LM
   hidden states (visual token positions), PMR-binary target. Complements the
   vision-encoder probe (M3) to show where the gap "opens" inside the LM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch


# Tokens we care about for the logit lens. Each is matched to the first
# sub-token id the tokenizer emits for the *lowercase* string. Multi-token
# words fall back to their first sub-token — Qwen tokenizer typically splits
# common English words into 1-2 sub-tokens.
PHYSICS_TOKENS: tuple[str, ...] = (
    "fall", "falls", "falling", "drop", "drops", "roll", "rolls", "rolling",
    "bounce", "slide", "land", "tumble", "move", "moving", "orbit",
)
GEOMETRY_TOKENS: tuple[str, ...] = (
    "circle", "shape", "line", "drawing", "image", "figure", "abstract",
    "geometric", "still", "static",
)
LABEL_TOKENS: tuple[str, ...] = (
    "ball", "planet", "object",
)


@dataclass
class TokenIds:
    physics: dict[str, int]
    geometry: dict[str, int]
    label: dict[str, int]

    def all_ids(self) -> list[int]:
        out: list[int] = []
        for d in (self.physics, self.geometry, self.label):
            out.extend(d.values())
        return out


def resolve_token_ids(tokenizer) -> TokenIds:
    """Map each token string to a single integer id (first sub-token).

    For a word with leading space variant, prefer the space-prefixed form
    (Qwen/SentencePiece tokenizers often use `▁word` for a natural-position
    word). Silently drops any string that tokenizes to multiple sub-tokens
    of which none is unique.
    """
    def _first_id(s: str) -> int | None:
        for cand in (f" {s}", s):
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        ids = tokenizer.encode(s, add_special_tokens=False)
        return int(ids[0]) if ids else None

    def _build(names: tuple[str, ...]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in names:
            tid = _first_id(n)
            if tid is not None:
                out[n] = tid
        return out

    return TokenIds(
        physics=_build(PHYSICS_TOKENS),
        geometry=_build(GEOMETRY_TOKENS),
        label=_build(LABEL_TOKENS),
    )


def _load_lm_hidden(run_dir: Path, sample_id: str, layer: int) -> np.ndarray:
    from safetensors.torch import load_file

    data = load_file(str(run_dir / f"{sample_id}.safetensors"))
    key = f"lm_hidden_{layer}"
    if key not in data:
        raise KeyError(f"{key} not in {sample_id}.safetensors; keys: {list(data)}")
    return data[key].to(dtype=torch.float32).numpy()


# ---------------------------------------------------------------------------
# Logit lens
# ---------------------------------------------------------------------------


def logit_lens_layer(
    lm_head: torch.nn.Linear,
    hidden: np.ndarray,
    token_ids: list[int],
    pool: Literal["mean", "last"] = "mean",
) -> np.ndarray:
    """Apply lm_head to (n_tokens, dim) hidden state; return logits for token_ids.

    Returns a 1-D array of length len(token_ids) (mean- or last-pooled across
    the visual token axis before applying lm_head).
    """
    h = torch.from_numpy(hidden)  # (n_tokens, dim), float32
    if pool == "mean":
        h = h.mean(dim=0, keepdim=True)  # (1, dim)
    elif pool == "last":
        h = h[-1:]
    else:
        raise ValueError(pool)
    dev = next(lm_head.parameters()).device
    h = h.to(dev, dtype=next(lm_head.parameters()).dtype)
    with torch.inference_mode():
        logits = lm_head(h)[0].float().cpu().numpy()  # (vocab,)
    return logits[token_ids]


def run_logit_lens_trajectories(
    activations_dir: Path,
    sample_ids: list[str],
    lm_head: torch.nn.Linear,
    token_ids: TokenIds,
    layers: Iterable[int],
    pool: Literal["mean", "last"] = "mean",
) -> pd.DataFrame:
    """For each (sample, layer), compute logits for all tokens in token_ids.

    Returns a long-form DataFrame with columns:
      sample_id, layer, token, category, logit
    """
    layers = tuple(int(x) for x in layers)
    all_tokens: list[tuple[str, str, int]] = []
    for t, tid in token_ids.physics.items():
        all_tokens.append((t, "physics", tid))
    for t, tid in token_ids.geometry.items():
        all_tokens.append((t, "geometry", tid))
    for t, tid in token_ids.label.items():
        all_tokens.append((t, "label", tid))
    ids = [tid for _, _, tid in all_tokens]

    records: list[dict] = []
    for i, sid in enumerate(sample_ids):
        if i % 50 == 0:
            print(f"  logit-lens: {i}/{len(sample_ids)}")
        for li in layers:
            hidden = _load_lm_hidden(activations_dir, sid, li)
            logits = logit_lens_layer(lm_head, hidden, ids, pool=pool)
            for (tok, cat, _), lo in zip(all_tokens, logits):
                records.append(
                    {"sample_id": sid, "layer": int(li), "token": tok,
                     "category": cat, "logit": float(lo)}
                )
    return pd.DataFrame(records)


def switching_layer_per_sample(
    trajectories: pd.DataFrame,
    physics_tokens: Iterable[str] | None = None,
    geometry_tokens: Iterable[str] | None = None,
    aggregator: Literal["max", "mean"] = "max",
) -> pd.DataFrame:
    """For each sample, find the *smallest* layer where physics logit ≥ geometry logit.

    aggregator: how to collapse multiple tokens into a category score
      (max = most-likely physics-word; mean = mean across physics words).
    Returns DataFrame with columns sample_id, switching_layer (-1 if never).
    """
    df = trajectories.copy()
    phys_mask = df["category"] == "physics"
    geom_mask = df["category"] == "geometry"
    if physics_tokens is not None:
        phys_mask &= df["token"].isin(physics_tokens)
    if geometry_tokens is not None:
        geom_mask &= df["token"].isin(geometry_tokens)

    agg_fn = np.max if aggregator == "max" else np.mean
    per = (
        df[phys_mask | geom_mask]
        .assign(
            cat2=lambda f: np.where(f["category"] == "physics", "phys", "geom"),
        )
        .pivot_table(index=["sample_id", "layer"], columns="cat2",
                     values="logit", aggfunc=agg_fn)
        .reset_index()
    )

    out = []
    for sid, sub in per.groupby("sample_id"):
        sub = sub.sort_values("layer")
        crossed = sub[sub["phys"] >= sub["geom"]]
        sw = int(crossed["layer"].min()) if len(crossed) else -1
        out.append({"sample_id": sid, "switching_layer": sw})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Per-layer PMR probe (parallel to probing/vision.py)
# ---------------------------------------------------------------------------


def load_lm_probing_dataset(
    activations_dir: Path | str,
    predictions_path: Path | str,
    layers: Iterable[int],
    pmr_source: str = "forced_choice",
):
    """Load per-layer LM hidden states and per-sample PMR binary label.

    ``pmr_source`` is matched against the ``prompt_variant`` column — any
    string the config uses works (``"open"``, ``"forced_choice"``,
    ``"open_no_label"``, ...). Pass an empty string to use all rows.
    """
    activations_dir = Path(activations_dir)
    preds = pd.read_parquet(predictions_path)
    if pmr_source:
        sub = preds[preds["prompt_variant"] == pmr_source]
        if sub.empty:
            raise ValueError(
                f"No predictions rows with prompt_variant={pmr_source!r}. "
                f"Available variants: {sorted(preds['prompt_variant'].unique())}"
            )
    else:
        sub = preds
    per_sample = sub.groupby("sample_id")["pmr"].mean()
    y_bin = (per_sample >= 0.5).astype(int).rename("y").reset_index()
    axes = preds[["sample_id", "object_level", "bg_level", "cue_level"]].drop_duplicates("sample_id")
    meta = axes.merge(y_bin, on="sample_id")

    from safetensors.torch import load_file

    rows_by_layer: dict[int, list[np.ndarray]] = {int(li): [] for li in layers}
    kept_ids: list[str] = []
    meta_kept_rows: list[pd.Series] = []
    for _, row in meta.iterrows():
        f = activations_dir / f"{row['sample_id']}.safetensors"
        if not f.exists():
            continue
        data = load_file(str(f))
        if not all(f"lm_hidden_{li}" in data for li in layers):
            continue
        kept_ids.append(row["sample_id"])
        meta_kept_rows.append(row)
        for li in layers:
            h = data[f"lm_hidden_{li}"].to(dtype=torch.float32).numpy()
            rows_by_layer[int(li)].append(h.mean(axis=0))

    X_per_layer = {li: np.stack(vs) for li, vs in rows_by_layer.items()}
    y = np.array([r["y"] for r in meta_kept_rows], dtype=np.int64)
    meta_df = pd.DataFrame(meta_kept_rows).reset_index(drop=True)
    print(f"LM probing dataset: {len(kept_ids)} samples (pmr_source={pmr_source}).")
    print(f"  y=1: {int(y.sum())} / {len(y)}")
    return X_per_layer, y, meta_df


def run_lm_layer_sweep(
    X_per_layer: dict[int, np.ndarray],
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Same protocol as probing/vision.py — sklearn LogisticRegression, stratified CV."""
    from .vision import train_layer_probe

    rows = []
    for li, X in sorted(X_per_layer.items()):
        r = train_layer_probe(X, y, n_splits=n_splits, seed=seed)
        rows.append({
            "layer": li, "auc_mean": r.auc_mean, "auc_std": r.auc_std,
            "accuracy_mean": r.accuracy_mean, "n_pos": r.n_pos, "n_neg": r.n_neg,
        })
    return pd.DataFrame(rows)
