"""Train a sparse autoencoder on Qwen vision-encoder activations.

Inputs:
- `outputs/<run>/vision_activations/*.safetensors` — per-stim, multi-layer.
- We use `vision_hidden_31` (last vision encoder layer, pre-projection,
  shape (n_visual_tokens, 1280) for each stim).

Outputs:
- `outputs/sae/<tag>/sae.pt` — trained SAE state dict.
- `outputs/sae/<tag>/metrics.json` — training metrics over time.
- `outputs/sae/<tag>/feature_ranking.csv` — features ranked by physics-vs-abstract delta.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_train.py \
        --activations-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations \
        --predictions outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv \
        --layer-key vision_hidden_31 --n-features 5120 --tag qwen_vis31_5120
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import safetensors.torch as st
import torch

from physical_mode.sae.train import TrainConfig, train_sae, save_sae
from physical_mode.sae.feature_id import rank_physics_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_all_activations(activations_dir: Path, layer_key: str) -> tuple[torch.Tensor, list[str]]:
    """Concatenate (n_visual_tokens × n_stim, d_in) activations across all stim.

    Supports both 2D shape (single-tile: n_tokens, d_in) and 3D shape
    (multi-tile: n_tiles, n_tokens, d_in) — multi-tile is flattened
    to (n_tiles * n_tokens, d_in) per stim before concat.
    """
    files = sorted(activations_dir.glob("*.safetensors"))
    print(f"Loading {len(files)} activation files for {layer_key}...")
    all_acts = []
    sample_ids = []
    for f in files:
        d = st.load_file(f)
        if layer_key not in d:
            continue
        a = d[layer_key]
        if a.dim() == 3:
            a = a.reshape(-1, a.shape[-1])
        all_acts.append(a)
        sample_ids.append(f.stem)
    out = torch.cat(all_acts, dim=0).float()
    print(f"  → {out.shape[0]:,} tokens × {out.shape[1]} dim ({len(sample_ids)} stim).")
    return out, sample_ids


def split_phys_abstract(
    activations_dir: Path, layer_key: str,
    sample_ids: list[str], predictions_csv: Path,
    pmr_phys_threshold: float = 0.667,
    pmr_abs_threshold: float = 0.333,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-sample mean PMR; return concat of activations from
    physics-mode (mean_pmr ≥ pmr_phys_threshold) and abstract (≤ pmr_abs_threshold) stim."""
    if predictions_csv.suffix == ".parquet":
        df = pd.read_parquet(predictions_csv)
    else:
        df = pd.read_csv(predictions_csv)
    sample_pmr = df.groupby("sample_id")["pmr"].mean()
    phys_ids = set(sample_pmr[sample_pmr >= pmr_phys_threshold].index)
    abs_ids = set(sample_pmr[sample_pmr <= pmr_abs_threshold].index)

    phys_acts = []
    abs_acts = []
    for sid in sample_ids:
        f = activations_dir / f"{sid}.safetensors"
        if not f.exists():
            continue
        d = st.load_file(f)
        if layer_key not in d:
            continue
        a = d[layer_key].float()
        if a.dim() == 3:
            a = a.reshape(-1, a.shape[-1])
        if sid in phys_ids:
            phys_acts.append(a)
        elif sid in abs_ids:
            abs_acts.append(a)
    print(f"  Physics-mode: {len(phys_acts)} stim; Abstract: {len(abs_acts)} stim.")
    if not phys_acts or not abs_acts:
        raise RuntimeError("Empty phys or abs set — check thresholds.")
    return torch.cat(phys_acts, dim=0), torch.cat(abs_acts, dim=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--layer-key", default="vision_hidden_31")
    p.add_argument("--n-features", type=int, default=5120,
                   help="SAE feature count (4x expansion for d=1280)")
    p.add_argument("--pmr-phys-threshold", type=float, default=0.667,
                   help="per-stim mean PMR threshold to label as physics-mode")
    p.add_argument("--pmr-abs-threshold", type=float, default=0.333,
                   help="per-stim mean PMR threshold to label as abstract; "
                   "loosen to 0.5 for saturated models (LLaVA-Next / Idefics2)")
    p.add_argument("--n-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l1-lambda", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="qwen_vis31_5120")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / "outputs" / "sae" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    activations, sample_ids = load_all_activations(args.activations_dir, args.layer_key)
    cfg = TrainConfig(
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        l1_lambda=args.l1_lambda, device=args.device,
    )

    t0 = time.time()
    sae, metrics = train_sae(activations, d_features=args.n_features, cfg=cfg)
    elapsed = (time.time() - t0) / 60
    print(f"Trained in {elapsed:.1f} min.")

    save_sae(sae, out_dir / "sae.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Identifying physics-cue features...")
    phys_acts, abs_acts = split_phys_abstract(
        args.activations_dir, args.layer_key, sample_ids, args.predictions,
        pmr_phys_threshold=args.pmr_phys_threshold,
        pmr_abs_threshold=args.pmr_abs_threshold,
    )
    ranked = rank_physics_features(sae, phys_acts, abs_acts, top_k=50)
    rank_df = pd.DataFrame({
        "feature_idx": list(range(sae.d_features)),
        "mean_phys": ranked["mean_phys"].numpy(),
        "mean_abs": ranked["mean_abs"].numpy(),
        "std_phys": ranked["std_phys"].numpy(),
        "std_abs": ranked["std_abs"].numpy(),
        "pooled_std": ranked["pooled_std"].numpy(),
        "delta": ranked["delta"].numpy(),
        "cohens_d": ranked["cohens_d"].numpy(),
    }).sort_values("delta", ascending=False)
    rank_df.to_csv(out_dir / "feature_ranking.csv", index=False)
    print("\n=== Top 10 physics-cue features (by delta) ===")
    print(rank_df.head(10).to_string(index=False))
    print("\n=== Top 10 physics-cue features (by Cohen's d) ===")
    print(rank_df.sort_values("cohens_d", ascending=False).head(10).to_string(index=False))

    print(f"\nWrote SAE + metrics + ranking → {out_dir}")


if __name__ == "__main__":
    main()
