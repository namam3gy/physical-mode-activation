"""Re-rank SAE features using Cohen's d (in addition to raw delta).

Uses the already-trained SAE + cached activations; no SAE retrain. Writes a
new `feature_ranking.csv` (with `cohens_d`, `pooled_std`, `std_phys`,
`std_abs` columns) and reports Spearman rank correlation between delta-rank
and Cohen's-d-rank on the top-50.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_rerank_features.py \
        --sae-dir outputs/sae/qwen_vis31_5120 \
        --activations-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations \
        --predictions outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv \
        --layer-key vision_hidden_31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from physical_mode.sae.train import load_sae
from physical_mode.sae.feature_id import rank_physics_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sae_train import split_phys_abstract  # noqa: E402  (reuse exact splitter)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sae-dir", type=Path, required=True)
    p.add_argument("--activations-dir", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--layer-key", default="vision_hidden_31")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--top-k-report", type=int, default=50,
                   help="top-k for Spearman comparison + console preview")
    p.add_argument("--output", type=Path, default=None,
                   help="output CSV path (default: <sae-dir>/feature_ranking.csv, "
                        "overwriting the original)")
    args = p.parse_args()

    sae_dir = args.sae_dir
    out_csv = args.output or (sae_dir / "feature_ranking.csv")

    sae = load_sae(sae_dir / "sae.pt", device=args.device)
    print(f"Loaded SAE: d_in={sae.d_in}, d_features={sae.d_features}")

    # Re-derive sample_ids from the activation directory in the same order as training.
    files = sorted(args.activations_dir.glob("*.safetensors"))
    sample_ids = [f.stem for f in files]
    print(f"  Found {len(sample_ids)} activation files.")

    phys_acts, abs_acts = split_phys_abstract(
        args.activations_dir, args.layer_key, sample_ids, args.predictions,
    )
    print(f"  Phys tokens: {phys_acts.shape}, Abs tokens: {abs_acts.shape}")

    ranked = rank_physics_features(sae, phys_acts, abs_acts, top_k=args.top_k_report)
    rank_df = pd.DataFrame({
        "feature_idx": list(range(sae.d_features)),
        "mean_phys": ranked["mean_phys"].numpy(),
        "mean_abs": ranked["mean_abs"].numpy(),
        "std_phys": ranked["std_phys"].numpy(),
        "std_abs": ranked["std_abs"].numpy(),
        "pooled_std": ranked["pooled_std"].numpy(),
        "delta": ranked["delta"].numpy(),
        "cohens_d": ranked["cohens_d"].numpy(),
    }).sort_values("delta", ascending=False).reset_index(drop=True)
    rank_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    print("\n=== Top 10 by delta ===")
    print(rank_df.head(10).to_string(index=False))
    print("\n=== Top 10 by Cohen's d ===")
    print(rank_df.sort_values("cohens_d", ascending=False).head(10).to_string(index=False))

    k = args.top_k_report
    top_delta = rank_df.head(k)["feature_idx"].tolist()
    top_cohen = rank_df.sort_values("cohens_d", ascending=False).head(k)["feature_idx"].tolist()
    overlap = set(top_delta) & set(top_cohen)
    print(f"\nTop-{k} overlap (set intersection): {len(overlap)} / {k}")

    # Spearman on the *full* ranking (rank for each feature under each metric).
    rho_full, _ = spearmanr(rank_df["delta"].values, rank_df["cohens_d"].values)
    print(f"Spearman ρ (delta vs Cohen's d, all 5120 features): {rho_full:.4f}")

    # Spearman on top-k subset by delta.
    sub = rank_df.head(k)
    rho_topk, _ = spearmanr(sub["delta"].values, sub["cohens_d"].values)
    print(f"Spearman ρ on top-{k} (by delta): {rho_topk:.4f}")

    # Top-20 explicit comparison (intervention-relevant).
    top20_delta = rank_df.head(20)["feature_idx"].tolist()
    top20_cohen = rank_df.sort_values("cohens_d", ascending=False).head(20)["feature_idx"].tolist()
    only_delta = [f for f in top20_delta if f not in top20_cohen]
    only_cohen = [f for f in top20_cohen if f not in top20_delta]
    print(f"\n=== Top-20 set comparison ===")
    print(f"  In top-20 delta but NOT in top-20 Cohen's d: {only_delta}")
    print(f"  In top-20 Cohen's d but NOT in top-20 delta: {only_cohen}")
    print(f"  Top-20 overlap: {20 - len(only_delta)} / 20")


if __name__ == "__main__":
    main()
