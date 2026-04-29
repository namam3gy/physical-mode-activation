"""Extract per-model v_L (5, 10, 15, 20, 25) from each cross-model M2 capture run.

Reuses `physical_mode.probing.steering.compute_steering_vectors` with
`pmr_source="open"` (cross-model M2 runs only have the open prompt).
Each model's saved `steering_vectors.npz` lands under its own run dir.

Required inputs (per-run, glob latest):
- outputs/cross_model_llava_capture_*/      (LLaVA-1.5 — M6 r2b)
- outputs/cross_model_llava_next_capture_*/ (LLaVA-Next — this milestone)
- outputs/cross_model_idefics2_capture_*/   (Idefics2 — this milestone)
- outputs/cross_model_internvl3_capture_*/  (InternVL3 — this milestone)
- outputs/mvp_full_*/probing_steering/      (Qwen — already exists, M5a)

Output: <run_dir>/probing_steering/steering_vectors.npz with v_5, v_10,
v_15, v_20, v_25 + their unit-norm versions.

Usage:
    uv run python scripts/m2_extract_per_model_steering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_pmr
from physical_mode.probing.steering import compute_steering_vectors, save_steering_vectors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAYERS = (5, 10, 15, 20, 25)

# Model → run-dir glob pattern. Latest match wins.
MODEL_RUNS: dict[str, str] = {
    "LLaVA-1.5":  "cross_model_llava_capture_*",
    "LLaVA-Next": "cross_model_llava_next_capture_*",
    "Idefics2":   "cross_model_idefics2_capture_*",
    "InternVL3":  "cross_model_internvl3_capture_*",
    # M-PSwap: Idefics2-MLP-pool variant (Pillar B controlled counterfactual).
    # Skipped automatically by _latest() if no run exists yet.
    "Idefics2-MPSwap": "cross_model_idefics2_mpswap_capture_*",
}


def _ensure_pmr_in_predictions(run_dir: Path) -> Path:
    """Score PMR per row and write a parquet alongside the predictions.jsonl.

    `compute_steering_vectors` reads parquet; we score open-prompt PMR
    here so it's available as `pmr` column.
    """
    jsonl = run_dir / "predictions.jsonl"
    parquet = run_dir / "predictions_with_pmr.parquet"
    if parquet.exists() and parquet.stat().st_mtime >= jsonl.stat().st_mtime:
        return parquet
    df = pd.read_json(jsonl, lines=True)
    df["pmr"] = df["raw_text"].apply(score_pmr)
    df.to_parquet(parquet, index=False)
    return parquet


def _latest(pattern: str) -> Path:
    matches = sorted(PROJECT_ROOT.glob(f"outputs/{pattern}"))
    matches = [m for m in matches if (m / "predictions.jsonl").exists()
               and (m / "activations").exists()]
    if not matches:
        raise FileNotFoundError(f"No capture run matches {pattern}")
    return matches[-1]


def main() -> None:
    for model, pat in MODEL_RUNS.items():
        try:
            run_dir = _latest(pat)
        except FileNotFoundError as e:
            print(f"[SKIP] {model}: {e}")
            continue

        print(f"[{model}] run_dir: {run_dir.name}")
        preds_parquet = _ensure_pmr_in_predictions(run_dir)

        out_dir = run_dir / "probing_steering"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "steering_vectors.npz"

        vectors = compute_steering_vectors(
            activations_dir=run_dir / "activations",
            predictions_path=preds_parquet,
            layers=LAYERS,
            pmr_source="open",
        )
        save_steering_vectors(vectors, out_path)

        # Print summary
        for li in LAYERS:
            v = vectors[li]
            print(f"  L{li}: dim={v.v.shape[0]}, ||v||={np.linalg.norm(v.v):.3f}, "
                  f"n_pos={v.n_pos}, n_neg={v.n_neg}")
        print(f"  → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
