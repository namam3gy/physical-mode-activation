"""Extract per-model v_L from M8a captures (cleaner class balance than M2).

M2 captures had n_neg = 1 / 5 / 9 for InternVL3 / Idefics2 / LLaVA-Next —
too few negatives for clean class-mean diff. M8a captures have
n_neg ≈ 100-280 per model, much cleaner.

Reuses `physical_mode.probing.steering.compute_steering_vectors` with
`pmr_source="open"`. Saved at <run_dir>/probing_steering/steering_vectors.npz.

Usage:
    uv run python scripts/m8a_extract_per_model_steering.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_pmr
from physical_mode.probing.steering import compute_steering_vectors, save_steering_vectors


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LAYERS = (5, 10, 15, 20, 25)

MODEL_RUNS: dict[str, str] = {
    "LLaVA-Next": "encoder_swap_llava_next_m8a_capture_*",
    "Idefics2":   "encoder_swap_idefics2_m8a_capture_*",
    "InternVL3":  "encoder_swap_internvl3_m8a_capture_*",
}


def _ensure_pmr_in_predictions(run_dir: Path) -> Path:
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

        for li in LAYERS:
            v = vectors[li]
            print(f"  L{li}: dim={v.v.shape[0]}, ||v||={np.linalg.norm(v.v):.3f}, "
                  f"n_pos={v.n_pos}, n_neg={v.n_neg}")
        print(f"  → {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
