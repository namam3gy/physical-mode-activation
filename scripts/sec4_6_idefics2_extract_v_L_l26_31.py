"""Extract v_L at L26/28/30/31 from the Idefics2 deeper-layer capture.

Mirrors `m2_extract_per_model_steering.py` but targets the new
`cross_model_idefics2_capture_l26_31_*` capture run, whose
LM hidden states cover L26/28/30/31 (≥ 81 % of Mistral-7B's 32
layers). Required for §4.6 Idefics2 deeper-layer sweep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from physical_mode.metrics.pmr import score_pmr
from physical_mode.probing.steering import compute_steering_vectors, save_steering_vectors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYERS = (26, 28, 30, 31)
RUN_PATTERN = "cross_model_idefics2_capture_l26_31_*"


def main() -> None:
    matches = sorted((PROJECT_ROOT / "outputs").glob(RUN_PATTERN))
    matches = [m for m in matches if (m / "predictions.jsonl").exists()
               and (m / "activations").exists()]
    if not matches:
        raise FileNotFoundError(f"No capture run matches {RUN_PATTERN}")
    run_dir = matches[-1]
    print(f"run_dir: {run_dir.name}")

    jsonl = run_dir / "predictions.jsonl"
    parquet = run_dir / "predictions_with_pmr.parquet"
    if not parquet.exists() or parquet.stat().st_mtime < jsonl.stat().st_mtime:
        df = pd.read_json(jsonl, lines=True)
        df["pmr"] = df["raw_text"].apply(score_pmr)
        df.to_parquet(parquet, index=False)
        print(f"  scored predictions → {parquet.name}")

    out_dir = run_dir / "probing_steering"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "steering_vectors.npz"

    vectors = compute_steering_vectors(
        activations_dir=run_dir / "activations",
        predictions_path=parquet,
        layers=LAYERS,
        pmr_source="open",
    )
    save_steering_vectors(vectors, out_path)
    for li in LAYERS:
        v = vectors[li]
        print(f"  L{li}: dim={v.v.shape[0]}, ||v||={np.linalg.norm(v.v):.3f}, "
              f"n_pos={v.n_pos}, n_neg={v.n_neg}")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
