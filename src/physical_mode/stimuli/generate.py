"""Stimulus generation driver. Writes images + manifest.parquet for a given config."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import EvalConfig
from ..utils import config_hash, ensure_dir, timestamp
from .scenes import render_scene


def generate_stimuli(cfg: EvalConfig) -> Path:
    """Render all stimuli implied by cfg.factorial and write to inputs/<run_id>/.

    Returns the run directory path.
    """
    run_id = f"{cfg.run_name}_{timestamp()}_{config_hash(cfg.factorial)}"
    run_dir = ensure_dir(cfg.inputs_root / run_id)
    img_dir = ensure_dir(run_dir / "images")

    rows = list(cfg.factorial.iter())
    if cfg.limit is not None:
        rows = rows[: cfg.limit]

    manifest_records: list[dict] = []
    for row in tqdm(rows, desc=f"Rendering {len(rows)} stimuli"):
        img = render_scene(row, size=cfg.image_size)
        path = img_dir / f"{row.sample_id}.png"
        img.save(path, format="PNG", optimize=False)
        rec = asdict(row)
        rec["image_path"] = str(path.relative_to(run_dir))
        manifest_records.append(rec)

    manifest = pd.DataFrame(manifest_records)
    manifest.to_parquet(run_dir / "manifest.parquet", index=False)
    manifest.to_csv(run_dir / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} stimuli to {run_dir}")
    return run_dir
