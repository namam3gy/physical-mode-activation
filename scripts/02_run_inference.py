"""Run VLM inference on the most recent stimulus run matching a config.

Usage:
    uv run python scripts/02_run_inference.py --config configs/pilot.py [--stimulus-dir <path>] [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from physical_mode.inference.run import run_inference

from importlib.util import module_from_spec, spec_from_file_location
import sys


def load_config(path: Path):
    spec = spec_from_file_location("_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config {path}")
    mod = module_from_spec(spec)
    sys.modules["_run_config"] = mod
    spec.loader.exec_module(mod)
    return mod.CONFIG


def _latest_stimulus_dir(inputs_root: Path, run_name: str) -> Path:
    candidates = sorted(inputs_root.glob(f"{run_name}_*"))
    if not candidates:
        raise FileNotFoundError(f"No stimulus dir matching {run_name}_* under {inputs_root}")
    return candidates[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--stimulus-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    stim_dir = args.stimulus_dir or _latest_stimulus_dir(cfg.inputs_root, cfg.run_name)
    print(f"Using stimulus dir: {stim_dir}")
    run_inference(cfg, stim_dir)


if __name__ == "__main__":
    main()
