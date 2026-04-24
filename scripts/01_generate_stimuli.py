"""Generate PhysCue stimuli for a given config.

Usage:
    uv run python scripts/01_generate_stimuli.py --config configs/pilot.py [--limit N]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from physical_mode.stimuli.generate import generate_stimuli


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load config {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_run_config"] = mod
    spec.loader.exec_module(mod)
    return mod.CONFIG


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="truncate stimulus count for smokes")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    print(f"Factorial yields {cfg.factorial.total()} stimuli; effective {cfg.limit or cfg.factorial.total()}")
    generate_stimuli(cfg)


if __name__ == "__main__":
    main()
