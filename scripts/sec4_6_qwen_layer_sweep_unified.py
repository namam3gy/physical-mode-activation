"""§4.6 — Qwen2.5-VL minimal layer sweep, single-process (load model once).

Tests whether layers other than L10 also admit pixel-encodability of physics-
mode commitment. Loads model once and iterates over (layer × stim × config),
avoiding per-layer model-load and process-startup overhead.

Per layer: n_seeds (default 3) × {bounded eps=0.1 v_unit_<L>, control random
eps=0.1} = 6 runs/layer. At ~30-180 sec/run depending on layer.

Output: outputs/sec4_6_qwen_layer_sweep_unified_<ts>/<layer>/<config>/<sid>/
        + manifest.json at top level.

Usage:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_qwen_layer_sweep_unified.py \
        --layers 5,15,20 --n-seeds 3 --eps 0.1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.synthesis.counterfactual import (
    gradient_ascent,
    pixel_values_from_pil,
    reconstruct_pil,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"
DEFAULT_STEERING_NPZ = PROJECT_ROOT / "outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--baseline-pattern", default="line_blank_none_fall_*.png")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--steering-npz", type=Path, default=DEFAULT_STEERING_NPZ)
    p.add_argument("--layers", default="5,15,20",
                   help="comma-separated layer indices to sweep (each must have v_unit_<L> in --steering-npz)")
    p.add_argument("--eps", type=float, default=0.1, help="single L_inf eps used for both v_unit and random control")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = PROJECT_ROOT / f"outputs/sec4_6_qwen_layer_sweep_unified_{ts}"
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]
    print(f"Layer sweep: {layers}")

    steering_npz = np.load(args.steering_npz)
    layer_v_units: dict[int, np.ndarray] = {}
    for L in layers:
        key = f"v_unit_{L}"
        if key not in steering_npz:
            raise SystemExit(f"steering_npz missing {key}; available: {list(steering_npz.keys())}")
        layer_v_units[L] = steering_npz[key]
        print(f"  v_unit_{L}: shape={layer_v_units[L].shape} norm={np.linalg.norm(layer_v_units[L]):.6f}")

    rng = np.random.default_rng(args.seed)
    # One random direction per layer (different seed per layer for independence).
    layer_random: dict[int, np.ndarray] = {}
    for L in layers:
        r = rng.standard_normal(layer_v_units[L].shape).astype(np.float32)
        r /= np.linalg.norm(r) + 1e-8
        layer_random[L] = r

    baselines = sorted(args.baseline_dir.glob(args.baseline_pattern))[: args.n_seeds]
    print(f"Baselines ({len(baselines)}): {[b.name for b in baselines]}")

    print(f"Loading model {args.model_id} on {args.device}...")
    t_load = time.time()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    print(f"  Model loaded in {(time.time()-t_load):.1f} sec.")

    n_per_layer = len(baselines) * 2  # 2 conditions per stim
    total = len(layers) * n_per_layer
    print(f"Total runs: {total}")

    manifest_rows: list[dict] = []
    t_start = time.time()
    run_idx = 0
    for L in layers:
        v_unit = layer_v_units[L]
        v_random = layer_random[L]
        configs = [
            {"name": f"L{L}_v_unit_eps{args.eps}", "v_name": f"v_unit_{L}", "v": v_unit, "eps": args.eps},
            {"name": f"L{L}_random_eps{args.eps}", "v_name": f"v_random_L{L}", "v": v_random, "eps": args.eps},
        ]
        for stim_path in baselines:
            sid = stim_path.stem
            pil = Image.open(stim_path).convert("RGB")
            for cfg in configs:
                run_idx += 1
                elapsed = time.time() - t_start
                eta = (elapsed / max(run_idx - 1, 1)) * (total - run_idx + 1) if run_idx > 1 else None
                eta_str = f" eta={eta/60:.1f}min" if eta else ""
                print(f"  [{run_idx}/{total}] L{L} {sid} | {cfg['name']} elapsed={elapsed/60:.1f}min{eta_str}")
                out = gradient_ascent(
                    model, processor, pil, cfg["v"],
                    layer=L, n_steps=args.n_steps, lr=args.lr,
                    eps=cfg["eps"], mode="bounded", prompt=args.prompt,
                    log_every=50,
                )
                cfg_dir = args.output_dir / f"L{L}" / cfg["name"] / sid
                cfg_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "pixel_values_initial": out["pixel_values_initial"],
                    "pixel_values_final": out["pixel_values_final"],
                }, cfg_dir / "pixel_values.pt")
                np.save(cfg_dir / "trajectory.npy", np.array(out["projection_trajectory"], dtype=np.float64))
                _, grid_thw = pixel_values_from_pil(pil, processor, args.prompt)
                recon = reconstruct_pil(out["pixel_values_final"], grid_thw, processor)
                recon.save(cfg_dir / "synthesized.png")
                pil.save(cfg_dir / "baseline.png")
                manifest_rows.append({
                    "sample_id": sid, "config_name": cfg["name"], "v_name": cfg["v_name"],
                    "layer": L, "eps": cfg["eps"], "mode": "bounded", "n_steps": args.n_steps,
                    "baseline_projection": out["baseline_projection"],
                    "final_projection": out["final_projection"],
                    "synthesized_path": str((cfg_dir / "synthesized.png").relative_to(PROJECT_ROOT)),
                    "trajectory_path": str((cfg_dir / "trajectory.npy").relative_to(PROJECT_ROOT)),
                })

    manifest_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    manifest = {"args": manifest_args, "rows": manifest_rows}
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nDone in {(time.time()-t_start)/60:.1f} min. Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
