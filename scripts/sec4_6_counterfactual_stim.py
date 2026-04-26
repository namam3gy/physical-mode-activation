"""§4.6 — VTI-reverse counterfactual stimulus generation driver.

Runs `gradient_ascent` on each baseline stim × each (mode, eps)
configuration. Saves: per-seed reconstructed PNGs, per-config delta
tensors, projection trajectories, and a config manifest. Inference
re-evaluation (PMR pre/post) is delegated to scripts/sec4_6_summarize.py.

Usage:
    uv run python scripts/sec4_6_counterfactual_stim.py
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
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--steering-npz", type=Path, default=DEFAULT_STEERING_NPZ)
    p.add_argument("--steering-key", default="v_unit_10")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps-list", default="0.05,0.1,0.2",
                   help="comma-separated L_inf bounds for bounded mode")
    p.add_argument("--include-unconstrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-random-controls", type=int, default=3,
                   help="number of random-direction controls (eps=0.1 fixed)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="if None, autogenerates outputs/sec4_6_counterfactual_<ts>")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random-direction controls")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = PROJECT_ROOT / f"outputs/sec4_6_counterfactual_{ts}"
    if not args.output_dir.is_absolute():
        args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_id} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    v_unit = np.load(args.steering_npz)[args.steering_key]
    print(f"Steering: {args.steering_key} shape={v_unit.shape} norm={np.linalg.norm(v_unit):.6f}")

    rng = np.random.default_rng(args.seed)
    random_dirs = []
    for i in range(args.n_random_controls):
        r = rng.standard_normal(v_unit.shape).astype(np.float32)
        r /= np.linalg.norm(r) + 1e-8
        random_dirs.append((f"v_random_{i}", r))

    baselines = sorted(args.baseline_dir.glob(args.baseline_pattern))[: args.n_seeds]
    print(f"Baselines ({len(baselines)}): {[b.name for b in baselines]}")

    eps_list = [float(x) for x in args.eps_list.split(",")]
    configs: list[dict] = []
    for eps in eps_list:
        configs.append({"name": f"bounded_eps{eps}", "mode": "bounded", "eps": eps,
                        "v": ("v_unit_10", v_unit)})
    if args.include_unconstrained:
        configs.append({"name": "unconstrained", "mode": "unconstrained", "eps": None,
                        "v": ("v_unit_10", v_unit)})
    for r_name, r_vec in random_dirs:
        configs.append({"name": f"control_{r_name}", "mode": "bounded", "eps": 0.1,
                        "v": (r_name, r_vec)})

    print(f"Configs ({len(configs)}): {[c['name'] for c in configs]}")
    total_runs = len(baselines) * len(configs)
    print(f"Total runs: {total_runs}")

    manifest_rows = []
    t_start = time.time()
    for i, stim_path in enumerate(baselines):
        sid = stim_path.stem
        pil = Image.open(stim_path).convert("RGB")
        for j, cfg in enumerate(configs):
            run_idx = i * len(configs) + j + 1
            v_name, v_arr = cfg["v"]
            elapsed = time.time() - t_start
            eta = (elapsed / max(run_idx - 1, 1)) * (total_runs - run_idx + 1) if run_idx > 1 else None
            eta_str = f" eta={eta/60:.1f}min" if eta else ""
            print(f"  [{run_idx}/{total_runs}] {sid} | {cfg['name']} (v={v_name}, "
                  f"mode={cfg['mode']}, eps={cfg['eps']}) elapsed={elapsed/60:.1f}min{eta_str}")
            out = gradient_ascent(
                model, processor, pil, v_arr,
                layer=args.layer, n_steps=args.n_steps, lr=args.lr,
                eps=cfg["eps"], mode=cfg["mode"], prompt=args.prompt,
                log_every=20,
            )

            cfg_dir = args.output_dir / cfg["name"] / sid
            cfg_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "pixel_values_initial": out["pixel_values_initial"],
                "pixel_values_final": out["pixel_values_final"],
            }, cfg_dir / "pixel_values.pt")
            traj_arr = np.array(out["projection_trajectory"], dtype=np.float64)
            np.save(cfg_dir / "trajectory.npy", traj_arr)
            _, grid_thw = pixel_values_from_pil(pil, processor, args.prompt)
            recon = reconstruct_pil(out["pixel_values_final"], grid_thw, processor)
            recon.save(cfg_dir / "synthesized.png")
            pil.save(cfg_dir / "baseline.png")

            manifest_rows.append({
                "sample_id": sid,
                "config_name": cfg["name"],
                "v_name": v_name,
                "mode": cfg["mode"],
                "eps": cfg["eps"],
                "layer": args.layer,
                "n_steps": args.n_steps,
                "baseline_projection": out["baseline_projection"],
                "final_projection": out["final_projection"],
                "synthesized_path": str((cfg_dir / "synthesized.png").relative_to(PROJECT_ROOT)),
                "trajectory_path": str((cfg_dir / "trajectory.npy").relative_to(PROJECT_ROOT)),
            })

    manifest_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    manifest = {"args": manifest_args, "rows": manifest_rows}
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    elapsed_total = time.time() - t_start
    print(f"\nDone. Manifest at {args.output_dir / 'manifest.json'}")
    print(f"Total elapsed: {elapsed_total/60:.1f} min")


if __name__ == "__main__":
    main()
