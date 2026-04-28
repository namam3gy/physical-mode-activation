"""M5 Phase 2 — VTI-style residual-stream injection.

Load the per-layer "physics-mode direction" derived by
`physical_mode.probing.steering` and inject alpha * v at the output of
an LM decoder layer via a forward hook. For each (test stimulus, layer,
alpha) generate a forced-choice response and score PMR. Compare flip
rates against the unmodified baseline (alpha=0).

Usage:
    uv run python scripts/06_vti_steering.py \
        --run-dir outputs/mvp_full_<ts>_<hash> \
        --stimulus-dir inputs/mvp_full_<ts>_<hash> \
        --test-subset line/blank/none \
        --layers 10,15,20,25 \
        --alphas 0,5,10,20,40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from physical_mode.inference.prompts import render as render_prompt
from physical_mode.metrics.first_letter import extract_first_letter
from physical_mode.metrics.pmr import score_rows
from physical_mode.models.vlm_runner import InferenceArgs, PhysModeVLM
from physical_mode.probing.steering import load_steering_vectors


def _resolve_lm_layers(model) -> list:
    inner = getattr(model, "model", model)
    for attr in ("language_model", "text_model"):
        lm = getattr(inner, attr, None)
        if lm is not None and hasattr(lm, "layers"):
            return lm.layers
    raise RuntimeError("could not find language_model/text_model submodule with .layers")


def make_hook(alpha: float, v: torch.Tensor):
    """Forward hook that adds alpha * v to the hidden-states output of a decoder layer."""
    v_cached = {}

    def hook(_module, _inputs, output):
        if not alpha:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        key = (hs.device, hs.dtype)
        if key not in v_cached:
            v_cached[key] = v.to(device=hs.device, dtype=hs.dtype)
        new_hs = hs + alpha * v_cached[key]
        if isinstance(output, tuple):
            return (new_hs,) + output[1:]
        return new_hs

    return hook


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="M2 output dir containing probing_steering/steering_vectors.npz")
    p.add_argument("--stimulus-dir", type=Path, required=True)
    p.add_argument("--test-subset", default="line/blank/none",
                   help="obj/bg/cue triple defining which stimuli to steer")
    p.add_argument("--label", default="circle",
                   help="prompt label — 'circle' keeps baseline PMR low, maximizing room to flip")
    p.add_argument("--prompt-variant", default="forced_choice")
    p.add_argument("--layers", default="10,15,20,25")
    p.add_argument("--alphas", default="0,5,10,20,40")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="T=0 for deterministic flip-rate measurement")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output-subdir", default=None,
                   help="subdir under steering_experiments/ for this run "
                   "(keeps M5a's original outputs intact when set)")
    args = p.parse_args()

    # -------- test stimuli --------
    obj, bg, cue = args.test_subset.split("/")
    manifest = pd.read_parquet(args.stimulus_dir / "manifest.parquet")
    sub = manifest[
        (manifest["object_level"] == obj)
        & (manifest["bg_level"] == bg)
        & (manifest["cue_level"] == cue)
    ].reset_index(drop=True)
    print(f"Test subset: {obj}/{bg}/{cue} → {len(sub)} stimuli")
    assert len(sub), "empty test subset"

    # -------- steering vectors --------
    vec_path = args.run_dir / "probing_steering" / "steering_vectors.npz"
    vs = load_steering_vectors(vec_path)
    print(f"Loaded steering vectors for layers {sorted(vs.keys())}")

    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    for li in layers:
        assert li in vs, f"no steering vector for layer {li}"

    # -------- model --------
    vlm = PhysModeVLM(
        model_id=args.model_id,
        torch_dtype="bfloat16",
        device="cuda",
    )
    lm_layers = _resolve_lm_layers(vlm.model)
    print(f"Model LM has {len(lm_layers)} decoder layers")

    rp = render_prompt(args.prompt_variant, args.label)
    gen_args = InferenceArgs(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # -------- sweep --------
    total = len(sub) * len(layers) * len(alphas)
    pbar = tqdm(total=total, desc="Steering sweep")
    records: list[dict] = []

    for li in layers:
        v_np = vs[li]
        v_tensor = torch.from_numpy(v_np)
        for alpha in alphas:
            handle = lm_layers[li].register_forward_hook(make_hook(alpha, v_tensor))
            try:
                for _, row in sub.iterrows():
                    img_path = args.stimulus_dir / row["image_path"]
                    out = vlm.generate(
                        image=img_path, prompt=rp.user, args=gen_args,
                        system_prompt=rp.system, choice_tokens=rp.choice_letters,
                    )
                    records.append({
                        "sample_id": row["sample_id"],
                        "object_level": row["object_level"],
                        "bg_level": row["bg_level"],
                        "cue_level": row["cue_level"],
                        "event_template": row["event_template"],
                        "label": args.label,
                        "prompt_variant": args.prompt_variant,
                        "layer": int(li),
                        "alpha": float(alpha),
                        "raw_text": out["raw_text"],
                    })
                    pbar.update(1)
            finally:
                handle.remove()
    pbar.close()

    # -------- score + summarize --------
    df = pd.DataFrame(records)
    df = score_rows(df)
    df["first_letter"] = df["raw_text"].apply(extract_first_letter)

    outdir = args.run_dir / "steering_experiments"
    if args.output_subdir:
        outdir = outdir / args.output_subdir
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / "intervention_predictions.parquet", index=False)
    df.to_csv(outdir / "intervention_predictions.csv", index=False)

    print()
    print("=" * 72)
    print("PMR flip rate by (layer, alpha)")
    print("=" * 72)
    pv = df.groupby(["layer", "alpha"])["pmr"].mean().unstack("alpha").round(3)
    print(pv.to_string())
    pv.to_csv(outdir / "pmr_by_layer_alpha.csv")

    print()
    print("=" * 72)
    print("First-letter distribution by (layer, alpha)")
    print("=" * 72)
    fl_pivot = (
        df.groupby(["layer", "alpha", "first_letter"])
          .size()
          .unstack("first_letter", fill_value=0)
    )
    print(fl_pivot.to_string())
    fl_pivot.to_csv(outdir / "first_letter_by_layer_alpha.csv")

    # Also grab a representative raw-text sample per (layer, highest alpha)
    print()
    print("Sample response at strongest steering (α = max):")
    max_alpha = max(alphas)
    for li in layers:
        sub_df = df[(df["layer"] == li) & (df["alpha"] == max_alpha)]
        if len(sub_df):
            r = sub_df.iloc[0]
            text = r["raw_text"].replace("\n", " ")[:200]
            print(f"  L{li} α={max_alpha}: pmr={r['pmr']}  :: {text}")

    # Baseline comparison
    baseline_pmr = float(df[df["alpha"] == 0]["pmr"].mean())
    print(f"\nBaseline PMR (α=0): {baseline_pmr:.3f}")

    (outdir / "run_meta.json").write_text(json.dumps({
        "run_dir": str(args.run_dir),
        "stimulus_dir": str(args.stimulus_dir),
        "test_subset": args.test_subset,
        "label": args.label,
        "prompt_variant": args.prompt_variant,
        "layers": layers,
        "alphas": alphas,
        "temperature": args.temperature,
        "n_stimuli": len(sub),
        "n_total_inferences": len(df),
        "output_subdir": args.output_subdir,
    }, indent=2))
    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
