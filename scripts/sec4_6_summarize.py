"""§4.6 — re-infer PMR on baseline + synthesized stim, aggregate, plot.

Loads the manifest produced by sec4_6_counterfactual_stim.py, runs a
fresh inference on each baseline.png and synthesized.png using the
Qwen2.5-VL inference pipeline (independent of the gradient-ascent path),
scores PMR, and writes:
  - results.csv: per-(sid, config) raw + scored responses
  - results_aggregated.csv: per-config mean PMR pre/post + n_flipped
  - docs/figures/sec4_6_counterfactual_stim_panels.png: 4-panel canonical seed
  - docs/figures/sec4_6_counterfactual_stim_trajectory.png: per-config trajectories

Usage:
    uv run python scripts/sec4_6_summarize.py --run-dir outputs/sec4_6_counterfactual_<ts>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."


def _generate(model, processor, pil: Image.Image, prompt: str = PROMPT) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return raw


def _plot_canonical_panels(run_dir: Path, manifest_rows: list[dict], canonical_sid: str) -> None:
    """4-panel figure: baseline / eps=0.05 / eps=0.1 / unconstrained for one seed."""
    panels = [
        ("baseline", "baseline (eps=0)"),
        ("bounded_eps0.05", "ε = 0.05"),
        ("bounded_eps0.1", "ε = 0.1"),
        ("unconstrained", "unconstrained"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (cfg, title) in zip(axes, panels):
        if cfg == "baseline":
            row = next(r for r in manifest_rows if r["sample_id"] == canonical_sid)
            img_path = (PROJECT_ROOT / row["synthesized_path"]).parent / "baseline.png"
            img = Image.open(img_path)
        else:
            row = next((r for r in manifest_rows
                       if r["sample_id"] == canonical_sid and r["config_name"] == cfg), None)
            if row is None:
                ax.set_visible(False); continue
            img = Image.open(PROJECT_ROOT / row["synthesized_path"])
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"§4.6 — VTI-reverse counterfactual (seed: {canonical_sid})", fontsize=14)
    out = PROJECT_ROOT / "docs/figures/sec4_6_counterfactual_stim_panels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_trajectories(run_dir: Path, manifest_rows: list[dict]) -> None:
    """Per-config mean trajectory ± std across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cfgs = sorted({r["config_name"] for r in manifest_rows})

    # Order: bounded ε ascending, then unconstrained, then controls
    def _sort_key(c):
        if c.startswith("bounded_eps"):
            return (0, float(c.replace("bounded_eps", "")))
        if c == "unconstrained":
            return (1, 0)
        return (2, c)
    cfgs = sorted(cfgs, key=_sort_key)

    for cfg in cfgs:
        seed_rows = [r for r in manifest_rows if r["config_name"] == cfg]
        trajs = []
        for r in seed_rows:
            t = np.load(PROJECT_ROOT / r["trajectory_path"])
            trajs.append(t[:, 1])
        if not trajs:
            continue
        max_len = max(len(x) for x in trajs)
        padded = np.array([np.pad(x, (0, max_len - len(x)), mode="edge") for x in trajs])
        # x-axis from logged step indices (step 0, log_every, ..., n_steps-1)
        step_idx = np.load(PROJECT_ROOT / seed_rows[0]["trajectory_path"])[:, 0]
        if len(step_idx) < max_len:
            step_idx = np.concatenate([step_idx, [step_idx[-1]] * (max_len - len(step_idx))])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        ls = "--" if cfg.startswith("control_") else "-"
        ax.plot(step_idx, mean, ls, label=cfg, linewidth=1.5)
        ax.fill_between(step_idx, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Projection on v_L10")
    ax.set_title("§4.6 — projection trajectory per config (mean ± std over 5 seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out = PROJECT_ROOT / "docs/figures/sec4_6_counterfactual_stim_trajectory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if not args.run_dir.is_absolute():
        args.run_dir = (PROJECT_ROOT / args.run_dir).resolve()

    with open(args.run_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    rows = []
    n_total = len(manifest["rows"])
    for i, r in enumerate(manifest["rows"], 1):
        sid = r["sample_id"]
        cfg = r["config_name"]
        synth_path = PROJECT_ROOT / r["synthesized_path"]
        baseline_path = synth_path.parent / "baseline.png"

        baseline_resp = _generate(model, processor, Image.open(baseline_path).convert("RGB"))
        synth_resp = _generate(model, processor, Image.open(synth_path).convert("RGB"))
        baseline_pmr = score_pmr(baseline_resp)
        synth_pmr = score_pmr(synth_resp)

        rows.append({
            **r,
            "baseline_response": baseline_resp,
            "synthesized_response": synth_resp,
            "baseline_pmr": baseline_pmr,
            "synthesized_pmr": synth_pmr,
            "delta_pmr": synth_pmr - baseline_pmr,
        })
        print(f"  [{i}/{n_total}] {sid} | {cfg}: bl_pmr={baseline_pmr} synth_pmr={synth_pmr} Δ={synth_pmr - baseline_pmr:+d}")

    df = pd.DataFrame(rows)
    df_csv = args.run_dir / "results.csv"
    df.to_csv(df_csv, index=False)
    print(f"\nWrote {df_csv}")

    agg = df.groupby("config_name").agg(
        n=("sample_id", "count"),
        baseline_pmr_mean=("baseline_pmr", "mean"),
        synth_pmr_mean=("synthesized_pmr", "mean"),
        delta_mean=("delta_pmr", "mean"),
        n_flipped=("delta_pmr", lambda s: int((s > 0).sum())),
    ).reset_index()
    print("\n=== Aggregated PMR per config ===")
    print(agg.round(3).to_string(index=False))
    agg.to_csv(args.run_dir / "results_aggregated.csv", index=False)

    canonical = manifest["rows"][0]["sample_id"]
    _plot_canonical_panels(args.run_dir, manifest["rows"], canonical)
    _plot_trajectories(args.run_dir, manifest["rows"])


if __name__ == "__main__":
    main()
