"""§4.6 cross-model — Qwen-derived synthesized stim transfer test.

Tests whether the §4.6 Qwen v_L10 synthesized stim — originally produced
by gradient ascent on Qwen2.5-VL's pixel_values targeting Qwen's v_L10
direction — also flip PMR on the other 4 VLMs (LLaVA-1.5, LLaVA-Next,
Idefics2, InternVL3).

Different question from "per-model §4.6": this tests TRANSFER (whether
Qwen's adversarial fools other models) rather than each model's own
pixel-encodable v_L direction. A positive transfer suggests cross-model
shortcut alignment; a null result is the more typical case
(adversarial perturbations are notoriously model-specific).

Loads the §4.6 sweep `outputs/sec4_6_counterfactual_<ts>/` and per
(seed × config) feeds both `baseline.png` and `synthesized.png` to
each of the 4 non-Qwen models. Scores PMR. Aggregates.

Usage:
    uv run python scripts/sec4_6_cross_model_transfer.py \
        --run-dir outputs/sec4_6_counterfactual_20260426-050343
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."

MODELS = [
    ("LLaVA-1.5",  "llava-hf/llava-1.5-7b-hf"),
    ("LLaVA-Next", "llava-hf/llava-v1.6-mistral-7b-hf"),
    ("Idefics2",   "HuggingFaceM4/idefics2-8b"),
    ("InternVL3",  "OpenGVLab/InternVL3-8B-hf"),
]


def _generate(model, processor, pil: Image.Image) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _load_model(model_id: str, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    return model, processor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="§4.6 sweep dir, e.g. outputs/sec4_6_counterfactual_<ts>")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.run_dir.is_absolute():
        args.run_dir = (PROJECT_ROOT / args.run_dir).resolve()
    if args.out_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.out_dir = PROJECT_ROOT / f"outputs/sec4_6_cross_model_transfer_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.run_dir / "manifest.json") as f:
        manifest = json.load(f)
    rows_in = manifest["rows"]

    rows = []
    for model_name, model_id in MODELS:
        print(f"\n==== {model_name} ({model_id}) ====")
        model, processor = _load_model(model_id, device=args.device)

        for r in rows_in:
            sid = r["sample_id"]
            cfg = r["config_name"]
            synth_path = PROJECT_ROOT / r["synthesized_path"]
            baseline_path = synth_path.parent / "baseline.png"
            try:
                bl = _generate(model, processor, Image.open(baseline_path).convert("RGB"))
                sy = _generate(model, processor, Image.open(synth_path).convert("RGB"))
            except Exception as e:
                print(f"  [ERROR] {model_name} {sid} {cfg}: {e}")
                continue
            bl_pmr = score_pmr(bl)
            sy_pmr = score_pmr(sy)
            rows.append({
                "model": model_name,
                "sample_id": sid,
                "config_name": cfg,
                "baseline_response": bl,
                "synthesized_response": sy,
                "baseline_pmr": bl_pmr,
                "synthesized_pmr": sy_pmr,
                "delta_pmr": sy_pmr - bl_pmr,
            })
            print(f"  {sid} | {cfg}: bl={bl_pmr} synth={sy_pmr} Δ={sy_pmr - bl_pmr:+d}")

        del model, processor
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "results.csv", index=False)
    print(f"\nWrote {args.out_dir / 'results.csv'} ({len(df)} rows)")

    # Aggregate
    agg = df.groupby(["model", "config_name"]).agg(
        n=("sample_id", "count"),
        baseline_pmr_mean=("baseline_pmr", "mean"),
        synth_pmr_mean=("synthesized_pmr", "mean"),
        n_flipped=("delta_pmr", lambda s: int((s > 0).sum())),
        delta_mean=("delta_pmr", "mean"),
    ).reset_index()
    agg.to_csv(args.out_dir / "results_aggregated.csv", index=False)
    print("\n=== Aggregated cross-model transfer (Δ PMR) ===")
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
