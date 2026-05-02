"""§4.6 InternVL3 alt-baseline — re-infer PMR on baseline + synthesized stim.

Reads the manifest from sec4_6_internvl3_layer_sweep_unified output, runs
InternVL3 inference on each baseline.png and synthesized.png with the
open_no_label prompt (matches the prompt under which the square baseline
was found at PMR=0.0), scores PMR, aggregates per config.

Outputs: results.csv + results_aggregated.csv in the run dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physical_mode.metrics.pmr import score_pmr  # noqa: E402

PROMPT = "What do you see in the image? What might happen next? Answer in one short sentence."
MODEL_ID = "OpenGVLab/InternVL3-8B-hf"


def _generate(model, processor, pil: Image.Image) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="§4.6 sweep dir from sec4_6_internvl3_layer_sweep_unified")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if not args.run_dir.is_absolute():
        args.run_dir = (ROOT / args.run_dir).resolve()

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    rows = manifest["rows"]
    print(f"Loaded {len(rows)} rows from manifest")

    print(f"Loading {MODEL_ID} on {args.device}...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    out_rows: list[dict] = []
    t_start = time.time()
    for i, r in enumerate(rows, 1):
        sid = r["sample_id"]
        cfg = r["config_name"]
        sweep_subdir = args.run_dir / f"L{r['layer']}" / cfg / sid
        baseline_pil = Image.open(sweep_subdir / "baseline.png").convert("RGB")
        synth_pil = Image.open(sweep_subdir / "synthesized.png").convert("RGB")

        baseline_resp = _generate(model, processor, baseline_pil)
        synth_resp = _generate(model, processor, synth_pil)
        b_pmr = score_pmr(baseline_resp)
        s_pmr = score_pmr(synth_resp)

        out_rows.append({
            **r,
            "baseline_response": baseline_resp,
            "synthesized_response": synth_resp,
            "baseline_pmr": b_pmr,
            "synthesized_pmr": s_pmr,
            "delta_pmr": s_pmr - b_pmr,
        })

        if i % 10 == 0 or i == len(rows):
            elapsed = time.time() - t_start
            eta = (elapsed / i) * (len(rows) - i)
            print(f"  [{i}/{len(rows)}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")

    df = pd.DataFrame(out_rows)
    results_csv = args.run_dir / "results.csv"
    df.to_csv(results_csv, index=False)
    print(f"\nWrote {results_csv}")

    # Aggregate per config
    agg = df.groupby("config_name", as_index=False).agg(
        n=("sample_id", "size"),
        baseline_pmr_mean=("baseline_pmr", "mean"),
        synth_pmr_mean=("synthesized_pmr", "mean"),
        delta_mean=("delta_pmr", "mean"),
        n_flipped=("delta_pmr", lambda s: int((s > 0).sum())),
    )
    # Sort: layer ascending, then v_unit before random
    agg["_layer"] = agg["config_name"].str.extract(r"L(\d+)_").astype(int)
    agg["_kind"] = agg["config_name"].str.contains("v_unit").map({True: 0, False: 1})
    agg = agg.sort_values(["_layer", "_kind"]).drop(columns=["_layer", "_kind"]).reset_index(drop=True)

    agg_csv = args.run_dir / "results_aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Wrote {agg_csv}")
    print()
    print("=" * 80)
    print("AGGREGATED RESULTS:")
    print("=" * 80)
    print(agg.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
