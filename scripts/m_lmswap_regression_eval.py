"""M-LMSwap regression eval — gate Variant B training on Variant A's viability.

Variant A's purpose (per docs/m_lmswap_design.md): controlled CLIP+Vicuna+MLP
reproduction of LLaVA-1.5 with encoder + projector held fixed, so that A↔B
isolates LM family. For A↔B comparison to be interpretable, A must reproduce
LLaVA-1.5's saturation regime (PMR_nolabel ≈ 0.18) — not collapsed (broken
training), not saturated to mid/high-band (recipe drift to LLaVA-Next territory).

Three gates (sequential, abort on first fail):

1. Generation sanity — 5 M2 stim → outputs > 5 words, no degenerate repetition
2. M2 PMR_nolabel ∈ [0.03, 0.50] — full 480 stim, `open_no_label` prompt
3. M5a baseline room — `line/blank/none` cell PMR ≤ 0.6 (room to flip via steering)

Note on the gate range: design doc §6 says [0.2, 0.8], but LLaVA-1.5's
actual M2 PMR_nolabel sits at 0.18 (below 0.2). [0.03, 0.50] is the
correct LLaVA-1.5 saturation regime — see CHANGELOG / m_lmswap_design.md
update notes.

Exit codes
----------
0 — all gates PASS, proceed to Variant B
1 — gate FAIL, abort B and trigger fallback queue

Usage
-----
    uv run python scripts/m_lmswap_regression_eval.py \\
        --ckpt outputs/lmswap_run_a_stage2_*/step21000 \\
        --variant A \\
        --stim-dir inputs/mvp_full_20260424-093926_e9d79da3 \\
        --output-dir outputs/lmswap_a_regression_eval \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from physical_mode.inference.prompts import OPEN_TEMPLATE_NO_LABEL, SYSTEM_PROMPT_OPEN
from physical_mode.lora.load_lmswap import load_lmswap_variant
from physical_mode.metrics.pmr import score_pmr


# Gates per docs/m_lmswap_design.md §6 (corrected for LLaVA-1.5 actual PMR=0.18).
PMR_LOWER = 0.03
PMR_UPPER = 0.50
BASELINE_CELL_UPPER = 0.6
GEN_SANITY_MIN_WORDS = 5
GEN_SANITY_N_STIM = 5


_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions."
)
_MISTRAL_BOS = "<s>"


def _format_inference_prompt(variant: str, system: str, user_text: str) -> str:
    """Return the variant-specific chat-formatted prompt with `<image>` literal.

    Mirrors `m_lmswap_train.format_chat`'s `prompt_only` branch, but with the
    open_no_label system prompt (training used a generic captioning system).
    """
    full_user = f"<image>\n{user_text}"
    if variant == "A":
        return f"{_VICUNA_SYSTEM} {system} USER: {full_user} ASSISTANT:"
    if variant == "B":
        return f"{_MISTRAL_BOS}[INST] {system}\n\n{full_user} [/INST]"
    raise ValueError(f"unknown variant {variant!r}")


def _is_degenerate(text: str) -> bool:
    """True if the response is degenerate (excessive repetition).

    Heuristic: if any single token (word) makes up > 35% of the response
    after the prompt, treat as repetitive degeneracy. Common failure mode
    of broken VLMs: "the the the the..." or single-token loops.
    """
    words = [w.lower() for w in text.split() if w.strip()]
    if len(words) < 4:
        return False
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    return top_count / len(words) > 0.35


def _strip_prompt(generated: str, prompt: str) -> str:
    if generated.startswith(prompt):
        return generated[len(prompt):].strip()
    # fallback: split on ASSISTANT: or [/INST]
    for marker in ("ASSISTANT:", "[/INST]"):
        if marker in generated:
            return generated.split(marker, 1)[1].strip()
    return generated.strip()


def run_eval(
    ckpt: Path, variant: str, stim_dir: Path, output_dir: Path,
    device: str, max_stim: int | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "regression_eval.jsonl"
    summary_path = output_dir / "summary.json"
    log_f = log_path.open("w")

    def _log(record: dict) -> None:
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    print(f"[regression] loading variant {variant} from {ckpt} on {device}...")
    model, processor = load_lmswap_variant(
        ckpt_dir=ckpt, variant=variant, device=device, merge_lora=True,
    )

    manifest = pd.read_parquet(stim_dir / "manifest.parquet")
    if max_stim is not None:
        manifest = manifest.head(max_stim).copy()
    print(f"[regression] {len(manifest)} M2 stim loaded")

    user_text = OPEN_TEMPLATE_NO_LABEL
    prompt_str = _format_inference_prompt(variant, SYSTEM_PROMPT_OPEN, user_text)

    # Gate 1: generation sanity on first GEN_SANITY_N_STIM stim
    print(f"[regression] gate 1: generation sanity ({GEN_SANITY_N_STIM} stim)...")
    gen_sanity_ok = True
    gen_sanity_reasons: list[str] = []
    sample_outputs: list[dict] = []

    pmr_per_stim: list[dict] = []
    t0 = time.time()
    for idx, row in manifest.iterrows():
        img_path = stim_dir / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, text=prompt_str, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            )
        full = processor.batch_decode(out, skip_special_tokens=True)[0]
        response = _strip_prompt(full, prompt_str.replace("<image>", ""))

        pmr = score_pmr(response)
        rec = {
            "sample_id": row["sample_id"],
            "object_level": row["object_level"],
            "bg_level": row["bg_level"],
            "cue_level": row["cue_level"],
            "response": response,
            "pmr": pmr,
        }
        pmr_per_stim.append(rec)
        _log(rec)

        if len(sample_outputs) < GEN_SANITY_N_STIM:
            sample_outputs.append(rec)
            words = response.split()
            if len(words) < GEN_SANITY_MIN_WORDS:
                gen_sanity_ok = False
                gen_sanity_reasons.append(
                    f"{row['sample_id']}: only {len(words)} words: {response!r}"
                )
            elif _is_degenerate(response):
                gen_sanity_ok = False
                gen_sanity_reasons.append(
                    f"{row['sample_id']}: degenerate repetition: {response!r}"
                )

        # progress every 50 stim
        if (len(pmr_per_stim) % 50) == 0:
            elapsed = time.time() - t0
            n = len(pmr_per_stim)
            eta = elapsed / n * (len(manifest) - n)
            print(f"  [{n}/{len(manifest)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                  f"pmr_so_far={sum(r['pmr'] for r in pmr_per_stim) / n:.3f}")

    log_f.close()

    df = pd.DataFrame(pmr_per_stim)
    pmr_overall = float(df["pmr"].mean())

    # Gate 3: line/blank/none cell baseline
    cell_mask = (
        (df["object_level"] == "line") &
        (df["bg_level"] == "blank") &
        (df["cue_level"] == "none")
    )
    n_cell = int(cell_mask.sum())
    pmr_baseline_cell = float(df.loc[cell_mask, "pmr"].mean()) if n_cell > 0 else None

    # Apply gates
    pmr_in_range = PMR_LOWER <= pmr_overall <= PMR_UPPER
    baseline_room = (pmr_baseline_cell is None) or (pmr_baseline_cell <= BASELINE_CELL_UPPER)

    gates = {
        "gate_1_generation_sanity": {
            "pass": gen_sanity_ok,
            "n_stim_checked": GEN_SANITY_N_STIM,
            "reasons": gen_sanity_reasons,
        },
        "gate_2_pmr_nolabel": {
            "pass": pmr_in_range,
            "value": pmr_overall,
            "range": [PMR_LOWER, PMR_UPPER],
            "reason": (None if pmr_in_range else
                       f"PMR_nolabel={pmr_overall:.3f} outside [{PMR_LOWER}, {PMR_UPPER}]"),
        },
        "gate_3_baseline_room": {
            "pass": baseline_room,
            "value": pmr_baseline_cell,
            "ceiling": BASELINE_CELL_UPPER,
            "n_cell": n_cell,
            "reason": (None if baseline_room else
                       f"line/blank/none baseline PMR={pmr_baseline_cell:.3f} > {BASELINE_CELL_UPPER}"),
        },
    }
    overall_pass = gen_sanity_ok and pmr_in_range and baseline_room

    summary = {
        "ckpt": str(ckpt),
        "variant": variant,
        "stim_dir": str(stim_dir),
        "n_stim": len(manifest),
        "pmr_nolabel": pmr_overall,
        "pmr_baseline_cell": pmr_baseline_cell,
        "n_baseline_cell": n_cell,
        "gates": gates,
        "overall_pass": overall_pass,
        "elapsed_sec": time.time() - t0,
        "sample_outputs": sample_outputs,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print("=" * 60)
    print(f"REGRESSION EVAL — variant {variant} from {ckpt.name}")
    print("=" * 60)
    print(f"  PMR_nolabel: {pmr_overall:.3f} (gate: [{PMR_LOWER}, {PMR_UPPER}]) "
          f"{'PASS' if pmr_in_range else 'FAIL'}")
    print(f"  Baseline cell (line/blank/none, n={n_cell}): "
          f"{pmr_baseline_cell:.3f} (gate: ≤{BASELINE_CELL_UPPER}) "
          f"{'PASS' if baseline_room else 'FAIL'}")
    print(f"  Generation sanity: {'PASS' if gen_sanity_ok else 'FAIL'}")
    if not gen_sanity_ok:
        for r in gen_sanity_reasons:
            print(f"    - {r}")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  Summary: {summary_path}")
    return 0 if overall_pass else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Stage 2 checkpoint dir (must contain MLP + LoRA + processor)")
    p.add_argument("--variant", choices=["A", "B"], required=True)
    p.add_argument("--stim-dir", type=Path,
                   default=_REPO_ROOT / "inputs" / "mvp_full_20260424-093926_e9d79da3",
                   help="M2 stim directory with manifest.parquet + images/")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-stim", type=int, default=None,
                   help="Limit stim count (for debugging)")
    args = p.parse_args()

    rc = run_eval(
        ckpt=args.ckpt, variant=args.variant, stim_dir=args.stim_dir,
        output_dir=args.output_dir, device=args.device, max_stim=args.max_stim,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
