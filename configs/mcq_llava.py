"""MCQ categorical probe — LLaVA-1.5-7B (M-MP audit follow-up).

See `mcq_qwen.py` for the design rationale. **Caveat**: LLaVA-1.5 has a
known greedy "A" bias on multi-choice prompts (M4c forced_choice baseline:
LLaVA returns "A" for 477/480 cells). Expect option A (= physical event)
over-selection on this run; the H2 paired-delta (ball − circle) should
still be informative even with bias contamination, but the absolute
PMR ceiling will be biased upward.

Inference size: 480 stim × 3 labels × 1 prompt = 1440 inferences.
Expected wall-clock on H200: ~10-15 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="mcq_llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground", "scene"),
        cue_levels=("none", "cast_shadow", "motion_arrow", "both"),
        event_templates=("fall",),
        seeds_per_cell=10,
    ),
    labels=("circle", "ball", "planet"),
    prompt_variants=("meta_phys_mcq",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
