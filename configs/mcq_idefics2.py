"""MCQ categorical probe — Idefics2-8B (M-MP audit follow-up).

See `mcq_qwen.py` for the design rationale. Idefics2 multi-prompt run
for behavioral parity with the existing 5-model × 3-prompt Phase 2 chain.
Idefics2 is also a candidate for *cross-model* dissociation: if it
behaves like Qwen on MCQ (categorical task null on M5a/M5b), the
gen-vs-cat boundary generalizes; if Idefics2 MCQ is positive (unlike
its yesno NULL), the boundary may be Qwen-specific format-bound.

Inference size: 480 stim × 3 labels × 1 prompt = 1440 inferences.
Expected wall-clock on H200 (5-tile): ~12-18 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="mcq_idefics2",
    model_id="HuggingFaceM4/idefics2-8b",
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
