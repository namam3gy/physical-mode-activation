"""MCQ categorical probe — Qwen2.5-VL-7B (M-MP audit follow-up).

Adds the `meta_phys_mcq` prompt as a 4th cognitive task on the 5-model
M2 stim. Companion to `meta_phys_yesno`: same categorical task, MCQ-letter
format instead of yes/no binary. Used to dissociate "categorical task" from
"yes/no format" in the Phase 3 generative-vs-categorical finding.

Per `docs/insights/review_audit_2026-04-28.md` follow-up #8: load-bearing
for the dissociation claim. If MCQ behaves like `meta_phys_yesno` in
Phase 3 (M5a 0/10, M5b no break), the boundary is the *categorical task*;
if MCQ behaves like `open`/`describe` (M5a 10/10, M5b break), the
boundary is the *yes/no format*. Both outcomes are paper-defensible.

Inference size: 480 stim × 3 labels × 1 prompt = 1440 inferences.
Expected wall-clock on H200: ~10-15 min (no captures, single prompt).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="mcq_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
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
