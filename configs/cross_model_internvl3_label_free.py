"""M6 round-2 cross-model — InternVL3-8B-hf label-free companion.

Cross-model H2 null test for InternVL3. Paired delta against
`cross_model_internvl3.py` (open prompt) yields the per-label
contribution; comparison against the Qwen + LLaVA M4b/M6 r1 deltas
tells us whether the language-prior contribution at InternVL3 is
positive (LLaVA-like — visual prior unsaturated) or near-zero
(Qwen-like — visual prior saturated).

Inference size: 480 stim × 1 label × 1 prompt = 480 inferences.
Expected wall-clock on H200: ~10-15 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_internvl3_label_free",
    model_id="OpenGVLab/InternVL3-8B-hf",
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
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
