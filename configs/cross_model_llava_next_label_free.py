"""M2 cross-model — LLaVA-Next-Mistral-7B label-free open prompt.

Mirrors `cross_model_llava_label_free.py` for the LLaVA-Next H2 null
test (label-free open vs labeled open). No captures — pairs with
`cross_model_llava_next.py` (capture run with labels).

Inference size: 480 stim × 0 labels (no-label) × 1 prompt = 480
inferences. Expected wall-clock on H200: ~25-35 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava_next_label_free",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
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
