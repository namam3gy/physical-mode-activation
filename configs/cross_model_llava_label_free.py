"""M6 round-1 cross-model — LLaVA-1.5-7B label-free companion.

Cross-model H2 null test: re-runs M4b on LLaVA-1.5-7B with the same M2
stimuli and the `open_no_label` prompt. Paired delta against
`cross_model_llava.py` (M6 labeled) yields the per-label contribution
specific to LLaVA-1.5; comparison against the Qwen M4b deltas tells us
whether the M4b finding (`ball ≈ no-label`, `circle = suppressor`) is
Qwen-specific or generalizes.

Inference size: 480 stim × 1 label × 1 prompt = 480 inferences.
Expected wall-clock on H200: ~10 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava_label_free",
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
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
