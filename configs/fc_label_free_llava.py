"""FC label-free probe — LLaVA-1.5-7B side.

Counterpart to `configs/fc_label_free_qwen.py`. Tests whether the
"A" bias observed in M6 r1 (LLaVA returned `first_letter=A` for every
labeled FC stimulus) is sensitive to removing the label antecedent.
The new prompt uses "the depicted object" instead of `It`/`{label}`.

If the bias persists, FC is fundamentally degenerate on LLaVA-1.5
regardless of prompt design. If it relaxes, the comparison against
Qwen FC label-free + Qwen open label-free becomes informative.

Inference size: 480 stim × 1 label × 1 prompt = 480 inferences.
Expected wall-clock on H200: ~10 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="fc_label_free_llava",
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
    prompt_variants=("forced_choice_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
