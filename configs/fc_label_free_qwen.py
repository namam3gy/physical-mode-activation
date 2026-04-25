"""FC label-free probe — Qwen2.5-VL-7B side.

Companion to `configs/label_free.py` (open_no_label, M4b). Uses the new
`forced_choice_no_label` variant — the FC template with "the depicted
object" as antecedent for options A-C, no label token in the prompt.

Question: does the M4b H2 reframing (`ball ≈ no-label`,
`circle = suppressor`) survive when the prompt switches from open-ended
free text to forced-choice MCQ? I.e., is the per-label paired delta
preserved under FC?

Inference size: 480 stim × 1 label × 1 prompt = 480 inferences.
Expected wall-clock on H200: ~10 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="fc_label_free_qwen",
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
    labels=("_nolabel",),
    prompt_variants=("forced_choice_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
