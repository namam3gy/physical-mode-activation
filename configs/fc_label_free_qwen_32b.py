"""B5 — Qwen2.5-VL 32B forced-choice label-free on M2 stim.

Cross-scale H4 supplement: §4.8 already covers Qwen 7B vs 32B on the open
prompt; this config extends the comparison to FC label-free. Reuses the
prompt design from `fc_label_free_qwen.py` (Qwen 7B FC label-free).

Question: does the FC open-vs-FC PMR gap (§4.4) shrink at 32B scale, or
does scale leave it intact? Combined with §4.8's open-prompt 32B finding
(aggregate PMR ~unchanged but cue=none drops 8.6 pp), this should
either reinforce or refine the "scale doesn't fix grounding" headline.

Inference: 480 stim × 1 prompt × 1 label = 480 inferences.
Wall-clock on H200 (single GPU, bf16): ~30 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="fc_label_free_qwen_32b",
    model_id="Qwen/Qwen2.5-VL-32B-Instruct",
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
