"""M2 cross-model — InternVL3-8B with LM activation captures.

Supplements the existing `cross_model_internvl3.py` (M6 r2a behavioral)
by re-running with `capture_lm_layers` enabled. The M6 r2a run produced
the open-prompt PMR but no LM hidden states; this run adds them so
per-model v_L10 extraction (for §4.6 cross-model) becomes possible.

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440
inferences + 480 once-per-stimulus capture calls. Expected
wall-clock on H200: ~50-70 min.
Disk: ~10-15 GB safetensors (LM hidden states only, bf16).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_internvl3_capture",
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
    labels=("circle", "ball", "planet"),
    prompt_variants=("open",),
    capture_lm_layers=(5, 10, 15, 20, 25),
    capture_vision_layers=(3, 7, 11, 15, 19, 23),
    capture_lm_attentions=False,
    random_seed=42,
)
