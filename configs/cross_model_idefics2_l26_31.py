"""M2 cross-model — Idefics2-8B deeper-layer capture (L26-31).

Mirrors `cross_model_idefics2.py` but captures LM hidden states at
the last 6 transformer layers (Mistral-7B has 32 layers; L26-31 =
81-97 % relative depth). Required for §4.6 Idefics2 deeper-layer
sweep — the existing capture only covers L5/10/15/20/25 (≤ 78 %),
which the §4.6 5-model layer sweep showed yields zero PMR flip
despite v_L projection ascending cleanly. This new capture
discriminates the "wrong-relative-depth" vs "perceiver-resampler-
bottleneck" hypothesis pair for the Idefics2 anomaly.

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440
inferences + 480 once-per-stimulus capture calls. Expected
wall-clock on H200: ~50-70 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_idefics2_capture_l26_31",
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
    prompt_variants=("open",),
    capture_lm_layers=(26, 28, 30, 31),
    capture_vision_layers=(),
    capture_lm_attentions=False,
    random_seed=42,
)
