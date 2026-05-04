"""M2 cross-model — LLaVA-1.5-7B deeper-layer capture (L28/30/31).

Mirrors `cross_model_idefics2_l26_31.py` but for LLaVA-1.5-7B. The
existing capture only covers L5/10/15/20/25 (≤ 78 % of LLaMA-2-7B's
32 layers); §4.6 LLaVA-1.5 layer sweep at those depths showed L25
weak shortcut only (40 % at n=10), L5/L10/L15/L20 null. This new
capture at L28/30/31 (87.5 / 93.75 / 96.875 % relative depth)
discriminates "L25 only" from "missed deeper layers" — mirror of
the Idefics2 disambiguation that resolved that anomaly across L5-L31.

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440
inferences + 480 once-per-stimulus capture calls. Expected
wall-clock on H200: ~50-70 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava_capture_l28_31",
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
    labels=("circle", "ball", "planet"),
    prompt_variants=("open",),
    capture_lm_layers=(28, 30, 31),
    capture_vision_layers=(),
    capture_lm_attentions=False,
    random_seed=42,
)
