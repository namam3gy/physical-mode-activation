"""Pilot config — small de-risk run (~240 stimuli × 2 prompts = 480 inferences).

Expected wall-clock on H200 (after Qwen2.5-VL-7B download): ~30-60 min including
model load.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="pilot",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "wind", "arrow_shadow"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("ball",),
    prompt_variants=("open", "forced_choice"),
    # Activation capture off for the pilot — behavior is the primary signal.
    capture_lm_layers=None,
)
