"""Full MVP config for Sub-task 1 — all four object levels × 3 bg × 3 cue.

Factorial size: 4 × 3 × 3 × 2 events × 15 seeds = 1,080 stimuli; with 3 labels
and 2 prompt variants that's 6,480 inferences. Expected ~4-6h on H200 with
activation capture on 5 LM layers (~8 GB of .safetensors).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="mvp_full",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground", "scene"),
        cue_levels=("none", "wind", "arrow_shadow"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=15,
    ),
    labels=("circle", "ball", "planet"),
    prompt_variants=("open", "forced_choice"),
    # Capture LM hidden states at 5 layer checkpoints for later Sub-task 3 probing.
    capture_lm_layers=(5, 10, 15, 20, 25),
)
