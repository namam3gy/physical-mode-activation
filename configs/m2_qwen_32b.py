"""§4.8 PMR scaling — Qwen2.5-VL 32B on M2 stim (label-free + labeled).

Tests "scale doesn't help PMR" hypothesis (MechBench analog) by running
the same M2 protocol on Qwen 32B as on Qwen 7B. Reuses existing M2 stim.

Inference size: 480 stim × 3 labels × 1 prompt variant = 1440 inferences.
Expected wall-clock on H200 (single GPU, bf16): ~6-8 hr.

Memory budget: ~67 GB weights at bf16 + ~30 GB activations + KV → ~100 GB
of 143 GB H200. Should fit single-GPU.

Stimuli reused via `--stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3`.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m2_qwen_32b",
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
    labels=("circle", "ball", "planet"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
