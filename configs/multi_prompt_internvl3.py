"""Multi-prompt cross-task generalization — InternVL3-8B-hf (M-MP, Track B Pillar A).

See `multi_prompt_qwen.py` for the design rationale.

InternVL3 (InternViT + InternLM3) is at the saturated ceiling
(behavioral PMR 0.99 on M2 baseline_pmr=1.0; M5b SAE k=160 break).
Multi-prompt test asks whether the multi-prompt design itself can
*break the saturation ceiling* — if `meta_phys_yesno` produces "no"
responses on stim where `open` produces 100 % physics, that's
informative about whether physics-mode commitment is *forced* by the
prediction-style prompt or arises spontaneously.

Inference size: 480 stim × 3 labels × 3 prompts = 4320 inferences.
Expected wall-clock on H200 (single 448×448 tile, no captures): ~25-35 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="multi_prompt_internvl3",
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
    prompt_variants=("open", "describe_scene", "meta_phys_yesno"),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
