"""Multi-prompt cross-task generalization — LLaVA-Next-Mistral-7B (M-MP, Track B Pillar A).

See `multi_prompt_qwen.py` for the design rationale.

LLaVA-Next is the AnyRes / 2nd-CLIP-cluster point. M5a runtime steering
flips it 10/10 (LM-side) but M5b SAE intervention is NULL (encoder-side).
Multi-prompt test asks whether the LM-side mechanism transfers to
non-prediction prompts — informative for the encoder-vs-LM dissociation
claim across cognitive tasks.

Inference size: 480 stim × 3 labels × 3 prompts = 4320 inferences.
Expected wall-clock on H200 (5-tile AnyRes, no captures): ~30-45 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="multi_prompt_llava_next",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
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
