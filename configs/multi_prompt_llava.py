"""Multi-prompt cross-task generalization — LLaVA-1.5-7B (M-MP, Track B Pillar A).

See `multi_prompt_qwen.py` for the design rationale. This config mirrors
the Qwen one and only swaps `model_id`.

LLaVA-1.5 is the unsaturated CLIP-encoder pole of the 5-model ladder
(M3 vision probe AUC 0.73, behavioral PMR 0.18). The multi-prompt test
on LLaVA-1.5 is most informative for whether the *encoder-bottleneck*
prediction holds across non-prediction prompts.

Inference size: 480 stim × 3 labels × 3 prompts = 4320 inferences.
Expected wall-clock on H200: ~25-35 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="multi_prompt_llava",
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
    prompt_variants=("open", "describe_scene", "meta_phys_yesno"),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
