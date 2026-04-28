"""Multi-prompt cross-task generalization — Idefics2-8B (M-MP, Track B Pillar A).

See `multi_prompt_qwen.py` for the design rationale.

Idefics2 (SigLIP-SO400M + perceiver-resampler + Mistral-7B) is the
clearest forward/inverse pathway dissociation: M4 LM probe AUC 0.995 +
M5a 10/10 LM-side flip + §4.6 0/9 layers pixel flip + M5b SAE k=160 break.
Multi-prompt test asks whether the LM-side mechanism (M5a + M4) is
task-agnostic — the prediction is yes, since the LM has the information.

Inference size: 480 stim × 3 labels × 3 prompts = 4320 inferences.
Expected wall-clock on H200 (5-tile, no captures): ~35-50 min.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="multi_prompt_idefics2",
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
    prompt_variants=("open", "describe_scene", "meta_phys_yesno"),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
