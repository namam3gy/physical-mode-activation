"""§4.5 ext — InternVL3-8B (InternViT + InternLM2) on M8a stim.

Fourth point on the encoder-swap × M8a × open-prompt grid (after Qwen,
LLaVA, Idefics2). InternVL3 uses InternViT (yet another non-CLIP vision
encoder family) + InternLM2-7B (yet another LM family). Together with
M6 r3 vision-encoder probe, this round produces a 4-point AUC ↔ PMR
table for the H-encoder-saturation paper claim.

Reuses the M8a Qwen stim dir (`inputs/m8a_qwen_*` — 400 stim).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_internvl3_m8a",
    model_id="OpenGVLab/InternVL3-8B-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("circle", "square", "triangle", "hexagon", "polygon"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall",),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
