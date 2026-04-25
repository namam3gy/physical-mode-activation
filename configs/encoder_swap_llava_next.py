"""§4.5 ext — LLaVA-v1.6-Mistral-7B (CLIP + Mistral-7B) on M8a stim.

Second CLIP-based point (after LLaVA-1.5-Vicuna). Same vision encoder
family (CLIP-ViT-L), different LM (Mistral-7B instead of Vicuna-7B).
Pairs with Idefics2 (SigLIP-SO400M + Mistral-7B) to address the n=1
CLIP-side limitation noted in M9: with this run, the table covers
2 CLIP × 2 SigLIP across LMs.

Uses the M8a Qwen stim dir.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_llava_next_m8a",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
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
