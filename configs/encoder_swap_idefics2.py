"""§4.5 — Cross-encoder swap test on Idefics2-8b (SigLIP + Mistral-7B).

Tests whether the visual encoder type drives behavioral PMR(_nolabel)
saturation, holding stimulus type constant. The two existing
comparison points are:

  - Qwen2.5-VL-7B-Instruct  : SigLIP encoder + Qwen2-7B LM   (M2/M8a/M8d)
  - LLaVA-1.5-7B            : CLIP-ViT-L/14 + Vicuna-7B LM   (M6/M8a/M8d)

Idefics2-8b adds a third point:
  - Idefics2-8b             : SigLIP encoder + Mistral-7B LM

Prediction:
  - If encoder type is the primary driver of PMR(_nolabel) ceiling,
    Idefics2 should pattern with Qwen (high PMR(_nolabel) on synthetic).
  - If LM family is the primary driver, Idefics2 (Mistral-7B) is in
    between Vicuna-7B (LLaVA, low) and Qwen2-7B (Qwen, high).

Reuses M8a stim (`inputs/m8a_qwen_*`) for the labeled + label-free
runs.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_idefics2",
    model_id="HuggingFaceM4/idefics2-8b",
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
