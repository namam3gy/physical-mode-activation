"""C6-orig (2026-05-01) — M8d expansion: boat / fish / plant on llava-1.5-7b-hf.

3 NEW non-Qwen-original categories (not car/person/bird) with literal
physical / abstract / exotic labels per shape. Tests cross-category H7
generalization beyond the original car/person/bird set.

Inference: 3 shapes × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds = 480 stim
× 3 labels × 1 open prompt = 1440 inferences.
"""
from __future__ import annotations
from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_extra_llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("boat", "fish", "plant"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
