"""§4.10 ext — Idefics2-8b attention capture (limit=20)."""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="attention_viz_idefics2",
    model_id="HuggingFaceM4/idefics2-8b",
    torch_dtype="bfloat16",
    max_new_tokens=64,
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
    capture_lm_layers=(5, 15, 20, 25),
    capture_lm_attentions=True,
    random_seed=42,
    limit=20,
)
