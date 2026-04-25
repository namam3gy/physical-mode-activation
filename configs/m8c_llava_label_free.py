"""M8c — real photographs, LLaVA-1.5-7B label-free arm."""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8c_llava_label_free",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("ball", "car", "person", "bird", "abstract"),
        object_levels=("photo",),
        bg_levels=("natural",),
        cue_levels=("none",),
        event_templates=("fall",),
        seeds_per_cell=12,
    ),
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
