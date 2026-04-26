"""§4.5 ext — InternVL3 M8d stim, label-free arm."""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_internvl3_m8d_label_free",
    model_id="OpenGVLab/InternVL3-8B-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
