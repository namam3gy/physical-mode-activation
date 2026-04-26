"""§4.5 ext — InternVL3-8B on M8d stim (categories: car/person/bird).

Closes the §4.11 5-model regime distribution gap (InternVL3 missing M8d).
Reuses the M8d stim dir (`inputs/m8d_qwen_*`) — same 480 stimuli as the
other 4 models.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_internvl3_m8d",
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
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
