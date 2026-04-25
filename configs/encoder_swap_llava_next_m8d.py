"""§4.5 ext — LLaVA-Next-Mistral on M8d stim (categories: car/person/bird).

Cross-stim extension of M6 r6 — does the LLaVA-Next 2nd CLIP point hold
its position between LLaVA-1.5 floor and saturated cluster on non-ball
categories? Reuses the M8d stim (`inputs/m8d_qwen_*`) — same 480 stimuli
as the M8d Qwen + LLaVA + Idefics2 runs.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_llava_next_m8d",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
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
