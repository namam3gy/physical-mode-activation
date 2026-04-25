"""§4.5 ext — Idefics2-8b on M8d stim (categories: car/person/bird).

Tests whether the SigLIP-encoder ceiling pattern observed on M8a
(5 shapes) holds for non-ball categories. Reuses the M8d stim
(`inputs/m8d_qwen_*`) — same 480 stimuli as the M8d Qwen + LLaVA runs.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="encoder_swap_idefics2_m8d",
    model_id="HuggingFaceM4/idefics2-8b",
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
