"""M-PSwap M2 capture — Idefics2-MLP-pool variant (perceiver-resampler swapped).

Mirrors ``cross_model_idefics2.py`` but loads the swapped + LoRA-tuned
Idefics2 variant via ``swapped_ckpt``. Required for §4.6 layer sweep
(re-extract v_L from the swapped model's LM hidden states).

Pillar B / G3: ``references/paper_gaps.md`` ``references/submission_plan.md``.
"""

from __future__ import annotations

from pathlib import Path

from physical_mode.config import EvalConfig, FactorialSpec


# Set this to the trained checkpoint dir before running. Default points to
# the in-progress run; user can override at runtime via env or by editing.
DEFAULT_CKPT = Path("outputs/mpswap_run_20260429-033238/step5000")


CONFIG = EvalConfig(
    run_name="cross_model_idefics2_mpswap_capture",
    model_id="HuggingFaceM4/idefics2-8b",
    swapped_ckpt=str(DEFAULT_CKPT),
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground", "scene"),
        cue_levels=("none", "cast_shadow", "motion_arrow", "both"),
        event_templates=("fall",),
        seeds_per_cell=10,
    ),
    labels=("circle", "ball", "planet"),
    prompt_variants=("open",),
    capture_lm_layers=(5, 10, 15, 20, 25),
    capture_vision_layers=(3, 7, 11, 15, 19, 23),
    capture_lm_attentions=False,
    random_seed=42,
)
