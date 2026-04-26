"""M2 cross-model — Idefics2-8B (SigLIP-SO400M + Mistral-7B) with captures.

Extends the M2 protocol (5-axis factorial, T=0.7, 480 stim × 3 labels)
to Idefics2-8B. Mirrors `cross_model_llava_capture.py` so the 5-model
M2-stim chain (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3)
is apples-to-apples.

Idefics2 is the 2nd SigLIP-family encoder + 2nd Mistral-7B LM in the
project. Behavioral M8a / M8d / M8c results already exist (§4.5);
this config adds the M2-stim run with captures so per-model v_L10
extraction (for §4.6 cross-model) and 5-model M3 / M4 cross-comparison
become possible.

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440
inferences + 480 once-per-stimulus capture calls. Expected
wall-clock on H200: ~50-70 min.
Disk: ~10-15 GB safetensors (LM hidden states only, bf16).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_idefics2_capture",
    model_id="HuggingFaceM4/idefics2-8b",
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
