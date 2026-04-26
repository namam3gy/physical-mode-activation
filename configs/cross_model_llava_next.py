"""M2 cross-model — LLaVA-Next-Mistral-7B with LM activation captures.

Extends the M2 protocol (5-axis factorial, T=0.7, 480 stim × 3 labels)
to LLaVA-Next-7B-Mistral. Mirrors `cross_model_llava_capture.py` for
LLaVA-1.5 (M6 r2b) so the 5-model M2-stim chain (Qwen / LLaVA-1.5 /
LLaVA-Next / Idefics2 / InternVL3) is apples-to-apples.

LLaVA-Next note: 4-axis architectural change vs LLaVA-1.5 (AnyRes
tiling, fusion projector, training, LM family = Mistral-7B). Same
CLIP-ViT-L encoder family. M6 r6 already has M8a behavioral on this
model; this config adds the M2-stim run with captures.

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440
inferences + 480 once-per-stimulus capture calls. Expected
wall-clock on H200: ~50-70 min.
Disk: ~10-15 GB safetensors (LM hidden states only, bf16).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava_next_capture",
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
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
