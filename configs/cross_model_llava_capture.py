"""M6 round-2 — LLaVA-1.5-7B with LM activation captures for cross-model M3/M4.

Re-runs the M6 r1 behavioral protocol on LLaVA-1.5-7B with
`capture_lm_layers=(5, 10, 15, 20, 25)` enabled, so we can extend
M3 vision probing and M4 logit-lens / per-layer probing to LLaVA.

Same layer indices as M2 / M6 r1 Qwen capture, so direct cross-model
comparison at matching layer indices is possible. Note: LLaVA-1.5 uses
LLaMA-2-7B (32 LM layers) while Qwen2.5-VL uses 28 LM layers — same
absolute index = different *relative* depth. Both interpretations are
worth reporting.

Vision-tower capture is enabled too (CLIP-ViT-L has 24 layers; sample at
3, 7, 11, 15, 19, 23 to match the M3 Qwen capture density).

Inference size: 480 stim × 3 labels × 1 prompt (open) = 1440 inferences,
plus 480 once-per-stimulus capture calls.
Expected wall-clock on H200: ~50-70 min.
Disk: ~10-15 GB additional safetensors.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_llava_capture",
    model_id="llava-hf/llava-1.5-7b-hf",
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
