"""M6 round-2 cross-model — InternVL3-8B-hf labeled.

Third cross-model point after Qwen2.5-VL-7B (M2) and LLaVA-1.5-7B (M6 r1).
Tests the visual-saturation hypothesis at a third VLM that uses InternLM
as the language backbone and InternViT as the vision encoder — orthogonal
to both Qwen (Qwen LM + SigLIP) and LLaVA-1.5 (Llama-2 LM + CLIP).

Smoke test confirmed `forced_choice` produces full justified responses
(not the "A" bias seen in LLaVA-1.5). FC is **excluded** from the bulk
run for time savings (~2.4s/inference at H200 → 2880 inf would take ~2 hr);
the smoke result is enough qualitative evidence that LLaVA's "A" bias is
LLaVA-specific. Open prompt + label-free are sufficient for the H1, H2,
H7 cross-model story.

Inference size: 480 stim × 3 labels × 1 prompt variant = 1440 inferences.
Expected wall-clock on H200: ~60 min.

Stimuli reused via `--stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3`.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="cross_model_internvl3",
    model_id="OpenGVLab/InternVL3-8B-hf",
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
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
