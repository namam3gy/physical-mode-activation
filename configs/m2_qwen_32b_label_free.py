"""§4.8 follow-up — Qwen2.5-VL 32B label-free arm on M2 stim.

Companion to `m2_qwen_32b.py` (labeled, ✅ done 2026-04-28). Adds the H2
cross-scale check: under `open_no_label`, does 32B PMR_nolabel still match
7B's saturation, or does scale show up here?

Inference: 480 stim × 1 prompt × 1 label = 480 inferences.
Wall-clock on H200 (single GPU, bf16): ~5 min.

Reuses M2 stim (`inputs/mvp_full_20260424-093926_e9d79da3`).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m2_qwen_32b_label_free",
    model_id="Qwen/Qwen2.5-VL-32B-Instruct",
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
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
