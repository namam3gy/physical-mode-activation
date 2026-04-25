"""Label-free prompt config — H2 null-hypothesis probe (ROADMAP §4.9).

Re-runs the M2 factorial on the same stimuli with `open_no_label` only —
no "ball" / "circle" / "planet" token in the prompt. Paired (obj, bg,
cue, seed) comparison against M2's labeled runs quantifies the
language-prior contribution as the PMR delta at fixed image content.

Activation capture is identical to M2 (5 LM layers) so the M4 logit
lens + per-layer probe can be re-run and yield a switching-layer metric
that is no longer collapsed at L5 by the label's physics priming.

Matched to M2 for paired comparability:
- factorial: 4 obj × 3 bg × 4 cue × 1 event × 10 seeds = 480 stimuli
- temperature: 0.7, top_p: 0.95 (stochastic, same 10 seeds/cell)
- capture_lm_layers: (5, 10, 15, 20, 25)
- model: Qwen2.5-VL-7B-Instruct

The factorial spec here is informational — `run_inference` reads the
manifest from `--stimulus-dir`, so we reuse
`inputs/mvp_full_20260424-093926_e9d79da3/` directly and skip the
`01_generate_stimuli` step.

Inference size: 480 stimuli × 1 label × 1 prompt = 480 inferences.
Expected wall-clock on H200: ~10 min (no model cold start if prior M5a-ext
session loaded weights recently).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="label_free",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
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
    capture_lm_layers=(5, 10, 15, 20, 25),
    capture_lm_attentions=False,
    random_seed=42,
)
