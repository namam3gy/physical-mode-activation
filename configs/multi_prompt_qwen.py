"""Multi-prompt cross-task generalization — Qwen2.5-VL-7B (M-MP, Track B Pillar A).

Tests whether the physics-mode commitment mechanism (`v_L10` direction,
encoder-side SAE physics-cue features) is **task-agnostic**, by running
the same M2 stimuli under three cognitively distinct prompts:

  1. `open`             — kinetic next-state prediction (current paper baseline)
  2. `describe_scene`   — free-form scene description
  3. `meta_phys_yesno`  — direct meta-categorization probe ("is this a real-world physical event?")

Same stim, same labels, different question. If the same `v_L10` /
SAE features fire across all 3 prompts, the mechanism is genuinely
"physics-mode commitment" rather than "next-state prediction" specific.

Reuses M2 stim via `--stimulus-dir inputs/mvp_full_<id>` at run time.

Inference size: 480 stim × 3 labels × 3 prompts = 4320 inferences.
Expected wall-clock on H200 (single GPU, bf16, no captures): ~25-35 min.
No captures (LM activations already exist from M2; M5a/M5b re-runs use
those existing v_L / SAE).

Track B reference: `references/submission_plan.md` Pillar A → M-MP.
Gap fixed: G1 (single-task evaluation) per `references/paper_gaps.md`.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="multi_prompt_qwen",
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
    labels=("circle", "ball", "planet"),
    prompt_variants=("open", "describe_scene", "meta_phys_yesno"),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
