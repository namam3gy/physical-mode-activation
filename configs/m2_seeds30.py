"""C1 stim regen — M2 factorial with seeds_per_cell = 30 (was 10).

Used by `scripts/chain_b9_b13_c1.sh` to enable real n=30 intervention runs.
Identical to M2 except seeds_per_cell.

Inference is NOT run on this config (we only need the rendered stim for
sae_intervention.py to sample). 480 → 1440 stim.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m2_seeds30",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",  # not used (no inference)
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground", "scene"),
        cue_levels=("none", "cast_shadow", "motion_arrow", "both"),
        event_templates=("fall",),
        seeds_per_cell=30,
    ),
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
