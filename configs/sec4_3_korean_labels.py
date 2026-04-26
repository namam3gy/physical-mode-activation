"""§4.3 — Qwen2.5-VL with Korean labels (공/원/행성) on M8a circle stim.

Tests whether the language of the label (Korean vs English) shifts PMR
when the question prompt is the same English template. Qwen2.5-VL is
multilingual, so Korean labels should activate the same concepts as
English ones — but they may have different prior strength in the
model's training distribution.

Single-shape (circle only) so we can pass explicit Korean label tuple
without fighting the LABELS_BY_SHAPE dispatch.

Stim: circle from inputs/m8a_qwen_<ts> (80 stim = 4 obj × 2 bg × 2 cue × 5 seed).
Labels: 공 (gong = ball), 원 (won = circle), 행성 (haengseong = planet).
n_inferences = 80 × 3 = 240.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="sec4_3_korean_labels_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("circle",),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall",),
        seeds_per_cell=5,
    ),
    labels=("공", "원", "행성"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
