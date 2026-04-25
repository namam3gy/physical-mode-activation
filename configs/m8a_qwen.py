"""M8a — non-circle synthetic shapes, Qwen2.5-VL-7B run.

External-validity round 1: does the visual-saturation / paired-delta /
GAR-by-label pattern observed for `circle` generalize across other
geometric shape classes?

Factorial (compact, designed to fit a single GPU-1 session):
    5 shapes  × 4 obj × 2 bg × 2 cue × 1 event × 5 seeds = 400 stimuli
    × 3 label roles × 1 open prompt                    = 1200 inferences

Shapes:    circle, square, triangle, hexagon, polygon
Obj:       line, filled, shaded, textured
Bg:        blank, ground       (drop `scene` to halve the budget)
Cue:       none, both          (none = bare; both = cast_shadow + motion_arrow)
Event:     fall                (single event keeps GAR comparable across shapes)
Seeds:     5                   (5 seeds × 5 shapes = 25 obs per (obj,bg,cue) cell)

Labels are dispatched per-shape via `LABELS_BY_SHAPE` in
`physical_mode.inference.prompts`. The role triple is
(physical, abstract, exotic), so each shape contributes:
    physical : ball / brick / wedge / nut / rock
    abstract : circle / square / triangle / hexagon / polygon
    exotic   : planet / tile / sign / coin / shape

`cfg.labels` therefore lists *role names* rather than literal labels.

Sampling settings match the cross-model M6 round so that paired-delta
numbers are comparable to the circle-only baseline (`mvp_full` open).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8a_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("circle", "square", "triangle", "hexagon", "polygon"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall",),
        seeds_per_cell=5,
    ),
    # Role names — see prompts.LABELS_BY_SHAPE.
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
