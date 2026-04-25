"""M8d — non-ball physical-object categories, Qwen2.5-VL-7B labeled run.

External-validity round 2 (after M8a non-circle shapes): does the
H7 label-selects-regime dissociation generalize beyond the
ball ↔ planet axis to non-ball categories?

Factorial:
    3 shapes  × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds = 480 stimuli
    × 3 label roles × 1 open prompt                     = 1440 inferences

Shapes:    car, person, bird
Obj:       line, filled, shaded, textured
Bg:        blank, ground
Cue:       none, both          (none = bare; both = cast_shadow + motion_arrow)
Event:     fall, horizontal    (two events — `horizontal` is the natural-event
                                cell for car/person/bird; `fall` is the
                                gravity stress test reused from M8a)
Seeds:     5

Labels per shape via `LABELS_BY_SHAPE` (see prompts.py):
    car    -> (car,    silhouette,    figurine)
    person -> (person, stick figure,  statue)
    bird   -> (bird,   silhouette,    duck)

`cfg.labels` lists role names; `inference.run` resolves them per-row to
literal labels. Sampling settings match M8a so paired-delta numbers are
comparable to the circle baseline.

Analysis convention (see design spec §2.2):
- H1 ramp uses `event ∈ {fall, horizontal}` union (the abstraction signal
  is event-independent).
- H7 paired-delta is reported on the `horizontal` subset (natural-event
  cell where regime selection is cleanest), with the `fall` subset as a
  stress test.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
