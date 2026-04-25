"""M8c — real photographs, Qwen2.5-VL-7B labeled run.

External-validity round 3 (after M8a non-circle synthetic shapes and
M8d non-ball categories): does the H7 / H-encoder-saturation pattern
hold on real photographs?

The factorial is NOT used by inference (a pre-curated manifest under
`inputs/m8c_photos_<ts>/manifest.parquet` provides the 60 photos:
12 × {ball, car, person, bird, abstract}). It is included only because
`EvalConfig` requires it; `scripts/01_generate_stimuli.py` is *not*
invoked for M8c.

Per-photo label triplet via `LABELS_BY_SHAPE`:
    ball     -> (ball,    circle,        planet)        — same as circle
    car      -> (car,     silhouette,    figurine)
    person   -> (person,  stick figure,  statue)
    bird     -> (bird,    silhouette,    duck)
    abstract -> (object,  drawing,       diagram)

Sampling settings match M8d so paired-delta numbers are comparable.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8c_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("ball", "car", "person", "bird", "abstract"),
        object_levels=("photo",),
        bg_levels=("natural",),
        cue_levels=("none",),
        event_templates=("fall",),  # placeholder — actual events come from manifest
        seeds_per_cell=12,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
