# M8d — Non-ball physical-object categories (run log)

External-validity round 2, executed 2026-04-25.

## Setup

- 3 categories × 4 obj_levels × 2 bg_levels × 2 cue_levels × 2 events × 5 seeds
  = **480 stimuli** (one common stim dir; reused across all 4 inference
  configs).
- Stimulus directory: `inputs/m8d_qwen_<ts>_<hash>/`.
- Sampling: T=0.7, top_p=0.95, max_new_tokens=96 (matches M6 r1 / r2 / M8a).
- Single GPU 0, sequential. Total wall clock: **~30-35 minutes**.

## Configs

| Config | Run dir | n |
|---|---|---|
| `m8d_qwen.py` | `outputs/m8d_qwen_<ts>_<hash>/` | 1440 (480 × 3 roles) |
| `m8d_qwen_label_free.py` | `outputs/m8d_qwen_label_free_<ts>_<hash>/` | 480 |
| `m8d_llava.py` | `outputs/m8d_llava_<ts>_<hash>/` | 1440 |
| `m8d_llava_label_free.py` | `outputs/m8d_llava_label_free_<ts>_<hash>/` | 480 |

## Code changes

- `src/physical_mode/stimuli/primitives.py`: added 3 categories
  (car / person / bird) × 4 abstraction modes via 12 new draw functions
  (`_draw_*_car`, `_draw_*_person`, `_draw_*_bird` in line / filled /
  shaded / textured). `Shape` literal extended.
- `src/physical_mode/stimuli/scenes.py`: ground-bound shapes
  (car, person) sit *on* the ground for `horizontal` events so the
  natural-motion reading is geometrically consistent with the cast
  shadow; birds keep midair placement.
- `src/physical_mode/config.py`: `Shape` literal extended; `Label`
  literal extended with `(car, silhouette, figurine)`,
  `(person, "stick figure", statue)`, `(bird, duck)` — silhouette is
  reused across car and bird.
- `src/physical_mode/inference/prompts.py`:
  `LABELS_BY_SHAPE` augmented with car / person / bird triplets.
- `src/physical_mode/metrics/lexicons.py`: added
  `CATEGORY_REGIME_KEYWORDS` for car / person / bird (kinetic + static
  word stems). Existing `PHYSICS_VERB_STEMS`, `HOLD_STILL_STEMS`,
  `DOWN_DIRECTION_PHRASES`, `ABSTRACT_MARKERS` are unchanged so M8a
  scoring remains bit-identical.
- `src/physical_mode/metrics/pmr.py`: added `classify_regime(category,
  text) → {kinetic, static, abstract, ambiguous}`. Original `score_pmr`
  / `score_gar` unchanged.
- `scripts/m8d_spot_check.py`, `scripts/m8d_run_all.sh`,
  `scripts/m8d_analyze.py`, `scripts/m8d_figures.py`,
  `scripts/m8d_hand_annotate.py` — driver + analysis utilities.
- `tests/test_m8d_labels.py`, `tests/test_m8d_primitives.py`,
  `tests/test_m8d_regime.py`. 123 unit tests pass.

## Visual verification

- `docs/figures/m8d_shape_grid.png` — 3×4 spot-check rendered before
  generating the 480-stim set. Each (category × abstraction-level)
  cell is visually distinct.
- `docs/figures/m8d_full_scene_samples.png` — 3 categories × 4
  abstraction × 2 events full-scene samples.

## Pre-registered criteria scoring (final)

_Filled in after run completes:_

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | _TBD_ | _TBD_ |
| H7 (phys>abs)     | _TBD_ | _TBD_ |
| Visual-sat. delta | _TBD_ | _TBD_ |

Per-category detail: see `docs/insights/m8d_non_ball_categories.md` §Results.

## Headline numbers

_To be filled in after run completes._

`PMR_regime(_nolabel)` baseline by (model × category):

| category | Qwen  | LLaVA |
|----------|-------|-------|
| car      | _TBD_ | _TBD_ |
| person   | _TBD_ | _TBD_ |
| bird     | _TBD_ | _TBD_ |

`PMR_regime` paired-delta `physical − _nolabel` on `horizontal` subset:

| category | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | _TBD_  | _TBD_  |
| person   | _TBD_  | _TBD_  |
| bird     | _TBD_  | _TBD_  |

`PMR_regime` ramp `(textured − line)` per category × model:

| category | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | _TBD_  | _TBD_  |
| person   | _TBD_  | _TBD_  |
| bird     | _TBD_  | _TBD_  |

## Classifier validation (50-stim hand-annotation)

_Filled in after `m8d_hand_annotate.py` is run on the predictions._

- N hand-annotated rows: _TBD_
- Combined error rate: _TBD_ (threshold for paper-ready signal: < 0.150)
- Per-regime confusion: see `docs/experiments/m8d_hand_annotate.csv`.

## Files

- `outputs/m8d_summary/` — per-model rollups + concatenated annotated
  parquet (`m8d_qwen_annotated.parquet`, `m8d_llava_annotated.parquet`).
- `docs/figures/m8d_*.png` — 5 figures.
- `notebooks/m8d_non_ball_categories.ipynb` — cell-by-cell reproduction.
