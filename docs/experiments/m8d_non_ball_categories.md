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

| Config | Run dir | n | wall |
|---|---|---|---|
| `m8d_qwen.py` | `outputs/m8d_qwen_20260425-151811_6c200dc8/` | 1440 (480 × 3 roles) | 12.6 min |
| `m8d_qwen_label_free.py` | `outputs/m8d_qwen_label_free_20260425-153049_e1f19e0d/` | 480 | 6.2 min |
| `m8d_llava.py` | `outputs/m8d_llava_20260425-153701_ea751428/` | 1440 | 8.8 min |
| `m8d_llava_label_free.py` | `outputs/m8d_llava_label_free_20260425-154549_16bc0be7/` | 480 | 4.3 min |

Total: **31.9 min** wall clock on H200 GPU 0 (15:18:07 → 15:50:03).

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

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| Visual-sat. delta | 1/3 (bird) | 2/3 (car, bird; person flips negative) |

Per-category detail: see `docs/insights/m8d_non_ball_categories.md` §Results.

## Headline numbers

`PMR_regime(_nolabel)` baseline by (model × category) on the **horizontal** subset:

| category | Qwen  | LLaVA |
|----------|-------|-------|
| car      | 1.000 | 0.550 |
| person   | 0.975 | 0.838 |
| bird     | 0.862 | 0.688 |

`PMR_regime` paired-delta `physical − _nolabel` on `horizontal` subset:

| category | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.000 | **+0.275** |
| person   | +0.025 | -0.100 |
| bird     | +0.125 | **+0.262** |

`PMR_regime` ramp `(textured − line)` per category × model (event union):

| category | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.008 | -0.033 |
| person   | -0.009 | -0.033 |
| bird     | +0.008 | -0.017 |

H7 paired-difference `PMR_regime(physical) − PMR_regime(abstract)` on `horizontal` subset:

| category | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.012 | **+0.525** |
| person   | +0.012 | +0.138 |
| bird     | +0.038 | **+0.550** |

H7 paired-difference at the *kinetic-fraction* level (Qwen ceiling-rescue):

| category | Qwen Δ kin_frac (physical − abstract) | Qwen Δ kin_frac (physical − exotic) |
|----------|--------|--------|
| car      | +0.063 | **+0.138** |
| person   | -0.013 | **+0.162** |
| bird     | +0.106 | +0.019  |

LLaVA `physical − exotic` kin_frac diff (for completeness):

| category | LLaVA Δ kin_frac (physical − exotic) |
|----------|--------|
| car      | +0.262 |
| person   | +0.138 |
| bird     | +0.062 |

## Classifier validation (54-stim hand-annotation)

`scripts/m8d_hand_annotate.py --mode sample --n-per-cell 3 --seed 42`
sampled **54 stratified rows** (model × category × role × 3 = 54). Hand
annotation applied a richer English vocabulary than the keyword
classifier — kinetic verbs, static states, abstract-reject phrases —
mirroring genuine human reading.

- N hand-annotated rows: 54
- Combined error rate: **0.056** (3 mismatches; threshold < 0.150 — **PASS**)
- Per-regime precision / recall:
  - kinetic:  precision 0.949 / recall 0.974
  - static:   precision 1.000 / recall 0.778
  - abstract: precision NaN / recall NaN  (no abstract responses in sample)
  - ambiguous: precision 0.875 / recall 1.000

Mismatches (3/54):
- 2 × Qwen person/abstract: "stick figure will *remain stationary*, no
  indication of *movement*" — the keyword classifier sees `mov` (kinetic)
  AND `remain`/`stationary` (static) and resolves kinetic-first; human
  reading resolves static. Stem-matching limitation.
- 1 × LLaVA person/exotic: "the statue will be *pulled* away from the
  line" — `pull` is in PHYSICS_VERB_STEMS but not in the per-category
  kinetic set, so classify_regime returns ambiguous. Per-category
  kinetic lexicon could be widened, but the rate is well below threshold.

CSV: `docs/experiments/m8d_hand_annotate.csv` (54 rows with predicted
+ hand columns).

## Files

- `outputs/m8d_summary/` — per-model rollups + concatenated annotated
  parquet (`m8d_qwen_annotated.parquet`, `m8d_llava_annotated.parquet`).
- `docs/figures/m8d_*.png` — 5 figures.
- `notebooks/m8d_non_ball_categories.ipynb` — cell-by-cell reproduction.
