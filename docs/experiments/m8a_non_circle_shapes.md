# M8a — Non-circle synthetic shapes (run log)

External-validity round 1, executed 2026-04-25.

## Setup

- 5 shapes × 4 obj_levels × 2 bg_levels × 2 cue_levels × 1 event × 5 seeds
  = **400 stimuli** (one common stim dir; reused across all 4 inference
  configs).
- Stimulus directory: `inputs/m8a_qwen_20260425-091713_8af4836f/`.
- Sampling: T=0.7, top_p=0.95, max_new_tokens=96 (matches M6 r1 / r2).
- Single GPU 1, sequential. Total wall clock: **~43 minutes**.

## Configs

| Config | Run dir | n |
|---|---|---|
| `m8a_qwen.py` | `outputs/m8a_qwen_20260425-092423_bf03832e/` | 1200 (400 × 3 roles) |
| `m8a_qwen_label_free.py` | `outputs/m8a_qwen_label_free_20260425-094239_26c66949/` | 400 |
| `m8a_llava.py` | `outputs/m8a_llava_20260425-095133_a2b5f318/` | 1200 |
| `m8a_llava_label_free.py` | `outputs/m8a_llava_label_free_20260425-100253_99a20dd8/` | 400 |

## Code changes

- `src/physical_mode/stimuli/primitives.py`: added `Shape` Literal +
  4 new shape classes × 4 abstraction modes
  (`_draw_square / _draw_triangle / _draw_hexagon / _draw_polygon`,
  each with line / filled / shaded / textured). Directional shading
  (Lambert-ish, light from upper-left) replaces radial shading for
  the non-spheres. Polygon vertex set is seeded (5–7 vertices, jittered
  radii / angles).
- `src/physical_mode/stimuli/scenes.py`: pass `shape` to `draw_object`.
- `src/physical_mode/config.py`: added `Shape` Literal,
  `StimulusRow.shape` field (default `"circle"` — backward compat),
  `FactorialSpec.shapes` axis. Sample-id only includes shape when
  `len(shapes) > 1`, so existing single-shape configs reproduce
  unchanged.
- `src/physical_mode/inference/prompts.py`: added
  `LABELS_BY_SHAPE` dict + `labels_for_shape()` helper. Shape-keyed
  triplets `(physical, abstract, exotic)`. After advisor review,
  polygon's exotic was changed from `shape` (more abstract than the
  geometric class — would have inverted role ordering) to `boulder`.
- `src/physical_mode/inference/run.py`: when `len(cfg.factorial.shapes) > 1`,
  `cfg.labels` is interpreted as a list of *role names*
  (`physical / abstract / exotic`) rather than literal labels.
  Per-row literal labels are dispatched via `LABELS_BY_SHAPE`.
- `src/physical_mode/metrics/pmr.py`: added `shape` to the per-axis
  summary loop.
- `scripts/m8a_spot_check.py`, `scripts/m8a_run_all.sh`,
  `scripts/m8a_analyze.py`, `scripts/m8a_figures.py` — driver +
  analysis utilities.

50 unit tests still pass (`uv run python -m pytest`).

## Visual verification

- `docs/figures/m8a_shape_grid.png` — 5×4 spot-check rendered before
  generating the 400-stim set.
- `docs/figures/m8a_full_scene_samples.png` — 5 representative
  full-scene cells (textured + ground + arrow + cast shadow).

## Pre-registered criteria scoring (final)

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | 3/5 ✗ | 4/5 ✓ |
| H7 (phys>abs)     | 1/5 ✗ | 4/5 ✓ |
| H7-GAR            | 1/5 ✗ | 5/5 ✓ |
| Visual-sat. delta | 3/5 ✓ borderline | 5/5 ✓ |

Per-shape detail: see `docs/insights/m8a_non_circle_shapes.md` §Results.

## Headline numbers

PMR(_nolabel) baseline by (model × shape):

| shape    | Qwen  | LLaVA |
|----------|-------|-------|
| circle   | 0.825 | 0.288 |
| square   | 0.925 | 0.088 |
| triangle | 0.788 | 0.075 |
| hexagon  | 0.875 | 0.150 |
| polygon  | 0.775 | 0.275 |

Qwen sits at 0.78–0.93 across shapes — vision encoder already commits
to physics-mode. LLaVA at 0.075–0.288 — labels are doing most of the
behavioral work.

Paired-delta `PMR(physical) − PMR(_nolabel)`:

| shape    | Qwen   | LLaVA  |
|----------|--------|--------|
| circle   | -0.013 | +0.575 |
| square   | -0.200 | +0.625 |
| triangle | -0.025 | +0.125 |
| hexagon  | -0.125 | +0.550 |
| polygon  | +0.025 | +0.487 |

LLaVA's `physical` label gives **+0.125 to +0.625** PMR boost on every
shape. Qwen's `physical` label gives **near-zero or negative** change —
the encoder already encodes the physical reading.

## Notable per-shape findings

- **Qwen `square`**: paired-delta -0.200 / -0.275 / -0.212 — a clean
  cross-shape replication of M4b's circle "label suppresses physics"
  effect. The label adds nothing the visual didn't already provide,
  *and* mildly suppresses physics-mode language (the model is more
  likely to write "the brick is on the ground, gray, weathered" than
  "the brick will fall").
- **LLaVA `triangle`**: paired-delta only +0.125 / +0.100 / +0.100,
  PMR(physical=wedge) = 0.200 vs ~0.7 for ball/brick/nut/rock. Almost
  certainly a label-quality issue: "wedge" is a weak physical-object
  cue. Future runs should test alternative physical labels for
  triangle (`pyramid`, `sandbag`, `ramp`).
- **LLaVA `polygon` abstract = -0.050**: only LLaVA paired-delta to
  go negative. "Polygon" reads as a math term, not a physical
  descriptor. The role taxonomy leaks for irregular shapes without
  common-vocabulary geometric nouns.
- **Qwen `circle` planet (exotic) → GAR = 0.175**: while GAR(ball) =
  0.675 and GAR(circle) = 0.700. The "planet" prior actively
  *suppresses* the falling-down language ("orbits the sun, rotates on
  axis"), and the model's `_nolabel` baseline is closer to the
  ball/circle reading than to the planet reading. Cross-shape this
  doesn't manifest as cleanly because nut / coin / boulder don't
  carry as strong a non-falling prior as planet.
- **Qwen `square` GAR**: physical=0.475, abstract=0.500, exotic=0.525
  — a flat row driven by Qwen's near-saturation. The encoder is
  already telling the LM "this is something that falls when there's a
  ground line below it"; the role-name doesn't move that.

## Raw artifacts

- `outputs/m8a_qwen_*/predictions{.jsonl,.parquet,.csv}`,
  `m8a_pmr_by_shape_*.csv`, `m8a_ramp_per_shape.csv`.
- `outputs/m8a_paired_deltas.csv` — Qwen + LLaVA × 5 shapes × 3 roles.
- `outputs/m8a_run_all.log` — full 4-run log (~487 KB).
- `inputs/m8a_qwen_20260425-091713_8af4836f/` — 400 PNG stimuli +
  manifest.parquet.
