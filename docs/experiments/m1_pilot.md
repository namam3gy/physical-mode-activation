# M1 — Pilot run (2026-04-24)

- **Command**: `uv run python scripts/02_run_inference.py --config configs/pilot.py`
- **Stimulus dir**: `inputs/pilot_20260424-072216_308c86fc` (240 stimuli)
- **Output dir**: `outputs/pilot_20260424-072418_2c16efb6`
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (first-run download, bf16, sdpa on H200)
- **Wall clock**: ~8 min total (first-run HF download ~15 s, 729-shard weight load ~8 s, 480 inferences at ~1.0 it/s)
- **N predictions**: 480 (240 stimuli × 1 label "ball" × 2 prompt variants)
- PMR scored twice: once with the initial lexicon, then with the patched
  `move` → `mov` family. Numbers below are from the final
  `predictions_scored.parquet`.
- **Deep dive**: `docs/insights/m1_pilot.md`.

## Headline PMR / GAR

**By object_level** (axis A — abstraction):

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 120 | 0.575 | 0.333 | 0.325 | 0.667 |
| filled | 120 | 0.658 | 0.333 | 0.225 | 0.867 |
| shaded | 120 | 0.642 | 0.408 | 0.183 | 1.000 |
| textured | 120 | **0.808** | 0.142 | 0.167 | 0.600 |

→ H1 (monotone S-curve line → textured) **partially supported**:
endpoints behave as predicted (line 0.575 < textured 0.808) but `shaded`
and `filled` are effectively tied mid-curve. Either noise (n=120 per
cell → ~4.5 pp std error) or the shading cue alone doesn't beat a
uniform fill at this scale.

**By bg_level** (axis B):

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 240 | 0.488 | N/A |
| ground | 240 | **0.854** | 0.783 |

→ **Ground presence adds +36 pp to PMR.** Largest single-factor effect
measured; matches H3 and the cog-sci prediction that a support plane
recruits physical-object interpretation.

**By cue_level** (axis C):

| cue_level | n | pmr |
|---|---|---|
| none | 160 | 0.500 |
| wind | 160 | 0.513 |
| arrow_shadow | 160 | **1.000** |

→ `arrow_shadow` saturates PMR at 1.0 (the trajectory arrow is a
complete give-away — the model reads it as "this is where the ball will
go"). **Wind marks do essentially nothing** — the VLM does not interpret
our programmatic wind streaks as airflow. See surprise #1.

**By prompt_variant** (methodological):

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 240 | **0.800** | 0.000 | 0.917 |
| forced_choice | 240 | 0.542 | 0.450 | 0.650 |

→ When option D ("abstract shape") is offered, the model uses it 45 %
of the time; in open-ended mode it *never* spontaneously calls the
stimulus abstract. The language prior from the "ball" label fully
dominates the open variant. Instrumentation warning: PMR from
open-ended prompts is inflated by the label axis D, so the behavioral
S-curve is best read off the forced-choice subset.

## Surprises / notes

1. **Wind cue is invisible to Qwen2.5-VL-7B.** The 15 small grey arcs
   anchored to one side of the object don't move PMR relative to "no
   cue" (0.513 vs 0.500). Consider a stronger visual: blurred motion
   trail in the object's wake, or actual particle streaks oriented with
   perspective. Before the MVP-full run, improve
   `primitives.draw_wind_marks` or drop the wind level from axis C in
   favor of `motion_blur` / `dust_cloud`.
2. **The arrow+shadow cue is too strong.** PMR=1.0 means no information
   left to measure. For MVP-full, split axis C into
   `{none, cast_shadow_only, trajectory_arrow_only, both}` so we can
   see how much of the boost comes from the shadow (supports the
   Kersten/Mamassian prediction about ground-attachment) vs the arrow
   (pure directional cue).
3. **Lexicon tuning matters.** Initial stem set missed "moving" (because
   "move" ≠ prefix of "moving") and "continue", costing ~2 pp on the
   textured cell. Patched stems committed to `lexicons.py`; regression
   tests added. Future lexicon edits should go through
   `tests/test_pmr_scoring.py`.
4. **At temperature=0 all seeds produce identical generations** per
   (stimulus, prompt). RC is therefore 1.0 for every cell — not a
   useful signal at T=0. For the MVP-full run, set `temperature=0.7`
   and increase `seeds_per_cell` so RC becomes meaningful (Sub-task 1
   metric in `references/project.md` §2.2).
5. **Raw responses look sensible** (e.g., "The ball will collide with
   the surface below it" on ground cells; "The ball will remain
   stationary unless acted upon by an external force" on blank cells
   — a Newton's-first-law framing). The model's errors are systematic,
   not random.

## Next actions

- MVP-full run with the three fixes above (wind → motion trail / dust;
  split cue axis; temperature 0.7 with more seeds). Spec it out in
  `configs/mvp_full.py` before the next run.
- Enable `capture_lm_layers = (5, 10, 15, 20, 25)` so Sub-task 3
  logit-lens analysis has hidden states ready.
- Before running MVP-full, expand axis D to `("circle", "ball", "planet")`
  to measure the H2 language-prior effect.
