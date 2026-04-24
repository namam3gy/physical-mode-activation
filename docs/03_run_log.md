# Run log

Append a timestamped entry for each pilot / full run. Keep entries short:
exact command, config, row counts, headline numbers, anything surprising.
Do **not** paste full prediction tables — they live in `outputs/<run_id>/`.

Template:

```
## YYYY-MM-DD <run_name>

- Command: `uv run python scripts/02_run_inference.py --config configs/<name>.py`
- Stimulus dir: inputs/<run_id>
- Output dir: outputs/<run_id>
- Model: Qwen/Qwen2.5-VL-7B-Instruct (or other)
- Wall clock: <hh:mm>
- N predictions: <count>

### Headline PMR by object_level

| object_level | n | pmr | gar |
|---|---|---|---|
| line | … | … | … |
| filled | … | … | … |
| shaded | … | … | … |
| textured | … | … | … |

### Notes / surprises

- …
```

---

<!-- New entries below this line, newest on top. -->

## 2026-04-24 vision probing (M3, Sub-task 2)

- Command: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 --output-dir outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations --layers 3,7,11,15,19,23,27,31`
- Capture time: ~70 s (10 it/s forward-only) + model load 20 s
- Disk: 12 GB (480 stimuli × 8 layers × (1296 tokens × 1280 dim × 2 bytes))
- Probes: sklearn LogisticRegression, StratifiedKFold (5), mean-pool token axis.
  Code: `src/physical_mode/probing/vision.py`
- Outputs: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_vision/*.csv`

### Headline: the encoder-decoder boomerang is real and **saturated**

**Stimulus-property probes (AUC by layer × target)**: the vision encoder
linearly separates **every** factorial axis at AUC=1.00 from layer 3
onward:

| target | layer 3 | 15 | 31 |
|---|---|---|---|
| y_bg_ground (bg != blank) | 1.00 | 1.00 | 1.00 |
| y_bg_scene | 1.00 | 1.00 | 1.00 |
| y_obj_3d (shaded/textured) | 1.00 | 1.00 | 1.00 |
| y_obj_textured | 1.00 | 1.00 | 1.00 |
| y_cue_any | 1.00 | 1.00 | 1.00 |
| y_cue_shadow | 1.00 | 1.00 | 1.00 |
| y_cue_arrow | 1.00 | 1.00 | 1.00 |

→ **Any downstream system with linear access to these features could
recover every factorial axis perfectly.** The encoder has zero
information bottleneck on the stimulus descriptor.

Meanwhile, forced-choice **behavioral PMR** on the same stimuli:

| axis | level | beh. PMR |
|---|---|---|
| bg | blank / ground / scene | 0.51 / 0.71 / 0.77 |
| object | line / filled / shaded / textured | 0.58 / 0.65 / 0.71 / 0.71 |
| cue | none / cast_shadow / motion_arrow / both | 0.28 / 0.49 / 0.93 / 0.95 |

The **LM is imperfectly sensitive** to properties the encoder perfectly
encodes. The gap is entirely downstream of the visual features.

### Controlled probe: the no-cue subset

Restrict to `cue_level=none` (120 stimuli, no red-arrow shortcut, no
cast shadow). Train encoder probe on behavioral forced-choice PMR.

| layer | encoder AUC | behavioural PMR |
|---|---|---|
| 3 | 0.793 | 0.28 |
| 11 | 0.852 | 0.28 |
| **19** | **0.890** | 0.28 |
| 27 | 0.852 | 0.28 |
| 31 | 0.859 | 0.28 |

The encoder's layer-19 activations determine which "no-cue" stimuli the
LM will call physics-mode with **AUC 0.89** — yet only **28 %** of those
stimuli actually trigger physics-mode in behavior. Read as calibration:
*"encoder knows which cells trigger physics-mode, but LM only lets through
a fraction of those."*

### Per-object-level encoder AUC vs behavior (forced-choice)

| object_level | encoder AUC @ L31 | behavioural PMR | gap |
|---|---|---|---|
| line | 0.944 | 0.583 | **+0.361** |
| filled | 0.950 | 0.647 | +0.303 |
| shaded | 0.943 | 0.711 | +0.232 |
| textured | 0.952 | 0.714 | +0.238 |

Boomerang gap is **largest for the most abstract object** — consistent
with H4 (the language-prior-vs-vision tension worsens with abstraction).

### Methodological caveat

Programmatic stimuli make encoder AUC 1.0 trivially attainable (a simple
mean-pooled representation is enough). The sharpness of the 1.0 vs
behavioral-PMR gap is evidence that **when the information is unambiguously
present, the LM still fails to route it into physical-mode behavior** — but
in photographic stimuli the encoder side may itself be imperfect, and the
effect size on real-world inputs needs cross-validation. M6 (cross-model)
and the axis A photorealistic extension will address this.

### Hypothesis scorecard post-M3

| H | status after M2 | status after M3 | change |
|---|---|---|---|
| **H-boomerang** | 후보 (연구계획 §1.4) | **지지 (증거 포화)** | encoder AUC 1.0 on every axis; behavioral PMR 0.28-0.95 |
| H4 (open-forced gap) | 지지 | **지지 + mechanism** | Per-object-level encoder AUC ~ constant (~0.95) while behavior varies (0.58-0.71); gap concentrated in LM |
| H6 (shadow standalone) | 지지 (수정) | **지지** | Encoder AUC=1.0 on y_cue_shadow → information full; LM uses it partially (0.49 PMR) |

### Unlocks

- **Sub-task 3 (M4) now has full machinery ready**: LM hidden states at
  5 layers (from M2) + vision hidden states at 8 layers (from M3).
  Logit lens on LM hidden states at visual-token positions is the next
  natural figure.
- **Sub-task 4 (M5, activation patching)** prereq: need attention
  capture. Flip `capture_lm_attentions=True` and rerun mini-batch.
- **Photorealistic stimulus extension** (§4 additional idea) will test
  whether the boomerang survives when encoder AUC is not saturated.

## 2026-04-24 mvp_full (M2)

- Command: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`
- Stimulus dir: `inputs/mvp_full_20260424-093926_e9d79da3` (480 stimuli)
- Output dir: `outputs/mvp_full_20260424-094103_8ae1fa3d`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, bf16, sdpa, **T=0.7, top_p=0.95**
- Factorial: 4 obj × 3 bg × 4 cue × 1 event (fall) × 10 seeds × 3 labels × 2 prompts = **2880 inferences**
- Activation capture: LM layers (5, 10, 15, 20, 25), hidden states only (no attentions)
  → 5.2 GB across 480 `.safetensors` files (~11.5 MB/stimulus, 324 visual tokens)
- Wall clock: ~55 min end-to-end (1.1 s/inference including capture)
- M2 config differences vs pilot documented in `configs/mvp_full.py` header and `docs/05_insights.md` §6

### Success-criteria scorecard (from ROADMAP M2)

| criterion | target | observed | status |
|---|---|---|---|
| Monotone S-curve over object_level (forced-choice) | monotone | forced: line 0.583 < filled 0.647 < shaded 0.711 < textured 0.714 | ✅ |
| Open-vs-forced gap at every object_level | >0 everywhere | 22-32 pp (line 32, filled 29, shaded 22, textured 24) | ✅ |
| cast_shadow alone > none + 20 pp | +20 pp | +18.4 pp averaged (+23.4 at blank, +18.4 at ground, +10.8 at scene) | ✅ (close; edge conditions satisfied) |
| RC < 1 cells exist | some | 103/288 cells (35.8%) with RC<1; mean RC=0.918 | ✅ |
| `outputs/*/activations/` populated | yes | 480 safetensors, LM hidden only | ✅ |

### Headline PMR tables

**Overall**: n=2880, PMR=0.797, hold_still=0.152, abstract_reject=0.160, GAR=0.656.

**By object_level (axis A)** — H1 now cleanly supported:

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 720 | 0.744 | 0.193 | 0.203 | 0.594 |
| filled | 720 | 0.790 | 0.153 | 0.168 | 0.646 |
| shaded | 720 | 0.822 | 0.136 | 0.139 | 0.671 |
| textured | 720 | **0.832** | 0.126 | 0.131 | 0.713 |

Monotone across all 4 levels (no more mid-curve tie). Endpoints gap = +8.8 pp.

**By bg_level (axis B)** — scene > ground > blank, replicating pilot's ground effect:

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 960 | 0.669 | — |
| ground | 960 | 0.842 | 0.648 |
| scene | 960 | **0.881** | 0.664 |

Blank → scene = +21 pp (similar to pilot's +36 pp blank→ground; T=0.7 softened the delta).

**By cue_level (axis C)** — H6 decomposition successful:

| cue_level | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| none | 720 | 0.540 | 0.347 | 0.479 |
| cast_shadow | 720 | **0.715** | 0.238 | 0.546 |
| motion_arrow | 720 | 0.964 | 0.031 | 0.860 |
| both | 720 | 0.969 | 0.025 | 0.738 |

Cast shadow alone: **+17.5 pp above none**. Arrow saturates at 0.96; adding shadow on top nudges to 0.97. The pilot's `arrow_shadow=1.00` is explained: arrow does essentially all the work; shadow's contribution is measurable but secondary.

**Per-bg decomposition** (shadow effect shrinks in richer backgrounds — saturation pattern):

| bg | none | shadow | arrow | both |
|---|---|---|---|---|
| blank | 0.287 | 0.521 (+23.4) | 0.912 | 0.954 |
| ground | 0.608 | 0.792 (+18.4) | 0.992 | 0.975 |
| scene | 0.725 | 0.833 (+10.8) | 0.988 | 0.979 |

**By prompt_variant** — open-vs-forced gap even larger than pilot:

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 1440 | **0.931** | 0.002 | 0.593 |
| forced_choice | 1440 | 0.664 | 0.318 | 0.719 |

Open-ended never self-identifies stimulus as abstract (3 out of 1440). Forced-choice rejects 32 %. Per-object_level gap ranges +22 pp (textured) to +32 pp (line) — **larger for more abstract objects**.

**By label (axis D)** — H2 directly quantified:

| label | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| ball | 960 | **0.892** | 0.072 | 0.786 |
| circle | 960 | 0.746 | 0.186 | 0.698 |
| planet | 960 | 0.754 | 0.222 | 0.483 |

**Label × object_level interaction**:

| obj \\ label | circle | ball | planet |
|---|---|---|---|
| line | 0.692 | **0.846** | 0.696 |
| filled | 0.729 | **0.900** | 0.742 |
| shaded | 0.779 | **0.888** | 0.800 |
| textured | 0.783 | **0.933** | 0.779 |

**Remarkable**: `line + ball` (PMR 0.846) > `textured + circle` (0.783) — language prior dominates visual cue.

### Striking qualitative finding — label flips the *kind* of physics

Same stimulus (textured ball + ground + no cue, open-ended prompt), three labels:

| label | response |
|---|---|
| circle | "The circle is likely to remain static unless acted upon by an external force." |
| ball | "The ball will continue rolling down the incline." |
| planet | "The planet will continue moving along its orbital path around the Sun." |

The label doesn't just toggle physics-mode on/off — it selects *which physics regime* the model applies. `planet` invokes orbital mechanics (GAR=0.48), `ball` invokes gravity (GAR=0.79). This is a paper-worthy qualitative result for Figure 2.

### Hypothesis scorecard post-M2

| H | pilot status | M2 status | change |
|---|---|---|---|
| H1 (S-curve) | 부분 지지 | **지지** | Middle tie resolved with T=0.7 + 10 seeds |
| H2 (ball label) | 강하게 지지 | **정량화** | +15 pp; `ball>circle` at every object_level |
| H3 (scene inconsistency) | 미검증 | 여전히 미검증 | axis E deferred from M2 |
| H4 (open-forced gap) | 후보 | **지지** | Gap +22 to +32 pp; monotone in abstraction |
| H5 (ground vs texture) | 일방향 | **혼재** | bg delta (+21 pp) > object delta (+9 pp); supports H5 but scene > ground now |
| H6 (shadow alone) | 분해 필요 | **지지** | Shadow +17.5 pp above none; not just annotation |

### New observations (candidate hypotheses)

- **Per-label GAR varies dramatically** (ball 0.79 / circle 0.70 / planet 0.48). "Planet" response invokes orbital physics, not gravity. Label effect on PMR is *binary-ish* but on GAR is *categorical*.
- **Saturation structure**: `motion_arrow` ~≈ `both` at 0.96-0.97. Arrow is the dominant cue; shadow's marginal contribution is strong only when the base is abstract (blank bg).
- **Open-ended is not broken** — the language-prior dominance is systematic (stronger for more abstract objects). This is consistent with the "hallucinated grounding" pattern in Vo et al. 2025.

### Next actions

- **M3 (Sub-task 2 — vision encoder probing)** is unblocked: LM activations captured. Vision encoder capture still needs implementation (`PhysModeVLM.capture` extension). Draft: add 3-5 layers of vision encoder (Qwen2.5-VL's SigLIP tower) to a targeted re-run of ~100 stimuli at key factorial cells.
- **Extra headline for the paper**: "When you call a circle a planet, it orbits" (the label → physics-regime categorical flip). Not in the original research_plan — this is a pilot-to-MVP-full emergent finding. Log it in ROADMAP §4 additional ideas.
- **Axis E (scene consistency)** still not tested; stays deferred pending a focused mini-experiment.

## 2026-04-24 pilot

- Command: `uv run python scripts/02_run_inference.py --config configs/pilot.py`
- Stimulus dir: `inputs/pilot_20260424-072216_308c86fc` (240 stimuli)
- Output dir: `outputs/pilot_20260424-072418_2c16efb6`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (first-run download, bf16, sdpa on H200)
- Wall clock: ~8 min total (first-run HF download ~15 s, 729-shard weight load ~8 s, 480 inferences at ~1.0 it/s)
- N predictions: 480 (240 stimuli × 1 label "ball" × 2 prompt variants)
- PMR scored twice: once with initial lexicon, once after patching `move` → `mov`
  family; numbers below are from the final `predictions_scored.parquet`.

### Headline PMR / GAR

**By object_level** (axis A — abstraction):

| object_level | n | pmr | hold_still | abstract_reject | gar |
|---|---|---|---|---|---|
| line | 120 | 0.575 | 0.333 | 0.325 | 0.667 |
| filled | 120 | 0.658 | 0.333 | 0.225 | 0.867 |
| shaded | 120 | 0.642 | 0.408 | 0.183 | 1.000 |
| textured | 120 | **0.808** | 0.142 | 0.167 | 0.600 |

→ H1 (monotonic S-curve line → textured) **partially supported**:
the endpoints behave as predicted (line 0.575 < textured 0.808) but
`shaded` and `filled` are effectively tied mid-curve. Either noise
(n=120 per cell → ~4.5 pp std error) or the shading cue alone
doesn't beat a uniform fill at this scale.

**By bg_level** (axis B):

| bg_level | n | pmr | gar |
|---|---|---|---|
| blank | 240 | 0.488 | N/A |
| ground | 240 | **0.854** | 0.783 |

→ **Ground presence adds +36 pp to PMR.** Largest single-factor effect we
measured; matches H3 and the cognitive-science prediction that a support
plane recruits physical-object interpretation.

**By cue_level** (axis C):

| cue_level | n | pmr |
|---|---|---|
| none | 160 | 0.500 |
| wind | 160 | 0.513 |
| arrow_shadow | 160 | **1.000** |

→ `arrow_shadow` saturates PMR at 1.0 (the trajectory arrow is a
complete give-away — the model reads it as "this is where the ball
will go"). **Wind marks do essentially nothing** — the VLM does not
interpret our programmatic wind streaks as airflow. See surprise #1.

**By prompt_variant** (methodological):

| prompt_variant | n | pmr | abstract_reject | gar |
|---|---|---|---|---|
| open | 240 | **0.800** | 0.000 | 0.917 |
| forced_choice | 240 | 0.542 | 0.450 | 0.650 |

→ When option D ("abstract shape") is offered, the model uses it
45 % of the time; in open-ended mode it *never* spontaneously
calls the stimulus abstract. The language prior from the "ball"
label fully dominates the open variant. This is an instrumentation
warning: PMR from open-ended prompts is inflated by the label axis D,
so the behavioral S-curve is best read off the forced-choice subset.

### Surprises / notes

1. **Wind cue is invisible to Qwen2.5-VL-7B.** The 15 small grey arcs
   anchored to one side of the object don't move PMR relative to "no
   cue" (0.513 vs 0.500). Consider a stronger visual: blurred motion
   trail in the object's wake, or actual particle streaks oriented
   with perspective. Before the MVP-full run, improve
   `primitives.draw_wind_marks` or drop the wind level from axis C
   in favor of `motion_blur` / `dust_cloud`.
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
   metric in research_plan.md §2.2).
5. **Raw responses look sensible** (e.g., "The ball will collide with
   the surface below it" on ground cells; "The ball will remain
   stationary unless acted upon by an external force" on blank cells
   — a Newton's-first-law framing). The model's errors are systematic,
   not random.

### Next actions

- MVP-full run with the three fixes above (wind → motion trail / dust;
  split cue axis; temperature 0.7 with more seeds). Spec it out in
  `configs/mvp_full.py` before the next run.
- Enable `capture_lm_layers = (5, 10, 15, 20, 25)` so Sub-task 3
  logit-lens analysis has hidden states ready.
- Before running MVP-full, expand axis D to `("circle", "ball", "planet")`
  to measure the H2 language-prior effect.

---
