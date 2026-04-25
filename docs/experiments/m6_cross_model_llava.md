# M6 Round 1 — LLaVA-1.5-7B Cross-Model Run Log

Re-runs M2 + M4b on the same stimuli with LLaVA-1.5-7B-hf. Primary
question: does the M4b H2 reframing (`ball ≈ no-label`, `circle = suppressor`)
generalize to a second open-source VLM, or is it Qwen-specific?

Run date: 2026-04-25.

## Setup

- Configs:
  - `configs/cross_model_llava.py` — labels `(circle, ball, planet)` × `prompt_variants=("open",)`.
  - `configs/cross_model_llava_label_free.py` — `labels=("_nolabel",)` × `prompt_variants=("open_no_label",)`.
- Stimuli: M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/` (480 stim).
- Model: `llava-hf/llava-1.5-7b-hf`, bf16, sdpa attention.
- Generation: T=0.7, top_p=0.95, max_new_tokens=96.
- Forced-choice excluded (see "Smoke test → FC bias" below).
- Activation capture disabled (round 1 is behavioral-only).
- Outputs:
  - `outputs/cross_model_llava_20260425-035506_7ff0256b/` — labeled (1440 rows).
  - `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/` — label-free (480 rows).

## Smoke test → FC bias

A 4-cell × 3-label × 3-variant smoke (36 inferences) revealed that
`forced_choice` on LLaVA-1.5-7B returns first_letter `A` for **every**
(image, label) combination (12/12 across `line/blank/none`,
`textured/blank/none`, `textured/ground/both`, `line/blank/both` ×
{ball, circle, planet}). The bias is non-stochastic at T=0.7 and
independent of stimulus content.

Decision: drop FC from this round entirely. Open prompts give
diverse, label/image-sensitive responses that are usable for cross-
model H1, H2, H7 testing. H4 (open-FC gap) is not testable in this
round; deferred.

## Behavioral results — LLaVA labeled (open)

### Overall (mean over labels)

| metric | value |
|---|---|
| n | 1440 |
| PMR | 0.681 |
| GAR | 0.194 |
| hold_still | 0.010 |
| abstract_reject | 0.001 |

### By label

| label  | PMR  | GAR  | hold_still |
|---|---|---|---|
| ball   | 0.858 | 0.356 | 0.012 |
| circle | 0.556 | 0.153 | 0.010 |
| planet | 0.627 | 0.072 | 0.006 |

### By object_level × label (PMR)

| object | ball  | circle | planet |
|---|---|---|---|
| line     | 0.833 | 0.275 | 0.417 |
| filled   | 0.825 | 0.575 | 0.567 |
| shaded   | 0.875 | 0.658 | 0.725 |
| textured | 0.900 | 0.717 | 0.800 |

### By cue_level × label (PMR)

| cue          | ball  | circle | planet |
|---|---|---|---|
| both         | 0.883 | 0.608 | 0.658 |
| cast_shadow  | 0.858 | 0.658 | 0.592 |
| motion_arrow | 0.842 | 0.508 | 0.692 |
| none         | 0.850 | 0.450 | 0.567 |

## Behavioral results — LLaVA label-free

### Overall

| metric | value |
|---|---|
| n | 480 |
| PMR | 0.383 |
| GAR | 0.181 |
| hold_still | 0.123 |
| abstract_reject | 0.006 |

### By object_level

| object | PMR | hold_still |
|---|---|---|
| line     | 0.142 | 0.058 |
| filled   | 0.317 | 0.067 |
| shaded   | 0.592 | 0.225 |
| textured | 0.483 | 0.142 |

(Note: shaded > textured is anomalous; possibly LLaVA-1.5 reads "rendered
3D shading" as more physical-content-laden than a "photorealistic
texture". Worth flagging for round 2 but not blocking.)

### By cue_level

| cue          | PMR | hold_still |
|---|---|---|
| both         | 0.442 | 0.067 |
| cast_shadow  | 0.508 | 0.167 |
| motion_arrow | 0.292 | 0.083 |
| none         | 0.292 | 0.175 |

## H2 cross-model — paired PMR delta

Same `(obj, bg, cue, seed)` pairing as M4b. Each cell averaged over 10
seeds, T=0.7.

### LLaVA-1.5 (this run)

| label  | mean `PMR(label) − PMR(_nolabel)` | n_pairs |
|---|---|---|
| ball   | **+0.475** | 480 |
| planet | **+0.244** | 480 |
| circle | **+0.173** | 480 |

### Qwen2.5-VL-7B (M4b, for comparison)

| label  | mean `PMR(label) − PMR(_nolabel)` | n_pairs |
|---|---|---|
| ball   | +0.006 | 480 |
| planet | +0.006 | 480 |
| circle | **−0.065** | 480 |

## Cross-model S-curve — PMR by object_level

| object   | Qwen labeled | Qwen no-label | LLaVA labeled | LLaVA no-label |
|---|---|---|---|---|
| line     | 0.906 | 0.942 | 0.508 | 0.142 |
| filled   | 0.933 | 0.933 | 0.656 | 0.317 |
| shaded   | 0.933 | 0.942 | 0.753 | 0.592 |
| textured | 0.950 | 0.975 | 0.806 | 0.483 |

- Qwen labeled is saturated near 0.93 across all object levels — no S-curve.
- Qwen no-label shows the same flat pattern (visual saturation).
- LLaVA labeled shows a clear monotone S-curve (0.51 → 0.81).
- LLaVA no-label is non-monotone (shaded > textured) — see anomaly note above.

## H7 cross-model — GAR by label (open prompt)

| label  | Qwen | LLaVA |
|---|---|---|
| ball   | 0.706 | 0.356 |
| circle | 0.753 | 0.153 |
| planet | **0.319** | **0.072** |

The H7 pattern (`planet` GAR << `ball`/`circle` GAR) holds in both
models — planet label diverts physics narration toward orbital / cosmic
events, suppressing gravity-aligned motion. Magnitudes differ (LLaVA
GAR is ~½ of Qwen for every label) but the gradient is qualitatively
the same.

## Cell-level — `line/blank/none` × label (LLaVA)

| label    | PMR | hold_still |
|---|---|---|
| _nolabel | 0.0 | 0.0 |
| ball     | 0.9 | 0.0 |
| circle   | 0.0 | 0.0 |
| planet   | 0.5 | 0.0 |

(GAR not computable here; bg_level=blank → no ground.)

The `line/blank/none × ball` case shows LLaVA committing 9/10 to
physics ("ball will fall", "ball will roll off the edge"), vs Qwen's
4/10 commit on the same cell (and 1 of those goes to hold_still).
This is the cleanest expression of the cross-model H2 fork:

- **Qwen**: image is fully abstract → visual evidence weak → no
  prior pushes the model strongly → some physics-mode + hold-still
  spread, total PMR ~0.40.
- **LLaVA**: image is fully abstract → visual evidence even weaker
  for LLaVA → no-label baseline collapses to 0; ball label is then a
  *strong* physics activator → 9/10 commit.

## Sample raw responses

`line/blank/none` (fully abstract) cell:

- LLaVA × ball:
  - "The ball will roll off the edge of the circle."
  - "The ball will fall."
  - "The ball will fall into a hole."
- LLaVA × circle:
  - "The circle will become more visible."
  - "The circle is going to get smaller and smaller."
  - "The circle will become a dot."
- LLaVA × planet:
  - "In the next moment, the planet will continue to spin and orbit around the sun."
  - "The planet will be consumed by a black hole."
- LLaVA × _nolabel:
  - "A round object is in the middle of the image, and the next part of the image is white."
  - "A black circle is seen in the background and the center of the circle is white."

`textured/blank/none` (physical-looking object, no scene):

- LLaVA × ball:
  - "The ball is in the air and about to fall."
  - "The ball is suspended in midair... it is likely that it will remain in th[e air]."
  - "The ball will roll."
- LLaVA × circle:
  - "The circle will roll on the ground."
  - "The circle is going to be on the ground."
  - "The circle will spin and change direction."

(Note: even with `circle` label, LLaVA on a textured-ball image often
still applies physics-mode predicates to "the circle". So LLaVA does
not strictly route circle → abstract; it routes by image content +
label jointly, with image content dominating once it carries enough
physical signal.)

## Raw artifacts

- `outputs/cross_model_llava_20260425-035506_7ff0256b/predictions{_scored,}.{jsonl,parquet,csv}` — 1440 rows.
- `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/cross_model_llava_*/summary_*.csv` — factor-level rollups.
