# M8d — Non-ball physical-object categories (external-validity round 2)

**Status**: Pre-registered (criteria locked before measurement). Results
section appended after the run.

## Motivation

Through M8a the cross-shape sweep (square / triangle / hexagon /
irregular polygon) validated the visual-saturation hypothesis and
identified H1 + H7 as *unsaturated-only* (LLaVA-clean / Qwen-suppressed).
Every shape so far is still a 2-D geometric primitive whose label
triplet is an in-class linguistic alias of itself
(`circle ↔ ball ↔ planet`, `polygon ↔ rock ↔ boulder`). M8d asks the
adjacent generalization question: do the H7 (label-selects-regime) and
H1 (object-abstraction ramp) findings extend to **non-ball physical-
object categories** — car / person / bird — where the regime
distinctions are not gravity-fall but kinetic actions
(drives / walks / flies) vs depiction-style abstract labels
(silhouette, stick figure)?

If H7 generalizes, the published claim moves from "label dispatches the
specific physics regime within the gravity family
(circle / ball / planet)" to "label dispatches the *category*-
appropriate physics regime across object kinds." That is a strictly
larger contribution and one that lifts the M5a / M5a-ext steering
result from a circle-only mechanism statement to a category-general
one.

## Stimulus design

3 categories × 4 abstraction levels × 2 backgrounds × 2 cue conditions
× **2 events** × 5 seeds = **480 stimuli**. The factorial doubles M8a's
event axis — `fall` is the M8a-comparable gravity stress test;
`horizontal` is the natural-event cell where regime selection is
cleanest (cars drive horizontally, persons walk horizontally, birds
fly horizontally).

```
shapes:         car, person, bird
object_levels:  line, filled, shaded, textured
bg_levels:      blank, ground
cue_levels:     none, both     (none = bare; both = cast_shadow + motion_arrow)
event:          fall, horizontal
seeds:          5
```

Per-category label triplets (physical / abstract / exotic) are
dispatched at prompt time via `LABELS_BY_SHAPE`:

| category | physical | abstract       | exotic   |
|----------|----------|----------------|----------|
| car      | car      | silhouette     | figurine |
| person   | person   | stick figure   | statue   |
| bird     | bird     | silhouette     | duck     |

The `abstract` role uses depiction-style labels (silhouette / stick
figure) rather than a forced geometric class because non-ball
categories don't have natural geometric-class names. The `exotic` role
shifts the regime away from each category's default kinetic event:
`figurine` → static toy on display, `statue` → static stone person,
`duck` → mixed (swims / waddles / walks alongside the default flies).
Known weak point: `bird/duck` is partial — duck is a flying bird whose
exotic shift is "swims/waddles" but flying is still in the regime
distribution. A flightless bird (penguin / ostrich) would give a
cleaner signal but is harder to render recognizably at low resolution.

A 3×4 visual grid (`docs/figures/m8d_shape_grid.png`) confirms each
category-level cell is visually distinct; the M8a-style full-scene
grid (`docs/figures/m8d_full_scene_samples.png`) shows category × event
combinations. For `horizontal` events with car / person, the object is
positioned **on** the ground line so the natural-motion reading
(drives / walks) is geometrically consistent with the cast shadow;
birds remain in midair (the natural-flight reading is airborne).

## Regime classifier

Because the standard PMR (gravity-verb-biased) systematically
undercounts the kinetic verbs that car (`drives`) and person (`walks`,
`runs`) responses produce, M8d defines a parallel category-aware
classifier `classify_regime(text, category) → {kinetic, static,
abstract, ambiguous}` (`metrics/lexicons.py::CATEGORY_REGIME_KEYWORDS`,
`metrics/pmr.py::classify_regime`). The lexicon is hand-curated:

```
car kinetic    : driv roll spee mov race accel trav head
car static     : park stop stay still stationary display remain
person kinetic : walk run jog step stride mov march pace
person static  : stand stay still stationary motionless frozen sit rest remain
bird kinetic   : fly fli flew flown swim swam soar waddl mov glid flap hop
bird static    : perch sit stay still stationary rest remain
```

`abstract` markers are cross-category (existing `ABSTRACT_MARKERS` set
shared with M8a — "abstract", "geometric", "drawing", "diagram",
"won't move", "nothing happens", etc.). The classifier short-circuits
on abstract markers before checking kinetic / static, so a response
like "this is just a drawing of a car — it stays parked" classifies as
`abstract`, not `static`.

The original PMR / GAR scorers in `metrics/pmr.py` are **unchanged** so
M8a analyses remain bit-identical. M8d analysis code calls
`classify_regime` directly via `m8d_analyze.py::annotate`.

For binary-PMR-style analyses, M8d defines:

```
PMR_regime = 1  iff  classify_regime(text, category) ∈ {kinetic, static}
PMR_regime = 0  iff  regime ∈ {abstract, ambiguous}
```

This captures "the model treats the object as physical regardless of
which regime fires" — equivalent in spirit to the original PMR. The
H1 ramp uses `PMR_regime` over the event union; H7 paired-delta uses
`PMR_regime` on the `horizontal` subset (natural-event cleanest cell).

## Pre-registered success criteria (locked 2026-04-25 before measurement)

Three criteria, parallel to M8a, declared cross-category-replicated if
the corresponding criterion is met *as stated*; otherwise reported as
failed (or partially replicated). Criteria are intentionally tight —
this is a real test, not a rubber stamp.

### H1 — object_level abstraction ramp generalizes (per category)
For each category, compute PMR_regime(line / filled / shaded / textured)
under the labeled `open` prompt averaged across all (bg, cue,
label_role, event, seed) cells.
**Replication**: ≥2/3 categories satisfy
`PMR_regime(textured) − PMR_regime(line) ≥ 0.05`
AND no internal inversion of magnitude greater than 0.05 in the
line→filled→shaded→textured sequence.

### H7 — label-role drives PMR_regime (physical > abstract)
For each category, compute mean PMR_regime by `label_role` on the
`horizontal` subset (natural-event cleanest cell).
**Replication**: ≥2/3 categories satisfy
`PMR_regime(physical) − PMR_regime(abstract) ≥ 0.05`.

### Visual-saturation paired-delta generalizes
For each category × model, compute paired-delta = PMR_regime(label_role)
− PMR_regime(_nolabel) per stimulus seed (averaged within category) on
the `horizontal` subset.
**Replication**: ≥2/3 categories per model exhibit the predicted
direction within the M6 r2 / M8a precedent — Qwen near-zero (saturated)
or shrunk to noise; LLaVA ≥+0.05 for `physical` role on most categories.

### Failure modes
- If H1 fails on most categories, the ramp is shape-specific (or
  category-specific) — itself a publishable nuance. We do not treat
  null results as embarrassing.
- If H7 fails for a category, label tagging may be too weak to
  override visual content (cf. M8a's `wedge` and `polygon` weak labels).
  We flag the per-category labels involved and discuss the label-design
  caveat in the paper.

## Setup

```bash
# Stimuli (run once; reused across 4 inference configs).
uv run python scripts/01_generate_stimuli.py --config configs/m8d_qwen.py

# Inference (single GPU 0, sequential).
M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_qwen.py            --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_qwen_label_free.py --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_llava.py           --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_llava_label_free.py --stimulus-dir "$M8D_DIR"

# Equivalently:
bash scripts/m8d_run_all.sh

# Analyze + figures.
uv run python scripts/m8d_analyze.py \
    --qwen-labeled  outputs/m8d_qwen_<ts>/predictions.jsonl \
    --qwen-nolabel  outputs/m8d_qwen_label_free_<ts>/predictions.jsonl \
    --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \
    --llava-nolabel outputs/m8d_llava_label_free_<ts>/predictions.jsonl \
    --out-dir       outputs/m8d_summary
uv run python scripts/m8d_figures.py --summary-dir outputs/m8d_summary

# Hand-annotation classifier validation (50 stratified rows).
uv run python scripts/m8d_hand_annotate.py --mode sample \
    --qwen-labeled  outputs/m8d_qwen_<ts>/predictions.jsonl \
    --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \
    --out           docs/experiments/m8d_hand_annotate.csv
# ... fill the hand_regime column ...
uv run python scripts/m8d_hand_annotate.py --mode score \
    --csv           docs/experiments/m8d_hand_annotate.csv
```

## Results

_To be appended after the run completes (~30 min wall-clock on H200
GPU 0). Filled in by `scripts/m8d_analyze.py` headlines + figures._

### Pre-registered scoring summary

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | _TBD_ | _TBD_ |
| H7 (phys>abs)     | _TBD_ | _TBD_ |
| Visual-sat. delta | _TBD_ | _TBD_ |

### PMR_regime(_nolabel) baseline

| category | Qwen  | LLaVA |
|----------|-------|-------|
| car      | _TBD_ | _TBD_ |
| person   | _TBD_ | _TBD_ |
| bird     | _TBD_ | _TBD_ |

### Headline interpretation

_To be filled in._

### Per-category notes

_To be filled in. Expect:_
- _car: cleanest (most visually distinct + most kinetic-verb-rich responses)._
- _person: medium (label `person` may be slightly under-specific)._
- _bird: noisiest H7 (the `duck` exotic role is partial — flying not fully suppressed)._

## Hypothesis updates (filled after results)

- H1 — _TBD_
- H7 — _TBD_ (cross-category)
- H-encoder-saturation — _TBD_

## Roadmap implications

_To be filled in after results._
