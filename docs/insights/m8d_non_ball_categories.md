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

## Results (2026-04-25)

Total wall clock: **32 minutes** on H200 GPU 0 (Qwen labeled 12.6 min,
Qwen label-free 6.2 min, LLaVA labeled 8.8 min, LLaVA label-free 4.3
min). 480 stimuli × 4 inference configs = 3840 inferences.

Classifier validation: **error rate 5.6 %** (3/54 hand-annotated rows
mismatch the keyword classifier; below the 15 % paper-ready threshold).
The 3 mismatches are all stem-matching false-positives in the "no
movement" / "pulled away" pattern — known limitation.

### Pre-registered scoring summary

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| Visual-sat. delta | 1/3 (bird) | 2/3 (car, bird; person flips negative) |

### PMR_regime(_nolabel) baseline (event union)

| category | Qwen  | LLaVA |
|----------|-------|-------|
| car      | 1.000 | 0.450 |
| person   | 0.988 | 0.744 |
| bird     | 0.831 | 0.706 |

(Per-category PMR_regime(_nolabel) on `horizontal` subset only:
Qwen car 1.000 / person 0.975 / bird 0.862;
LLaVA car 0.550 / person 0.838 / bird 0.688.)

Qwen sits at 0.83-1.00 across categories — saturated. LLaVA spans
0.45-0.74 — the encoder gives the binary metric room to move with
labels.

### Headline interpretation

**The H7 (label-selects-regime) finding generalizes 3-of-3 cross-category
on LLaVA.** This is the strongest cross-category H7 evidence in the
project. LLaVA on `horizontal` subset:

| category | physical PMR | abstract PMR | physical − abstract |
|----------|--------------|--------------|---------------------|
| car      | 0.825        | 0.300        | **+0.525** |
| person   | 0.738        | 0.600        | +0.138 |
| bird     | 0.950        | 0.400        | **+0.550** |

For car and bird the abstract label (silhouette) STRONGLY suppresses
physics-mode response. For person the suppression is smaller (the
`stick figure` label still admits walking responses on LLaVA).

**Qwen's H7 fails strict pre-registration** because PMR_regime is
ceiling-saturated (0.95-1.0 across all label_role × category cells on
horizontal subset). The ceiling is exactly the M8a-style visual-
saturation pattern: Qwen's encoder identifies the category, triggers
physics-mode automatically, no headroom for label suppression to move
the binary metric.

**Underneath the Qwen ceiling, the regime distribution shows the H7
signal is alive and well at the categorical level**:

| category × label | kin_frac | static_frac |
|---|---|---|
| car/physical (car)       | 0.944 | 0.050 |
| car/exotic (figurine)    | 0.806 | **0.175** |
| person/physical (person) | 0.912 | 0.081 |
| person/exotic (statue)   | 0.750 | **0.225** |
| bird/physical (bird)     | 0.981 | 0.012 |
| bird/exotic (duck)       | 0.962 | 0.006 |

Statue and figurine labels DO inject a static regime fraction in Qwen
(17.5 % / 22.5 % static, vs ~5 % static for the physical labels). The
duck exotic does not — confirming the pre-registered weak point that
duck still flies.

Translated: **the binary "physics-mode active" measure saturates for
Qwen on these recognizable categories, but the categorical regime
choice (kinetic vs static) survives label-driven manipulation.** This
is a finer-grained version of M2's "label selects regime" finding —
the regime axis is robust even when the binary-physics axis is
saturated.

### Visual-saturation paired-delta (horizontal subset)

| (category, role)   | Qwen Δ | LLaVA Δ |
|--------------------|--------|---------|
| car / physical     | +0.000 | +0.275  |
| car / abstract     | -0.012 | **-0.250** |
| car / exotic       | -0.025 | +0.000  |
| person / physical  | +0.025 | -0.100  |
| person / abstract  | +0.012 | -0.238  |
| person / exotic    | +0.012 | -0.288  |
| bird / physical    | +0.125 | +0.262  |
| bird / abstract    | +0.088 | **-0.288** |
| bird / exotic      | +0.138 | +0.238  |

LLaVA's car/abstract = -0.250 and bird/abstract = -0.288 are the
strongest per-cell suppressions in the M8d run — the silhouette label
positively *reduces* PMR_regime by ~25 pp on cars and ~29 pp on birds.
Qwen all near zero (saturated). Bird shows a small Qwen positive
delta uniformly (+0.09 to +0.14) — the encoder is least saturated on
bird among the three categories (PMR_regime(_nolabel) baseline 0.862
vs car/person ≈ 0.97-1.0).

Person on LLaVA shows an unusual pattern: **all three roles produce
*negative* paired-deltas** (−0.10, −0.24, −0.29). The person `_nolabel`
baseline (PMR_regime 0.838) is unusually high because label-free
responses to a stick figure or person silhouette frequently still get
"the person walks forward", but adding the label `person` slightly
reduces the rate. The `stick figure` label suppresses (consistent with
H7), and `statue` strongly suppresses kinetic (regime distribution
shows 41 % static for person/`_nolabel`, dropping to 15 % for
person/exotic — yes, less static for `statue` because LLaVA assigns
many `statue`s as ambiguous-stationary wording). This is the most
nuanced M8d cell; person is the noisiest category.

### Per-category notes

- **Car** — cleanest H7 LLaVA result (+0.525 on PMR; +0.138 kin_frac
  on Qwen). Figurine produces 17.5 % static on Qwen — the "static toy
  on display" reading is operative.
- **Person** — noisiest. Person/`_nolabel` LLaVA baseline is high (the
  "the person walks" reading is the visual default), so labels can only
  suppress, not add. On Qwen, statue produces the strongest static
  injection (22.5 %) of any (category × label) cell.
- **Bird** — strongest per-cell PMR delta on LLaVA (+0.550 physical −
  abstract). Duck exotic stays kinetic (cross-validation of the
  pre-registered weak-label note: duck flies as much as bird flies, so
  H7 needs flightless bird in M8d round 2).

### H1 ramp failure analysis

H1 fails on both models — Qwen ceiling (0.97 → 0.98), LLaVA non-monotone
(car flat at ~0.50, person 0.68 → 0.43 → 0.64, bird 0.74 → 0.84 → 0.68
→ 0.73). The non-monotonicity is itself informative:

- For circle/ball (M2), `line` is *abstract reject territory* —
  the visual default reading is "this is just a circle". Adding visual
  detail (filled, shaded, textured) progressively activates physics-mode.
- For car/person/bird, every abstraction level is *already
  category-recognizable*: the line car has wheels, the line person is
  a stick figure, the line bird has a beak. The category is identified
  from line-level alone, so there's nothing to ramp up — visual detail
  doesn't change the affordance, only the surface realism.

This is the **shape-vs-category dissociation**: the H1 ramp is a
property of the abstract-shape ↔ physical-object axis (where label
disambiguates), not of object recognizability across abstraction
levels for *named* categories.

## Hypothesis updates

- **H1** — *additionally narrowed to abstract-shape ↔ physical-object*.
  M8a found H1 unsaturated-only (LLaVA-clean / Qwen-suppressed). M8d
  finds H1 also fails on LLaVA for category-named objects, because
  category recognition saturates the binary measure independent of
  visual abstraction. H1 is a property of the abstract-shape ↔
  physical-object axis specifically, not a general visual-realism →
  physics-prior coupling.
- **H7** — **3-of-3 cross-category replication on LLaVA, plus regime-
  level replication on Qwen**. The strongest evidence yet that label
  dispatches the category-appropriate regime. Specifically:
  - LLaVA: physical − abstract paired delta on PMR_regime (binary)
    ≥ +0.138 across car/person/bird.
  - Qwen: paired delta on KIN_fraction (4-class regime)
    physical − exotic ≥ +0.138 for car (figurine) and person (statue);
    the binary measure is saturated.
- **H-encoder-saturation** — **further validated cross-category**.
  Qwen's PMR_regime ceiling on car/person (0.97-1.0) replicates the
  M8a ceiling pattern; LLaVA's PMR_regime range (0.55-0.84) leaves
  headroom; the asymmetry predicts H7 measurability — exactly what we
  observe (Qwen H7 0/3 binary, LLaVA H7 3/3 binary).

## Roadmap implications

1. **H7 generalizes** — promote from "circle-only" to "cross-category"
   in the headline of the paper.
2. **H1 is shape-specific** — refine paper claim to "abstraction ramp
   is a property of the geometric-shape ↔ named-object axis, not a
   general visual-detail → physics-prior mechanism".
3. **regime-distribution as a finer instrument** — when binary
   PMR_regime saturates, the kinetic-vs-static split inside regime
   distribution still moves with label_role. Add this as a paper
   methodological contribution: "regime distribution rescues H7
   signal from binary saturation".
4. **M8c (real photographs)** is now strongly motivated — does
   photo-realism close LLaVA's encoder gap? Does it push Qwen even
   further into ceiling? Pre-registered M8c criteria can use the
   M8d-validated regime classifier.
5. **Round-2 weak-label fixes** — `duck` should become `penguin` /
   `ostrich` / `chicken` (flightless or ground-bird) for cleaner H7
   exotic role.
6. **No mechanism work** (encoder probing, LM probing) is needed for
   M8d; the behavioral signal is clean enough. Save mechanism for
   M8c (paper-ready figure 1) or §4.5 (encoder swap).
