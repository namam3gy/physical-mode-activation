# Scoring rubric — PMR / GAR / RC

Exact definitions for the three behavioral metrics in research_plan.md §2.2.
Implementation: `src/physical_mode/metrics/pmr.py`.

## PMR — Physics-Mode priming Rate

Binary per response. **PMR = 1 iff**:

1. The lowercased response contains *no* phrase in `ABSTRACT_MARKERS`
   (e.g., "this is just a circle", "won't move", "abstract shape"); **AND**
2. At least one whitespace-separated word in the response starts with a stem
   in `PHYSICS_VERB_STEMS` (e.g., "falls" matches `fall`, "rolling" matches
   `roll`, "accelerates" matches `accelerat`).

The abstract-reject gate comes **first** so that a response like "this is an
abstract shape — it won't move" scores PMR = 0 despite the word "move"
matching the `move` stem.

### Expanding the lexicon

If inspection of `predictions_scored.csv` reveals a false negative
(a clearly physical response scored 0), append the missing stem to
`PHYSICS_VERB_STEMS` in `src/physical_mode/metrics/lexicons.py` and add
an assertion to `tests/test_pmr_scoring.py::PMR_POSITIVE`. Do the reverse
for false positives. Treat the lexicon as living, not canonical.

## GAR — Gravity-Align Rate

Ternary: **1 / 0 / None**. Only defined when:

- `bg_level ∈ {ground, scene}` (a ground plane must exist), **AND**
- `event_template ∈ {fall, roll_slope}` (gravity is the salient force)

For responses where GAR is defined, **GAR = 1 iff** the response contains any
phrase in `DOWN_DIRECTION_PHRASES` (e.g., "down", "to the ground", "onto the
floor"). Aggregation skips `None` rows.

## RC — Response Consistency

For each factorial cell (object × bg × cue × event × label × prompt_variant),
RC is the fraction of the majority PMR value across seeds:

    RC(cell) = max(count(PMR=1), count(PMR=0)) / n(cell)

RC ∈ [0.5, 1.0]. Low RC means the model flips between physics-mode and
abstract-mode *on identical factor levels* — a sign of prompt instability or
a borderline cue.

## Secondary columns

The scorer also emits:

- `hold_still` — 1 if a "stay / remain / rest / sit" verb fires.
  Co-occurrence with PMR=0 is the "explicit no-motion" case.
- `abstract_reject` — 1 if an `ABSTRACT_MARKERS` phrase matches. Equivalent
  to the PMR gate but surfaced so analysis can separate "rejected as
  abstract" from "nothing predicted".

## Cell-level expected outcomes (for sanity checks)

| Object | Bg | Cue | Event | Expected PMR | Expected GAR |
|---|---|---|---|---|---|
| `line` | `blank` | `none` | `fall` | low (< 0.3) | N/A |
| `textured` | `ground` | `none` | `fall` | high (> 0.7) | high (> 0.7) |
| `line` | `blank` | `wind` | `horizontal` | mid (the wind cue alone may trigger motion language) | N/A |
| `shaded` | `ground` | `arrow_shadow` | `fall` | high | high |

These are prior expectations, not ground truth. A strong departure from this
table is a *scientific* finding, not a bug — log it in `03_run_log.md` with
sample model outputs.
