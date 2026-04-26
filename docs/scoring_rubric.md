# Scoring rubric — PMR / GAR / RC

Exact definitions for the three behavioral metrics in `references/project.md` §2.2.
Implementation: `src/physical_mode/metrics/pmr.py`.

## PMR — Physics-Mode Priming Rate

Binary per response. **PMR = 1 iff**:

1. The lowercased response contains *no* phrase in `ABSTRACT_MARKERS`
   or `KOREAN_ABSTRACT_MARKERS` (e.g., "this is just a circle", "won't
   move", "abstract shape", `그대로`, `움직이지 않`); **AND**
2. *Either* an English path: at least one whitespace-separated word in
   the response starts with a stem in `PHYSICS_VERB_STEMS` (e.g.,
   "falls" matches `fall`, "rolling" matches `roll`); *or* a Korean
   path: a stem in `KOREAN_PHYSICS_VERB_STEMS` appears as a substring
   of the response (e.g., `떨어` catches 떨어진다 / 떨어지는 / 떨어졌다).

The abstract-reject gate comes **first** so that a response like "this is an
abstract shape — it won't move" scores PMR = 0 despite the word "move"
matching the `mov` stem.

The Korean substring fallback was added in the §4.3 cross-model run
(2026-04-26): on Hangul-only responses (rare, ~1% of cross-model §4.3
predictions, observed in LLaVA-Next + Idefics2), the English regex
`[A-Za-z]+` returns no tokens, silently scoring PMR = 0. Korean is
agglutinative and not space-segmented, so substring matching on a stem
like `떨어` reliably catches all inflections of 떨어지다.

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
a borderline cue. Note: at temperature = 0, RC is degenerate (always 1.0);
M2 raised temperature to 0.7 to make it informative.

## Secondary columns

The scorer also emits:

- `hold_still` — 1 if a "stay / remain / rest / sit" verb fires.
  Co-occurrence with PMR = 0 is the "explicit no-motion" case.
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
table is a *scientific* finding, not a bug — log it in the appropriate
`docs/experiments/m{N}_*.md` with sample model outputs.

## Known scoring artifacts

Forced-choice responses often enumerate the unselected options ("D — it
*cannot fall, move, or change direction*"); the enumerated verbs trip
`PHYSICS_VERB_STEMS` and produce false PMR=1 readings. When evaluating
forced-choice runs, pair PMR with the **first-letter** of the response
(A/B/C/D). The first-letter signal is the cleaner causal indicator; PMR
is the right metric for open-ended responses.
