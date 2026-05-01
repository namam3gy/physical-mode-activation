# M5b — Idefics2 post-projection non-monotonic anomaly

**Date**: 2026-05-01.
**Source**: `outputs/sae_intervention/idefics2_post_proj_open_circle_filled_blank_both/results.csv`.
**Status**: diagnosed — scoring artifact, not real regime restoration.

## Observation

Post-projection SAE intervention on Idefics2 (`circle / filled / blank+both`)
shows a non-monotonic PMR curve as a function of `k_zeroed`:

| k_zeroed | intervention_text                                | intervention_pmr |
|---|---|---|
| 20  | "The circle will disappear."                      | **0** |
| 40  | "The circle will continue to expand outward."     | **1** |
| 80  | "The circle will continue to expand outward."     | **1** |
| 160 | "The circle will continue to expand outward."     | **1** |

Naively this suggests that ablating MORE features `k=40+` somehow *restores*
physics-mode commitment that `k=20` had broken. That would be implausible —
removing more features cannot bring back what removing fewer destroyed.

## Diagnosis

The recovery is a **scoring artifact**, not a real regime restoration.

`src/physical_mode/metrics/lexicons.py` includes the stem `"continu"` in
`PHYSICS_VERB_STEMS`:

```
48:    "continu",   # continue(s/d/ing)
```

The PMR scorer (`src/physical_mode/metrics/pmr.py:score_pmr`) uses
`startswith` matching on word stems:

```python
words = _words(text)
if _any_stem_hit(words, PHYSICS_VERB_STEMS):
    return 1
```

So `"continue"` triggers PMR=1 even when the surrounding clause is geometric:

- `"The circle will continue to expand outward."`
  - `continue` matches `continu` stem → PMR=1.
  - But `"expand outward"` is geometric (a static drawing growing in size),
    not kinetic motion. A human reader would call this an abstract-mode
    response, not physics-mode.

For comparison, `k=20` produces `"The circle will disappear."` —
`disappear` is not in `PHYSICS_VERB_STEMS` and `"will disappear"` is not in
the `HOLD_STILL_STEMS` list, so it scores PMR=0. That correctly captures
the abstract regime.

## Implication

The Idefics2 `circle / filled / blank+both` cell does **not** show a
non-monotonic break-then-restore — it shows a clean break at all k ≥ 20,
mis-scored as PMR=1 from `k=40` onward because of the auxiliary verb
"continue" in the model's stylized completion ("The circle will continue
to ...").

## Connection to M5b paper-headline reframe (C3)

This is a concrete instance of the broader scoring artifact reframe (see
`docs/review_ppt/slide_notes_full_review_ko.md` slide 27):

- M5b "NULL" headline includes 3 distinct phenomena, one of which is
  **binary PMR concealing real regime shifts**.
- The Idefics2 non-monotonicity is the cleanest single example: the *text*
  shifts from `"Falling."` (baseline) → `"disappear"` (k=20) → `"expand
  outward"` (k=40+). All three texts represent different regimes; only
  the binary PMR fails to track this.

## Recommended downstream action

1. **Paper-level (C3)**: report regime-shift score (text-distance from
   baseline) alongside PMR for any "NULL" claim.
2. **Lexicon hygiene (low priority)**: consider whether `continu` should
   require a kinetic neighbor (e.g., `continue to fall|move|roll`) rather
   than match any `continue *`. This would tighten PMR but require
   re-scoring all M2 / cross-model runs to maintain comparability — likely
   not worth the cost mid-paper.

For the post-projection round 2 ladder (slide 26), the Idefics2
non-monotonicity is annotated as "non-monotonic; scoring artifact" but the
underlying regime-cross signature is treated as `★★ partial break` (same
verdict as a clean low-k break that doesn't recover).
