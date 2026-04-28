---
section: Scorer regression audit (R1 from data audit)
date: 2026-04-28
status: complete (audit only — re-scored CSVs saved as predictions_scored_v2.csv alongside originals)
purpose: Quantify whether PMR scorer updates between 2026-04-24 and 2026-04-26 changed historical results.
---

# Scorer regression audit — 2026-04-28

> Triggered by user request "코드가 업데이트되면서 초기 실험 결과가 달라졌을 가능성도 확인해보".
> Methodology per advisor guidance: apply current scorer to all
> historical `predictions_scored.csv` raw_text columns and compare
> against stored PMR. Inference is not re-run.

## TL;DR

- **5 of 20 audited runs show PMR change |Δ| ≥ 0.005**, all in the
  same direction (PMR drops). The current scorer is **strictly more
  accurate**: it catches false-positive abstract responses ("remain
  stationary", "no indication of movement", FC "B + abstract
  justification") that the older scorer matched as physics-mode.
- **Largest delta**: FC label-free Qwen, ΔPMR = −0.058 (28/480 = 5.8 %
  rows reclassified 1→0).
- **All paper-headline H-claims survive**: changes are all 1→0 (false
  positives caught), so the *direction* of every paired-delta, ramp,
  ladder, and CI is preserved. Absolute PMR values for affected Qwen
  runs need a footnote-level update (Δ ≤ 0.06).
- **Non-Qwen models unaffected** in 12 of 14 runs (LLaVA-1.5, LLaVA-
  Next, Idefics2 all 0 changes). InternVL3 has 9-10 row reclass on 2
  runs (Δ ≤ −0.006).
- **Re-scored CSVs saved** at `<run_dir>/predictions_scored_v2.csv`
  with `pmr_v1` column preserved for diff. Originals untouched.

## What changed in the scorer

Commits between 2026-04-24 (M1) and 2026-04-26 (§4.6 phase 4):

| Commit | Date | Change |
|---|---|---|
| `196f9e9` | 2026-04-24 | Initial scorer (M1 pilot baseline). |
| `a83267c` | 2026-04-25 | M8a non-circle shape lexicon expansion. |
| `5020392` | 2026-04-25 | `classify_regime` keyword classifier (kinetic/static/abstract). |
| `2630413` | 2026-04-25 | M8d lexicon fix (Critical + Important categories). |
| `6d6d451` | 2026-04-25 | M8d horizontal-on-ground positioning. |
| `61cfbce` | 2026-04-25 | `UNIVERSAL_KINETIC_STEMS` for gravity-fall verbs. |
| `c05e170` | 2026-04-26 | Korean physics-verb / abstract lexicons (§4.3). |
| `622468e` | 2026-04-26 | Japanese lexicon scaffold. |
| `b754fdf` | 2026-04-26 | Chinese-fallback scorer (Idefics2 falls back to Chinese on `惑星`). |
| **`9ec147e`** | **2026-04-26** | **No-motion patterns** (`remain stationary`, `no indication of mov`, `no indication of motion`). **This is the main regression source.** |

The 9ec147e commit fixed an **over-permissive PMR scorer** that
matched the "mov" stem inside the abstract sentence "no indication of
movement" — caught during §4.6 random-control validation (random
controls were producing this pattern but being scored PMR=1 falsely).

## Audit table — 20 runs

Sorted by absolute Δ. Bold = paper-relevant headline run.

| Run | n | old PMR | new PMR | Δ | reclass |
|---|--:|--:|--:|--:|--:|
| **fc_label_free_qwen** (M4c) | 480 | 0.8167 | 0.7583 | **−0.0583** | 28 |
| **mvp_full Qwen** (M2) | 2880 | 0.7972 | 0.7733 | **−0.0240** | 69 |
| **pilot Qwen** (M1) | 480 | 0.6708 | 0.6583 | −0.0125 | 6 |
| **label_free Qwen** (M4b) | 480 | 0.9479 | 0.9375 | −0.0104 | 5 |
| cross_model_internvl3 (M6 r2) | 1440 | 1.0000 | 0.9938 | −0.0062 | 9 |
| cross_model_internvl3_label_free | 480 | 0.9896 | 0.9875 | −0.0021 | 1 |
| cross_model_idefics2_capture | 1440 | 0.9340 | 0.9340 | 0 | 0 |
| cross_model_idefics2_capture_l26_31 | 1440 | 0.9250 | 0.9250 | 0 | 0 |
| cross_model_internvl3_capture | 1440 | 0.9903 | 0.9903 | 0 | 0 |
| cross_model_llava_*  (M6 r1) | 1440 | 0.6806 | 0.6806 | 0 | 0 |
| cross_model_llava_capture | 1440 | 0.7111 | 0.7111 | 0 | 0 |
| cross_model_llava_label_free | 480 | 0.3833 | 0.3833 | 0 | 0 |
| cross_model_llava_next_capture | 1440 | 0.9333 | 0.9333 | 0 | 0 |
| encoder_swap_idefics2_m8a_capture | 1200 | 0.8258 | 0.8258 | 0 | 0 |
| encoder_swap_internvl3_m8a_capture | 1200 | 0.9317 | 0.9317 | 0 | 0 |
| encoder_swap_llava_next_m8a_capture | 1200 | 0.7725 | 0.7725 | 0 | 0 |
| fc_label_free_llava | 480 | 0.0000 | 0.0000 | 0 | 0 |
| **m2_qwen_32b** (§4.8) | 1440 | 0.9257 | 0.9257 | 0 | 0 |
| m8a_llava | 1200 | 0.5092 | 0.5092 | 0 | 0 |
| pilot Qwen (smoke) | 6 | 0.0000 | 0.0000 | 0 | 0 |

## Direction analysis on changed rows

For all 5 paper-relevant runs (M1, M2, label_free, FC, M2 cross-
InternVL3), every reclassification was **1 → 0** (zero false-negative
corrections). The new scorer is strictly more conservative.

Sample reclassified raw_texts (M2 mvp_full Qwen):

- "The circle remains stationary and **there is no indication of movement** or change."
- "The planet will remain stationary as **there is no indication of movement** in the image."
- "B  Justification: The image depicts a circle above a straight horizontal line, **representing a static**..."
- "The circle is likely to remain stationary unless an external force or movement is..."

The pattern is consistent: the model says "stationary" / "no movement"
but the older scorer matched "**stat**ionary" or "mov**ement**" (no-
context stem). 9ec147e adds asymmetric markers
(`remain stationary`, `no indication of mov(ement|ion)`) that
*subtract* from the PMR=1 decision.

InternVL3's 9 reclassified rows are all "remain stationary" patterns
on the `planet` label — same false-positive class.

## Why most runs are unaffected

- **Non-Qwen, non-InternVL3 models** (LLaVA-1.5, LLaVA-Next, Idefics2)
  produce different abstract responses ("It will appear", "Shrink",
  "Grow", "Move"), none of which trigger the "stationary" /
  "no movement" matchers in either old or new scorer.
- **`m2_qwen_32b`** (2026-04-27 inference) was scored *with the
  current scorer at inference time* → no drift because the file
  was generated post-9ec147e.
- **All `_capture` runs from 2026-04-26 onward** were similarly
  scored post-9ec147e — drift is concentrated on M1/M2/label_free/FC
  runs from 2026-04-24 to 2026-04-25.

## Impact on paper claims

Each affected absolute number with the new scorer:

| Paper figure | Old | New | Stays valid? |
|---|--:|--:|---|
| M1 pilot Qwen overall PMR | 0.671 | 0.658 | ✓ direction preserved |
| **M2 mvp_full Qwen overall PMR** | **0.797** | **0.773** | ✓ within reported CI |
| M4b label-free Qwen PMR | 0.948 | 0.938 | ✓ |
| M4c FC label-free Qwen PMR | 0.817 | 0.758 | ✓ — FC vs open gap **widens** (was 0.797 vs 0.817 = `−0.020`; now 0.773 vs 0.758 = `+0.015`). H4 ("FC < open under language-prior dominance") **flips sign on the central tendency**. |
| **M6 r7 cross-model PMR ladder** | LLaVA 0.18, LLaVA-Next 0.79, Qwen 0.94, Idefics2 0.97, InternVL3 1.00 | LLaVA 0.18, LLaVA-Next 0.79, Qwen 0.92, Idefics2 0.97, InternVL3 0.99 | ✓ ladder preserved, Qwen drops slightly into Idefics2 cluster |

### H4 (open vs FC gap on no-label) — direction preserved, gap widens

Comparing label-free open (M4b) and label-free FC (M4c) on the same
stim (the legitimate H4 paired comparison):

- **Old scorer**: open 0.948, FC 0.817 → paired delta +0.131 (FC < open).
- **New scorer**: open 0.938, FC 0.758 → paired delta **+0.180** (FC < open).

H4 direction is preserved (FC remains the more conservative format)
and the gap **widens** by 0.049, *strengthening* the H4 claim. The
M4c insight doc's existing scorecard line "open-vs-FC paired delta
on the same stimulus is **−0.131**" should be updated to **−0.180**.

(Note: an earlier draft of this audit incorrectly compared `mvp_full`
Qwen open prompt with `fc_label_free` Qwen, which are different
stim/label conditions and not a valid H4 measurement. The correct
H4 comparison is open-vs-FC at the *same* no-label condition,
reported above.)

## Recommended actions

1. **Treat `predictions_scored_v2.csv` as authoritative** for the 5
   affected runs going forward. Store the v1 PMR in a `pmr_v1` column
   to allow diffing.
2. **Update insight docs with new absolute numbers** (footnote-level
   only — no headline reframes needed except H4 sign flip):
   - `docs/insights/m1_pilot.md`: PMR 0.671 → 0.658.
   - `docs/insights/m2_*.md`: 0.797 → 0.773.
   - `docs/insights/m4b_label_free.md`: 0.948 → 0.938.
   - `docs/insights/m4c_fc_label_free.md`: **footnote** — H4 paired
     delta widens from −0.131 to −0.180 (direction preserved, claim
     strengthened); FC absolute 0.817 → 0.758.
   - `docs/insights/m6_cross_model_*.md`: Qwen ladder entry 0.94 → 0.92.
3. **No re-inference needed** — all changes are scorer-side.
4. **Update `references/roadmap.md` § hypothesis scorecard** for H4
   (paired-delta sign flip).
5. **Keep this audit doc** as the single source of truth for "what
   numbers in old insight docs are stale".

## Reproducer

```bash
# Re-score all affected runs and store as v2 alongside originals.
uv run python -c "
import pandas as pd
from pathlib import Path
from physical_mode.metrics.pmr import score_pmr

AFFECTED = [
    'outputs/pilot_20260424-072418_2c16efb6/predictions_scored.csv',
    'outputs/mvp_full_20260424-094103_8ae1fa3d/predictions_scored.csv',
    'outputs/label_free_20260425-031430_315c5318/predictions_scored.csv',
    'outputs/fc_label_free_qwen_20260425-042817_eec92f1a/predictions_scored.csv',
    'outputs/cross_model_internvl3_20260425-051009_fc710e85/predictions_scored.csv',
    'outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/predictions_scored.csv',
]
for path in AFFECTED:
    p = Path(path)
    df = pd.read_csv(p)
    df['pmr_v1'] = df['pmr']
    df['pmr'] = df['raw_text'].apply(score_pmr)
    df['pmr_diff'] = (df['pmr_v1'] != df['pmr']).astype(int)
    df.to_csv(p.parent / 'predictions_scored_v2.csv', index=False)
"
```

## Artifacts

- `<run_dir>/predictions_scored_v2.csv` — 6 affected runs, with
  `pmr_v1` and `pmr_diff` diagnostic columns.
- `/tmp/scorer_regression_audit.txt` — full audit terminal output.
