# M4c FC Label-Free Prompt — Run Log

Companion to M4b — extends the H2 null-hypothesis test from open-ended
prompts to forced-choice prompts. Adds the new `forced_choice_no_label`
variant (4-way MCQ with "the depicted object" as antecedent for options
A-C) and runs it on the same M2 stimuli on both Qwen2.5-VL-7B and
LLaVA-1.5-7B.

Run date: 2026-04-25.

## Setup

- New prompt variant: `forced_choice_no_label` — same A/B/C/D options as
  `forced_choice` but the subject of A/B/C is "the depicted object"
  instead of "It"/"the {label}". D is kept verbatim ("This is an
  abstract shape — nothing physical happens.") because it does not
  reference a depicted object.
- Configs:
  - `configs/fc_label_free_qwen.py` — Qwen2.5-VL-7B-Instruct.
  - `configs/fc_label_free_llava.py` — llava-hf/llava-1.5-7b-hf.
- Stimuli: M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/`.
- Generation: T=0.7, top_p=0.95, max_new_tokens=96.
- Activation captures disabled.
- Outputs:
  - `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/` — 480 rows.
  - `outputs/fc_label_free_llava_20260425-044517_81ae56d5/` — 480 rows.

## Prompt template

```
Look at the image. Which option best describes what will happen next?
A) The depicted object falls down.
B) The depicted object stays still.
C) The depicted object moves sideways.
D) This is an abstract shape — nothing physical happens.
Answer with a single letter (A, B, C, or D), then briefly justify.
```

## Qwen FC label-free — overall

| metric | value |
|---|---|
| n | 480 |
| PMR (text-based) | 0.817 |
| GAR | 0.781 |
| hold_still | 0.225 |
| abstract_reject | 0.135 |

### First-letter distribution

| letter | count | %    |
|---|---|---|
| A | 292 | 60.8 |
| B |  77 | 16.0 |
| D |  57 | 11.9 |
| other | 54 | 11.3 |

(C did not appear at all, consistent with M2 FC labeled.)

### First-letter by object_level

| object   | A  | B  | D  | other |
|---|---|---|---|---|
| filled   | 67 | 25 | 17 | 11 |
| line     | 60 | 16 | 30 | 14 |
| shaded   | 80 | 18 |  4 | 18 |
| textured | 85 | 18 |  6 | 11 |

### First-letter by cue_level

| cue          | A  | B  | D  | other |
|---|---|---|---|---|
| both         | 98 |  0 |  0 | 22 |
| cast_shadow  | 70 | 23 | 27 |  0 |
| motion_arrow | 88 |  0 |  0 | 32 |
| none         | 36 | 54 | 30 |  0 |

### Paired delta vs Qwen open label-free (M4b, same `sample_id`)

| comparison | mean Δ |
|---|---|
| `PMR(FC, _nolabel) − PMR(open, _nolabel)` | **−0.131** |

FC is more conservative than open at the same (image, label-free) cell —
it routes more responses to D (abstract reject) when the image is ambiguous.

### Paired delta vs Qwen FC labeled (M2, by label)

| labeled M2 | mean Δ `PMR(FC, label) − PMR(FC, _nolabel)` |
|---|---|
| ball   | **+0.013** |
| circle | **−0.208** |
| planet | **−0.263** |

- `ball − _nolabel` ≈ 0 — same null-finding as M4b's open prompt:
  ball does not enhance over the no-label baseline.
- `circle − _nolabel` = −0.208 — much stronger than M4b's open
  prompt delta of −0.065. FC's D option provides a clean abstract
  outlet that circle exploits more aggressively.
- `planet − _nolabel` = −0.263 — **new finding under FC**. The
  M4b open prompt showed `planet ≈ no-label` because planet's orbital
  responses ("orbits the sun", "consumed by black hole") were
  scored as physics-mode under the verb lexicon. Under FC, those
  same orbital intuitions cannot be expressed because the option set
  is gravity-centric (falls/stays/moves sideways); they collapse to
  D ("abstract shape — nothing physical happens"), suppressing PMR.

## Qwen — `line/blank/none` cell (fully abstract, FC vs open vs labeled)

| condition          | PMR  | hold_still | abstract_reject | first_letter (n=10) |
|---|---|---|---|---|
| open × _nolabel    | 0.40 | 0.20 | 0.00 | n/a |
| FC × _nolabel      | 0.00 | 0.40 | 1.00 | D=9, B=1 |
| FC × ball          | 0.00 | 0.60 | 1.00 | D=10 |
| FC × circle        | 0.00 | 1.00 | 1.00 | D=10 |
| FC × planet        | 0.00 | 0.10 | 1.00 | D=10 |

Under FC at this fully abstract cell, **all four label conditions
collapse to D**. The first-letter table shows the no-label run is
slightly less collapsed (D=9, B=1) than the labeled runs (D=10), so
the no-label condition gives the model very slightly more flexibility,
but the FC option set's D escape is dominant. This is the cleanest
demonstration of FC's "abstract sink" pull at fully abstract images.

Note: hold_still=0.60 / 1.00 with PMR=0.00 is not contradictory — the
model picks D and writes a justification like "this is a static
representation"; "static" triggers hold_still but the response is
still a D selection.

## LLaVA FC label-free — degenerate

`first_letter` distribution across all 480 stimuli:

| letter | count |
|---|---|
| A | 477 |
| B | 3 |
| C | 0 |
| D | 0 |

The "A" bias observed in M6 round 1 (12/12 on a 4-cell smoke with
labeled FC) **persists** under the re-templated label-free prompt.
Re-wording from "It falls down" / "the {label} falls down" to
"the depicted object falls down" does not move LLaVA off the A
default. Only 3/480 responses pick B; never C or D.

This rules out the prompt-template hypothesis as the source of the
LLaVA FC bias. The bias is at the model level — likely related to
LLaVA-1.5's training data preferring "A" answers in MCQ contexts, or
to a tokenization artefact in the FC choice generation step. Either
way, FC is unusable on LLaVA-1.5 with verb-PMR or first-letter
metrics.

Sample LLaVA FC label-free responses:

| cell | response |
|---|---|
| line/blank/none | `'A'` (× 9), `'A'` for the rest |
| textured/ground/both | `'A'` (× 10) |
| textured/blank/none | `'A'` (× 10) |

(The 3 `B` responses across all 480 stimuli are scattered without
obvious cell pattern — random noise.)

## Raw artifacts

- `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/fc_label_free_llava_20260425-044517_81ae56d5/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows (degenerate; not used for H2 testing).
- `outputs/fc_label_free_*/summary_*.csv` — factor-level rollups.
