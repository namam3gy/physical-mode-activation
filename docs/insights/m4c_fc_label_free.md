# M4c — Forced-Choice Label-Free Prompt

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H2** — The label (ball / circle / planet) independently raises PMR even on minimal stim — a language-prior contribution beyond the visual evidence.
> - **H4** — The open-ended vs. forced-choice PMR gap is a stable signature of the language-prior ↔ visual-evidence conflict.
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **M2** — ST1 MVP-full — 5-axis factorial (2880 stim); H1 monotone S-curve, H7 emerged.
> - **M4b** — M4 + label-free prompt as H2 null test; revealed H2 is asymmetric on Qwen (circle override, not ball enhancement).
> - **M4c** — Forced-choice label-free variant — confirms M4b under FC; surfaces LLaVA "A" greedy bias.
> - **M6** — ST5 cross-model sweep — see M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next).
> - **M6 r1** — ST5 cross-model — LLaVA-1.5-7B replicates H2 cleanly (unsaturated CLIP encoder lets the label-prior shift PMR).


**The stim** — same M2 line/blank/none baseline as M4b, under forced-choice prompting:

![M4c reference stim: line / blank / none](../figures/01_line_blank_none.png)

Companion to M4b. Tests whether the H2 reframing (`ball ≈ no-label`,
`circle = suppressor`) survives moving from open-ended to forced-choice
prompts; also tests whether re-templating the FC options without a
label antecedent (`The depicted object falls down...`) relaxes
LLaVA-1.5's pathological "A" bias from M6 r1.

Raw numbers: `docs/experiments/m4c_fc_label_free.md`.
Code: `src/physical_mode/inference/prompts.py` (new
`forced_choice_no_label` variant), `configs/fc_label_free_{qwen,llava}.py`.

## 1. One-line summary

On Qwen2.5-VL the M4b H2 finding **strengthens under FC** — `ball`
remains ≈ no-label (Δ=+0.013), `circle` suppresses harder (Δ=−0.208 vs
M4b's open Δ=−0.065), and `planet` newly suppresses (Δ=−0.263) because
the FC option set is gravity-centric and collapses orbital physics into
the D ("abstract") sink. On LLaVA-1.5 the FC "A" bias **persists**
(477/480 = 99.4 % `A`) under the re-templated prompt — the bias is at
the model level and FC is unusable on LLaVA-1.5 regardless of prompt
design.

## 2. Qwen FC label-free — confirmation + amplification of M4b H2

### 2.1 Direction is consistent

| comparison | M4b open | M4c FC |
|---|---|---|
| `PMR(ball) − PMR(_nolabel)`   | +0.006 | +0.013 |
| `PMR(circle) − PMR(_nolabel)` | −0.065 | **−0.208** |
| `PMR(planet) − PMR(_nolabel)` | +0.006 | **−0.263** |

`ball ≈ no-label` survives the prompt-format shift cleanly. `circle`
moves further negative under FC. `planet` flips from "≈ no-label" to
strongly negative.

### 2.2 The FC "abstract sink" — why circle and planet move further negative under FC

FC's options A-C are all gravity-centric: A "falls down", B "stays
still", C "moves sideways". D is the abstract escape: "This is an
abstract shape — nothing physical happens." When the image is
ambiguous and the label evokes a regime that doesn't fit
{falls, stays, moves sideways}:

- The open prompt lets the model write its preferred narration
  (`circle` → "becomes smaller", `planet` → "orbits the sun") which
  the verb-PMR lexicon scores as physics-mode (PMR = 1) for some
  responses and not for others.
- The FC prompt forces the model to commit to one of {A, B, C, D}.
  Orbital physics has no native FC option, so `planet` → D. Abstract
  geometry has no native option, so `circle` → D. The D rate
  per-label rises sharply, dragging text-PMR down.

Numerical example at `line/blank/none`:

| condition       | first_letter | text PMR |
|---|---|---|
| open × _nolabel | n/a (open)   | 0.40 |
| FC × _nolabel   | D=9, B=1     | 0.00 |
| FC × ball       | D=10         | 0.00 |
| FC × circle     | D=10         | 0.00 |
| FC × planet     | D=10         | 0.00 |

At a fully abstract image, FC collapses every condition to D
regardless of label — the model interprets "abstract shape" as
applicable whenever the image content does not actively support a
gravity-centric reading. This is FC's option-set bias, not a
behavioral statement about labels per se.

### 2.3 Implication for H2 reading

The visual-saturation hypothesis (M6 r1) is reinforced: M4b's "circle
suppression only" was a Qwen-specific consequence of Qwen's PMR ceiling
under the open prompt; under FC, the model has explicit access to the
abstract option, so the suppressive direction is more visible across
labels. The unified statement holds: **language prior contributes
positively (or zero) for every label**, and the apparent negative
contributions in Qwen are FC's option-set bias + Qwen's visual
saturation, not actual negative semantic priors.

## 3. LLaVA FC label-free — pathology persists, FC is dead on this model

| `first_letter` | count |
|---|---|
| A | 477 |
| B |   3 |
| C |   0 |
| D |   0 |

LLaVA gives `A` for 477/480 stimuli even with the re-templated prompt
that removes the label antecedent ("the depicted object falls down"
instead of "the ball falls down"). The bias is independent of:
- image content (`line/blank/none` and `textured/ground/both` both 100% A),
- label (12/12 on M6 labeled smoke; 477/480 on label-free),
- prompt antecedent (`It` vs `the {label}` vs `the depicted object`).

Therefore the LLaVA FC bias is at the model level. Hypothesis: LLaVA's
instruction-tuning data over-weighted "A" answers in MCQ format, or
the FC instruction interacts with the start-of-answer token statistics
in a way that locks the first letter. This is a sticky model-property
finding for the cross-model section of the paper but not fixable in
this round.

## 4. Hypothesis scorecard update

| H | Pre-M4c | Post-M4c |
|---|---|---|
| **H2** | revised under visual-saturation hypothesis (M6 r1) | **further reinforced** — Qwen FC label-free reproduces the M4b "ball ≈ no-label, circle suppresses" pattern in a different prompt format. The FC sink under "planet" reveals that the per-label suppression seen in Qwen is *partly* an option-set artefact, supporting the visual-saturation framing rather than a literal "abstract override" claim about specific labels. |
| **H4** (open vs FC gap) | supported (Qwen) | **measurable cross-format on Qwen** — paired open-vs-FC delta on the same stimulus (no-label) is **−0.131**. FC is consistently more conservative than open. Ready for cross-model H4 once the LLaVA FC bias is resolved. |
| **H7** (label selects regime) | supported but narrower; cross-model replicated (M6 r1) | **caveat added** — the regime distinction (planet → orbital, ball → gravity) is only visible under prompts that allow narrative latitude. Under FC, all non-gravity regimes collapse to D, masking H7. The FC option set as currently written is biased toward gravitational physics; an extended option set ("D) The depicted object orbits or rotates", "E) Other") would be required to test H7 under FC. |
| **LLaVA FC bias** (M6 r1 finding) | observation | **confirmed model-level pathology** — re-templating the FC antecedent does not move LLaVA off the A default. FC is unusable on LLaVA-1.5; the cross-model H4 test is fully blocked here. |

## 5. Paper implications

- M4b's "ball ≈ no-label, circle = suppressor" finding is **prompt-
  robust on Qwen** — it appears under both open prompt and FC prompt,
  with the FC version showing a stronger circle suppression and a
  new "planet" suppression that's an option-set artefact.
- The FC option set is gravity-centric and pulls non-gravitational
  regimes into D. This has methodological implications: any FC-based
  PMR comparison across regime-flexible labels (e.g. `planet`) will
  underestimate physics-mode commitment for orbital-routed labels.
  Paper should flag this with a one-line note in the methods section.
- LLaVA-1.5 FC bias is a model-level pathology. Reportable as a
  cross-model methodology point: behavioral metrics that depend on
  forced-choice answer selection are not portable to LLaVA-1.5
  without significant prompt-side scaffolding (different option set,
  different instruction style) that would itself confound the
  cross-model comparison. Recommend the open-prompt protocol as the
  cross-model standard going forward.
- Open-vs-FC gap (H4) at no-label is now measurable: **−0.131 paired
  PMR** for Qwen at the same stimuli, confirming the M2 H4 pattern
  cleanly without label confounding.

## 6. Limitations

- LLaVA's FC degenerate. Round-2 idea: replace the FC stage entirely
  with a **first-token logit ratio** (P(`A`) vs P(`B`) vs P(`C`) vs
  P(`D`)) instead of greedy first-letter argmax. This bypasses the
  bias if it lives in the temperature-sampling stage but still falls
  to it if the bias is in the underlying logits.
- Planet's collapse under FC is option-set-driven, not necessarily a
  shift in underlying physics-mode commitment. Need an extended FC
  option set (orbit / rotate / other) to disentangle.
- This round did not re-derive M2's FC labeled PMR using
  `first_letter` (the M6 r1 advisor recommendation). Cross-model
  first-letter PMR comparison still TODO; M4c provides the Qwen FC
  label-free first-letter table as a starting point.
- Single FC prompt design. Sensitivity to other re-wordings ("Choose
  the most plausible outcome", "Pick A/B/C/D", removing "Answer with
  a single letter" instruction) not tested.
