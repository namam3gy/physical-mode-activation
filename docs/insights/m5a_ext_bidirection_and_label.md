# M5a Extensions — Bidirectionality & Label Interaction

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **H-boomerang** — Vision encoder linearly separates physics-mode classes even where behavior fails — encoder knows, decoder gates. (Qwen-scoped: refuted on LLaVA-1.5 because its CLIP encoder is the bottleneck.)
> - **H-direction-bidirectional** — v_L10 is a regime axis within physics-mode (+α → kinetic, −α → static); revised from the initial "one-way activator" framing.
> - **H-locus** — The bottleneck is at the LM mid layers (L10 specifically), not earlier or in the decoding head.
> - **H-regime** — The steering direction is binary "object-ness" — refuted; replaced by H-direction-bidirectional.
> - **M2** — ST1 MVP-full — 5-axis factorial (2880 stim); H1 monotone S-curve, H7 emerged.
> - **M5a** — ST4 VTI steering — adding +α·v_L10 at LM L10 over visual tokens flips line/blank/none from "stays still" → physics-mode.
> - **M5b** — ST4 Phase 3 (SIP + activation patching + SAE feature decomposition) — deferred / optional.
> - **M6** — ST5 cross-model sweep — see M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next).
> - **v_L10** — Steering direction in LM hidden space (dim 3584) at layer 10, derived from M5a class-mean diff (physics − abstract). Unit norm.

Follow-up to `m5_vti_steering.md`. Addresses two §7 limitations:
negative-α bidirectionality and label × steering-direction interaction.

Raw numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.
Implementation: `scripts/06_vti_steering.py` (extended), `src/physical_mode/metrics/first_letter.py`.

> **Revision 2026-04-25**: Exp 1's "no effect from negative α" was confirmed
> to be a ceiling artifact. Exp 3 on a moderate baseline revealed that negative
> α **does** drive a behavioral shift — it flips the model into the "stays
> still" (B) regime rather than suppressing physics-mode back to abstract.
> §1, §2.3, §3.3, and §4 have been rewritten accordingly; §2.2 retains the
> original Exp 1 numbers for completeness.

## 1. One-line summary

The L10 direction is a **regime axis within physics-mode**, not a
physics-vs-abstract activator: large `|α|` at L10 breaks the model out of the
baseline "abstract / won't move" (D) response into physics-mode, while the
*sign* of α selects **which** physics regime the model narrates — `+α` pushes
toward a kinetic "falls" (A) response, `-α` pushes toward a static "stays
still" (B) response. Label and image priors modulate the positive-α target
(label=ball or obj=textured is enough to elicit A at +α=40; both abstract
elicit B at +α=40), but `-α=40` uniformly elicits B regardless of label or
object level.

## 2. Bidirectionality test

### 2.1 Exp 1 setup (ceiling baseline — 2026-04-24)

See run log §Experiment 1.

### 2.2 Exp 1 result

| α | A | other | D | PMR |
|---|---|---|---|---|
| 0   | 9  | 1 | 0 | 1.0 |
| -5  | 10 | 0 | 0 | 1.0 |
| -10 | 10 | 0 | 0 | 1.0 |
| -20 | 10 | 0 | 0 | 1.0 |
| -40 | 10 | 0 | 0 | 1.0 |

### 2.3 Exp 3 recheck (moderate baseline — 2026-04-25)

Exp 1's `textured/ground/both` baseline was at PMR=1 ceiling; the
null-movement under negative α could be either (a) a ceiling artifact or
(b) genuine asymmetry. Exp 3 moved to `textured/blank/none` (α=0 baseline
≈ 10/10 D, near floor) and swept α ∈ {−40, −20, −10, −5, 0, 5, 10, 20, 40}
across both `label ∈ {ball, circle}` and both `obj ∈ {line, textured}` —
yielding a full (α × label × obj) grid at L10, T=0. Raw tables in the run
log §Experiment 3.

Cross-run summary at |α|=40:

| obj × label | α=−40 | α=0 | α=+40 |
|---|---|---|---|
| line × circle     | **10 B** | 10 D | 10 B |
| line × ball       | **10 B** | 10 D | 10 A |
| textured × circle | **10 B** | 10 D | 10 A |
| textured × ball   | **10 B** |  9 D + 1 B | 10 A |

- **`-α=40` → 10 B across all four cells**, regardless of label or object.
- **`+α=40` → 10 A** in three of the four cells (textured/*, and line/ball);
  only `line × circle` gives 10 B at +α=40 (the original M5a result).
- `α=0` is always 10 D (modulo one Exp 3a stimulus that was 1 B / 9 D).

### 2.4 Revised interpretation — regime axis within physics-mode

The Exp 1 "no effect" finding was a **ceiling artifact**, not inherent
asymmetry. Once the baseline drops to the D floor, large |α| at L10
breaks the model out of D into a physics-mode response, and the **sign**
selects which physics regime:

- **`+α · v_L10`** → "falls" (A) — kinetic / gravity-active regime. At
  +α=40, A is elicited whenever *either* the image carries a physical
  appearance (textured) *or* the label carries a physical prior (ball).
  A is suppressed only when both image and label are abstract (line +
  circle), in which case the model instead defaults to B ("stays still").
- **`-α · v_L10`** → "stays still" (B) — static / gravity-passive regime.
  The -α=40 target is B uniformly across all four (obj × label) cells in
  Exp 3; neither label nor image prior moves the target.

So `v_L10` is neither a "physics vs abstract" axis nor an "object-ness
on/off activator". It is an **axis within the physics-mode subspace**,
with two opposing regimes (kinetic vs static) at its endpoints and the
baseline D ("won't move — it's abstract") response sitting *below* the
|α| threshold rather than at one end of the axis.

This changes the M5a causal story from "object-ness gets turned on / off"
to "we know the model has a `v_L10`-aligned physics-mode subspace, and we
can causally pick which regime the model narrates by choosing the sign —
but only once |α| crosses ~15-20". The D response is outside this
subspace: neither direction of α returns the model to D once it is pushed
out.

## 3. Label × steering (Exp 2)

### 3.1 Setup

See run log §Experiment 2.

### 3.2 Result

| α | A | D | PMR |
|---|---|---|---|
| 0  | 0  | 10 | 0.0 |
| 5  | 0  | 10 | 0.0 |
| 10 | 0  | 10 | 0.0 |
| 20 | 0  | 10 | 0.0 |
| 40 | 10 | 0  | 1.0 |

### 3.3 Interpretation — label interacts with the +α regime target, not the -α regime target

At `line/blank/none × +α=40`, the regime target depends on the label:
- label=`circle` (M5a) → 10/10 B.
- label=`ball` (Exp 2) → 10/10 A.

Holding the image, stimulus cell, steering vector, and magnitude constant,
the flip target changes from B to A when the label changes. This is
dissociation evidence for H7 (label selects physics regime) — but,
informed by Exp 3, the H7 claim is now **narrower** than originally
stated:

- **Scope of label-selects-regime**: holds when the image itself is
  abstract (`line`). In `textured/blank/none × +α=40`, both labels give
  A (Exp 3a: ball → 10 A; Exp 3b: circle → 10 A) — label is fully
  dominated by the image-level physical signal once it is present.
- **-α regime target is label-independent**: `-α=40` elicits B across
  all four (obj × label) cells in Exp 3. Label does not disambiguate
  the static regime.

Restated: regime is chosen by a **joint** (image, label, α sign) function,
not by label alone. Label-driven regime-flipping is visible only when
the other two channels are weak (abstract image, moderate |α|). The
cleanest Exp 2 finding — "same steering, only label differs, regime
flips" — is still a valid causal demonstration, but its generalization is
restricted to the abstract-image regime.

Subsidiary: the α=0 baseline for ball+line+blank+none is 10/10 D — the
label prior alone does not make this (abstract) stimulus physics-mode
under the steering script's forced-choice prompt. This matches the M5a
observation that `circle+line+blank+none × α=0` was also 10/10 D
(`docs/insights/m5_vti_steering.md` §3.2). Both together suggest the
steering-script's forced-choice prompt template is more conservative
than M2's default prompts — worth reconciling in a future prompt
variant audit, but not a threat to the present result.

## 4. Hypothesis scorecard update

| H | Pre-M5a-ext | Post-M5a-ext (2026-04-24) | Post-recheck (2026-04-25) |
|---|---|---|---|
| H-boomerang | extended + causal | unchanged | **unchanged** — causal leg reinforced (Exp 2 + Exp 3 grid). |
| H-locus | supported (early-mid) | unchanged | **unchanged** — L10 effective across all Exp 3 cells. |
| H-regime | candidate | supported (causally) | **supported but narrower** — label-only regime flip holds when image is abstract; with textured image, +α=40 gives A regardless of label. Regime is chosen by a joint (image, label, α sign) function. |
| **H-direction-bidirectional** (new 2026-04-24) | — | refuted as bidirectional, supported as one-way activator | **revised**: `v_L10` is a **regime axis within physics-mode** — +α → kinetic/falls (A), -α → static/stays-still (B), baseline D sits below the \|α\| activation threshold. The original "one-way activator" framing was itself a ceiling artifact of Exp 1. |

## 5. Paper implications

- **Figure 6 (M5a causal steering) gains a multi-panel companion**: the
  (α × label × obj) grid from Exp 2 + Exp 3 gives a 2×2 of "same image,
  same α sign, label flip" and "same image, same label, α sign flip"
  demonstrations. The cleanest single figure is the `-α=40` row: four
  different (obj, label) cells all collapse to B, independent of any
  prior signal — this is the strongest isolated causal effect in the
  paper so far.
- **Language discipline in the text**: `v_L10` should be described as a
  "regime axis within physics-mode" rather than a "physics-mode activator"
  (too narrow, given Exp 3) or a "physics-vs-abstract direction" (simply
  wrong, given that negative α does not restore D). The paper must also
  explicitly note that the baseline D response is below the activation
  threshold rather than at one end of the axis — otherwise readers
  unfamiliar with VTI will default to the common "bidirectional concept
  direction" frame.
- **M5b (SAE) motivation is strengthened**: the regime-axis structure is
  more compelling for SAE decomposition than a simple activator would
  be. Expected SAE features: (a) a "kinetic / falls" feature, (b) a
  "static / stays" feature, and (c) a distinct "abstract / not a
  physical object" feature that sits outside the physics-mode axis.
  If SAE recovers (a) + (b) as oppositely-loaded features along the
  mean-diff direction, that validates the regime-axis interpretation at
  a finer grain than behavioral steering can.
- H7 needs a qualifier in the paper: "label selects regime **when image
  is abstract**" — not a global claim. The Exp 3 data shows label
  dominance fails as soon as the image carries physical-object signal
  (textured).

## 6. Limitations still open

- The |α| activation threshold was located to |α|∈[15, 25] by the Exp
  3a α=+10 (7A+1B+2D) / α=+20 (10A) transition; a finer sweep (10, 12,
  15, 18, 22, 25) to pin the threshold precisely is deferred.
- Only L10 tested in Exp 3. The α-sign regime split at L15 / L20 / L25
  is not replicated; by the M5a single-layer result, later layers may
  not participate in the regime axis.
- Exp 3 covers (line, textured) × (ball, circle); the full M2 axis is
  4 × 3 (obj × label) × `planet`. `planet` and `shaded / filled` are not
  tested. M2 data suggests `planet` activates orbital physics (not
  gravity-fall) under baseline — whether `-α=40` still pushes `planet` to
  B is unknown.
- The prompt template is still `forced_choice` at T=0. A prompt audit
  (open-ended / T>0 / label-free §4.9) remains deferred.
- Does not touch SAE / patching / cross-model — those are M5b/M6.
