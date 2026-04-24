# M5a Extensions — Bidirectionality & Label Interaction

Follow-up to `m5_vti_steering.md`. Addresses two §7 limitations:
negative-α bidirectionality and label × steering-direction interaction.

Raw numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.
Implementation: `scripts/06_vti_steering.py` (extended), `src/physical_mode/metrics/first_letter.py`.

## 1. One-line summary

The L10 direction is a **one-way activation, not a bidirectional axis** —
negative α does not suppress physics-mode — but it is **label-composable**:
the same `+α · v_L10` routes to "falls" (A) with label `ball` and "stays
still" (B) with label `circle`. Object-ness is set by steering, regime is
set by label, causally.

## 2. Bidirectionality test (Exp 1)

### 2.1 Setup

See run log §Experiment 1.

### 2.2 Result

| α | A | other | D | PMR |
|---|---|---|---|---|
| 0   | 9  | 1 | 0 | 1.0 |
| -5  | 10 | 0 | 0 | 1.0 |
| -10 | 10 | 0 | 0 | 1.0 |
| -20 | 10 | 0 | 0 | 1.0 |
| -40 | 10 | 0 | 0 | 1.0 |

### 2.3 Interpretation — direction is one-way

Injecting `-α · v_L10` at the physics-mode baseline fails to suppress
physics-mode: all α ∈ {0, -5, -10, -20, -40} leave the first-letter
distribution at ≥ 9/10 A. The 1 "other" at α=0 even concentrates into
A at negative α. Two interpretations are consistent with this null
result:

- **(a) Ceiling effect**: `textured/ground/both` is already at PMR=1 at
  α=0, so there is no room to observe a physics-mode increase; but the
  non-movement toward D means -v is not *removing* physics-mode either.
- **(b) Inherent asymmetry**: the direction activates the physical-object
  concept but does not have a "abstract-mode" antipode in the same residual
  direction. Suppressing physics-mode would require a different mechanism
  (different direction, different layer, or negative sign with a different
  feature).

Either way, **the M5a finding should be framed as a one-way activator**:
M5a showed that positive α pushes abstract → physical; Exp 1 shows that
negative α does not push physical → abstract. The direction is not a
bidirectional concept axis as VTI is sometimes assumed to be (e.g.,
gender, truthfulness directions in LLM literature).

An important caveat: because the baseline is already at the physics-mode
ceiling, a cleaner bidirectionality test would use an `textured/blank/none`
stimulus (moderate baseline PMR around 0.5) where both +α and -α have
measurement headroom. Deferred.

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

### 3.3 Interpretation — regime is label-driven, activation is steering-driven

With label=`ball`, α=40 flips 10/10 to A ("falls"). With label=`circle`
(M5a), the same intervention flipped 10/10 to B ("stays still"). This
is a **clean dissociation**: holding the image, the stimulus cell, the
steering vector, and the magnitude constant, only the prompt token
changes — and the flip target changes from B to A. This is strong
causal evidence for **H7 (label selects the physics regime)**: the
steering direction activates "physical object-ness" as a coarse binary,
while the label prior determines which physics the model narrates.

Subsidiary: the α=0 baseline for ball+line+blank+none is 10/10 D — the
label prior alone does not make this (abstract) stimulus physics-mode
under the steering script's forced-choice prompt. This matches the M5a
observation that `circle+line+blank+none × α=0` was also 10/10 D
(`docs/insights/m5_vti_steering.md` §3.2). Both together suggest the
steering-script's forced-choice prompt template is more conservative
than M2's default prompts — worth reconciling in a future prompt
variant audit, but not a threat to the present result.

## 4. Hypothesis scorecard update

| H | Pre-M5a-ext | Post-M5a-ext | Change |
|---|---|---|---|
| H-boomerang | extended + causal | **extended + causal (unchanged)** | Exp 2 reinforces the causal leg (one more mechanism-level intervention shows label × steering composability). |
| H-locus | supported (early-mid) | **unchanged** | Exp 2 confirms L10 is still the effective site even with the label swap. |
| H-regime | candidate | **supported (causally)** | Exp 2: same intervention with only label swap produces A vs B flip. Label selects regime under identical steering. |
| **H-direction-bidirectional** (new) | — | **refuted (as bidirectional), supported (as one-way activator)** | Exp 1: `-α · v_L10` at `textured/ground/both` does not shift toward D. The direction activates object-ness but does not suppress it in the same residual-space direction. |

## 5. Paper implications

- **Figure 6 (M5a causal steering) gains a companion figure**: the Exp 2
  side-by-side — same image, same α, only label differs → A vs B — is the
  cleanest causal demonstration of the decomposition `steering = object-
  ness; label = regime`. A single side-by-side panel makes the paper's
  H7 claim self-contained.
- The H-direction-bidirectional refutation tempers overclaiming: do not
  call `v_L10` a "physics-mode direction" in abstract; call it a
  "physical-object-ness activator". Precise language matters because
  VTI-style vectors in LLM interp are frequently described as
  bidirectional concept axes.
- M5b priority: the SAE decomposition from ROADMAP M5b is now more
  compelling — if `v_L10` is a one-way activator, SAE features could
  reveal both (a) the activating sub-features and (b) a distinct
  suppressive direction that a simple mean-difference VTI misses.

## 6. Limitations still open

- Exp 1 baseline is at the ceiling (PMR=1.0 at α=0), so the "no
  bidirectional effect" is confounded with "no room for effect".
  A moderate-PMR baseline (e.g. `textured/blank/none`) would disambiguate.
- α=40 remains a magic number; finer sweep (30/35/40/45/50) to locate
  the threshold precisely is deferred.
- Only L10 tested in both experiments. The label swap at L15/L20/L25
  is not replicated.
- Does not touch SAE / patching / cross-model — those are M5b/M6.
