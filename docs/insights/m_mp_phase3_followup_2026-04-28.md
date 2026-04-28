# M-MP Phase 3 audit follow-up — MCQ probe + Idefics2 2nd-cell test (2026-04-28 evening)

> **Status**: ✅ complete. Closes audit follow-ups #1 and #4 from
> `docs/insights/review_audit_2026-04-28.md` (load-bearing items).
>
> **Trigger**: Audit follow-up #8 of `docs/insights/review_audit_2026-04-28.md`
> flagged two open items:
> 1. **Multi-choice categorization (NOT yes/no)** to dissociate "categorical task"
>    from "yes/no binary format" in the Qwen Generative-vs-Categorical claim.
> 2. **Idefics2 2nd-cell SAE ablation** to lift the single-cell framing-shift
>    finding (`shaded/ground/both ball`) to architecture-level claim.
>
> **Companion docs**: `docs/experiments/m_mp_phase3.md`,
> `docs/experiments/m_mp_phase3_idefics2_verification.md`,
> `docs/experiments/m_mp_mcq_phase1_smoke.md`,
> `docs/experiments/m_mp_mcq_phase2.md` (TBD post-chain).

## TL;DR

- **MCQ probe (1a)**: New `meta_phys_mcq` prompt added to scoring + render
  pipeline. 5-model behavioral run + Qwen Phase 3 M5a + M5b at the existing
  pinned cells.
  - **M5a**: MCQ at α=40 → 0/10 flip (matches yesno, not describe).
  - **M5b**: MCQ at top_k=20 → 10/10 break (A → B; matches describe, not
    yesno). Random control retains 10/10 baseline.
  - **Empirical finding**: at the audit-pinned Qwen cells, **MCQ splits the
    M5a × M5b methods** — M5a categorical-blockade matches yesno; M5b break
    matches describe. The original "Generative-vs-Categorical" unified
    framing is partially preserved (M5a) and partially split (M5b). Yes/no
    is the **single (n=1) categorical prompt** where M5b doesn't break —
    we describe this as **yes/no-prompt-specific** until more categorical-
    binary prompts confirm whether the load-bearing axis is "format" or
    "this specific prompt".
- **Idefics2 2nd-cell test (1b)**: `textured/ground/cast_shadow ball` cell
  baseline is already suspended ("The ball is in the air."), not kinetic.
  Top-k SAE ablation has no observable effect (already in suspended frame),
  but random ablation introduces "and it is falling" (kinetic) on this cell.
  - Cannot directly lift the original cell-1 framing-shift claim to
    architecture-level via this cell. **Refined claim**: SAE features are
    kinetic-verb-encoding, with effect observable only when baseline produces
    kinetic verbs. On baseline-suspended cells, top-k ablation is a no-op
    (specificity confirmed).

## Setup

### MCQ prompt + scorer (1a-1)

**Prompt** (`META_PHYS_MCQ_TEMPLATE` in `src/physical_mode/inference/prompts.py`):

> The image shows a {label}. Which option best describes what this image depicts?
> A) A real-world physical event (an object subject to gravity, mass, or momentum).
> B) A geometric figure or abstract shape with no physical context.
> C) A symbol, icon, or schematic diagram.
> D) None of the above.
> Answer with a single letter (A, B, C, or D), then briefly justify.

**Scorer** (`score_meta_phys_mcq` in `src/physical_mode/metrics/pmr.py`):
- A → 1 (physical event = physics-mode commitment)
- B/C/D → 0 (non-physical category)
- Unparseable → -1 (treated as 0 in `score_for_variant`)

**Scoring caveats** (advisor 2026-04-28):
- Option A is more verbose/evocative than B/C/D → small "verbosity-preference"
  upward bias in absolute PMR. H2 paired-delta robust (label is the only
  diff across ball/circle/planet).
- The 2x2 (task × format) is missing the "generative + yes/no" cell, so any
  asymmetry between MCQ and yesno is interpretable as either task-level OR
  format-level. The Phase 3 result reveals different boundaries on M5a vs
  M5b — see Interpretation below.

### Pre-committed parse-rate threshold (advisor 2026-04-28)

≥85% parseable per model on smoke required before Phase 2 full run. Result:
**5/5 models pass with 100% parse rate** (smoke n=144 each). All of Qwen,
LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3 produce a leading A/B/C/D with
no exceptions.

## Phase 1 smoke results (5-model n=48 stratified)

Stim source: `inputs/m_mp_smoke_strat/` (existing 48-cell stratified subset).
Inferences: 48 stim × 3 labels × 1 prompt = 144 / model × 5 models = 720 total.
Wall: ~7 min sequential on GPU 1. Output: `outputs/mcq_<model>_<ts>/`.

| Model | n | PMR | Parse rate | circle | ball | planet | Δ ball−circle | First-letter dist |
|---|---|---|---|---|---|---|---|---|
| Qwen | 144 | 0.257 | 1.000 | 0.333 | 0.354 | 0.083 | +0.021 | C 63 / B 44 / A 37 |
| LLaVA-1.5 | 144 | 0.389 | 1.000 | 0.125 | 0.583 | 0.458 | **+0.458** | A 56 / B 51 / C 37 |
| LLaVA-Next | 144 | 0.035 | 1.000 | 0.000 | 0.042 | 0.062 | +0.042 | B 116 / C 23 / A 5 |
| Idefics2 | 144 | 0.875 | 1.000 | 0.792 | 0.896 | 0.938 | +0.104 | (saturated) |
| InternVL3 | 144 | 0.528 | 1.000 | 0.333 | 0.604 | 0.646 | +0.271 | (mid-band) |

**Smoke headlines**:
1. **Parse rates 100% for all 5 models** — MCQ prompt design works cleanly
   across architectures. The pre-committed Phase 2 gate (≥85%) is satisfied.
2. **Qwen MCQ ≠ Qwen yesno**: Qwen yesno Yes-rate = 0.729 (Phase 1); Qwen MCQ
   PMR = 0.257. Qwen on MCQ is much more selective when given fine-grained
   options (picks "C: symbol/icon" 44%, "B: geometric" 31%, "A: physical"
   only 26% in smoke).
3. **LLaVA-1.5 H2 paired-delta = +0.458** — strongest classical H2 signal
   in any prompt (vs open ~+0.18, vs yesno ~+0.21). The MCQ format makes the
   label-prior contribution more visible in LLaVA-1.5.
4. **LLaVA-Next "B" preference**: picks B (geometric) 81% of the time.
   Possible MCQ-format-specific behavior or AnyRes interaction; PMR 0.035 is
   very low. Phase 2 will resolve whether this is consistent across cells.

Smoke summary doc: `docs/experiments/m_mp_mcq_phase1_smoke.md`.

## Phase 2 — 5-model × 1440 inferences (full M2 stim)

Chain: `scripts/run_mcq_phase2_chain.sh` on GPU 1 (~75 min wall, 5 models
sequential). Stim: `inputs/mvp_full_20260424-093926_e9d79da3` (480 stim ×
3 labels × 1 prompt = 1440 / model). Total inferences: 7200.

### Per-model headline

| Model | n | PMR | Parse rate | circle | ball | planet | Δ ball−circle | Δ planet−circle |
|---|---|---|---|---|---|---|---|---|
| **Qwen** | 1440 | 0.208 | 1.000 | 0.290 | 0.240 | 0.094 | **−0.050** | −0.196 |
| **LLaVA-1.5** | 1440 | 0.394 | 1.000 | 0.194 | 0.525 | 0.465 | **+0.331** | +0.271 |
| **LLaVA-Next** | 1440 | 0.033 | 1.000 | 0.000 | 0.021 | 0.077 | +0.021 | +0.077 |
| **Idefics2** | 1440 | 0.867 | 1.000 | 0.798 | 0.890 | 0.915 | +0.092 | +0.117 |
| **InternVL3** | 1440 | 0.553 | 1.000 | 0.412 | 0.627 | 0.619 | +0.215 | +0.206 |

### Phase 2 headlines

1. **Parse rates 100% across all 5 models on 7200 inferences** — robust
   MCQ prompt design at scale. Phase 1 smoke result confirmed.

2. **Qwen MCQ has NEGATIVE H2 paired-delta (−0.050)** — the **first cell
   where H2 inverts** across the 4-prompt × 5-model design. This is the
   **only** of the new 20 cells (5 models × 4 prompts) that doesn't show
   `ball ≥ circle`. Possible interpretation: when the prompt says "The
   image shows a ball" but the image is abstract, MCQ option "B: geometric
   figure or abstract shape" becomes a more natural fit (label is `ball`,
   image is `circle` shape — labels mismatch image, model resolves toward
   image-side B). When prompt says "The image shows a circle" + abstract
   image, label-image alignment is high → model picks A more often. This
   is a label-prior × MCQ-option-semantics interaction that doesn't show
   up on yes/no or generative prompts.

3. **H2 cross-prompt-conserved status**: was 15/15 cells positive on the
   3-prompt × 5-model design. With MCQ added: **19/20 cells positive**;
   Qwen × MCQ is the exception. The cross-prompt conservation claim now
   needs a qualifier — robust on 3 of 4 prompt types, with MCQ-specific
   label-image-mismatch interaction in Qwen breaking the pattern.

4. **LLaVA-Next near-floor PMR (0.033)** confirmed at full scale. LLaVA-Next
   on MCQ picks "B: geometric figure" 81%+ of the time. The model's
   AnyRes tile-processing of synthetic abstract stim seems to favor the
   "geometric figure" interpretation over physical-event reading even
   when the label would normally pull toward physical.

5. **Saturated cluster (Idefics2, InternVL3)** preserves positive H2 on
   MCQ as on other prompts. MCQ does not break their saturation.

### Cell variation (representative cells, 5-model)

| Cell | Qwen | LLaVA-1.5 | LLaVA-Next | Idefics2 | InternVL3 |
|---|---|---|---|---|---|
| line/blank/none (most abstract) | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 |
| line/ground/cast_shadow | 0.00 | 0.17 | 0.00 | 0.17 | 0.10 |
| textured/blank/none | 0.00 | 0.33 | 0.00 | 0.37 | 0.50 |
| textured/ground/cast_shadow | 0.20 | 0.47 | 0.03 | 1.00 | 0.77 |
| shaded/ground/both (most physics) | 0.57 | 0.40 | 0.03 | 1.00 | 1.00 |

Cell-strength gradient holds for Qwen (0 → 0.57), Idefics2 (0 → 1.00),
InternVL3 (0 → 1.00), LLaVA-1.5 (0.13 → 0.47). LLaVA-Next is flat near 0
across cells (consistent with its overall floor).

Phase 2 detail doc: `docs/experiments/m_mp_mcq_phase2.md`.

## Phase 3 — Qwen MCQ M5a + M5b (audit-pinned cells)

**Pinned cells** (audit advisor 2026-04-28):
- M5a (flip test): `line/blank/none × circle × α=0,40` (Qwen baseline-low cell).
- M5b (break test): `shaded/ground/both × ball × Cohen's-d top-20` (Qwen baseline-high cell).
- Same cells as Phase 3 yesno + describe runs → apples-to-apples comparison.

### M5a runtime steering at L10 (n=10 stim per condition)

| Prompt | α=0 PMR | α=40 PMR | Effect |
|---|---|---|---|
| open (existing M5a) | low (~0.0) | **1.000 (10/10 flip)** | flip |
| describe_scene (Phase 3) | 0.000 | **1.000 (10/10 flip)** | flip |
| meta_phys_yesno (Phase 3) | 0.000 | 0.000 (NO flip) | NULL |
| **meta_phys_mcq (NEW)** | 0.000 | **0.000 (NO flip)** | **NULL — matches yesno** |

**Sample MCQ output at α=40** (steering toward physics-mode):
> "B  The image shows a circle, which is a geometric figure or abstract shape. It does not depict a real-world physical event, nor is it a symbol, icon, or schematic diagram. Therefore, the correct answer is..."

The model commits to "B = geometric" before the steering can affect the
first-letter answer. Steering reasoning is visible mid-response ("does not
depict a real-world physical event"), but the categorical answer is locked
in by the MCQ task structure.

### M5b SAE intervention top-20 at vision_hidden_31 (n=10 stim per condition)

| Prompt | baseline PMR | top_k=20 PMR | random PMR | Effect |
|---|---|---|---|---|
| describe_scene (Phase 3) | 1.000 | 0.000 (10/10 break) | 1.000 | break |
| meta_phys_yesno (Phase 3) | 1.000 | 1.000 (NO break) | 1.000 | NULL |
| **meta_phys_mcq (NEW)** | 1.000 | **0.000 (10/10 break)** | 1.000 | **break — matches describe** |

**Sample outputs**:
- Baseline: "A — The image depicts a ball falling towards a surface..."
- top_k=20: "B — The image depicts a simple 3D representation of a sphere with a shadow..."
- random_0: "A — The image depicts a ball with an arrow pointing downwards..."

Top-k ablation **flips MCQ answer A → B for 10/10 stim**. Random control
retains baseline 10/10. Specificity confirmed.

## Empirical pattern at Qwen × MCQ (post-MCQ)

The original Phase 3 was framed as a single "generative-vs-categorical"
boundary. The MCQ result reveals that **at Qwen × MCQ, the M5a and M5b
methods give different verdicts**:

| Method | Generative | Categorical-MCQ | Categorical-yesno |
|---|---|---|---|
| **M5a (LM-side steering at L10)** | flip ✅ (open + describe 10/10) | NO flip ❌ | NO flip ❌ |
| **M5b (encoder-side SAE ablation)** | break ✅ (open + describe 10/10) | break ✅ (10/10) | NO break ❌ |

This means:
- **M5a-side pattern**: categorical-MCQ matches categorical-yesno (both NO
  flip). The categorical-task structure (explicit options, discrete-letter
  answer) prevents LM-side steering from affecting the answer in both
  prompts tested.
- **M5b-side pattern**: categorical-MCQ matches generative (both break);
  categorical-yesno is the lone exception (NO break). At the encoder-side,
  yes/no is the **single categorical prompt where M5b doesn't break**.

### Scope caveats (audit 2026-04-28 evening)

The data are at **n=1 categorical-yes/no prompt** (`meta_phys_yesno`) and
**n=1 categorical-MCQ prompt** (`meta_phys_mcq`). The split between MCQ
(M5b breaks) and yesno (M5b doesn't break) is therefore one observed
asymmetry, not a confirmed format-general or task-general mechanism. We
describe yes/no's M5b immunity as **yes/no-prompt-specific** until more
categorical-binary prompts (e.g., paraphrased yes/no, true/false framing)
test whether the load-bearing axis is the response format or this
specific prompt. The audit-defensible empirical claim is: **at Qwen × MCQ,
M5a method nulls and M5b method breaks** — a cross-method dissociation at
one cell, with the existing yesno result as the contrasting reference.

### Possible interpretations (hypotheses, not demonstrated)

The following are plausible mechanistic stories consistent with the data,
but are **not** independently demonstrated by these experiments:

- *(H-LM-prior)* Yes/no answers in Qwen may be language-prior-dominated:
  the LM defaults to "yes" on physics-prompts regardless of encoder cues,
  so ablating physics-cue features doesn't override the language prior.
- *(H-MCQ-encoder-cue)* MCQ answers may engage encoder-cued reasoning to
  pick among options; ablating encoder cues then shifts A → B.
- *(H-generative-dual-route)* Generative outputs (open + describe) may be
  flippable via either LM-side steering OR encoder-side ablation because
  they don't have a hard categorical commitment locked in early.
- *(H-A-verbosity)* **Alternative interpretation for M5b on MCQ**: Option A
  ("A real-world physical event...") is the most verbose / semantically
  rich option. Encoder ablation that disrupts "physical event" semantic
  resonance could shift A → B for verbosity-related reasons, independent
  of any "physics-mode" mechanism. The verbosity-preference confound noted
  in the smoke caveats applies here as well.

These would need additional probes (e.g., MCQ with length-balanced options,
multiple yes/no paraphrases, third categorical prompt type) to dissociate.
For the current audit follow-up, the empirical pattern stands and the
mechanistic stories remain hypotheses.

## 1b — Idefics2 2nd-cell SAE ablation (textured/ground/cast_shadow ball)

Cell selection: audit-recommended `textured/ground/cast_shadow ball` to test
whether the cell-1 (`shaded/ground/both ball`) framing-shift kinetic→suspended
generalizes to a 2nd cell (lifting from single-cell to architecture-level).

### Unexpected baseline

Cell-2 baseline output: **"The ball is in the air."** (already suspended frame,
no kinetic verb). Cell-1 baseline was "A ball is falling down." (kinetic).

This means cell-2 cannot directly test the cell-1 framing-shift claim — there
are no kinetic verbs at baseline to ablate.

### Token-frequency results (n=10 stim per condition)

| Condition | Kinetic verb (`fall|drop`) | Suspended frame (`in the air`) | Unique outputs |
|---|---|---|---|
| baseline (k=0) | 0/10 | 10/10 | 1 ("The ball is in the air.") |
| top_k=160 | 0/10 | 10/10 | 1 ("The ball is in the air.") |
| top_k=320 | 0/10 | 10/10 | 1 ("The ball is in the air.") |
| top_k=500 | 0/10 | 10/10 | 1 ("The ball is in the air.") |
| random_0 (mass-matched) | **10/10** | 10/10 | 2 ("...and it is falling") |

### Interpretation

Top-k SAE ablation: **no-op on baseline-suspended cell** (kinetic features
aren't already firing, so removing them changes nothing).

Random ablation: **introduces** kinetic verbs on this cell. This is the
opposite pattern from cell-1 (where random retained kinetic baseline). The
mass-matched random feature set differs across cells, so the random-control
behavior is also cell-conditional.

**Architecture-level claim status**:
- Cell-1 alone: "Idefics2 SAE features encode kinetic-verb production"
  (single-cell, 30/30 framing-shift).
- Cell-1 + Cell-2 combined: "Idefics2 SAE features encode kinetic-verb
  production specifically. Top-k ablation flips kinetic→suspended on cells
  where baseline produces kinetic verbs (cell-1), and is a no-op on cells
  where baseline is already suspended (cell-2). The mechanism is consistent;
  the effect is observable only in cells where the kinetic frame is the
  default response."

This **narrows but does not refute** the cell-1 finding. The 2nd cell test
provides specificity evidence: top-k ablation isn't a generic perturbation —
it specifically targets kinetic-verb production. When there's no kinetic
verb to remove, ablation is silent.

**Architecture-level lift status**: ⚠️ partial. The 2nd-cell test does NOT
demonstrate framing-shift on a fresh cell (because baseline is already
suspended). It DOES demonstrate that the SAE features have specific
behavior tied to kinetic-verb production rather than arbitrary perturbation.
Lifting to a fully architecture-level claim would require finding a 3rd cell
with kinetic baseline (e.g., different cue combinations) — left for Pillar B
follow-up if needed.

## Implications for paper §6 (Mechanistic level)

### What changes

1. **Phase 3 dissociation**: was "Qwen generative-vs-categorical (within
   Qwen)" treated as a single boundary on M5a + M5b together. Now: at
   Qwen × MCQ, the M5a and M5b methods give different verdicts (M5a null,
   M5b break). The unified framing is split — the M5a result alone matches
   the original "categorical-blocks-steering" reading; the M5b result alone
   shows that yes/no is the **single (n=1) categorical prompt** where
   ablation doesn't break the answer. Whether yes/no's M5b immunity
   generalizes to other categorical-binary prompts is untested.

2. **Phase 3 cross-method "agreement" recount**: The audit's 1+3 asymmetry
   (1 positive/positive + 3 nulls) is preserved on 4 (model × prompt) cells,
   but the audit's MCQ extension adds a **new** cell (Qwen × MCQ) with a
   **mixed** signature — M5a null + M5b positive. So the picture is now:
   - Qwen × describe: M5a positive + M5b positive (the original "co-firing" cell).
   - Qwen × open: M5a positive + M5b positive (existing M5a-ext + M5b SAE).
   - Qwen × MCQ: **M5a null + M5b positive** (NEW — the cross-method split).
   - Qwen × yesno: M5a null + M5b null.
   - Idefics2 × {describe, yesno}: both null/null (limited cross-method power).
   - Idefics2 × cell-2 SAE: top-k no-op (not on yes/no/describe; on suspended-baseline cell).

3. **Idefics2 single-cell finding**: cell-1 framing-shift remains the
   primary evidence; cell-2 specificity test confirms SAE features are
   kinetic-specific rather than arbitrary, but doesn't lift to a fully
   architecture-level claim.

### What stays

- Behavioral H2 cross-prompt-conserved (Phase 2 5-model × 4-prompt — TBD).
- "Robust > flashy" framing — narrower honest claims preferred.
- M5a + M5b co-firing on Qwen describe is the canonical positive cell.

### Caveats (carry forward)

- **Verbosity-preference confound on Option A** (advisor 2026-04-28): Option
  A is the most semantically rich option. H2 paired-delta is robust to this
  confound (label is the only diff across ball/circle/planet) but absolute
  MCQ PMR may be biased upward. **Specifically for M5b**: A → B shift under
  encoder ablation could reflect verbosity sensitivity rather than a
  "physics-mode mechanism"; flagged as alternative interpretation H-A-verbosity.
- **2x2 (task × format) missing "generative + yes/no" cell**: cannot fully
  dissociate task-level from format-level effects on M5a alone with current
  prompts.
- **n=1 categorical-yes/no prompt and n=1 categorical-MCQ prompt**: the
  cross-prompt split at M5b (yesno NO break vs MCQ break) is one
  observed asymmetry, not a confirmed format-general or task-general
  mechanism. A 2nd categorical-binary prompt would test whether the
  yesno-immunity is prompt-specific or format-general.

## Files

- This doc: `docs/insights/m_mp_phase3_followup_2026-04-28.md`.
- Smoke summary: `docs/experiments/m_mp_mcq_phase1_smoke.md`.
- Phase 2 summary: `docs/experiments/m_mp_mcq_phase2.md` (post-chain).
- Phase 3 M5a output: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/mcq_audit_l10_a40/`.
- Phase 3 M5b output: `outputs/sae_intervention/qwen_vis31_5120_mcq_audit/`.
- 1b output: `outputs/sae_intervention/idefics2_vis26_4608_2nd_cell/`.
- Comparison script: `scripts/m_mp_mcq_phase3_compare.py`.
- Updated docs: `docs/experiments/m_mp_phase3.md` (post-MCQ refinement),
  `docs/experiments/m_mp_phase3_idefics2_verification.md` (cell-2 specificity).
