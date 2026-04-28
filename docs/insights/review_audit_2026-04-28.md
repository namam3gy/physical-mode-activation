# Comprehensive review audit — methodology + claims (2026-04-28)

> **Status**: ✅ audit complete (Phase 1+2+3 multi-prompt + post-projection SAE +
> M-PSwap feasibility + cross-model M5b — all 4 most-recent additions reviewed).
> **Trigger**: User request "지금까지 실험과 인사이트, 연구내용도 잘못된게 없는지
> 한번 전체적으로 검토해" (comprehensive review for methodological errors,
> inconsistencies, contradictions across recent work).
> **Companion docs**: `data_audit_2026-04-28.md` (M2 data audit, 2026-04-28
> earlier in day) + `scorer_regression_audit_2026-04-28.md` (scorer v1→v2 audit).
> This audit covers the **post-Track-B-decision additions** (M-MP, M-PSwap,
> Idefics2 verification).

## TL;DR

The Track-B-era findings (Phase 1+2+3 multi-prompt, post-projection SAE,
M-PSwap feasibility) **survive audit with minor corrections**:

- ✅ All 15 H2 cells in Phase 2 still positive after the post-Phase-3 lexicon
  expansion (max delta ≤ +0.006 on label-averaged PMR; max +0.033 on a single
  cell). Headline "5-model × 3-prompt H2 conserved" stands.
- ✅ Idefics2 NULL on M5a + M5b describe is **robust, not under-perturbation**.
  Top-k SAE ablation on the *single tested cell* (`shaded/ground/both` ball)
  produces a categorical 30/30 framing shift (kinetic verbs → "in the air");
  random control retains 10/10 kinetic verbs. **Single-cell scope** — the
  quantification is 10 stim × 3 k values of one cell with 1 unique
  intervention text, not 30 independent stimuli. Architecture-level
  "kinetic-verb production specifically" claim requires a 2nd-cell test
  (e.g. `textured/ground/cast_shadow` ball).
- ⚠️ **Cross-method M5a/M5b "agreement" claim is asymmetric**: 1 positive/positive
  cell (Qwen-describe) + 3 null/null cells, not 4 independent positives. Doc
  language tightened.
- ⚠️ **Generative-vs-Categorical claim is Qwen-specific, not architecture-level**:
  the dissociation only fits Qwen (open + describe positive, yesno null).
  Idefics2 describe and yesno are BOTH null/null on M5a + M5b — Idefics2 can't
  test gen-vs-cat. Plus, the design has 2 generative prompts + 1 categorical
  prompt, so even within Qwen we can't dissociate "categorical task" from
  "yes/no binary format". Follow-up #1 (multi-choice categorization) is now
  load-bearing for the dissociation claim.
- ⚠️ **Cell selection in Phase 3** is constraint-driven, not free choice (M5a
  needs baseline-low, M5b needs baseline-high — different cells unavoidable).
  Documented in `m_mp_phase3.md` + `m_mp_phase3_idefics2_verification.md`.
- ⚠️ **Roadmap language tightened**: was "5-model M-MP coverage", now "5-model
  behavioral coverage + 2-model causal coverage" — Phase 3 does not extend to
  3 of the 5 models.

No claims were **withdrawn**; several were qualified.

## What I checked

### 1. Phase 2 lexicon-expansion re-score (BLOCKING per advisor)

**Concern**: `score_describe` was extended mid-Phase-3 with stems `impact`,
`about to (fall|hit|impact|land|drop|bounce)`, `is about to`, `going to
(fall|hit)`. If the expansion changed the Phase 2 ladder, the "all 15 cells
positive" headline could be invalidated.

**Test**: Re-ran `scripts/m_mp_summarize.py paths` with all 5 model dirs after
the lexicon expansion.

**Result**: Microscopic deltas, all 15 cells preserved positive.

| Quantity | Pre-expansion | Post-expansion | Δ |
|---|---|---|---|
| Qwen describe label-averaged | 0.459 | 0.465 | +0.006 |
| Idefics2 describe label-averaged | 0.595 | 0.597 | +0.002 |
| Idefics2 textured/blank/none describe (single cell) | 0.000 | 0.033 | +0.033 |
| H2 ball−circle range across 15 cells | +0.006 to +0.344 | +0.006 to +0.344 | unchanged |
| All 15 cells positive | 15/15 | 15/15 | unchanged |
| Per-(model, prompt) ladder ordering | preserved | preserved | unchanged |

`docs/experiments/m_mp_phase2.md` updated with re-scored numbers + audit
footnote.

### 2. Idefics2 framing-shift quantification (BLIND-SPOT per advisor)

**Concern**: `m_mp_phase3_idefics2_verification.md` claimed "top-k SAE ablation
shifts framing from kinetic to suspended" but only showed sample outputs at
each k. Anecdotal, not quantified.

**Test**: Loaded `outputs/sae_intervention/phase3_idefics2_describe_higher_k/
results.csv` (40 rows = 10 stim × 4 conditions: top_k={160,320,500} +
random_0). Counted kinetic-verb tokens (`fall|drop`) vs suspended-frame tokens
(`in the air|suspended|hover`) per row.

**Result**: Categorical and complete shift (table reproduced in
`m_mp_phase3_idefics2_verification.md` Sanity-Check-2 section).

| Group | n | `fall\|drop` | `in the air\|suspended\|hover` |
|---|---|---|---|
| Baseline (k=0) | 10 | **10/10** | 0/10 |
| Intervention k=160 | 10 | 0/10 | **10/10** |
| Intervention k=320 | 10 | 0/10 | **10/10** |
| Intervention k=500 | 10 | 0/10 | **10/10** |
| Random control (k≈300, mass-matched) | 10 | **10/10** | 0/10 |

The framing shift is **30/30 categorical** under top-k SAE ablation, and
random retains kinetic 10/10 — the shift is not a generic "suspended-frame
attractor" the model defaults to under any ablation. Caveat: low output
diversity (baseline = 2 unique strings, intervention = 1 unique string), so
the test covers only one cell's lexical surface; generalization to other
cells / suspended frames is untested.

### 3. Cross-method M5a/M5b "agreement" asymmetry (BLOCKING per advisor)

**Concern**: Phase 3 framing presented M5a + M5b as **agreeing** on the same
prompt boundaries as evidence the mechanism is real. But:

| Cell | M5a | M5b | Type |
|---|---|---|---|
| Qwen × `describe_scene` | 10/10 flip | 0/10 break | **positive/positive** |
| Qwen × `meta_phys_yesno` | 0/10 NO flip | NO break | null/null |
| Idefics2 × `describe_scene` | 0/10 NO flip | NO break | null/null |
| Idefics2 × `meta_phys_yesno` | 0/10 NO flip | NO break | null/null |

The "agreement" rests on **1 positive coincidence + 3 shared nulls**, not 4
independent confirmations. 3 null/null cells are also consistent with both
methods being insensitive in the same way (e.g., no method moves the
yes/no decision).

**Action**: Doc language tightened. Added explicit asymmetry table to
`m_mp_phase3.md` + roadmap M-MP row + Track B priorities row. Follow-up #3 +
#4 added: extending to a 3rd model with M5a+M5b positive on describe (e.g.
InternVL3) would strengthen from n=1 to n=2.

### 4. Generative-vs-Categorical asymmetric design (BLIND-SPOT)

**Concern**: The "Generative vs Categorical" claim rests on:
- 2 generative prompts (`open`, `describe_scene`) — both flip/break in Qwen.
- 1 categorical prompt (`meta_phys_yesno`) — neither flips/breaks.

This conflates 2 hypotheses:
- (H_task) The categorical TASK (meta-categorization) routes through different
  LM circuitry.
- (H_format) The yes/no BINARY FORMAT routes through different LM circuitry.

The current data can't dissociate them.

**Action**: Phase 3 follow-up #1 ("4th cognitive task: multi-choice
categorization, NOT yes/no") now flagged as **load-bearing for the Generative
vs Categorical claim**, not just exploratory. Cheap (one new prompt variant +
1-cell × 2-model run), high payoff. Follow-up #2 ("re-worded yesno as
generative") tests the inverse direction.

### 5. Cell-selection rationale (BLIND-SPOT)

**Concern**: M5a uses `line/blank/none × circle` (Qwen 0.000 baseline) and M5b
uses `shaded/ground/both × ball` (Qwen 0.800 baseline) — different cells.
Reader could ask "why?"

**Resolution**: Methodologically necessary, not a free choice:

- M5a is a *flip-direction* test → needs baseline-low cell for positive PMR
  delta to be detectable.
- M5b is a *break-direction* test → needs baseline-high cell for negative PMR
  delta to be detectable.

A single cell can't be both. **Note**: an early M5b attempt used
`filled/blank/both` (Qwen Phase 2 describe = 0.000) before recognizing it
was wrong-direction; switched to `shaded/ground/both` (= 0.800). The switch
was driven by the cell-selection logic above, not by inspecting intervention
results. No M5b break test on `filled/blank/both` was performed-and-discarded.

**Action**: Documented in `m_mp_phase3.md` Setup / Cell-selection rationale.

### 6. Roadmap "5-model M-MP coverage" claim (BLIND-SPOT)

**Concern**: Roadmap said "5-model M-MP" without distinguishing behavioral
(5-model) from causal (2-model — Qwen + Idefics2). Reader skimming the
roadmap could conclude Phase 3 was 5-model.

**Action**: Roadmap M-MP row rewritten:
- "Behavioral coverage = 5-model × 3-prompt"
- "Causal coverage = 2-model (Qwen + Idefics2) × 2 new prompts (`describe_scene`
  + `meta_phys_yesno`) on M5a + M5b"
- Headlines split into "Behavioral headline (5-model)" + "Causal headline
  (2-model, refined post-verification)".

### 7. Velocity / confirmation-bias risk (BLIND-SPOT)

**Concern**: Phase 1+2+3 + post-projection SAE + M-PSwap feasibility +
Idefics2 verification all happened in a single 2026-04-28 push. Confirmation
bias risk: every step ratifies the previous step.

**Action**: This audit is the deliberate slow-down. Findings:
- Lexicon re-score (#1): no confirmation bias — could have invalidated 15
  cells; didn't.
- Framing-shift quantification (#2): no confirmation bias — could have shown
  random-control also shifts; didn't.
- Cross-method asymmetry (#3): explicit qualification added.
- Generative-vs-categorical (#4): explicit follow-up flagged as load-bearing.

**Process improvement**: schedule a 24-48h cooling-off + audit pass before
locking any major framing in the paper draft (Pillar C / week 9-11).

## What I did NOT find

- No M5b numbers I could re-check showed errors (top-k=20 break / random
  controls / Wilson CIs all consistent with cited sources).
- No M3/M4/§4.6/§4.8 results contradicted the M-MP findings.
- No memory entries are stale (paper_strategy.md updated after Phase 3 with
  refined claims).
- No commits were pushed without context (each commit message references the
  insight doc + run dirs).
- M5b post-projection SAE "k=20 same threshold as pre-projection" is
  consistent with the merger module's role (LM-space refinement, not
  feature-distribution change). Sanity: pre-proj 5120-feature SAE, k=20 =
  0.4% of features; post-proj 14336-feature SAE, k=20 = 0.14% of features —
  both extremely sparse, consistent with the same handful of physics-cue
  features being the bottleneck.
- M-PSwap identity-bypass garbage output ("tip tip tip...") is consistent
  with the architecture analysis: post-modality_projection raw context is
  out-of-distribution for the LM's expected 64-token cross-attentional
  output.

## Action items

| # | Item | Status | Doc |
|---|---|---|---|
| 1 | Re-score Phase 2 with updated lexicon | ✅ done | `m_mp_phase2.md` |
| 2 | Quantify Idefics2 framing shift | ✅ done | `m_mp_phase3_idefics2_verification.md` |
| 3 | Document cross-method asymmetry | ✅ done | `m_mp_phase3.md` + roadmap |
| 4 | Document generative-vs-categorical design asymmetry | ✅ done | `m_mp_phase3.md` follow-ups |
| 5 | Document cell-selection rationale | ✅ done | `m_mp_phase3.md` |
| 6 | Tighten roadmap behavioral-vs-causal coverage | ✅ done | `roadmap.md` M-MP row + Track B table |
| 7 | Velocity / confirmation-bias process note | ✅ done (this doc) | `review_audit_2026-04-28.md` |
| 8 | (FOLLOW-UP, NOT BLOCKING) Run 4th cognitive task (multi-choice categorization, NOT yes/no) | open — load-bearing for generative-vs-categorical | scheduled with Pillar B |

## Recommendation for paper draft (week 9-11 Pillar C)

The Phase 3 finding is **paper-defensible** with the following framing:

1. **Computational level (Marr §6.1)**: H2 cross-prompt-conserved at 5-model ×
   3-prompt = 15 cells (behavioral evidence; n=21,600 inferences total).

2. **Mechanistic level (Marr §6.3)**: Cross-method M5a + M5b agreement on
   prompt-level boundaries, qualified as:
   - 1 positive coincidence (Qwen × describe: M5a 10/10 + M5b 0/10) +
     3 shared nulls (Qwen-yesno, Idefics2-describe, Idefics2-yesno).
   - **Qwen-specific** generative-vs-categorical dissociation; Idefics2
     cannot test gen-vs-cat from current data (both prompts null).
   - Idefics2 single-cell finding on `shaded/ground/both` ball: top-k SAE
     ablation shifts framing kinetic→suspended (30/30 quantification covers
     10 stim × 3 k values × 1 cell × 1 unique intervention text, not 30
     independent stimuli). Architecture-level claim about Idefics2 encoder
     features is **pending** a 2nd-cell test.
   - Generative-vs-Categorical interpretation in Qwen **pending** the
     multi-choice categorization follow-up (Pillar B, week 4-7).

3. **Limitations (paper §10)**:
   - Phase 3 causal coverage is 2-model (Qwen + Idefics2), not 5-model.
   - Cross-method agreement evidence is asymmetric (1 positive coincidence +
     3 shared nulls, not 4 independent positives).
   - Generative-vs-Categorical dissociation is Qwen-specific, not
     architecture-level; cross-architecture extension requires a 3rd model
     where M5a + M5b both fire on describe.
   - Idefics2 framing-shift quantification is single-cell; 2nd-cell test
     required for architecture-level claim about encoder feature content.
   - Cell selection is constraint-driven (M5a baseline-low + M5b
     baseline-high → different cells).
   - Generative-vs-Categorical design has 2 generative + 1 categorical prompt
     → can't dissociate task from format until follow-up #1 runs.

This framing **strengthens** the paper rather than weakening it: a 5-model
behavioral + 2-model causal pattern with explicit limitations reads as more
honest and reviewer-defensible than an over-claim.

## Files

- This doc: `docs/insights/review_audit_2026-04-28.md`
- Companion: `docs/insights/data_audit_2026-04-28.md`,
  `docs/insights/scorer_regression_audit_2026-04-28.md`
- Updated: `docs/experiments/m_mp_phase2.md`, `m_mp_phase3.md`,
  `m_mp_phase3_idefics2_verification.md`, `references/roadmap.md`
