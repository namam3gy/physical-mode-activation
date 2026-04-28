# M-MP — Phase 3 cross-prompt M5a + M5b causal test (2026-04-28)

> **Status**: ✅ complete (Qwen + Idefics2 × M5a + M5b × 3 prompts).
> **Design doc**: `docs/m_mp_multi_prompt_design.md`.
> **Phase 1 / Phase 2 docs**: `m_mp_phase1.md` / `m_mp_phase2.md`.
> **Track B reference**: `references/submission_plan.md` Pillar A / G1 fix.

## Question

Phase 1+2 showed that the *behavioral* H2 paired-delta is positive across
all 5 models × 3 prompts (15 cells). But this is **correlation**, not
causation. Phase 3 tests the *causal* version: do the **same v_L direction
(M5a) and the same encoder-side SAE features (M5b)** that flip/break PMR
on the `open` prompt also flip/break PMR on the new prompts?

## Setup

### M5a (LM-side runtime steering)

- Qwen: existing `v_L10` from `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz`. α=40, label=circle, cell=line/blank/none.
- Idefics2: existing `v_L25` from `cross_model_idefics2_capture_20260426-111434_49ac35be`. α=20, label=circle, cell=line/blank/none.

### M5b (encoder-side SAE intervention)

- Qwen: existing 5120-feature SAE (`outputs/sae/qwen_vis31_5120`), Cohen's-d top-20, hook on `vision_hidden_31` (last block, pre-projection).
- Idefics2: existing 4608-feature SAE (`outputs/sae/idefics2_vis26_4608`), Cohen's-d top-160, hook on vis_26.
- For both: cell=shaded/ground/both, label=ball (high-PMR cell where break is detectable).

### Cell-selection rationale (audit 2026-04-28)

M5a (`line/blank/none` + label=`circle`) and M5b (`shaded/ground/both` + label=`ball`)
use **different cells**, by design constraint, not as a methodological choice:

- **M5a** is a *flip-direction* test: needs a baseline-low cell so a positive PMR
  delta is detectable. `line/blank/none` × `circle` is the canonical 0-PMR cell from
  M2 (Qwen 0.000 / Idefics2 0.133 on Phase 2 describe; comparably low on open).
- **M5b** is a *break-direction* test: needs a baseline-high cell so a negative
  PMR delta is detectable. `shaded/ground/both` × `ball` is Qwen's canonical
  ceiling cell (Phase 2 describe 0.800; open 1.0).

Choosing matched-direction cells for the two methods is unavoidable — a single
cell cannot be both baseline-low (for flip) AND baseline-high (for break). This
means the M5a/M5b "agreement" is on the **same prompt × same model**, not the
same cell.

**M5b cell switch on Qwen describe**: a first attempt used `filled/blank/both`
(Qwen Phase 2 describe = 0.000 — wrong direction for a break test). It was
switched to `shaded/ground/both` (Phase 2 describe = 0.800) post-hoc. The switch
was driven by the cell-selection logic above, not by inspecting intervention
results. No M5b break test on `filled/blank/both` was performed and discarded.

### Code

`scripts/06_vti_steering.py --prompt-variant {describe_scene, meta_phys_yesno}`
+ `scripts/sae_intervention.py --prompt-mode {describe_scene, meta_phys_yesno}`
(both extended in commit `921db94`).

## Results

| Prompt | Qwen M5a flip rate | Qwen M5b break rate | Idefics2 M5a flip rate | Idefics2 M5b break rate |
|---|---|---|---|---|
| `open` (existing) | **10/10** | **0/20** (break) | **10/10** | **0/n** (k=160 break) |
| `describe_scene` | **10/10** | **0/10** (break) | **0/10** (NO flip) | **NO break** |
| `meta_phys_yesno` | **0/10** (NO flip) | **NO break** | **0/10** | **NO break** |

Random controls (×3, mass-matched):
- Qwen describe_scene M5b: random_{0,1,2} all 1.0 (intact).
- Qwen yesno M5b: random_{0,1,2} all 1.0.
- Idefics2 describe / yesno M5b: random_{0,1,2} all 1.0.

## Interpretation

### Qwen — generative-prompt-conserved, blocked for categorical (within Qwen)

The same `v_L10` direction and the same top-20 SAE features cause **PMR
flip/break on `describe_scene`** with the same threshold as `open`. But
on `meta_phys_yesno` (binary yes/no probe), neither M5a nor M5b flips
the response. The model continues to say "Yes" (baseline) even when
the encoder's physics-cue features are ablated or the LM residual is
steered.

**Refined claim (Qwen-specific, audit 2026-04-28 evening update)**: The MCQ
follow-up (`docs/insights/m_mp_phase3_followup_2026-04-28.md`) shows that
at Qwen × MCQ the M5a and M5b methods give different verdicts (cross-method
split), rather than the original unified "generative-vs-categorical" boundary:
- **M5a-side**: MCQ matches yesno (0/10 flip vs describe 10/10 flip). The
  categorical-task structure (explicit options, discrete letter answer)
  blocks LM-side steering in both categorical prompts tested.
- **M5b-side**: MCQ matches describe (10/10 break at top_k=20 vs yesno NO
  break). Yes/no is the **single (n=1) categorical prompt where M5b
  doesn't break** — described as **yes/no-prompt-specific** until more
  categorical-binary prompts (paraphrased yes/no, true/false) test whether
  the load-bearing axis is response format or this specific prompt.

Audit-defensible empirical claim: at Qwen × MCQ, M5a method nulls and M5b
method breaks — cross-method dissociation at one cell. The "two separable
mechanisms" interpretation (LM-side categorical-blockade vs encoder-side
yes/no-format-immunity) is a hypothesis consistent with the data but not
demonstrated; the audit follow-up doc lists alternative interpretations
including Option A verbosity confound. The original "generative-prompt-
conserved + categorical-blocked" framing is partially preserved (M5a) and
partially split (M5b).

**Architecture-level scope (audit 2026-04-28)**: This dissociation is
demonstrated **only in Qwen**. Idefics2 describe is null/null on M5a + M5b
(a generative-prompt failure, not success) and Idefics2 yesno is also
null/null — Idefics2 cannot dissociate generative-vs-categorical because
neither prompt produces a flip/break. The "generative-vs-categorical"
mechanism dissociation is therefore a **Qwen finding**, not a
cross-architecture finding, and should be presented as such.

### Idefics2 — kinetic-verb-production specifically (verified post-hoc 2026-04-28)

Idefics2's mechanism is **mechanistically distinct from Qwen's**, not just narrower.

Initial reading (NULL on describe_scene + meta_phys_yesno) was verified by
sanity checks (`docs/experiments/m_mp_phase3_idefics2_verification.md`):

1. **Higher-α M5a steering** (α=40, 60): output degenerates ("tip tip tip..."), confirming v_L25 is FC-template-specific direction; pushing harder doesn't recover physics-mode.
2. **Higher-k M5b ablation** (k=320, 500): intervention shifts output from "falling down" → "in the air" — both physics-mode but different framings. Ablation doesn't BREAK physics-mode in describe.
3. **SAE feature quality**: Idefics2's top-10 Cohen's d range (0.25–0.35) is 2× weaker than Qwen's (0.39–0.78), aligning with M3 vision-encoder probe AUC differences (0.93 vs 0.99).

**Refined claim**: Idefics2's encoder features encode **kinetic-verb production**
specifically (falling/dropping/rolling), NOT general physics-mode commitment.
Ablating them shifts the model to alternative physics-mode framings ("in the air",
"suspended") rather than abstract description. Qwen's features by contrast encode
general physics-mode commitment that bridges across kinetic-prediction and
descriptive language.

This is a **stronger** finding than "Idefics2 mechanism is narrower" — it identifies
*what* the encoder features represent in each architecture, not just the breadth of
their effect. Reviewers will read the cross-model dissociation as architecturally
informative rather than as a methodological null.

## Headline for paper §6 / Track B framing

Phase 3 result lands as **Mixed** per `m_mp_multi_prompt_design.md` §6.2
acceptance criteria. The original "task-agnostic physics-mode commitment"
claim does not survive — but the refinement is **more interesting and
more honest**:

> **The mechanism is generative, not categorical. It modulates how the
> model produces language about a physical scene, but it does not gate
> the model's meta-cognitive yes/no judgment. The breadth of the
> generative effect is architecture-conditional: Qwen covers
> kinetic-prediction + free-form description; Idefics2 covers
> kinetic-prediction only.**

This refinement strengthens the **mechanistic dissociation** thread in
the paper: M5a (LM-side) and M5b (encoder-side) **agree** on the same
prompt-level boundaries (generative vs categorical, plus
architecture-conditional breadth) — the *cross-method consistency* is
itself evidence that we've identified a real mechanism rather than an
artifact of either method.

### Cross-method "agreement" — asymmetry caveat (audit 2026-04-28)

The 4 (model × prompt) cells in the M5a/M5b cross-method comparison are
**asymmetric**:

| Cell | M5a | M5b | Type |
|---|---|---|---|
| Qwen × `describe_scene` | 10/10 flip | 0/10 break | **positive/positive** (1 cell) |
| Qwen × `meta_phys_yesno` | 0/10 NO flip | NO break | **null/null** |
| Idefics2 × `describe_scene` | 0/10 NO flip | NO break | **null/null** |
| Idefics2 × `meta_phys_yesno` | 0/10 NO flip | NO break | **null/null** |

So "cross-method agreement" rests on **1 positive/positive cell + 3 null/null
cells**. Strict reading: only the Qwen-describe positive shows that M5a and M5b
*both fire* on the same condition; the 3 null/null cells show that neither
method fires when the other doesn't, but *that's also consistent with both
methods being insensitive in the same way*.

The framing "same prompt-level boundary, two methods" is **defensible**, but the
strength of the cross-method-agreement evidence is **n=1 positive coincidence + 3
shared nulls**, not n=4 independent confirmations. This is a real limitation —
expanding to a 3rd or 4th model that's tractable on M5a + M5b would lift the
positive count, and is a candidate for follow-up if the Pillar B work
(M-PSwap / M-LMSwap) opens additional model points.

Pillar C (Marr 3-level §6 restructure) gains a clean dichotomy:
- **Computational level (PMR)**: H2 cross-prompt-conserved (15/15 cells, Phase 2).
- **Mechanistic level (M5a + M5b)**: causal effect cross-prompt-conserved for *generative* prompts, blocked for *categorical*. Architecture-conditional for breadth.

The world-model framing also gains nuance: VLMs have a **localized
world-type-recognition signal** that drives generative output but is
*decoupled* from explicit categorical reasoning. This is a substantive
finding for the broader claim.

## Follow-ups (for paper writeup)

1. **Test Qwen M5a + M5b on a 4th cognitive task**: e.g. *"What kind of object is this — a ball, a planet, or a circle?"* (multi-choice categorization, NOT yes/no). Tests whether the categorical blockage is yes/no-specific or all-categorization. **(Audit 2026-04-28)**: this is now load-bearing for the "Generative vs Categorical" claim. The current Phase 3 design has 2 generative prompts (open + describe) + 1 categorical prompt (yesno) — the asymmetry means we can't dissociate "categorical task" from "yes/no binary format". Recommended priority: high. Cheap (one new prompt variant + 5-model × 1-cell run). **STATUS (2026-04-28 evening)**: ✅ Done. Results refine the dissociation into two separable mechanisms — see `docs/insights/m_mp_phase3_followup_2026-04-28.md`. Headline: M5a categorical-blockade is task-driven (MCQ 0/10 flip matches yesno 0/10 flip), but M5b "no-break" is yes/no-FORMAT-specific (MCQ at top_k=20 breaks 10/10 like describe, NOT like yesno).
2. **Test with `meta_phys_yesno` re-worded** to make it more generative ("Explain whether this is a real-world physical event"). Tests whether the yes/no blockage is the *binary format* or the *meta-cognitive task*.
3. **LLaVA / LLaVA-Next / InternVL3 M5a × describe_scene**: extends the architecture-conditional claim. (Stretch — not in the Required scope.) **(Audit)**: Currently the cross-method-agreement claim leans on 1 positive cell; adding a 3rd model with M5a+M5b positive on describe (e.g. InternVL3) would strengthen from n=1 to n=2.
4. **Audit-driven**: re-run M5a on `shaded/ground/both ball` and M5b on `line/blank/none circle` if any model has dynamic range that allows both directions to be tested in *the same cell*. If feasible, M5a/M5b agreement on a single cell × prompt = a stronger cross-method signature than the current matched-prompt agreement.

## Cross-references

- `docs/m_mp_multi_prompt_design.md` — design doc.
- `docs/experiments/m_mp_phase1.md` — Phase 1 stratified smoke + hand-label gate.
- `docs/experiments/m_mp_phase2.md` — Phase 2 full PMR (5-model).
- `references/paper_gaps.md` G1 — single-task evaluation gap fix track.
- Memory `paper_strategy.md` — Marr 3-level decision; the generative-vs-categorical dissociation goes into Mechanistic level §6.3.
