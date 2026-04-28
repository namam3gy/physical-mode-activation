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

### Qwen — task-agnostic for generative, blocked for categorical

The same `v_L10` direction and the same top-20 SAE features cause **PMR
flip/break on `describe_scene`** with the same threshold as `open`. But
on `meta_phys_yesno` (binary yes/no probe), neither M5a nor M5b flips
the response. The model continues to say "Yes" (baseline) even when
the encoder's physics-cue features are ablated or the LM residual is
steered.

**Refined claim**: Qwen's physics-mode commitment mechanism operates on
**generative language production** (kinetic prediction + descriptive
language), not on **meta-categorization** (yes/no judgment). The yes/no
decision goes through different LM circuitry that the encoder-side
features don't gate.

### Idefics2 — kinetic-prediction-specific (narrower than Qwen)

Idefics2's mechanism is **even narrower**: only `open` (kinetic prediction)
flips/breaks. `describe_scene` does NOT flip/break despite the same v_L
direction. The α=20 steering produces a different non-physics response
("The shape of the tip of the arrow is not shown"), confirming the
steering has *some* effect but doesn't push toward physics-mode under
descriptive prompting.

**Refined claim**: Idefics2's mechanism is **kinetic-prediction-mode-
specific**, not generic physics-mode-commitment. The CLIP-cluster
architecture (or the perceiver-resampler projector) couples the encoder
features more tightly to the kinetic-prediction prompt template.

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
boundaries (generative vs categorical, plus architecture-conditional
breadth) — the *cross-method consistency* is itself evidence that we've
identified a real mechanism rather than an artifact of either method.

Pillar C (Marr 3-level §6 restructure) gains a clean dichotomy:
- **Computational level (PMR)**: H2 cross-prompt-conserved (15/15 cells, Phase 2).
- **Mechanistic level (M5a + M5b)**: causal effect cross-prompt-conserved for *generative* prompts, blocked for *categorical*. Architecture-conditional for breadth.

The world-model framing also gains nuance: VLMs have a **localized
world-type-recognition signal** that drives generative output but is
*decoupled* from explicit categorical reasoning. This is a substantive
finding for the broader claim.

## Follow-ups (for paper writeup)

1. **Test Qwen M5a + M5b on a 4th cognitive task**: e.g. *"What kind of object is this — a ball, a planet, or a circle?"* (multi-choice categorization, NOT yes/no). Tests whether the categorical blockage is yes/no-specific or all-categorization.
2. **Test with `meta_phys_yesno` re-worded** to make it more generative ("Explain whether this is a real-world physical event"). Tests whether the yes/no blockage is the *binary format* or the *meta-cognitive task*.
3. **LLaVA / LLaVA-Next / InternVL3 M5a × describe_scene**: extends the architecture-conditional claim. (Stretch — not in the Required scope.)

## Cross-references

- `docs/m_mp_multi_prompt_design.md` — design doc.
- `docs/experiments/m_mp_phase1.md` — Phase 1 stratified smoke + hand-label gate.
- `docs/experiments/m_mp_phase2.md` — Phase 2 full PMR (5-model).
- `references/paper_gaps.md` G1 — single-task evaluation gap fix track.
- Memory `paper_strategy.md` — Marr 3-level decision; the generative-vs-categorical dissociation goes into Mechanistic level §6.3.
