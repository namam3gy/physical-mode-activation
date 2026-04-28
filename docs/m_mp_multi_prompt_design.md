# M-MP — Multi-prompt cross-task generalization (design doc)

> **Status**: design (week 1 of `references/submission_plan.md` Track B).
> **Owner**: namam3gy. **Created**: 2026-04-28.
> **Pillar**: A (multi-task generalization). **Gap fixed**: G1 (single-task evaluation).
> See `references/paper_gaps.md` for the full gap context.

---

## 1. Motivation

The paper currently localizes a "physics-mode commitment" mechanism (Qwen `v_L10`
direction, encoder-side SAE physics-cue features, M5a runtime steering, M5b
SAE intervention) using **only one prompt family**: kinetic next-state
prediction (*"What will happen next?"*).

This leaves the alternative reading open:

> *Qwen's mechanism is a 'predict next state' shortcut, not a 'physics-mode' shortcut.*

Track B Pillar A tests **task-agnosticism** by varying the cognitive question
asked of the model while holding stim + label constant. If the same `v_L10`
direction / SAE features fire across multiple prompts, the mechanism is
genuinely physics-mode commitment.

---

## 2. Three prompts

Same stim, same label, different cognitive task.

| ID | Variant name | Template | Cognitive task |
|---|---|---|---|
| P1 | `open` | *"The image shows a {label}. Describe what will happen to the {label} in the next moment, in one short sentence."* | **Kinetic prediction** — current paper baseline. |
| P2 | `describe_scene` | *"The image shows a {label}. Describe what you see in the image, in one short sentence."* | **Free-form description** — what does the model perceive, not predict. |
| P3 | `meta_phys_yesno` | *"The image shows a {label}. Is this a depiction of a real-world physical event (an object subject to gravity, mass, momentum, or other physical forces)? Answer with 'yes' or 'no', followed by a brief justification."* | **Direct meta-categorization** — most direct probe of physics-mode commitment. |

**Dropped from initial 5-prompt menu** (per user, 2026-04-28):
- Counting prompt (*"How many objects?"*) — not relevant to physics-mode commitment.
- Object-identity prompt (*"What is this object?"*) — already partially covered by M2 forced-choice.
- Michotte 2-frame causality — peripheral; was speculative `§4.4` plan; retired.

---

## 3. Stimulus selection

**M2 stim** (480 stim, 5-axis factorial: 4 obj × 3 bg × 4 cue × 1 event × 10
seeds, circle only). Reuse via `--stimulus-dir inputs/mvp_full_<id>` at run
time.

**Why M2 stim**: M5a + M5b were originally calibrated on M2 stim (or a
20-stim SIP subset of it). Using M2 stim makes the cross-prompt comparison
maximally apples-to-apples with existing v_L10 / SAE features.

**Phase 1 subset** (smoke run, ~80 stim): the 80 unsaturated baseline cells
that show clear PMR variance across (cue, label) on Qwen open. Tractable: ~10
min/model on H200.

**Phase 2 full** (480 stim × 3 labels × 3 prompts × 5 models = 21,600 inferences,
~3.5 hr total wall): only if Phase 1 confirms scorability + variance per prompt.

---

## 4. Scoring strategy

Each prompt requires its own PMR scorer. The existing `score_pmr` rule-based
scorer assumes kinetic-prediction style outputs.

| Prompt | Scoring approach | Status |
|---|---|---|
| P1 `open` | Existing PMR scorer (kinetic verbs, physics stems, abstract markers) | ✅ ready |
| P2 `describe_scene` | New PMR scorer extension: physics-mode terms in description (gravity, fall, mass, ball-like, sphere, momentum, abstract-shape markers, etc.) | **TODO** — extension required before Phase 2 |
| P3 `meta_phys_yesno` | First-token "yes"/"no" parsing (with stems "Yes" / "No" + justification) → `physics_mode = (first_token.lower() == "yes")` | ✅ tractable, simpler than P1/P2 |

**Phase 1 deliverable**: visual inspection of N=20 outputs per (model, prompt)
to validate that prompts produce parseable / differentiable outputs at all.
If the model outputs are degenerate (e.g., LLaVA's "A" bias on FC), pivot
to alternative prompt wording before Phase 2.

**Scorer-vs-hand-agreement gate (locked 2026-04-28, post-advisor)**: before
running Phase 2 at full 4320-inference scale, validate `score_pmr_describe`
on N=50 hand-labeled outputs per model. Required threshold: **scorer-vs-hand
agreement ≥ 0.85 (Cohen's κ ≥ 0.70)**. `describe_scene` outputs are
ambiguous (e.g., *"the ball is on the ground"* — physics-mode? is it
*describing rest* (static) or *physics-mode*? — needs adjudication rule).
If agreement < 0.85, the failure modes are:
1. Scorer too permissive → tighten lexicon (drop "ground", "below" matches without action verbs).
2. Scorer too strict → broaden physics-cue stems (add "rest", "settled", "support").
3. Prompt itself ambiguous → revise wording (e.g., add *"focus on what physical state the object is in"*).

Block Phase 2 full run until the gate passes. `meta_phys_yesno` does not
need this gate (yes/no parse is unambiguous); `open` already validated.

---

## 5. Configs

5 configs, one per model, all sharing `prompt_variants=("open", "describe_scene", "meta_phys_yesno")`:

- `configs/multi_prompt_qwen.py`
- `configs/multi_prompt_llava.py`
- `configs/multi_prompt_llava_next.py`
- `configs/multi_prompt_idefics2.py`
- `configs/multi_prompt_internvl3.py`

All use the same M2 5-axis factorial (480 stim × 3 labels), no captures
(LM activations from M2 already exist for M5a / M5b re-runs).

---

## 6. Two-phase plan

### Phase 1 — Sanity check (week 1)

**Goal**: verify each prompt produces parseable + differentiable outputs on
each of the 5 models. Lightweight, no scoring extensions yet.

1. Generate stim (already exist via M2 mvp_full).
2. Run smoke (`scripts/02_run_inference.py --config configs/multi_prompt_qwen.py --limit 20 --stimulus-dir inputs/mvp_full_<id>`).
3. Visually inspect 60 outputs (5 models × 3 prompts × 4 sample stim).
4. Decide: do P2 / P3 produce non-degenerate outputs? If yes → Phase 2. If no → revise prompt wording.

### Phase 2 — Full multi-prompt PMR (week 2)

**Goal**: produce per-(model, prompt, label) PMR table for paper §6.1.

1. Implement `score_pmr_describe` and `score_pmr_meta_yesno` extensions to `src/physical_mode/metrics/scoring.py`.
2. Run full configs (5 models × 4320 inferences each ≈ 3.5 hr).
3. Score each prompt with its respective scorer.
4. Tabulate: per-(model, prompt) PMR vs baseline `open` PMR. Check correlation.

### Phase 3 — Cross-prompt M5a / M5b re-runs (week 3)

**Goal**: test whether the same `v_L10` direction / SAE features fire across
prompts, not just behavioral PMR correlation. **Phase 3 is the only phase
that actually tests "task-agnostic mechanism"; Phases 1+2 alone produce
behavioral correlations which are a weaker form of evidence (could just be
model-specific response-style consistency).**

#### Minimum-viable Phase 3 scope (locked 2026-04-28, post-advisor)

To avoid Phase 3 ballooning past week 3 of `submission_plan.md`, the
required deliverable is restricted to 2 models × 2 interventions:

| Required (Required) | Stretch (Time-permitting) |
|---|---|
| **Qwen** M5a runtime steering on 3 prompts | LLaVA-Next M5a (LM-side) on 3 prompts |
| **Qwen** M5b SAE intervention on 3 prompts | LLaVA-1.5 M5a + M5b on 3 prompts |
| **Idefics2** M5a runtime steering on 3 prompts | InternVL3 M5b on 3 prompts |
| **Idefics2** M5b SAE intervention on 3 prompts | |

Why **Qwen + Idefics2**: these two have the cleanest M5a + M5b signals in
current 5-model results. Qwen is the canonical case (M5a 10/10 flip on
`open`; M5b k=20 break). Idefics2 is the cleanest cross-architecture
confirmation (M5a 10/10 flip at L25; M5b k=160 break) AND has the
forward/inverse-pathway dissociation that makes its multi-prompt result
maximally informative. Other models go to stretch column.

**Phase 3 minimum success criterion (Required)**: at least Qwen + Idefics2 ×
3 prompts × M5a + M5b complete with quantified flip-rate / break-rate
table. Stretch models add to §6 confirmation breadth but are not on the
critical path.

#### Implementation

1. Extend `scripts/06_vti_steering.py` and `scripts/sae_intervention.py` with
   `--prompt-variant {open, describe_scene, meta_phys_yesno}` flag.
2. Re-run M5a steering on **Qwen + Idefics2** (Required) using the new
   prompts. Same `v_L`, same α as `open`-tuned values (Qwen L10 α=40, Idefics2 L25 α=20).
3. Re-run M5b SAE intervention on **Qwen + Idefics2** (Required) using the
   new prompts. Same SAE features, same k (Qwen k=20, Idefics2 k=160).
4. Tabulate cross-prompt M5a flip rates + cross-prompt M5b break-rates.
5. (Stretch) extend to LLaVA-Next + LLaVA-1.5 + InternVL3 if week-3 bandwidth allows.

**Acceptance criteria**:

- **Strong (Required Qwen + Idefics2 success)**: same `v_L` / SAE features flip behavior-equivalent on all 3 prompts × 2 models → mechanism is task-agnostic in the canonical (Qwen) and cleanest cross-arch (Idefics2) case. Paper claim stands; multi-prompt becomes §6 confirmation. Stretch results extend breadth.
- **Mixed**: same direction works on 2 of 3 prompts (or only 1 of 2 required models) → claim refined to *"physics-mode commitment under prediction-loaded prompts (open + meta_phys_yesno)"* if `describe_scene` is the failure mode, or *"physics-mode commitment in saturated-encoder regime"* if Idefics2 fails but Qwen succeeds. Each refinement is publishable but narrower.
- **Weak (pivot)**: only kinetic prediction fires on both required models → claim narrows to *"next-state-prediction-mode commitment"*. Multi-prompt becomes the *principled boundary* of the claim. Still a publishable result; framing weakens but world-model angle survives because next-state prediction is itself a core world-model primitive.

---

## 7. Implementation TODO

Phase 0 — design (DONE 2026-04-28):
- [x] Add `describe_scene` and `meta_phys_yesno` to `prompts.py`.
- [x] Update `PromptVariant` Literal in `config.py`.
- [x] Create 5 multi-prompt configs.

Phase 1 — sanity check (week 1, in queue):
- [ ] Phase 1 smoke runs (5 models × ~5 min each on H200).
- [ ] Visual inspection of 60 outputs (5 × 3 × 4 stim sample).
- [ ] Decision gate: do P2 / P3 produce non-degenerate outputs?

Phase 2 — full multi-prompt PMR (week 2, blocked on Phase 1):
- [ ] Implement `score_pmr_describe` and `score_pmr_meta_yesno` extensions.
- [ ] **Scorer-vs-hand-agreement gate** for `describe_scene`: hand-label N=50 per model, require ≥0.85 agreement / Cohen's κ ≥ 0.70 before full run.
- [ ] Phase 2 full inference (5 models × 4320 inferences × ~30-45 min each).
- [ ] Per-(model, prompt) PMR table.

Phase 3 — cross-prompt M5a / M5b re-runs (week 3, blocked on Phase 2):
- [ ] Extend `06_vti_steering.py` and `sae_intervention.py` with `--prompt-variant` flag.
- [ ] **Required**: Qwen + Idefics2 × M5a + M5b × 3 prompts.
- [ ] (Stretch) LLaVA-Next + LLaVA-1.5 + InternVL3 if bandwidth allows.
- [ ] Cross-prompt flip / break-rate table.

Paper-side:
- [ ] Paper §6.1 multi-prompt PMR table.
- [ ] `notebooks/m_mp_multi_prompt.ipynb` reproduction notebook.

---

## 8. Cross-references

- `references/submission_plan.md` — Track B 20-week schedule (M-MP at weeks 1–3).
- `references/paper_gaps.md` — G1 (single-task) → fix mapping.
- `references/roadmap.md` — milestone overview row M-MP.
- `docs/scoring_rubric.md` — existing PMR scoring (extension target).
- Memory: `paper_strategy.md` — Marr-3-level decision (multi-prompt feeds Computational level).
