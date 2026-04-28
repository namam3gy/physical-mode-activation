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
prompts, not just behavioral PMR correlation.

1. Extend `scripts/06_vti_steering.py` and `scripts/sae_intervention.py` with
   `--prompt-variant {open, describe_scene, meta_phys_yesno}` flag.
2. Re-run M5a steering on Qwen + LLaVA-Next + Idefics2 (3 models that flipped
   on `open`) using the new prompts. Same `v_L`, same α.
3. Re-run M5b SAE intervention on Qwen + Idefics2 + InternVL3 (3 models that
   broke PMR on `open`). Same SAE features, same k.
4. Tabulate cross-prompt M5a flip rates + cross-prompt M5b break-rates.

**Acceptance criteria**:

- **Strong**: same `v_L` flips PMR-equivalent on all 3 prompts × ≥ 3 models → mechanism is task-agnostic. Paper claim stands; multi-prompt becomes §6 confirmation.
- **Mixed**: same direction works on 2 of 3 prompts → claim refined to *"physics-mode commitment under prediction-loaded prompts"*.
- **Weak (pivot)**: only kinetic prediction fires → claim narrows to *"next-state-prediction-mode commitment"*. Multi-prompt becomes the *principled boundary* of the claim.

---

## 7. Implementation TODO

- [x] Add `describe_scene` and `meta_phys_yesno` to `prompts.py` (DONE 2026-04-28).
- [x] Update `PromptVariant` Literal in `config.py` (DONE 2026-04-28).
- [x] Create 5 multi-prompt configs (DONE 2026-04-28).
- [ ] Phase 1 smoke runs (5 models × ~5 min each).
- [ ] Phase 2 scorer extensions (`score_pmr_describe`, `score_pmr_meta_yesno`).
- [ ] Phase 2 full inference (5 models × ~30-45 min each).
- [ ] Phase 3 M5a / M5b cross-prompt scripts.
- [ ] Paper §6.1 multi-prompt PMR table.
- [ ] `notebooks/m_mp_multi_prompt.ipynb` reproduction notebook.

---

## 8. Cross-references

- `references/submission_plan.md` — Track B 20-week schedule (M-MP at weeks 1–3).
- `references/paper_gaps.md` — G1 (single-task) → fix mapping.
- `references/roadmap.md` — milestone overview row M-MP.
- `docs/scoring_rubric.md` — existing PMR scoring (extension target).
- Memory: `paper_strategy.md` — Marr-3-level decision (multi-prompt feeds Computational level).
