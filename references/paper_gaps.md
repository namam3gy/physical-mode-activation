# Paper Gaps — ICLR/NeurIPS-grade weakness audit

> **Role of this document.** Track each known paper-level weakness, what reviewers will flag, the concrete fix track, the acceptance criteria, dependencies, and current status. Updated whenever a fix completes or a new weakness surfaces. Pair with `references/submission_plan.md` (Track B execution plan) and `references/roadmap.md` (milestone status).

**Created**: 2026-04-28. **Owner**: namam3gy. **Status**: active (4 open gaps, all with fix tracks).

---

## Summary table

| ID | Weakness | Severity | Fix pillar | Status | Target close |
|---|---|---|---|---|---|
| G1 | Single-task evaluation (only "what will happen next?") | High | Pillar A (multi-prompt) | open | Week 3 |
| G2 | Sparse non-Qwen mechanistic confirmations | Medium | Pillar B (B3 — extra model) | partial | Week 8 |
| G3 | Idefics2 perceiver-resampler hypothesis is n=1 | High | Pillar B (B1 + B2 — controlled LoRA swaps) | open | Week 7 |
| G4 | 5-signatures framing weakness (correlated by construction) | Medium | Pillar C (Marr-3-level restructure) | open | Week 9 |

Severities: **High** = blocking ICLR-grade unless addressed; **Medium** = reviewer-noticeable, addressable via framing OR experiments; **Low** = nice-to-have polish.

---

## G1 — Single-task evaluation

### Weakness

The paper's behavioral and mechanistic claims (PMR / `v_L` / SAE physics-cue features / M5a steering / M5b SAE intervention) are all built on **one prompt family**: `"What will happen next?"` (kinetic next-state prediction). Even though this is a clean physics-mode probe, it leaves room for the alternative reading:

> *Qwen has a 'predict next state' shortcut, not a 'physics-mode' shortcut.*

Under this alternative, our `v_L10` direction is *kinetic-prediction direction*, our SAE features are *kinetic-prediction features*, and the "physics-mode commitment" framing is over-claimed.

### What reviewers will flag

- "Why should I believe this generalizes to other physics-loaded prompts (description, identity, causality)?"
- "Is the shortcut about *physics-mode commitment* or about *next-state prediction*?"
- "What about static physics tasks (stability, affordance, causal attribution)?"

### Fix track — Pillar A

**A1 (Required)** — Multi-prompt evaluation on existing 5-model M8a stim. Reuse all infrastructure (no new stim, no new model loading). Three prompts:
1. **Kinetic prediction** (current baseline): *"What will happen next?"*
2. **Free-form description**: *"Describe the scene."*
3. **Direct meta-categorization**: *"Is this a depiction of a real-world physical event?"* (yes/no answer, more constrained than prediction or description).

**Dropped from initial 5-prompt menu** (per user, 2026-04-28):
- Counting prompt (`"How many objects?"`) — not relevant to the physics-mode commitment thesis.
- Object-identity prompt (`"What is this object?"`) — already partially covered by M2 forced-choice.
- Michotte 2-frame causality — peripheral; was speculative `§4.4` plan; retired.

**A2 (Required)** — Re-run M5a runtime steering + M5b SAE intervention on the 3 prompts. Same `v_L` direction, same SAE features. Tabulate cross-prompt steering effect / SAE-intervention effect.

**A3 (Backlog, time permitting)** — Validate on external benchmarks. Candidates:
- **Physion** (Bear et al., NeurIPS 2021) — 8 physical scenarios, video.
- **CLEVRER** (Yi et al., ICLR 2020) — causal + counterfactual.
- **IntPhys 2** (Riochet et al.) — object permanence + cohesion + continuity.

External benchmarks are stretch — only pursue if Track B has buffer at week 13.

### Acceptance criteria

- **Strong (best case)**: same `v_L` direction flips PMR-equivalent behavior on all 3 prompts × multiple models → mechanism is task-agnostic physics-mode commitment. Paper claim stands as-is; multi-prompt becomes a §6 confirmation.
- **Mixed (acceptable)**: same direction works on 2 of 3 prompts → paper claim refined to "physics-mode commitment under prediction-loaded prompts" — narrower but defensible.
- **Weak (pivot)**: only kinetic-prediction prompt fires → claim narrowed to "next-state-prediction-mode commitment". Multi-prompt then becomes the *principled boundary* of the claim. Still publishable, but framing weakens; world-model angle survives because next-state prediction is itself a core world-model primitive.

### Dependencies

- M8a stim (✅ done), 5-model loaders (✅ done), M5a steering infra (✅ done), M5b SAE feature ranking (✅ done).
- New: 2 multi-prompt config files (`configs/multi_prompt_*.py`), 1 multi-prompt run script.

### Current status

Open. Multi-prompt config design is the first task of week 1 (per `submission_plan.md` schedule).

---

## G2 — Sparse non-Qwen mechanistic confirmations

### Weakness

The full mechanistic chain (encoder → SIP-patching → MLP-knockout → SAE-intervention) is **Qwen-only**. Cross-model expansion has gaps:

- **LLaVA-1.5**: M5b SIP+patching ✅ (L20 lock-in confirmed). Per-head knockout / SAE intervention NULL (encoder-side).
- **LLaVA-Next**: M5a runtime steering ✅ (10/10 LM-side flip). M5b SAE intervention NULL (encoder-side).
- **Idefics2**: M5a runtime steering ✅ (10/10 at L25). M5b SAE intervention ✅ (k=160 break).
- **InternVL3**: M5b SAE intervention ✅ (k=160 break). Other tests blocked by baseline saturation.

The story "encoder-vs-LM dissociation" is supported but rests on partial data per model. Reviewers will ask why we didn't run the *full chain* on at least 3 models.

### What reviewers will flag

- "You ran SIP+patching only on Qwen and LLaVA-1.5 — why not on Idefics2 + InternVL3?"
- "Per-head knockout was Qwen-only — does the 'L9 MLP construction, L10 attention readout' picture replicate elsewhere?"
- "The 5-model ladder shows correlation between encoder probe AUC and SAE intervention effect — but you only have full mechanistic data on 1 model."

### Fix track — Pillar B (B3)

**B3 (Stretch)** — Add **1 additional non-Qwen model** with full M5a + M5b coverage. Candidates:
- **Pixtral 12B** (Mistral AI) — different architecture, large.
- **InternVL3.5** (newer release) — incremental over InternVL3.
- **Phi-3.5-Vision** — different scale family.

The minimal coverage: M5a runtime steering + M5b SAE intervention on the new model. Full SIP+patching on saturated baselines is often blocked by the same `n_neg=0` issue Idefics2/InternVL3 hit; that's an honest limitation, not a Track B requirement.

### Acceptance criteria

- ≥ 6 models in the M5a + M5b table (currently 5 with partial coverage).
- ≥ 3 models with full encoder-side SAE intervention positive results (currently Qwen + Idefics2 + InternVL3).

### Dependencies

- B3 is independent of B1/B2; can run in parallel on either GPU.
- Requires ~3 days of model loader integration + 2 days of run + analysis.

### Current status

Partial. 5-model M5b SAE intervention complete (3 of 5 break, 2 LLaVA NULL). B3 = adding a 6th model. **Lower priority than B1/B2** since the encoder-vs-LM dissociation is already supported by 5 models.

---

## G3 — Idefics2 perceiver-resampler hypothesis is n=1

### Weakness

The paper's most architecturally specific claim is:

> *Idefics2's perceiver-resampler removes pixel-space gradient routability, not LM-side information* — supported by M4 LM AUC 0.995 + M5a runtime steering 10/10 + §4.6 pixel-flip 0/9 layers (forward pathway works, inverse pathway blocked).

This is a **clean dissociation**, but it rests on *one* perceiver-architecture model (Idefics2). The 5-model design varies encoder (SigLIP, SigLIP-SO400M, CLIP, CLIP+AnyRes, InternViT) + projector (MLP, AnyRes, perceiver-resampler) + LM family **simultaneously**, so we cannot causally isolate the projector.

### What reviewers will flag

- "How do you rule out that the encoder family or LM family is the cause, given Idefics2 differs from MLP-projector models on 3 architectural axes simultaneously?"
- "What's your evidence that perceiver-resampler is the *cause* rather than a confound?"
- "Have you done a controlled comparison?"

### Fix track — Pillar B (B1 + B2)

**B1 (Required)** — Controlled **projector-swap LoRA on Idefics2**:
1. Hold encoder (SigLIP-SO400M) and LM (Mistral-7B) fixed.
2. Replace the perceiver-resampler with a learned MLP projector (matching LLaVA's 2-layer GELU-MLP design).
3. Train on a small mixture of Idefics2's pretraining data + LLaVA-style instruction-tune (LoRA rank-32, ~10K steps, ~5 GPU-hr).
4. Evaluate the new variant (Idefics2-MLP) on §4.6 pixel-flip + M5b SAE intervention.

**Predicted result**: if perceiver-resampler is causally responsible for blocking pixel-space routability, **Idefics2-MLP should now flip in §4.6** like LLaVA-Next does. If Idefics2-MLP still doesn't flip, the cause is elsewhere (encoder or LM).

**Feasibility risk**: perceiver-resampler is non-trivially integrated into Idefics2's forward pass (cross-attention with learned queries). LoRA-style replacement may require structural surgery, not just LoRA adapters. **1-day feasibility spike at week 4** before committing.

**B2 (Required)** — Controlled **LM-only-swap LoRA**:
1. Pair CLIP-ViT-L (the LLaVA-1.5 encoder) with Vicuna-7B (LLaVA-1.5's LM) and Mistral-7B (LLaVA-Next's LM).
2. Train both LoRA variants on identical instruction-tune mixture (~5 GPU-hr each).
3. Re-run M5a + M5b on both. Tests whether the LM family alone can flip a model from "M5b NULL" (LLaVA-1.5) to "M5b break" (Mistral-flavored variant).

**Predicted result**: if encoder-vs-LM dissociation is real, the LM-only swap should NOT flip M5b NULL → break (because the encoder is the bottleneck). If it does flip, then LM family is doing more than we thought.

### Acceptance criteria

- **Strong** (best case): B1 produces Idefics2-MLP that flips on §4.6 → causal projector claim. B2 confirms LM-only swap does not change M5b status → encoder-vs-LM dissociation isolated.
- **Mixed (acceptable)**: B1 architecturally infeasible → fall back to literature + theoretical argument (perceiver-resampler is well-studied; cite Flamingo, BLIP-2 architecture papers). B2 still runs and supports LM-side claim.
- **Weak (pivot)**: B1 + B2 both fail or produce unexpected results → narrate honestly as architectural confound, scope claim down.

### Dependencies

- B1 requires Idefics2 source-code modification (perceiver-resampler replacement). Need to verify in `transformers` library code (use Context7).
- B2 is straightforward LoRA training (well-trodden).
- Both can run in parallel on GPU 0 (B1) and GPU 1 (B2).

### Current status

Open. Highest-impact / highest-risk Track B work item. Spike feasibility at week 4 of `submission_plan.md` schedule.

---

## G4 — 5-signatures framing weakness

### Weakness

The paper currently presents 5 measurements in a parallel-bullets style:
1. Behavioral PMR (M2)
2. M3 vision-encoder probe AUC
3. M4 LM logit-lens probe AUC
4. M5a runtime steering effect
5. M5b SAE intervention effect

These are **correlated by construction** — all measure "is the model in physics-mode at layer L?" using different methods. Reviewers will say:

> *This is one finding measured 5 ways, not 5 independent pieces of evidence. Why should I count this as 5x evidence?*

### What reviewers will flag

- "These 5 'signatures' are tautological. They all probe the same internal state; cross-method agreement is mathematically expected."
- "Where are the *predictions* that one method makes that another contradicts?"
- "Why is this 5 bullets and not 1 + 4 confirmations?"

### Fix track — Pillar C (C1)

**C1 (Required)** — Restructure §6 of `docs/paper/draft_v1.md` into **3 Marr levels** (per user decision 2026-04-28). 5 measurements collapse to 3 logically distinct evidential categories:

| Marr level | Question | Signatures | Method-specific failure mode | Control |
|---|---|---|---|---|
| **Computational** | Does the model behaviorally enter physics-mode? | PMR (M2) | Prompt-wording bias | KO/JA labels (§4.3); open vs FC; multi-prompt (Pillar A) |
| **Representational** | Does the model encode physics-mode? | M3 vision probe + M4 LM probe | Low-level visual stats / token frequency | Random direction baseline; label-free (M4b/M4c); token-balanced |
| **Mechanistic** | Does that encoding cause behavior? | M5a steering + M5b SAE | Norm-scaling / feature noise | Mass-matched random (M5a); 3 random feature sets (M5b) |

**Reviewer-facing claim**: "We have **3 levels of evidence** (Marr's hierarchy), each with multiple measurements that protect against the level's specific failure modes. Cross-level convergence is *not* tautological because each level addresses a different reading of the data; cross-method redundancy *within* each level provides robustness against method-specific confounds."

**Pruning option (deferred)**: User chose appendix-relegation over pruning (2026-04-28). All 5 measurements stay in main paper organized by Marr level. **If late-stage narrative demands compression**, drop M3 + M4 to supplementary (kept as confirmations of M5a + M5b), leaving 3 main signatures (PMR + M5a + M5b) = minimum viable Marr 3-level instantiation.

### Acceptance criteria

- §6 of paper draft restructured into §6.1 (Computational) / §6.2 (Representational) / §6.3 (Mechanistic) / §6.4 (Cross-level triangulation).
- Each Marr level has explicit failure-mode subsection citing controls.
- §1 (Introduction) frames the evidence structure as "3 levels of evidence" (not "5 signatures").

### Dependencies

- Paper-side only. No new experiments.
- Multi-prompt (Pillar A) results feed into §6.1 (Computational level).
- Controlled architectural counterfactuals (Pillar B) feed into §6.3 (Mechanistic level).

### Current status

Open. Restructure scheduled for week 9 of `submission_plan.md` (after Pillar A and B experiments complete, so §6 numbers are final).

---

## Cross-references

- `references/submission_plan.md` — Track B execution schedule with pillar mapping.
- `references/roadmap.md` — milestone status, hypothesis scorecard, additional ideas.
- `docs/paper/draft_v1.md` — current paper draft to be restructured per G4.
- Memory: `paper_strategy.md` — Track B decision, world-model framing, Marr-3-level decision.

---

## Change log

| Date | Change |
|---|---|
| 2026-04-28 | Document created. 4 gaps identified (G1–G4) with severity, fix pillar, acceptance criteria, dependencies, status. |
