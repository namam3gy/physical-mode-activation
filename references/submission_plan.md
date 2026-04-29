# Submission Plan — Track B (ICLR/NeurIPS-grade)

> **Role of this document.** Captures the venue strategy, target deadline, Track B execution pillars, week-by-week schedule, resource budget, and pruning rules. Re-read at the start of every paper-track session. Pair with `references/paper_gaps.md` (weakness ↔ fix mapping) and `references/roadmap.md` (milestone status).

**Created**: 2026-04-28. **Owner**: namam3gy. **Status**: active.

---

## 1. Decision summary

### Venue calendar (verified 2026-04-28)

Sources: trybibby.com/conference-deadlines (cross-venue aggregator); iclr.cc/Conferences/2026/Dates (ICLR 2026 official).

| Venue | Deadline (UTC AoE) | Days from 2026-04-28 | Status |
|---|---|---|---|
| **NeurIPS 2026** | Abstract May 5, 2026 / Full May 7, 2026 | 9 days | **too tight — drop** |
| **AAAI 2027** | Abstract July 26 / Full Aug 2, 2026 | 96 days (~3 months) | possible mid-stage option |
| **ICLR 2027** (estimated) | Abstract Sep 19 / Full Sep 24, 2026 | 149 days (~5 months) | **primary target** |
| **NeurIPS 2027** (estimated) | ~May 16, 2027 | ~383 days (~12 months) | secondary / stretch |
| **EMNLP 2027** (estimated) | ARR May 29, 2027 | ~395 days | tertiary |
| **TMLR** | rolling | n/a | backup |

**Notes**:
- ICLR 2026 had Sep 24, 2025 deadline; ICLR 2027 estimate (Sep 24, 2026) follows the same pattern (per `iclr.cc` historical schedule; trybibby aggregator marks it "estimated"). Re-verify on `iclr.cc/Conferences/2027` once the official call posts (typical lead time: ~2 months before deadline → recheck mid-July 2026).
- NeurIPS 2026 deadline (May 7, 2026) is 9 days from today — Track B's controlled experiments (projector-swap LoRA, LM-only-swap LoRA) cannot fit. Squeezing into NeurIPS 2026 forces a Track A retreat.
- AAAI 2027 is a viable mid-stage backstop (Aug 2 deadline) if Track B's Pillar A + half of Pillar B complete by week 12.

### Decision

- **Track**: **B (ICLR/NeurIPS-grade, ~5 months execution)**, chosen 2026-04-28 over Track A (Findings/TMLR/Workshop).
- **Primary target**: **ICLR 2027** (Sep 24, 2026 estimated deadline, ~5 months from now). Realistic given Track B scope (multi-task + controlled projector-swap + LM-only-swap + Marr-3-level reframing).
- **Mid-stage backstop**: **AAAI 2027** (Aug 2, 2026 deadline) if Pillar A + half-Pillar B done by week 12.
- **Secondary target**: **NeurIPS 2027** (~May 16, 2027 deadline, ~12 months). Used as stretch / re-submission slot if ICLR 2027 misses.
- **Backup**: **TMLR (rolling)** — accepts longer mechanistic-interpretability papers, no fixed deadline; use if all top venues miss.
- **Why not NeurIPS 2026**: deadline May 7, 2026 (~9 days). Track B's controlled experiments cannot fit; squeezing forces a Track A retreat.
- **Why not EMNLP 2026 (June)**: borderline feasible (~6 weeks) but EMNLP audience prefers NLP-leaning framing; our paper's mechanistic-interp + world-model angle is a better fit for ICLR/NeurIPS reviewer pool.

**Resource available**: H200 × 2 GPUs (per user, 2026-04-28).

**Paper philosophy** (per user): **robust, not flashy.** Better to land a tight, defensible paper than a sprawling impact piece.

---

## 2. Three pillars of Track B

The paper currently has 4 weaknesses (per `references/paper_gaps.md`). Track B addresses them across three pillars; **all three must complete to be ICLR-grade.**

### Pillar A — Multi-task generalization (addresses gap #1)

**Claim**: physics-mode commitment mechanism is **task-agnostic**, not specific to "what will happen next?"

**Experiments**:
- **A1 (Required)** — Multi-prompt evaluation on existing 5-model M8a stim. Three prompts:
  1. *"What will happen next?"* — kinetic-prediction baseline (current).
  2. *"Describe the scene."* — free-form description.
  3. *"Is this a depiction of a real-world physical event?"* — direct meta-categorization probe.
  - **Drop**: counting prompt (not relevant to physics-mode commitment thesis), Michotte 2-frame causality (peripheral), object-identity prompt (already covered by M2 FC).
- **A2 (Stretch)** — Re-run M5a runtime steering + M5b SAE intervention on the 3 prompts. Show the same `v_L` direction / SAE features fire across prompts. If yes: mechanism is task-agnostic.
- **A3 (Backlog, time permitting)** — External benchmark validation (Physion or CLEVRER subset). Stretch only.

**Outputs**: paper §6.1 + §7 (multi-prompt cross-task table); supplementary §A (per-prompt PMR + steering effects).

**Resources**: H200 × 1 × ~1 week (A1+A2). External benchmarks (A3): +2–3 weeks if pursued.

### Pillar B — Controlled architectural counterfactuals (addresses gaps #2 + #3)

**Claim**: encoder-vs-LM mechanism dissociation in CLIP cluster (LLaVA family) is **causally** projector-driven, not n=1 anecdote.

**Experiments**:
- **B1 (Required)** — Controlled **projector-swap LoRA** on Idefics2: replace the perceiver-resampler with a learned MLP projector matching LLaVA's design, holding encoder + LM fixed. Then re-run §4.6 + M5b. **Feasibility check first**: 1-day spike to verify the swap is trainable on small data; if blocked (architectural incompatibility), pivot to B2.
  - **Status (2026-04-29)**: feasibility spike done — bypass-only fails (perceiver integral to forward pass), full LoRA training required. Infra built: `src/physical_mode/lora/{idefics2_mlp_resampler.py,load_swapped.py}` + `scripts/m_pswap_{train,smoke,regression_eval,post_training,diagnose_nan*,repro_nan_batch,discriminator}.py`. Smoke (50 step) PASS; **full training NaN-blocked at step 1000** (run `outputs/mpswap_run_20260429-033238/step1000`); diagnostic suite WIP. Until the NaN is resolved, the §4.6/M5b re-run on the swapped variant is blocked. See ROADMAP §6 changelog (2026-04-28/29 M-PSwap entry) and `paper_gaps.md` G3 for fallback.
- **B2 (Required)** — Controlled **LM-only-swap LoRA**: pair CLIP-ViT-L with Vicuna-7B vs Mistral-7B (matching LLaVA-1.5 vs LLaVA-Next-Mistral but holding everything else fixed). Re-run M5a + M5b. Tests whether LM family modulates encoder-side mechanism availability.
- **B3 (Stretch)** — Add 1 more non-Qwen model to the 5-model chain (e.g., **Pixtral**, **InternVL3.5**, **Phi-3.5-Vision**) to thicken the mechanistic confirmation set. Even partial M5a + M5b coverage (no full M3/M4) is acceptable.

**Outputs**: paper §6.3 (causal projector claim with controlled comparison); supplementary §B (LoRA training details).

**Resources**: H200 × 2 × 1–2 weeks per LoRA (B1, B2 in parallel on the two GPUs). Total 1.5–2.5 weeks.

### Pillar C — World-model framing & Marr-3-level restructure (addresses gap #4)

**Claim**: paper localizes a **production-VLM world-model commitment mechanism**, framed at three levels of evidence (Marr).

**Restructure work** (no new experiments — paper-side only):

- **C1 (Required)** — Restructure §6 of `docs/paper/draft_v1.md` into 3 Marr levels:
  - **§6.1 Computational level** — PMR threshold, prompt-bias controls (KO/JA, open vs FC).
  - **§6.2 Representational level** — M3 vision-encoder probe + M4 LM logit-lens, with low-level/token-frequency confound controls.
  - **§6.3 Mechanistic level** — M5a runtime steering + M5b SAE intervention, with norm/feature-noise controls.
  - **§6.4 Cross-level triangulation** — same layer L, same direction `v_L`, same SAE features; convergence table.
- **C2 (Required)** — Rewrite §1 (Introduction) to lead with the **world-model framing**: production VLMs implicitly act as world models in embodied/robotics applications (RT-2, OpenVLA, V-JEPA), and the *first step* of any world model is recognizing "what kind of world am I in?" (physical vs abstract). We localize *when, where, and how* this commitment fires in production VLMs.

  **§1.0 operational definition (drafted 2026-04-28, post-advisor; ~150 words)**: world-model framing must be operationally defined *before* §6 results are written, otherwise reviewers will read it as retrofitted narrative. Working definition for §1.0:

  > *We use **world-model commitment** to refer to the early-pipeline computation by which a vision-language model categorizes its visual input as either (i) a depiction of a real-world physical event with implicit dynamics (mass, gravity, momentum, contact), or (ii) an abstract symbol-system input where physical reasoning is inappropriate (geometry, OCR, mathematical notation). This binary commitment is required for any downstream computation that must condition on world-type — including next-state prediction, spatial-physical reasoning, causal attribution, and embodied action selection. We do not claim production VLMs implement world models in the strict sense of Ha & Schmidhuber (2018); rather, we identify a localizable mechanism that performs **world-type-recognition**, which is a necessary precursor to any explicit world-model architecture.*

  Validate this paragraph fits the current evidence base before writing §6. If we cannot defend it from current data alone, the framing is retrofitting and should be softened (drop "world-model" → "physical-mode commitment").
- **C3 (Required)** — Rewrite §9 (Discussion) to draw the broader implication: future world-model architectures should include explicit **world-type recognition** modules; our finding suggests current production VLMs solve this implicitly via a single layer + direction, with model-specific routing through encoder vs LM.
- **C4 (Stretch)** — One paragraph in §10 (Conclusion) connecting our localization claim to potential **interventions for safety / alignment** (e.g., suppressing physics-mode commitment when the model should treat input as abstract — useful for OCR / mathematical reasoning where physical intuition misfires).

**Outputs**: rewritten paper §1, §6, §9, §10. No new experiments.

**Resources**: 1–1.5 weeks of writing/restructuring time.

**Note on 5-signature framing**: User chose **appendix-relegation over pruning** (2026-04-28). All 5 measurements stay in main paper, organized by Marr level (Computational → PMR; Representational → M3+M4; Mechanistic → M5a+M5b). Cross-method redundancy is justified as **failure-mode robustness within each level**, not 5 independent claims. Pruning to 3-main + 2-appendix is reserved for very late stage if final narrative demands it.

---

## 3. Week-by-week schedule

Today: 2026-04-28. Target: ICLR 2027 deadline ~late Sep 2026. Buffer: ~5 months. The schedule is front-loaded with experiments; writing intensifies in months 4–5.

| Week | Date range | Focus | Pillar | Deliverable |
|---|---|---|---|---|
| 1 | 2026-04-28 → 05-04 | Multi-prompt config design + smoke runs | A1 | 3 prompt configs × 5 models, smoke results |
| 2 | 2026-05-05 → 05-11 | Full multi-prompt run + analysis | A1 | Per-prompt PMR table; A1 milestone close |
| 3 | 2026-05-12 → 05-18 | M5a + M5b re-evaluation on 3 prompts | A2 | Cross-prompt steering/SAE table; A2 close |
| 4 | 2026-05-19 → 05-25 | Projector-swap LoRA feasibility (1-day) + spec | B1 | Go/no-go on B1; LoRA training script |
| 5 | 2026-05-26 → 06-01 | Projector-swap LoRA training + analysis | B1 | Idefics2-MLP variant; §4.6 + M5b re-run |
| 6 | 2026-06-02 → 06-08 | LM-only-swap LoRA spec + training (in parallel on GPU 2) | B2 | CLIP+Vicuna vs CLIP+Mistral pair |
| 7 | 2026-06-09 → 06-15 | LM-only-swap LoRA analysis + re-runs | B2 | M5a + M5b table for LM-controlled pair |
| 8 | 2026-06-16 → 06-22 | Optional: 1 more non-Qwen model | B3 | M5a + M5b on 6th model |
| 9 | 2026-06-23 → 06-29 | Marr-3-level §6 restructure | C1 | Rewritten §6 |
| 10 | 2026-06-30 → 07-06 | World-model §1 introduction rewrite | C2 | Rewritten §1 |
| 11 | 2026-07-07 → 07-13 | §9 discussion + §10 conclusion | C3 + C4 | Rewritten §9, §10 |
| 12 | 2026-07-14 → 07-20 | Full paper consistency pass + figure regen | — | Polished v2 draft |
| 13 | 2026-07-21 → 07-27 | External benchmark stretch (Physion / CLEVRER subset) — only if ahead | A3 | Optional supplementary §C |
| 14 | 2026-07-28 → 08-03 | Human baseline (re-evaluate; deferred 2026-04-28) | — | If results have matured by here, run Prolific (`docs/m7_human_baseline_plan.md`) |
| 15–17 | 2026-08-04 → 08-24 | Buffer for unexpected: re-runs, advisor reviews, ablations | — | — |
| 18 | 2026-08-25 → 08-31 | Internal advisor review + revision | — | Advisor-passed draft |
| 19 | 2026-09-01 → 09-07 | Final figure polish + appendix completion | — | Submission-ready draft |
| 20 | 2026-09-08 → 09-14 | ICLR submission week | — | Submitted to ICLR 2027 |

**Deadline buffer logic**: ICLR 2027 deadline ~late Sep 2026. Schedule lands first ICLR-ready draft by week 18 (end of August), giving 3 weeks of slack for unforeseen rework / supplementary experiments / response to advisor.

---

## 4. Resource budget

Available: **H200 × 2 GPUs**. Track B inference + LoRA training fits comfortably.

Expected hot-path GPU usage:
- Multi-prompt re-runs (A1): 5 models × 3 prompts × (M2 + M5a + M5b) ≈ 4–6 GPU-hours per model = 20–30 GPU-hr total. Trivial.
- Projector-swap LoRA training (B1): rank-32 LoRA on Idefics2, ~10K steps × ~30 min ≈ 5 GPU-hr per training run, ~20 GPU-hr including search.
- LM-only-swap LoRA training (B2): 2 LoRAs (Vicuna + Mistral) × 5–10 GPU-hr each ≈ 20 GPU-hr.
- Re-running §4.6 + M5b on LoRA-tuned variants: ≈ 30 GPU-hr.

**Total estimated**: ~100 GPU-hr across 5 months ≈ <1 GPU-hr per day average. Resource is **not** the bottleneck; experimenter time is. Emphasize parallelization (B1 on GPU 0, B2 on GPU 1).

---

## 5. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Projector-swap LoRA architecturally infeasible (B1) | Medium | High | 1-day feasibility spike at week 4; pivot to B2-only + lit-review-based theoretical claim if blocked. |
| Multi-prompt A1 yields *task-specific* mechanism (not task-agnostic) | Low–Medium | Medium | Honest scope-narrowing: paper claim shifts from "physics-mode commitment" to "next-state-prediction-mode commitment" + multi-prompt becomes the principled boundary of the claim. Still a publishable finding, but framing weakens. |
| LM-only-swap LoRA produces too-similar models (B2) — no signal | Medium | Medium | Pivot to encoder-controlled comparison: same Vicuna LM with CLIP vs SigLIP encoders (more direct test of encoder family). |
| ICLR 2027 deadline slips past week 20 | Low | Medium | Pivot to NeurIPS 2027 (May 2027 deadline) — gains 7 months for further work without losing submission slot. TMLR rolling as final fallback. |
| Reviewer dings Marr-3-level framing as "still 5 redundant signatures" | Low | Medium | Strengthen failure-mode-controls subsection (per Strategy 2 from `paper_strategy.md` memory) — show each method has *different* failure modes, with empirical evidence in supplementary. |
| Advisor flags fundamental claim weakness late (~week 18) | Low | High | Internal advisor review at week 18 (3-week buffer); honest scope-narrowing if claim fragile. |

---

## 6. Pruning / fallback rules

If by **end of week 8 (mid-June)** the experiments have NOT produced clean Pillar B results:
- **Drop B1 (projector-swap LoRA)** → keep n=1 perceiver result with stronger theoretical/literature grounding.
- **Keep B2 only** (LM-only-swap) — narrow but defensible architectural claim.
- **Add 1 more non-Qwen model (B3)** as compensating breadth.
- Track pivots toward "broad cross-model survey + Marr-3-level localization" rather than "controlled causal architecture claim."

If by **end of week 14 (early August)** the paper is still missing key pieces:
- Pivot to **NeurIPS 2027** (gain 7 months).
- Use the additional time to land Pillar B fully + add Pillar A's external-benchmark stretch.

---

## 7. Cross-references

- `references/paper_gaps.md` — 4 weaknesses ↔ fix-track mapping.
- `references/roadmap.md` — milestone status, hypothesis scorecard, change log.
- `docs/paper/draft_v1.md` — current paper draft (will be restructured per Pillar C).
- `docs/m7_human_baseline_plan.md` — Prolific human baseline plan (deferred to week 14 conditional re-evaluation).
- Memory: `paper_strategy.md` — strategic decisions, world-model framing, Marr-3-level decision.

---

## 8. Change log

| Date | Change |
|---|---|
| 2026-04-28 | Document created. Track B chosen. ICLR 2027 primary, NeurIPS 2027 secondary. 3 pillars defined. 20-week schedule. |
| 2026-04-28 (advisor-fix) | Verified venue calendar via WebSearch — confirmed ICLR 2027 ~Sep 24, 2026 / NeurIPS 2027 ~May 16, 2027 / NeurIPS 2026 May 7, 2026 (9 days, dropped). Added AAAI 2027 (Aug 2, 2026) as mid-stage backstop. Added §1.0 world-model operational definition (~150 words) for C2 — must validate against current evidence before writing §6 to avoid retrofitting. Sources: trybibby.com/conference-deadlines, iclr.cc/Conferences/2026/Dates. |
