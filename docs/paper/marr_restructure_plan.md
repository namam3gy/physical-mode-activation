---
purpose: Plan for the §6 Marr-3-level restructure (paper_gaps.md G4 fix)
date: 2026-05-04
status: planning — pre-execution outline
references:
  - references/paper_gaps.md §G4
  - references/submission_plan.md §2 Pillar C / §3 weeks 9-11
  - docs/paper/draft_v1.md (current §4–§7, lines 292–1022)
---

# §6 Marr-3-level restructure plan

> **Goal.** Convert the parallel-bullets "5 signatures" framing (which reviewers will read as one finding measured 5 ways) into a **3-level evidence hierarchy** (Marr) where each level addresses a *different question* about the data, and within-level method redundancy serves as **failure-mode robustness**, not as independent claims. Closes paper gap G4.
>
> **Scope.** Paper-side restructure of §4 + §5 + §6 + §7 of `draft_v1.md` into one new §6 with 4 subsections. No new experiments. Per `submission_plan.md` §3 this is week-9 work; we are starting it ~7 weeks early because the user is feeling the framing has wandered and wants it locked before any further data collection.

---

## 1. Target structure

The restructured §6 collapses what is currently §4 (Behavioral) + §5 (Encoder vs LM) + §6 (Causal localization M5a/M5b) + §7 (Pixel encodability §4.6) into a single 4-subsection §6 organized by Marr level.

| New subsection | Marr question | Methods | Failure mode | Within-level controls |
|---|---|---|---|---|
| **§6.1 Computational** | Does the model behaviorally enter physics-mode? | PMR (M2 + cross-model) | Prompt-wording bias / next-state-prediction shortcut alternative | KO/JA labels (§4.3); open vs FC (M4c); multi-prompt (Pillar A — M-MP) |
| **§6.2 Representational** | Does the model *encode* physics-mode in its activations? | M3 vision-encoder probe + M4 LM logit-lens probe | Low-level visual stats / token-frequency confounds | Stim-y vs behavioral-y AUC dissociation (M3); random-direction baseline; label-free (M4b/M4c); 5-fold StratifiedKFold |
| **§6.3 Mechanistic** | Does that encoding *cause* behavior? | M5a runtime steering (LM-side) + M5b SAE intervention (encoder-side) + §4.6 pixel-encodability (input-side) | Norm-scaling / feature noise / "any sufficient perturbation flips" | Mass-matched random (M5a); 3 random feature sets (M5b); random-direction control at matched ε (§4.6) |
| **§6.4 Cross-level triangulation** | Do the 3 levels converge on the *same layer / direction / features*? | Convergence table + dissociation cases | Cross-level redundancy mistaken for independent claims | Show within-level method-disagreement cases as evidence levels are *not* tautological (e.g., Idefics2 §4.6 0/9 + M5a 10/10; LLaVA-Next M5a+ / M5b NULL) |

**Reviewer-facing claim**: "Cross-level convergence is *not* tautological because each level addresses a different reading of the data; cross-method redundancy *within* each level provides robustness against method-specific confounds."

---

## 2. Mapping current sections → new structure

This is the concrete content-migration table. Existing prose is preserved; only the organizing skeleton + a few connective sentences change.

| Current section (line range) | Target section | Notes |
|---|---|---|
| §4. Behavioral findings — cross-model PMR ladder (292-360) | **§6.1 Computational** | Moves wholesale. |
| 4.1 PMR(_nolabel) ladder (296-317) | §6.1.1 The PMR ladder | Keep as-is. |
| 4.2 H1 (abstraction ramp) (318-331) | §6.1.2 Abstraction ramp (H1) | Keep as-is. |
| 4.3 H7 (label selects regime) (332-346) | §6.1.3 Label selects regime (H7) | Keep as-is. |
| 4.4 Photo collapse (M8c) (347-360) | §6.1.4 Photo collapse | Keep as-is. |
| **(NEW)** | §6.1.5 **Failure-mode controls** | New subsection: 1-page summary of KO/JA (§4.3 cross-language), open vs FC (M4c), multi-prompt 5-model × 4-prompt H2 paired-delta (M-MP) — defending against the "next-state-prediction shortcut" alternative reading. |
| §5. Encoder vs LM disambiguation (361-455) | **§6.2 Representational** + **§6.4** (split) | See split below. |
| 5.1 Vision encoder probes — uniform discriminability (365-388) | §6.2.1 Vision encoder probes | Keep prose; move stim-y vs behavioral-y dissociation here as the within-level control. |
| 5.4 LM logit-lens cross-model (415-446) | §6.2.2 LM logit-lens probes | Keep prose; same level (representational) as 6.2.1 — the convergence between vision and LM probes is the within-level robustness story. |
| 5.2 The 2-CLIP-point insight (389-400) | §6.4.2 Architecture-level identity | Moves to triangulation section (it's an architecture-level dissociation between encoder family and behavioral PMR — that's a cross-level reading). |
| 5.3 §4.5 cross-encoder swap (401-414) | §6.4.2 Architecture-level identity | Moves alongside 5.2 — both architecture-level reads. |
| 5.5 The architecture-level reframe (447-455) | §6.4.1 Convergence table preface | Becomes the lead paragraph of the cross-level triangulation section. |
| §6. Causal localization M5a + M5b (456-626) | **§6.3 Mechanistic** | Moves wholesale, restructured into LM-side / encoder-side / input-side. |
| 6.1 v_L direction extraction (461-466) | §6.3.0 Direction extraction (shared infra) | Keep as-is; promote to a setup paragraph for §6.3 because v_L is reused by 6.3.1, 6.3.2, and §4.6. |
| 6.2 + 6.2b M5a runtime steering + cross-model (467-547) | §6.3.1 LM-side intervention (M5a) | Keep prose; merge 6.2 (Qwen original) and 6.2b (cross-model) under one §6.3.1. |
| 6.3 v_L10 is a regime axis (548-567) | §6.3.1.1 Regime axis characterization | Becomes a sub-subsection under M5a — it's a refinement of what the M5a direction is. |
| 6.4 Encoder-side intervention M5b SAE (568-626) | §6.3.2 Encoder-side intervention (M5b) | Keep prose. |
| §7. Pixel encodability §4.6 (627-1022) | §6.3.3 Input-side intervention (§4.6 pixel-encodability) | Moves wholesale; reframed as "the third mechanistic test — can the LM-side direction be reached from the input via gradient ascent?" Pixel encodability is a *causal* claim about reverse routability, so it lives in mechanistic. |
| 7.6 Cross-model §4.6 — pixel-encodability is architecture-conditional | §6.3.3 final paragraph + §6.4.3 dissociation cases | The cross-model finding splits: the per-model results stay in §6.3.3; the architecture-conditional reading (LLaVA-1.5 weak / LLaVA-Next L20+L25 / Idefics2 0/9 / InternVL3 testable under modified protocol) feeds the dissociation table in §6.4.3. |
| **(NEW)** | §6.3.4 **Failure-mode controls (mechanistic)** | New subsection: 1-page summary of mass-matched random (M5a); 3 random feature sets (M5b); random-direction control at matched ε (§4.6); explicitly defending against "any sufficient perturbation" / "norm-scaling" / "feature noise" alternatives. |
| **(NEW)** | §6.4 **Cross-level triangulation** | New section. Three deliverables: (1) the convergence table (same layer L, same direction v_L, same SAE features across the 3 levels for Qwen + at least 1 cross-model); (2) the architecture-level identity reading (combines current 5.2 + 5.3 + 5.5); (3) the dissociation cases (LLaVA-Next M5a+ / M5b NULL; Idefics2 §4.6 0/9 / M5a 10/10) — these *prove* the levels are not tautological. |

**Net effect**: 4 existing sections (§4 + §5 + §6 + §7) collapse into 1 new §6 (with §6.1/6.2/6.3/6.4 + new failure-mode subsections). Word count likely shrinks 10-20% via dedup of the architecture-level reframe (which currently appears in §5.5, §6.4 final paragraph, and §7.5).

---

## 3. New convergence table (§6.4.1)

Single table that makes the cross-level convergence explicit per model. This is the centerpiece of the restructured §6.

|  Marr level → | Computational (§6.1) | Representational (§6.2) | Mechanistic (§6.3) |
|---|---|---|---|
| **Question** | enters physics-mode behaviorally? | encodes physics-mode in activations? | causally bound to behavior? |
| Qwen2.5-VL-7B | PMR 0.94 | M3 AUC 0.99 / M4 AUC 0.96 (peak L20) | M5a L10 α=40 10/10; M5b k=20 0/20; §4.6 5/5 ε=0.05 |
| LLaVA-Next-7B | PMR 0.70 | M3 AUC 0.81 / M4 AUC 0.79 | M5a L20+L25 10/10; M5b k=160 NULL; §4.6 L20+L25 10/10 |
| Idefics2-8B | PMR 0.88 | M3 AUC 0.93 / M4 AUC 0.995 | M5a L25 α=20 10/10; M5b k=160 0/20; §4.6 0/90 across L5-L31 |
| InternVL3-8B | PMR 0.92 | M3 AUC 0.89 / M4 untestable (n_neg=1) | M5a untestable; M5b k=160 0/20; §4.6 testable under M8a (L10 5/5) |
| LLaVA-1.5-7B | PMR 0.18 | M3 AUC 0.73 / M4 AUC 0.76 | M5a 0/10; M5b NULL ≤ k=800; §4.6 weak L25 only (40 % at n=10) |

**Convergence reading**: each model's row should read consistently across the 3 columns. Qwen + InternVL3 + Idefics2 + LLaVA-Next show varying but coherent within-row patterns; LLaVA-1.5 is the systematic low-end across all 3 levels. **The cross-level coherence is the convergence claim.**

**Dissociation cases** (the within-row mismatches that *prove* the levels are not tautological — these go in §6.4.3):

| Model | Dissociation | What it shows |
|---|---|---|
| Idefics2 | M4 AUC 0.995 + M5a 10/10 + §4.6 0/90 | Information presence ≠ pixel-space routability. Forward pathway works; inverse pathway blocked. (Perceiver-resampler signature.) |
| LLaVA-Next | M5a 10/10 + M5b NULL | LM-side direction operative; encoder-side features absent. (CLIP family routes physics-mode commitment through LM.) |
| LLaVA-1.5 | M5a 0/10 + M5b NULL + §4.6 weak only | The full encoder-bottleneck — both encoder-side localization and pixel-side routability missing. (CLIP-ViT-L weak encoder.) |

These are not "5 measurements giving the same answer" — they are 3 different probes giving *different* answers per model, with the pattern of agreement/disagreement characterizing each architecture.

---

## 4. Updates outside §6

The Marr framing also requires light edits to §1 and §9. These are scoped to **lead paragraphs** only — full §1/§9 rewrite is week-10/11 work per `submission_plan.md`.

### 4.1 §1.2 Contributions (lines 80-128)

Replace the 3-bullet contribution list with a Marr-framed equivalent. Current bullets ("Cross-architectural quantification", "Causal localization", "Pixel encodability") map to new bullets keyed to the 3 levels:

- (1) **Computational-level claim** — behavioral physics-mode commitment varies 5× across architectures (PMR 0.18-0.99) on identical stim. Cross-model + multi-prompt evidence.
- (2) **Representational + Mechanistic-level claim** — the commitment is *causally localized* to a single LM mid-layer × residual-stream direction × encoder-side SAE feature set, with architecture-conditional routing through encoder vs LM.
- (3) **Cross-level pixel-encodability** — the same direction is reachable from pixel space via gradient ascent in 3 of 5 architectures, falsifying "shortcut emerges only at runtime" alternatives.

Then add 1 short sentence: "We organize evidence at three Marr levels (Computational / Representational / Mechanistic) with within-level controls protecting against method-specific failure modes; cross-level convergence is not tautological because each level addresses a different reading of the data."

### 4.2 §9 Discussion lead

Lead paragraph reframe: "We localize a **production-VLM world-model commitment mechanism** at three levels of evidence. The Computational level shows the commitment varies architecturally; the Representational level identifies the layer + direction encoding it; the Mechanistic level shows three causal pathways (LM-side, encoder-side, pixel-side) with architecture-conditional routing. Future world-model architectures should..."

(Full §9 rewrite is week-11 work.)

---

## 5. Sequencing — proposed execution order

This is the work plan for executing the restructure incrementally rather than in one session.

| Step | Action | Estimated effort | Gate |
|---|---|---|---|
| **1** | Create new §6 skeleton (4 subsection headers + Marr-framing intro paragraph + the convergence table from §3 above as a placeholder). Keep current §4–§7 intact below it. | 30 min | User reviews + approves outline |
| **2** | Migrate §4 → §6.1 (cut/paste with ≤5 connective edits). Add §6.1.5 failure-mode subsection citing §4.3 + M4c + M-MP. | 1-2 hr | Reads cleanly end-to-end |
| **3** | Migrate §5.1 + §5.4 → §6.2. Move §5.2 + §5.3 + §5.5 → §6.4 placeholder. | 1-2 hr | §6.2 reads as a single coherent representational-level argument |
| **4** | Migrate §6 (M5a + M5b) → §6.3.1 + §6.3.2. Promote §6.1 to §6.3.0 setup. | 2-3 hr | M5a + M5b read as two methods at one level (mechanistic), not two unrelated sections |
| **5** | Migrate §7 (§4.6 pixel-encodability) → §6.3.3. Add §6.3.4 failure-mode subsection. | 2-3 hr | §6.3 has 3 mechanistic methods (LM / encoder / pixel) clearly framed under one Marr question |
| **6** | Write §6.4 cross-level triangulation: convergence table + architecture-level identity + dissociation cases. | 3-4 hr | The convergence + dissociation reading is explicit |
| **7** | Light edits to §1.2 Contributions (Marr-framed bullets + 1-sentence framing line) + §9 lead paragraph. | 1 hr | §1 promises what §6 delivers |
| **8** | Delete old §4 + §5 + §7 (now empty). Renumber downstream sections (§8 + onward shift). Update Roadmap §1.3 hypothesis status table refs if any §4/§5/§7 numbers are quoted. | 1 hr | Internal cross-refs consistent |
| **9** | Read-through pass for word-count; aim for net 10-20 % shrink via dedup of architecture-level reframe. | 1-2 hr | Section flows; no orphan claims |
| **10** | Insight doc + CHANGELOG entry: `docs/insights/marr_restructure_2026-05-04.md` + `docs/CHANGELOG.md` row. | 30 min | Per CLAUDE.md project rules §5 |

**Total**: ~13-19 hr of writing time. Per `submission_plan.md` schedule this is week 9 (1 week), so the estimate fits.

**Recommended cadence**: do steps 1-3 in one session (the easy wins, ~3 hr), pause for user review, then steps 4-6 in a second session (the harder mechanistic restructure, ~7 hr), then steps 7-10 in a final polish session (~3 hr).

---

## 6. Risks + things to be careful about

- **Renumbering breakage**: sections 4-7 collapse to 6, so 8+ shift. Cross-refs in §1.3 Roadmap, in figures, in supplementary, and in any other paper-side doc need updating. Use grep to find all §<n>.<m> references before final commit.
- **Dedup drift**: the architecture-level reading currently appears in §5.5 + §6.4 final + §7.5. After consolidation into §6.4.2, the prose needs one tight statement, not three lightly-paraphrased ones.
- **Multi-prompt (M-MP / Pillar A) integration**: M-MP results live in `docs/insights/m_mp_phase3_followup_2026-04-28.md` but are not in `draft_v1.md` yet. §6.1.5 needs them; they should be inlined as a paragraph + small table, not just cited.
- **§6.1.5 failure-mode controls — the "next-state-prediction shortcut" alternative**: this is paper_gaps.md G1. M-MP behavioral 19/20 cells positive answers it; cite explicitly. Don't leave G1 unaddressed in §6.1's narrative just because Pillar A is "done" elsewhere.
- **§6.4 dissociation cases — the LLaVA-1.5 deeper-layer test currently running** (PID 17258, expected ~30-40 min): result will land mid-week. If the deeper-layer L28/30/31 result is null (predicted), it strengthens §6.4.3's "LLaVA-1.5 full encoder-bottleneck" reading by closing depth coverage. Hold off finalizing §6.4 until that result is in.
- **G3 (Idefics2 perceiver-resampler n=1)**: M-PSwap is backlogged with NaN unresolved (per `docs/CHANGELOG.md`, M-LMSwap dropped 2026-05-03, M-PSwap unresolved as of 2026-04-29). The Marr restructure does NOT close G3 — it just frames the data we have. G3 needs a separate plan; leave the §6.3.3 prose architecture-conditional ("perceiver-resampler is the leading remaining candidate") rather than causal until Pillar B closes.

---

## 7. Open questions for the user

Before executing step 1, user should confirm:

1. **OK with collapsing §4 + §5 + §6 + §7 into one §6?** Alternative is keeping §4 separate as "Behavioral findings" and only restructuring §5-§7 into Marr levels. Per `submission_plan.md` §2 the explicit intent is "§6 of paper draft restructured into §6.1 / §6.2 / §6.3 / §6.4" — so the collapse is the intended structure. Confirm.
2. **Insertion order of §6.3.3 (pixel-encodability)**: should it come before or after §6.3.2 (M5b SAE encoder-side)? Argument for *before* (current §7 order, LM → pixel via reverse): cleaner narrative arc. Argument for *after* (current order in scoping → mechanism): pixel is the third probe, harder than M5a/M5b, builds on them. Recommend *after*; confirm.
3. **§6.4 dissociation table**: include all 3 dissociation cases, or only the 2 "positive at one level / negative at another" cases (Idefics2 + LLaVA-Next), framing LLaVA-1.5 as a "convergent low end" rather than dissociation? Recommend including all 3 for completeness; confirm.
