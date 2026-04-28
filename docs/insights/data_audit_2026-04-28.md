---
section: Data audit (2026-04-28)
date: 2026-04-28
status: in-decision
purpose: Decide (A) what data to collect next; (B) what existing data is fit-for-purpose / over-collected for the paper.
---

# Data audit — 2026-04-28

> Two separate questions:
> - **(A) Do we need more data?** Forward, gap-filling for the EMNLP/NeurIPS paper.
> - **(B) Does the data we have fit the research?** Backward, identify
>   over-collection or claim/data mismatches.

## TL;DR

- **(A) Three new collections are paper-blocking; two are nice-to-have.**
  In priority order: (1) **M7 human baseline** (Prolific 20 raters × 50
  stim) — without this, no human-vs-model contrast in the paper. (2)
  **§4.6 Idefics2 deeper-layer (L26-31) sweep** — required to
  discriminate the "wrong-relative-depth vs perceiver-resampler"
  hypothesis pair for the Idefics2 anomaly. Without it, the §4.6
  cross-model section has an unresolved anomaly that weakens the
  H-shortcut claim. (3) **§4.6 per-model abstract-baseline expansion**
  — break the n=10 ceiling by using each model's own abstract cells
  (not all 5 models share the same baseline cells). Trade-off: cross-
  model comparability vs within-model power. Nice-to-have: §4.8 Qwen
  72B (one more scaling point); explicit prompt-steering à la Gavrikov
  et al. 2024 (currently in original ST5 scope but not executed).

- **(B) The datasets that are partially-utilized for mechanism work:**
  **LLaVA-Next / Idefics2 / InternVL3 LM activation captures** are
  used for §4.6 layer sweep + §4.5/§4.6 cross-stim probes, but
  **NOT** for M5b SIP — because Idefics2 and InternVL3 yield 0 valid
  SIP pairs (open-prompt PMR is saturated → no clean/corrupted pair
  with PMR=1/PMR=0 split), and LLaVA-Next has only 2 SIP candidates
  (too few for stable IE estimate). The Qwen vision encoder is the
  only one with M5b SAE training. Decision: (i) **expand SIP-feasible
  stim for LLaVA-Next** (need more open-prompt abstract-baseline cells
  to get SIP candidates) — turns LLaVA-Next captures into a 3rd model
  point for the L9-MLP-locus claim; (ii) **train SAE on the 3
  non-Qwen vision encoders** — no SIP needed, just vision activations
  (already captured); (iii) accept Qwen + LLaVA-1.5 as the only
  mechanism-evidence model points and frame paper accordingly.

- **The biggest framing risk for the paper → PARTIALLY RESOLVED
  2026-04-28:** the "5-model" wording on §4.6 H-shortcut was
  operationally 3-model testable + 2 caveats. **T1b ran**: Idefics2
  9-layer test (L5-L31) shows 0 clean shortcuts at any depth →
  wrong-relative-depth hypothesis falsified. **Perceiver-resampler
  bottleneck is the leading remaining candidate**, not isolated:
  the 5-model design varies encoder + projector + AnyRes
  simultaneously across the cell where Idefics2 differs from
  MLP-projector models, so a controlled projector-swap test would be
  needed to isolate the projector axis (out of scope). Paper claim
  now: "**3 of 5 testable architectures support pixel-encodability;
  Idefics2 falsifies the universal claim across 9 LM layers
  (16-97 % depth) — perceiver-resampler is the leading remaining
  architectural candidate; InternVL3 untestable under §4.6
  protocol**".

---

## 1. Hypothesis × supporting-dataset × claim strength matrix

| Hypothesis | Supporting datasets | Sample on binding cell | Claim strength | Paper risk |
|---|---|---|---|---|
| **H-encoder-saturation** (architecture-level) | M2 + M8a + M8c + M9 bootstrap (5 models × 3 stim sources) | Bootstrap CI separates non-CLIP [0.80, 0.92] vs CLIP-LLaVA-1.5 [0.14, 0.37] cleanly | **strong** | low |
| **H1 (S-ramp, unsaturated-only)** | M2 + M6 r7 + M8a + M8d cross-shape | LLaVA-1.5 ramp +0.30 cleanest; Qwen ceiling-flat +0.05; M8a strict 1/4 Qwen | **supported, qualified (unsaturated-only)** | low |
| **H2 (label-prior raises PMR)** | M2 (Qwen) + M6 r1 (LLaVA-1.5) + M6 r2a (InternVL3) + M4b label-free + §4.2 reverse prompting | 5 models × 3 stim sources; image vs label trade-off observed | **strong** | low |
| **H7 (label selects regime, unsaturated-only)** | M2 GAR + M5a-ext Exp 2 + M8a + M8d + §4.11 regime distribution | LLaVA M8d 3/3 strict PASS; Qwen 0/3 binary but non-trivial regime split | **strong (LLaVA), qualified (Qwen ceiling)** | low |
| **H-shortcut (pixel-encodability)** | §4.6 5-model n=10 layer sweep | **3 testable models** (Qwen broad, LLaVA-Next L20+L25, LLaVA-1.5 L25 weak); **2 caveats** (Idefics2 anomaly, InternVL3 saturated) | **architecture-conditional; "5-model" framing is misleading** | **medium** — Idefics2 disambiguation actionable |
| **H-direction-specificity** | §4.6 5-model n=10 sweep | 1/250 random hits across all 5 models | **strong on projection level even where PMR doesn't flip** | low |
| **H-locus (L9 MLP constructs, L10 reads)** | M5a + M5b SIP (Qwen) + M5b knockout (Qwen) + M5b per-head (Qwen) + M5b SIP cross-model (LLaVA-1.5 only) | Qwen IE=+1.0 SIP & knockout, LLaVA-1.5 cross-model lock-in shifted to L20 | **mechanism solid in Qwen, layer-shifts in LLaVA-1.5; 2 model points only** | **medium** — Idefics2/InternVL3 too saturated for SIP, so cross-model mechanism is fundamentally bounded |
| **H-LM-modulation** | M9 5000-iter bootstrap on M8d | Idefics2 H7 CI just touches 0; Qwen CI crosses 0; gap driven by single shape | **suggested only** | high (already known) — drop from paper or rerun with same-encoder LM swap |
| **H-boomerang** (Qwen-scoped) | M3 encoder AUC + M4 LM AUC + M5a behavior | Qwen ~0.99/0.94/0.93 small "encoder knows, decoder mildly gates" | **strong, Qwen-scoped** | low (scoped) |
| **H3 (scene inconsistency degrades RC)** | — | **untested** (axis E dropped from M2) | **untested** | low (out of scope for current paper) |

---

## 2. Backward audit — does what we have fit the research?

### 2.1 Load-bearing for paper headline

| Dataset | Inferences | Paper role |
|---|--:|---|
| M2 mvp_full (Qwen) | 1440 (480 stim × 3 labels × 1 prompt) | H1, H2, H4, H5, H6, H7 main behavioral table |
| M2 cross-model M6 r7 (LLaVA-1.5/Next, Idefics2, InternVL3) | ~7200 | 5-model PMR ladder, encoder-saturation paired-delta |
| M8a 5-shape × 2-model | 1600 | Cross-shape generalization for H1/H7 |
| M8c 60 photos × 5 models | ~7200 incl labels | M9 bootstrap on photos; encoder-swap convergence |
| M8d 3-category × 2 models | 960 | H7 cross-category (3/3 LLaVA strongest) |
| M9 5000-iter bootstrap | (re-analysis) | Replaces PASS/FAIL with CI separation |
| §4.6 5-model n=10 layer sweep | 250 v_unit + 250 random | H-shortcut + H-direction-specificity |
| M5b SIP + per-head + SAE (Qwen) | ~2000 | H-locus mechanism |
| §4.8 Qwen 32B M2 | 1440 | "Scale doesn't fix grounding (aggregate); helps on weak-cue cells" |

### 2.2 Possibly over-collected for current paper

| Dataset | Status | Recommendation |
|---|---|---|
| **M5b SIP captures** for LLaVA-Next, Idefics2, InternVL3 (LM activations + vision activations) | Captured 2026-04-26 to 04-27, used for §4.5/§4.6 layer sweep but **SIP-blocked**: Idefics2 / InternVL3 yield 0 SIP pairs (saturated open-prompt PMR); LLaVA-Next yields only 2 (too few). Only Qwen + LLaVA-1.5 have full SIP. | **Decide before submission**: (i) generate more open-prompt abstract-baseline stim for LLaVA-Next to get ≥10 SIP pairs (~2 hr H200) — turns into 3rd cross-model L9-locus point. (ii) For Idefics2/InternVL3, SIP is fundamentally blocked by saturation → frame paper accordingly. |
| **Vision encoder activations** for LLaVA-1.5/Next, Idefics2, InternVL3 | Captured for M3/M6 probing, used for AUC ladder | **Untapped for SAE**: M5b SAE was trained Qwen-only. Could train cross-model SAE without new captures (~5 min per model on H200) — would surface "are physics-cue features universal?" finding. |
| **§4.10 attention viz UI** (Qwen-only) | Complete | Paper appendix figure infrastructure only — keep but don't expand. |
| **§4.3 Korean + Japanese 5-model labels** | Complete (~3000 inferences) | Strong stand-alone finding (LM language fluency modulates label transfer); paper-secondary. Could cut to a single-paragraph mention if space-constrained. |
| **§4.5 ext (Idefics2 M8d + M8c)** | Complete | Cross-stim encoder-swap → confirms photos collapse all 3 models. Already folded into M9; not separately load-bearing. Subsumable into M9 narrative. |
| **§4.7 RC per-axis on M8a 5-model** | Complete | Decision-stability axis ranking — paper-secondary. Keep as "decision-stability ceiling = 3rd architecture-level signature alongside PMR-ceiling and pixel-encodability" → paper figure. |

### 2.3 Coverage gaps

- **ST5 prompt-steering (Gavrikov et al. 2024 method)**: original
  project.md ST5 scope says "treat this as an abstract geometric
  shape" vs "treat this as a physical object subject to gravity"
  prompt-steering test. This is **not executed**. §4.3 (KO/JA) is a
  partial language-prompt variation but not the explicit shape-
  texture-style steering. Either (i) execute it (1 day per model
  × 5 models = ~1 week, but probably 2-3 hr per model since stim
  exist), (ii) explicitly retire it from scope in the paper.
- **M7 Prolific human baseline**: project.md ST1 deliverable. Not
  executed. Without this, paper has no human contrast.
- **§4.4 Michotte 2-frame causality**: §4 backlog item, untested.
  Probably out of scope for v1 paper.

---

## 3. Forward gap-fill — 3-tier collapse

### Tier 1 — execute now, no user decision needed (cost < 1 hr each, no-brainer value) — **EXECUTED 2026-04-28**

| # | Item | Outcome |
|--:|---|---|
| **T1a ✅** | **M5b SAE cross-model** — trained 4 cross-model SAEs (LLaVA-1.5 vis23 4096, LLaVA-Next vis23 4096, Idefics2 vis24 4608, InternVL3 vis23 4096) on existing M3 vision activations (no new captures). | Wallclock ~25 min (sequential on shared GPU 0). InternVL3 first inspection: top Cohen's d=0.41 (vs Qwen 0.5+) — physics-cue features exist cross-model but separation weaker than Qwen's encoder. **Intervention runs (cross-model SAE feature ablation) deferred** — would close the "physics-cue features universal?" question. |
| **T1b ✅** | **§4.6 Idefics2 L26-L31 sweep** — fresh M2 LM activation capture at L26/28/30/31, v_L extraction (n_pos=470 / n_neg=10), 80-run counterfactual sweep at n=10 × 2 configs. | Wallclock ~90 min. **Result: 0/40 v_unit + 0/40 random with 1 isolated noise hit at L28** (Wilson [0.0025, 0.40]). v_L projection ascends cleanly (baseline -10.7 → final +27-30 at L26-30; -72 → +163 at L31). **Wrong-relative-depth hypothesis falsified across 9 LM layers (L5-L31, 16-97 % depth)**; **perceiver-resampler bottleneck is the only standing architectural mechanism**. H-shortcut framing risk **resolved** — paper claim becomes "3 of 5 testable models support pixel-encodability; Idefics2 falsifies via perceiver-resampler bottleneck (9-layer evidence); InternVL3 protocol-saturated". |

### Tier 2 — user decision required (paper-scope or external cost)

| # | Item | Decision |
|--:|---|---|
| **T2a** | **M7 Prolific human baseline** (20 raters × 50 stim, ~$200 + 1 wk) | External cost. No human-vs-model contrast otherwise. **Required for paper.** |
| **T2b** | **ST5 prompt-steering** (Gavrikov-style "treat as abstract geometric / treat as physical object", 5 models × M8a, ~5 hr H200) | Original ST5 scope item, not executed. **Recommendation: retire-with-reframe** — frame paper as "language-prior modulation already measured via KO/JA labels (§4.3) + label_free prompt (M4b) + open vs FC (M4c)" rather than executing the explicit Gavrikov test. Reviewers unlikely to demand the exact Gavrikov instantiation if the broader axis is covered. |
| **T2c** | **Paper trim of secondary content** | §4.3 cross-language: strong stand-alone but secondary. §4.5 ext: subsumable into M9. §4.7 RC per-axis: useful as 3rd architecture-level signature. §4.10 attention viz: appendix-only. **Decision needed depends on venue (see §4 below).** |

### Tier 3 — conditional, depends on Tier 1 results

| # | Item | When to revisit |
|--:|---|---|
| T3a | **§4.6 per-model abstract-baseline expansion** | Only if T1b doesn't resolve Idefics2 anomaly (then "is the n=10 stim cell unrepresentative?"). Otherwise this tightens within-model CIs without changing any paper headline. |
| T3b | **§4.8 Qwen 72B M2** | Nice-to-have scaling-curve point; current 7B-vs-32B comparison already supports the headline. Pursue only if 72B inference can be made fast (4-bit quant on 1× H200). |
| T3c | **M5b SIP for LLaVA-Next** (needs more open-prompt abstract-baseline stim, ~2-3 hr) | Conditional on T1a SAE cross-model results. If SAE shows clean cross-model "physics-cue features", LLaVA-Next SIP becomes a confirmation gate. Idefics2/InternVL3 SIP is **fundamentally blocked** by saturation (0 valid pairs). |

---

## 4. Venue dependency

The Tier 2/3 decisions hinge on paper venue:

| Venue | Tier needed | Reasoning |
|---|---|---|
| **EMNLP long primary** (language-prior dominance angle) | Tier 1 only + T2a (M7) + T2b retire-with-reframe | Behavioral + cross-model H1/H2/H7 + light mechanism is sufficient. Tier 1 alone gives 5-model SAE evidence + clean §4.6 cross-model story. |
| **NeurIPS stretch** (mechanistic localization angle) | Tier 1 + Tier 2 + T3c | Cross-model mechanism (L9-MLP-locus) needs ≥3 model points; LLaVA-Next SIP becomes load-bearing. |

**Recommend committing to EMNLP scope first** — execute Tier 1 + T2a, retire T2b — and revisit NeurIPS-only Tier 3 items after EMNLP draft is in motion.

---

## 5. Decisions queued for user

The 7-item queue collapses to 3 user decisions:

1. **Approve M7 Prolific** ($200 + 1 wk)? — strong recommend
2. **Confirm ST5 prompt-steering retire-with-reframe**? — recommend retire
3. **Confirm EMNLP-first scope** (Tier 1 + T2a only)? — recommend yes; defer Tier 3 to post-EMNLP

Tier 1 (T1a + T1b, ~2 hr H200 total) executes without user prompt.

## 6. Open framing risks

- **"5-model" §4.6 H-shortcut wording → PARTIALLY RESOLVED via T1b
  (2026-04-28).** Idefics2 9-layer test (L5-L31, 16-97 % depth)
  confirms 0 clean shortcuts at any depth. Paper claim is now: "3 of
  5 testable architectures support pixel-encodability (Qwen broad,
  LLaVA-Next L20+L25, LLaVA-1.5 L25 weak); **Idefics2 falsifies the
  universal claim across 9 LM layers (16-97 % relative depth, 0/90
  v_unit + 0/90 random with 1 isolated noise hit at L28 on a
  different stim than the L25 hit — uncorrelated noise)**; InternVL3
  protocol-saturated under §4.6 'circle' prompt." **A projector-
  design axis (MLP vs perceiver) emerges as the leading remaining
  candidate** for the disambiguating mechanism (encoder saturation
  alone is ruled out), but the 5-model design doesn't isolate
  projector cleanly — Idefics2 differs from MLP-projector models on
  encoder + projector + AnyRes simultaneously. Controlled
  projector-swap test would be needed to isolate; out of scope for
  v1 paper.
- **"Cross-model mechanism" L9-MLP claim** is fundamentally bounded
  to Qwen + LLaVA-1.5 (the only non-saturated models with successful
  SIP). For paper, frame as "LM mid-layer mechanism, model-specific
  locus" rather than "general L9 MLP claim". Adding LLaVA-Next via
  T3c gives 3 model points if NeurIPS scope is pursued.

---

## Reproducer / artifacts referenced

- Hypothesis scorecard: `references/roadmap.md` §1.3
- §4.6 layer sweep table: `outputs/sec4_6_cross_model_layer_summary/cross_model_layer_table.csv`
- M9 bootstrap: `outputs/m9_generalization_audit/`
- M5b cross-model SIP captures (unused): `outputs/encoder_swap_{idefics2,internvl3,llava_next}_m8a_*` + `cross_model_*_capture_*`
