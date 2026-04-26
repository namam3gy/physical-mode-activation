---
type: project synthesis
date: 2026-04-25
status: as-of M6 r6 + §4.2 + §4.10 (paper Section 4 lock)
audience: paper reviewer / new collaborator / future self
---

# Physical-mode activation in VLMs — research overview

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H1** — PMR rises in an S-shape along the abstraction axis (line → filled → shaded → textured); ground introduction adds the largest single jump.
> - **H2** — The label (ball / circle / planet) independently raises PMR even on minimal stim — a language-prior contribution beyond the visual evidence.
> - **H4** — The open-ended vs. forced-choice PMR gap is a stable signature of the language-prior ↔ visual-evidence conflict.
> - **H5** — A single ground line shifts PMR more than the visual difference between an abstract circle and a textured ball.
> - **H6** — Cast-shadow alone drives the cue saturation; the arrow is closer to annotation than physics signal — partially refuted (arrow alone also saturates).
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **H-boomerang** — Vision encoder linearly separates physics-mode classes even where behavior fails — encoder knows, decoder gates. (Qwen-scoped: refuted on LLaVA-1.5 because its CLIP encoder is the bottleneck.)
> - **H-direction-bidirectional** — v_L10 is a regime axis within physics-mode (+α → kinetic, −α → static); revised from the initial "one-way activator" framing.
> - **H-encoder-saturation** — Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the architecture level (joint encoder + LM), not encoder representational capacity alone.
> - **M0** — Infrastructure scaffold (package layout, scripts, configs, tests).
> - **M1** — ST1 pilot — Qwen2.5-VL-7B on 480 stim; established H1 partial / H2 strong / H4 / H5 / H6 candidates.
> - **M2** — ST1 MVP-full — 5-axis factorial (2880 stim); H1 monotone S-curve, H7 emerged.
> - **M3** — ST2 vision-encoder probing — encoder AUC ≈ 1.0 trivially separates factorial axes ("boomerang").
> - **M4** — ST3 LM logit lens / per-layer probes — LM AUC plateaus at ~0.95 at visual-token positions from L5.
> - **M4b** — M4 + label-free prompt as H2 null test; revealed H2 is asymmetric on Qwen (circle override, not ball enhancement).
> - **M4c** — Forced-choice label-free variant — confirms M4b under FC; surfaces LLaVA "A" greedy bias.
> - **M5a** — ST4 VTI steering — adding +α·v_L10 at LM L10 over visual tokens flips line/blank/none from "stays still" → physics-mode.
> - **M5b** — ST4 Phase 3 (SIP + activation patching + SAE feature decomposition) — deferred / optional.
> - **M6** — ST5 cross-model sweep — see M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next).
> - **M7** — Human Prolific baseline (20 raters × 50 stim) + paper writing — deferred / optional.
> - **M8** — Stim diversification family — see M8a (synthetic shapes), M8c (real photos), M8d (non-ball categories), M8e (cross-source).
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim diversification — real photographs (60 photos × 5 categories from COCO + WikiArt). Photos REDUCE Qwen PMR(_nolabel) 18-48 pp.
> - **M8d** — Stim diversification — non-ball physical-object categories (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M8e** — Cross-source paired analysis (M8a + M8d + M8c consolidated). Model × category × source_type heatmap is the paper Table 1 candidate.
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.
> - **M6 r1** — ST5 cross-model — LLaVA-1.5-7B replicates H2 cleanly (unsaturated CLIP encoder lets the label-prior shift PMR).
> - **M6 r2** — ST5 round 2 — InternVL3 super-saturated, LLaVA captures expose CLIP-encoder bottleneck, FC logit ratio confirms LLaVA "A" bias is logit-level.
> - **M6 r3** — Idefics2 SigLIP-SO400M probe — vision encoder probe AUC 0.93 closes the encoder-AUC ↔ PMR chain (3-point).
> - **M6 r4** — InternVL3 InternViT probe — AUC 0.89 / PMR 0.92, extends the chain to 4 model points; H-encoder-saturation "non-CLIP-general".
> - **M6 r5** — M8c photo encoder probe (4 models, cross-stim) — behavioral-y AUC inverts but stim-y AUC stays at 1.0 → encoder discriminability is uniform; architecture-level reframe.
> - **M6 r6** — LLaVA-Next-Mistral 5th model point (2nd CLIP) — PMR 0.700 [0.65, 0.74] sits between LLaVA-1.5 floor and saturated cluster; rules out vision-encoder-family as sole determinant.
> - **v_L10** — Steering direction in LM hidden space (dim 3584) at layer 10, derived from M5a class-mean diff (physics − abstract). Unit norm.

A single-doc synthesis of what this project found, how it found it, and
what is still open. Intended for readers who haven't followed the
session-by-session work.

## TL;DR

Open-source VLMs **do** read minimal synthetic stimuli (a black circle on
white) as physical objects, **but** the reading is determined at the
**architecture level** (joint vision-encoder + LM), not at the encoder
representational level. Across 5 tested models (Qwen2.5-VL, LLaVA-1.5,
LLaVA-Next, Idefics2, InternVL3) on 3 stim sources (M8a synthetic shapes
× 5, M8d synthetic categories × 3, M8c real photos × 5):

- Every encoder linearly separates physics-vs-abstract factorial cells
  at AUC = 1.0 — encoder discriminability is uniform.
- Behavioral PMR (the rate at which the model commits to physics-mode
  language) ranges from 0.18 to 0.92 on the same stim, depending on the
  joint architecture.
- The 2-CLIP-point comparison (LLaVA-1.5 PMR 0.18 vs LLaVA-Next PMR 0.70)
  rules out vision-encoder-family as the sole driver.
- All 5 LMs allocate only 3–26% of last-token attention to visual tokens
  even though visual tokens are 79–98% of the input. Visual attention
  peaks at mid-layers.
- Real photographs **compress** the encoder gap (all 5 models converge
  to PMR [0.18, 0.67]) and **halve** the label-driven H7 effect
  (LLaVA-1.5 M8d +0.31 → M8c +0.10) — image-prior dominates label-prior
  when the image is rich.

## Headline figure

![5-model × 3-stim PMR ladder](../figures/session_5model_cross_stim_pmr.png)

5-model × 3-stim PMR ladder with bootstrap CIs. The encoder-family split
on synthetic stim (M8a + M8d) and the photo-collapse on M8c are the two
paper-grade facts.

## What we measure

**PMR (physics-mode reading rate)**: fraction of model responses that
describe the next-state of the stimulus in physical terms (e.g., "the
ball will fall", "it bounces", "rolls down"). Rule-based scorer with
~5% disagreement against hand-annotation. See `docs/scoring_rubric.md`.

Variants:
- **PMR(_nolabel)** — open-ended prompt without a label cue
  ("What do you see? What might happen next?"). Direct measure of the
  joint encoder+LM tendency.
- **PMR(_physical)** — same prompt with a physical-role label
  ("the ball...", "the car...", "the person...").
- **PMR(_abstract)** — same prompt with an abstract-role label
  ("the circle...", "the silhouette...", "the stick figure...").
- **H7 paired-difference**: PMR(_physical) − PMR(_abstract) per shape,
  averaged across shapes. Tests whether the label selects the physics
  regime.

Bootstrap CIs (5000 iters, prediction-level resampling within
shape × role cells) are used in M9 / M6 r6 / §4.2 to replace
PASS/FAIL binarization with separability tests.

## Stimulus design

Three sources, in order of complexity:

- **M8a — synthetic shapes (factorial 400 stim per model)**.
  5 shapes (circle, square, triangle, hexagon, polygon) ×
  4 abstraction levels (line / filled / shaded / textured) ×
  2 backgrounds (blank / ground) × 2 cues (none / both) × 5 seeds.
  Single event template (`fall`).
- **M8d — synthetic categories (factorial 480 stim per model)**.
  3 categories (car, person, bird) × 4 abstraction × 2 bg × 2 cue ×
  2 events (`fall`, `horizontal`) × 5 seeds. Same protocol as M8a but
  with non-ball physical objects.
- **M8c — real photographs (60 photos)**.
  12 photos × 5 categories (ball, car, person, bird, abstract) from
  COCO 2017 + WikiArt. Single label triplet per category from
  LABELS_BY_SHAPE.

Each labeled-arm run × model × stim produces ~400-1440 inferences;
each label-free arm produces an n=400 PMR(_nolabel) baseline.

## What we built (4 M0–M5a phases, then 4 M6 + M8 + M9 phases)

### Phase 1: pilot + scoring (M0–M2)

- **M0**: programmatic stimulus generation in PIL (`primitives.py`).
- **M1**: pilot inference on Qwen2.5-VL-7B-Instruct, label-free arm.
  Established the basic PMR ladder across object_level / bg_level /
  cue_level cells. Paired-delta (`physical_label` − `_nolabel`)
  near zero on `textured/ground/both` cells (ceiling), strongly
  positive on `line/blank/none` cells (headroom). `docs/insights/m1_pilot.md`.
- **M2**: full PhysCue grid for H1/H2/H4/H5/H6/H7.
  - **H1 (ramp)**: PMR rises monotonically with object_level.
  - **H2 (label)**: physical labels raise PMR (later refined to
    "raise PMR only where headroom exists").
  - **H7 (label-selects-regime)**: physical labels (ball) elicit
    "fall/roll" verbs; abstract labels (planet) elicit "orbit".

### Phase 2: probing (M3–M4)

- **M3 — vision encoder probe**: per-layer logistic regression on
  pooled vision-encoder activations. Qwen SigLIP AUC = 0.99 at every
  captured layer — the encoder linearly separates physics-mode and
  abstract-mode stim trivially. **H-boomerang** discovered: encoder
  AUC ≈ 1.0 but behavioral PMR varies — the LM "gates" the encoder
  signal. `docs/insights/m3_encoder_boomerang.md`.
- **M4 — LM logit lens + per-layer probes**: per-layer LM activations
  fed into a logit lens (logits projected to next-token vocabulary).
  Label-physics margin develops at mid-layers (peak L20 of 28).
  `docs/insights/m4_logit_lens.md`.
- **M4b — label-free H2 null test**: Qwen with no label barely changes
  PMR vs Qwen with `ball` label (paired-delta +0.006). M2's "labels
  add PMR" pattern attributed to **circle suppression**, not ball
  enhancement. `docs/insights/m4b_label_free.md`.
- **M4c — forced-choice label-free**: confirmed the FC scoring
  reproduces M4b's null. LLaVA "A" bias is logit-level pathology, not
  greedy-sampling artifact. `docs/insights/m4c_fc_label_free.md`.

### Phase 3: causal steering (M5a)

- **M5a — VTI steering**: causal intervention at LM L10 with the
  M3-derived "object-ness" direction. α = +40 flips 10/10
  `line/blank/none` Qwen responses from D (abstract) to B (static
  physical). `docs/insights/m5_vti_steering.md`.
- **M5a-ext**: −α steering pushes physics-mode responses from
  kinetic to static (negative α), confirming `v_L10` is a
  **regime axis within physics-mode** (+α kinetic, −α static, baseline
  D below |α| threshold).
  `docs/insights/m5a_ext_bidirection_and_label.md`.

### Phase 4: cross-model + cross-stim (M6 + M8 + M9)

- **M6 r1 — LLaVA-1.5-7B cross-model**: M2 + M4b protocol replicated.
  LLaVA-1.5 shows the **original H2** (`ball` +0.475 vs no-label) —
  Qwen's "circle suppression" was Qwen-specific, traceable to encoder
  saturation. **Visual-saturation hypothesis** introduced.
  `docs/insights/m6_cross_model_llava.md`.
- **M6 r2 — 3-model expansion + capture + FC**: InternVL3 cross-model
  behavioral (paired-delta +0.010 for every label, super-saturated);
  LLaVA-1.5 vision encoder probe AUC ~0.73 (well below Qwen's 0.99).
  **H-encoder-saturation hypothesis** anchored to vision-encoder probe
  AUC. `docs/insights/m6_r2_cross_model.md`.
- **M8a — non-circle synthetic shapes**: 5 shapes × Qwen + LLaVA,
  strict pre-registration scoring. **Qwen 1/4 PASS, LLaVA 4/4 PASS** —
  asymmetry validates H-encoder-saturation cross-shape. H1 + H7 demoted
  to **unsaturated-only**. `docs/insights/m8a_non_circle_shapes.md`.
- **M8d — non-ball categories**: car/person/bird × abstraction × bg
  × cue × 2 events × 5 seeds. **LLaVA 3/3 H7 ✓** (project's strongest
  H7), Qwen 0/3 binary (ceiling) but regime distribution preserves the
  pattern. New `classify_regime` keyword classifier (5.6% hand-
  annotation error). `docs/insights/m8d_non_ball_categories.md`.
- **M8c — real photographs**: 60 photos × 5 categories. **Photos REDUCE
  Qwen PMR** by 18–48 pp across categories — synthetic-stim minimality
  is a co-factor of behavioral saturation, not just encoder representation.
  `docs/insights/m8c_real_photos.md`.
- **§4.5 — cross-encoder swap (Idefics2)**: Idefics2-8b (SigLIP-SO400M
  + Mistral-7B) on M8a — PMR(_nolabel) = 0.882, matching Qwen 0.838 (vs
  LLaVA 0.175). H-encoder-saturation **causally confirmed at the
  encoder-family level**: SigLIP family saturates regardless of LM.
  `docs/insights/encoder_swap_idefics2.md`.
- **M6 r3 — Idefics2 SigLIP-SO400M probe**: AUC 0.93. 3-point AUC
  ladder Qwen 0.99 / Idefics2 0.93 / LLaVA 0.73.
  `docs/insights/m6_r3_idefics2_probe.md`.
- **M6 r4 — InternVL3 InternViT probe + 4-model chain**: AUC 0.89
  / PMR 0.92. 4-point chain on identical stim. **Stim-y check (added
  late round)** discovered: all 4 encoders separate stim-defined
  factorial cells at AUC = 1.0. **Reframes H-encoder-saturation to
  architecture level** (encoder + LM fusion).
  `docs/insights/m6_r4_internvl3_probe.md`.
- **M6 r5 — M8c photo encoder probe**: cross-stim 4-model. Behavioral-y
  AUC inverts on photos (Qwen 0.88→0.44, but LLaVA stays). Stim-y AUC
  stays at 1.0. Cross-stim confirmation of architecture-level reframe.
  `docs/insights/m6_r5_m8c_photo_probe.md`.
- **M9 — generalization audit / paper Table 1**: 9 (model × stim) cells
  with bootstrap CIs (5000 iters). Replaces PASS/FAIL binarization
  with separability tests. `docs/insights/m9_generalization_audit.md`.
- **M6 r6 — LLaVA-Next 5th model + cross-stim**: 2nd CLIP point (CLIP-
  ViT-L + Mistral-7B + AnyRes). PMR(M8a) = 0.700 [0.65, 0.74], between
  LLaVA-1.5 floor and saturated cluster. M8d 0.625, M8c 0.417 (= Idefics2).
  Stim-y AUC = 1.0 across all 3 stim. **The 2nd CLIP point rules out
  vision-encoder-family as the sole determinant**. 5×3 grid lock.
  `docs/insights/m6_r6_llava_next.md`.

### Phase 5: §4 add-ons (this session)

- **§4.2 — reverse prompting on real photos**: existing M8c labeled-arm
  data re-analyzed. Image-prior dominates label-prior on real physical
  photos: phys − abs ≤ +0.146 across 5 models, vs LLaVA-1.5 M8d
  synthetic phys − abs +0.306. **Label dominance requires image
  impoverishment**. `docs/insights/sec4_2_reverse_prompting.md`.

  ![§4.2 H7 effect halves on photos](../figures/session_image_vs_label_h7.png)

- **§4.10 — attention visualization UI**: 5-model attention captures
  on M8a subset. **All 5 VLMs allocate only 3–26% of last-token
  attention to visual tokens** despite visual tokens being 79–98% of
  the input. Visual attention peaks at mid-layers (15 or 20).
  `docs/insights/sec4_10_attention_viz.md`.

  ![Cross-model attention to visual tokens](../figures/session_attention_cross_model.png)

## Hypothesis status (post-M6 r6 + §4.2 + §4.10)

| Hypothesis | Status | Evidence |
|---|---|---|
| **H-encoder-saturation** (architecture-level) | ✅ confirmed at 5 model points × 3 stim sources | All 5 stim-y AUCs = 1.0; PMR ladder is downstream-conditional. CLIP+Vicuna 0.18 vs CLIP+Mistral+AnyRes 0.70 rules out encoder-family as sole driver. |
| **H1** (ramp) | ✅ unsaturated-only | LLaVA 5/5 M8a, Qwen 1/4 M8a (saturated). |
| **H7** (label-selects-regime) | ✅ unsaturated-only AND architecture-conditional | LLaVA-1.5 M8d +0.31 (project max). LLaVA-Next M8d −0.05 — same encoder family, architecture switch breaks H7. |
| **H-direction-bidirectional** (M5a-ext) | ✅ confirmed | v_L10 is a regime axis within physics-mode (+α kinetic, −α static). |
| **H-boomerang** | ✅ Qwen-scoped (revised) | Holds for Qwen (encoder AUC 0.99 + behavioral 0.95). Refuted for LLaVA-1.5 (encoder AUC 0.73 = bottleneck). |
| **H-LM-modulation** | ⚠ suggested only | Two-Mistral M8d H7 ≈ 0 clustering (Idefics2 +0.05 / LLaVA-Next −0.05) is multi-axis-confounded. Not paper-defensible. |
| **§4.2 image-dominates-label** | ✅ confirmed cross-stim | Synthetic label effect halves to ≤ +0.15 on photos across all 5 models. |

## Method-level contributions (paper-relevant)

1. **PhysCue stimulus protocol** — 5 shapes × 4 abstraction × 2 bg × 2
   cue factorial (M8a) + 3 categories × 2 events extension (M8d) +
   60 real photos (M8c). Programmatic, deterministic, reproducible.
2. **PMR scoring rubric** — rule-based with hand-annotation validation
   (~5% disagreement). `docs/scoring_rubric.md`.
3. **Bootstrap CI methodology (M9)** — 5000-iter resampling at the
   prediction level within (shape × role) cells. Replaces PASS/FAIL
   binarization with separability tests.
4. **Stim-y vs behavioral-y probe distinction (M6 r4)** — encoder
   probe AUC has two interpretations: with stim-defined y (factorial
   cells), it measures encoder discriminability; with behavioral y
   (model's PMR distribution), it measures encoder-behavior alignment.
   The two diverge sharply (1.0 vs 0.77–0.93), and the divergence is
   the H-encoder-saturation reframe.
5. **classify_regime keyword classifier (M8d)** — 5.6% hand-annotation
   error; rescues H7 signal from binary-PASS saturation by reading the
   physics-regime distribution underneath the ceiling.
6. **VTI steering at L10 (M5a)** — causally confirms `v_L10` as the
   physics-mode regime axis. α = ±40 flips response between static
   and kinetic regimes within physics-mode.

## What this synthesis does NOT cover

- **M5b — SIP / activation patching / SAE feature decomposition**.
  The next mechanism-level milestone, not yet started. Would
  causally locate which LM features carry the physics-mode commitment.
- **§4.6 — counterfactual stimulus generation via SAE / VTI reverse**.
  Adversarial physics-mode prompt synthesis. Not started.
- **§4.3 — Korean vs English label prior** (1-hour experiment). Open.
- **§4.4 — Michotte-style 2-frame causality**. Open.
- **§4.7 — RC reinterpretation as per-axis decision stability**. Open.
- **§4.8 — PMR scaling** (Qwen 32B/72B). Open.
- **§4.11 — categorical H7 follow-up** (regime confusion matrix). Open.
- **M7 — paper draft + human baseline** (Prolific 20 raters × 50
  stimuli + EMNLP/NeurIPS draft). Open.
- **External validity beyond Qwen2.5-VL / LLaVA-1.5 / LLaVA-Next /
  Idefics2 / InternVL3** — Pixtral, Phi-V, Gemini-VL, GPT-4V,
  Claude. Open.

## Limitations

1. **Multi-axis confound between LLaVA-1.5 and LLaVA-Next**. The 0.18 → 0.70
   PMR jump is the largest behavioral move in the project but confounds 4
   axes (AnyRes tiling, fusion projector, training data + recipe, LM
   family). No tested model provides a same-architecture LM-only swap;
   we cannot isolate which axis carries the load.
2. **n=12 photos per category on M8c** is underpowered for H7 detection.
3. **Synthetic factorials are M8a/M8d-style** (line / blank / none ↔
   textured / ground / both). Real-world stim distributions are more
   varied; M8c is a small step toward this.
4. **Attention attribution is approximate**. Section §4.10 measures
   "fraction of last-token attention on visual tokens" — a coarse
   signal of visual-information access. Activation patching (M5b) is
   needed for causal claims.
5. **No human baseline yet**. PMR is a model-internal metric; mapping
   to human "physics-mode reading" judgments is M7 work.

## Next research directions (priority-ordered)

1. **M5b — SIP / activation patching / SAE**. Mechanism-level evidence
   for which LM features carry physics-mode commitment.
2. **§4.6 — SAE counterfactual stimulus generation**. Adversarial
   physics-mode prompt to test shortcut interpretation.
3. **M7 — paper draft**. Section 4 (encoder-saturation chain) is now
   paper-grade complete; Section 5 (mechanism, M5b) is the gap.
4. **External validity sweep** (Pixtral, Phi-V, Gemini-VL) once
   M5b establishes the 5-model mechanism baseline.

## Reproduction map (where to find what)

- **Stimuli**: `inputs/m8a_qwen_*` (synthetic), `inputs/m8d_qwen_*`
  (categories), `inputs/m8c_photos_*` (real photos).
- **Predictions**: `outputs/<config_name>_<ts>_<hash>/predictions.{jsonl,parquet,csv}`.
- **Vision activations**: `outputs/encoder_swap_<model>_<stim>_vision_activations/*.safetensors`.
- **Probe outputs**: `outputs/encoder_swap_<model>_<stim>_probe{,_stim_y}/*.csv`.
- **Aggregated tables**: `outputs/m9_audit/m9_{table1,summary}.csv`,
  `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`.
- **Notebooks**: `notebooks/encoder_saturation_chain.ipynb` (5-model ×
  3-stim chain + §4.2), `notebooks/attention_viz.ipynb` (§4.10).
- **Insight docs**: `docs/insights/*.md` (per-milestone, paired ko/en).
- **Roadmap**: `references/roadmap.md` (status table, hypothesis
  scorecard, additional ideas, change log).

## Pointers to per-milestone insight docs

- M0: stimulus protocol — `docs/stimulus_spec.md`
- M1: pilot — `docs/insights/m1_pilot.md`
- M2: PhysCue grid (H1/H2/H4/H5/H6/H7) — implicit in M3/M4 docs
- M3: encoder boomerang — `docs/insights/m3_encoder_boomerang.md`
- M4: LM logit lens — `docs/insights/m4_logit_lens.md`
- M4b: label-free null — `docs/insights/m4b_label_free.md`
- M4c: FC label-free — `docs/insights/m4c_fc_label_free.md`
- M5a: VTI steering — `docs/insights/m5_vti_steering.md`
- M5a-ext: bidirectional steering — `docs/insights/m5a_ext_bidirection_and_label.md`
- M6 r1 / r2: cross-model — `docs/insights/m6_cross_model_llava.md`,
  `docs/insights/m6_r2_cross_model.md`
- M6 r3 / r4 / r5 / r6: encoder probe chain —
  `docs/insights/m6_r{3,4,5,6}_*.md`
- M8a / M8c / M8d / M8e: stim diversification —
  `docs/insights/m8{a,c,d,e}_*.md`
- §4.5 / §4.5 ext: encoder swap — `docs/insights/encoder_swap_idefics2.md`
- M9: generalization audit — `docs/insights/m9_generalization_audit.md`
- §4.2: reverse prompting — `docs/insights/sec4_2_reverse_prompting.md`
- §4.10: attention viz — `docs/insights/sec4_10_attention_viz.md`
- 5-model synthesis (paper): `docs/insights/encoder_saturation_paper.md`
- This session (2026-04-25): `docs/insights/session_2026-04-25_summary.md`
