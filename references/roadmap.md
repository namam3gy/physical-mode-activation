# ROADMAP — Physical-Mode Activation

> **Role of this document.** Single source of truth for "where we are now and what's next." When starting a new session, **read this file first**. Update §3 every time a milestone completes. Detailed material is linked out to the relevant doc / code.
>
> - Research philosophy & hypotheses (canonical): `references/project.md`
> - **Track B (ICLR/NeurIPS-grade) execution plan**: `references/submission_plan.md` (chosen 2026-04-28; ICLR 2027 primary, NeurIPS 2027 secondary)
> - **Paper gaps & fix-track mapping**: `references/paper_gaps.md` (4 gaps: G1 single-task, G2 sparse non-Qwen, G3 n=1 perceiver, G4 5-sig framing)
> - Architecture: `docs/architecture.md`
> - Stimulus spec: `docs/stimulus_spec.md` / Scoring rubric: `docs/scoring_rubric.md`
> - Run history: `docs/experiments/m{1,2,...}_*.md`
> - Code-level next-step entry points: `docs/next_steps.md`
> - Per-milestone insight deep dives: `docs/insights/m{1,3,4,5}_*.md`

---

## 1. Research definition

### 1.1 Central question

**Above what visual-cue threshold does an open-source VLM stop processing an abstract shape (a circle) as geometry and start processing it as a physical object (a ball)?**

Measured at two layers:

- **Behavior**: PMR (physics-mode priming rate) / GAR (gravity-align rate) / RC (response consistency) of next-state-prediction prompt responses.
- **Mechanism**: linear probe AUC of vision-encoder activations, layer-wise logit-lens trajectories in the LM backbone, causal bottleneck layers / heads revealed by activation patching.

### 1.2 Sub-task structure (per `references/project.md` §2)

| # | Title | Content | Input | Output |
|---|---|---|---|---|
| ST1 | **PhysCue** behavioral thresholds | 4-5-axis factorial stimulus + next-state-prediction prompt | programmatic / photo stimuli | PMR/GAR/RC tables, per-factor curves |
| ST2 | Vision-encoder probing | linear probe + Gandelsman head decomposition + SAE on CLIP / SigLIP / InternViT activations | ST1 captured vision activations | layer×head AUC, monosemantic features |
| ST3 | LM backbone layer-wise emergence | logit lens + per-layer probes at visual-token positions (Neo et al. 2024 recipe) | ST1 captured LM activations | layer×token heatmap, switching-layer |
| ST4 | Causal localization | Semantic Image Pairs + activation patching + VTI steering + SAE intervention | pilot pairs | IE curve, steering vector, head ranking |
| ST5 | Cross-model + prompt-steering | same factorial on LLaVA-1.5 / LLaVA-Next / Qwen2-VL / InternVL2 + prompt steering (Gavrikov et al. 2024) | extended EvalConfig | model-comparison tables, prompt-bias curves |

### 1.3 Hypothesis scorecard

Original H1-H3 from `references/project.md` §2.2 plus H4-H7 derived during the pilot and MVP-full milestones. Pilot evidence in `docs/insights/m1_pilot.md`.

| ID | Hypothesis | Status (post-M5a-ext recheck) | Evidence / next test |
|---|---|---|---|
| **H1** | PMR rises S-shaped along the abstraction axis (line → textured); 3D shading and ground introduction produce the largest step-changes. | **supported, unsaturated-only AND shape-axis-only (cross-shape, M8a + M8d; cross-model M6 r7)** | M2 (Qwen, 2026-04-24): monotone across 4 object_levels (0.744 → 0.832) but saturated. M6 r1 (LLaVA-1.5, 2026-04-25): clean S-curve 0.51 → 0.81. **M6 r7 cross-model on M2 stim (2026-04-26)**: H1 ramp range LLaVA-1.5 +0.30 (cleanest) > LLaVA-Next +0.14 > Idefics2 +0.09 > Qwen +0.05 > InternVL3 +0.02 — unsaturated-only confirmed at 5 model points on the same M2 stim. **M8a (2026-04-25)**: cross-shape strict scoring — Qwen 3/5 (square/triangle fail; ceiling-effect compression), LLaVA 4/5 (polygon fail at filled→shaded inversion). **M8d (2026-04-25)**: cross-category strict scoring — Qwen 0/3 (ceiling), LLaVA 0/3 (non-monotone). H1 is a property of the geometric-shape ↔ named-object axis: every abstraction level of a car/person/bird is already category-recognizable (line car still has wheels), so visual detail doesn't change the affordance — only surface realism. The ramp is operationally measurable only when the vision encoder is unsaturated AND the input is on the abstract-shape ↔ physical-object axis. |
| **H2** | The "ball" label substantially raises PMR even on line drawings → independent contribution of the language prior. | **fully validated, three-point + encoder-anchored** | Qwen (saturated, M4b): ball/planet ≈ 0, circle = −0.065. LLaVA (unsaturated, M6 r1): ball +0.475, planet +0.244, circle +0.173. InternVL3 (super-saturated, M6 r2a): all labels +0.010 ≈ noise. The 3-model paired-delta pattern matches the encoder-saturation prediction. M6 r2b shows the saturation difference is rooted in the vision encoder probe AUC (Qwen 0.99 vs LLaVA 0.73). M4b's "circle suppression only" pattern is the Qwen-specific symptom of the encoder being already saturated. |
| **H3** | Scene inconsistency degrades RC. | **untested** | Axis E was dropped from M2 (complexity); reserved for a focused mini-experiment. RC infrastructure was validated in M2 (103/288 cells with RC<1). |
| **H4** (pilot-derived; **2026-04-28 scorer audit strengthens**) | The open vs forced-choice PMR gap is a stable signature of the **language-prior ↔ visual-evidence** conflict. | **supported (Qwen-only — cross-model FC untested); claim strengthened under v2 scorer** | M2 (Qwen): gap present at every object_level (line 32 pp → textured 22 pp). Higher abstraction ⇒ larger gap. **2026-04-28 scorer audit**: paired open-vs-FC delta on no-label condition widens from −0.131 (v1 scorer) to **−0.180** (v2 scorer, post-`9ec147e` no-motion patterns) — H4 direction preserved, gap larger. M6 r7 cross-model M2 was open-prompt only (no FC for non-Qwen due to LLaVA "A" greedy bias and uncertain Mistral behavior). Cross-model FC remains open. See `docs/insights/scorer_regression_audit_2026-04-28.md`. |
| **H5** (pilot-derived) | The single ground line causes a **larger** PMR shift than going from no-ground textured ball to with-ground textured ball. | **mixed** | M2: bg delta (blank 0.67 → scene 0.88 = +21 pp) > object delta (line 0.74 → textured 0.83 = +9 pp). Direction matches; however, scene also surpasses ground. |
| **H6** (pilot-derived) | The arrow+shadow cue saturation is driven entirely by **cast shadow alone**; the arrow is closer to annotation. | **supported (revised)** | M2 decomposition: cast_shadow alone = +17.5 pp above none (Kersten ground-attachment cue confirmed); **but the arrow alone also saturates at 0.96** — partially refutes the "arrow = annotation" sub-claim. Arrow is the dominant cue, shadow is secondary. |
| **H7** (M2-derived) | The label does not toggle PMR — it selects **which physics regime** to apply. | **supported, unsaturated-only AND cross-category (M8a + M8d) AND 5-model H2 patterns (M6 r7)** | M2 GAR (Qwen): ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2: label flip @ +α=40 swaps B vs A on `line/blank/none`. M6 r1 + r2a cross-model: `planet GAR << ball/circle GAR` holds in Qwen (0.32 vs 0.71/0.75), LLaVA-1.5 (0.07 vs 0.36/0.15), and InternVL3 (0.43 vs 0.82/0.79) — circle-only. **M6 r7 cross-model M2 H2 paired-delta (2026-04-26)**: 3 distinct architecture-conditional patterns of label effect — LLaVA-1.5/LLaVA-Next all positive (classical H2), Qwen/Idefics2 asymmetric (circle/planet < 0, "circle override"), InternVL3 ≈ 0 (saturated). H7 isn't "label always adds PMR" — encoder saturation determines which path. **M8a (2026-04-25)**: cross-shape role-PMR strict scoring — Qwen 1/5 (only square; rest are -0.10 to +0.075 = ceiling-flat), LLaVA 4/5 (triangle fails at +0.025; `wedge` is a weak physical label, not a shape failure). H7-GAR strict: Qwen 1/5, LLaVA 5/5. Orbital-routing dissociation generalizes only when the encoder is unsaturated. **M8d (2026-04-25)**: cross-category role-PMR strict scoring — LLaVA **3/3** (car +0.525, person +0.138, bird +0.550 on PMR_regime physical−abstract; strongest cross-category H7 evidence in the project). Qwen 0/3 binary (ceiling) but regime distribution shows the same pattern at the kinetic-vs-static split: figurine 17.5 % static, statue 22.5 % static (vs ~5 % static for physical labels). The label-selects-regime claim is now category-general, not circle-specific. |
| **H-boomerang** | Encoder knows, decoder gates: vision encoder linearly separates physics-mode classes even where behavior fails. | **Qwen-scoped (revised)** | Holds in Qwen2.5-VL: M3 encoder AUC ~0.99 at every layer; M4 LM AUC ~0.94 at visual tokens; behavioral PMR ~0.93 — small "encoder knows, decoder mildly gates" gap. M5a: causal intervention at L10 flips behavior. **Refuted in LLaVA-1.5** (M6 r2b): vision encoder AUC ~0.73, LM AUC ~0.75, behavioral ~0.78 — flat through pipeline, encoder is the bottleneck. The boomerang as a phenomenon requires encoder saturation. |
| **H-encoder-saturation** (M6 r2-derived; M8c-refined; §4.5-causal; M9-bootstrap; M6 r4 + apples-to-apples M8a-stim; **stim-y check moves locus to architecture-level (encoder + LM)**; **M6 r6 adds 2nd CLIP point**; **M6 r7 + §4.6 cross-model add 5-model M2-stim apples-to-apples + pixel-encodability signature**) | Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the **architecture level** (joint encoder + LM), *not* at encoder representational capacity. **Stim-defined y check (2026-04-25)**: all 5 tested encoders (SigLIP, CLIP-ViT-L ×2, SigLIP-SO400M, InternViT) linearly separate factorial cells at AUC = 1.0 (rendered_vs_line, physics_cell_vs_abstract_cell, within-object-level minimal pairs). Encoder discriminability is uniform; what differs is LM-side consumption of encoder output. The 5-model behavioral PMR ladder (non-CLIP: 0.84–0.92; CLIP-LLaVA-1.5: 0.18; CLIP-LLaVA-Next: 0.70) and the behavioral-y probe AUC (0.77–0.93) reflect each LM's reading of encoder output as physics-mode signal — *downstream-conditional*, not encoder-info. **M9 bootstrap CIs** (3 models × 3 stim sources): non-CLIP CIs [0.80, 0.92] vs CLIP [0.14, 0.37] on synthetic stim — fully separated; on photos all 3 collapse into [0.18, 0.67]. **M6 r6 (2026-04-25)**: 2nd CLIP point (LLaVA-Next, Mistral LM, AnyRes) at PMR 0.70 [0.65, 0.74] rules out CLIP-as-encoder explanation; the LLaVA-1.5 → LLaVA-Next 0.52-PMR jump confounds 4 axes (AnyRes / projector / training / LM family) — consistent with architecture-level reframe but not LM-isolated. | **architecture-level confirmed at 5 model points (3 non-CLIP + 2 CLIP) + bootstrap-validated cross-stim (M9 + M6 r3-r6)** | M6 r2b: Qwen vision AUC 0.99 / behavioral PMR(_nolabel) 0.95; LLaVA-1.5 AUC 0.73 / behavioral 0.38; InternVL3 not captured but behavioral PMR(_nolabel) 0.99 (matches saturation profile). **M8a (2026-04-25)**: cross-shape paired-delta `PMR(physical) − PMR(_nolabel)`: Qwen 5/5 shapes near-zero or negative; LLaVA 5/5 shapes ≥+0.125. **M8d (2026-04-25)**: cross-category paired-delta — Qwen +0.000 / +0.025 / +0.125; LLaVA +0.275 / -0.100 / +0.262. **M9 (2026-04-25)**: 3-model × 3-stim bootstrap CI for mean PMR(_nolabel): non-CLIP [0.800, 0.917] separates from CLIP [0.140, 0.371]; photos converge to [0.183, 0.667]. **M6 r3 (2026-04-25)**: Idefics2 SigLIP-SO400M probe AUC 0.93 (mean 4 layers). **M6 r4 (2026-04-25)**: InternVL3 InternViT probe AUC 0.89 / PMR 0.92 — 4-point chain non-CLIP-generalizes the saturation. **M6 r6 (2026-04-25)**: LLaVA-Next CLIP-ViT-L behavioral-y AUC 0.81, stim-y AUC 1.0, PMR 0.70 [0.65, 0.74] — 5-model chain locked. |
| **H-LM-modulation** (M9-derived, 2026-04-25) | At encoder saturation, LM family may modulate residual H7 measurability — Mistral-7B (Idefics2) shows H7 mean +0.048 [+0.000, +0.094] on M8d vs Qwen2-7B mean +0.008 [−0.033, +0.052] on the same stim. | **suggested only — CIs touch on M8d, fully overlap on M8a/M8c** | M9 bootstrap (5000 iters × 9 cells): Idefics2 M8d H7 CI just touches 0; Qwen M8d H7 CI crosses 0. The 33-pt PASS-rate gap (0.667 vs 0.333) is driven by a single shape (`car`: +0.025 vs +0.094) crossing the strict threshold. Not paper-defensible from current data; needs same-encoder LM swap or 3–5× more shapes. |
| **H-locus** (M4-derived; revised 2026-04-26 evening via M5b SIP; refined 2026-04-26 night via layer-level knockout; **further confirmed 2026-04-27 via per-head knockout**; **2026-04-28 cross-model M5a steering — 3 of 4 testable models confirm LM mid-late layer locus**) | The bottleneck is at LM mid layers, with model-specific layer (Qwen L10 / LLaVA-Next L20-L25 / Idefics2 L25) — not earlier or in the decoding head. | **supported, triply-refined and cross-model confirmed** | M5a (Qwen): L10 α=40 flips 10/10 abstract → physical responses; later layers do not move. M5a-ext Exp 3: L10 regime-flip holds across cells. M5b SIP patching (sufficiency): patching corrupted's L0-L9 with clean's full visual-token hidden state recovers physics-mode in 20/20 pairs (IE=+1.0); L10-L11 60% recovery; L14+ zero. M5b layer-level attention/MLP knockout (necessity): L9 MLP zero → 20/20 flip (IE=+1.0); attention knockout 0/20 break at every L0-L27. **M5b per-head attention knockout (2026-04-27, this round)**: 20 stim × 7 layers (L8-L14) × 28 heads = **196 (L,h) ablations all show IE = 0** — every single head is dispensable. Partial MLP necessity ring (L8 +0.4, L10 +0.6, L11 +0.4, L14 +0.4) preserved. Triangulation: L9 is **sufficient (SIP) AND necessary (knockout)** — L9's MLP *constructs* physics-mode representation, L10 *reads* it out via redundant attention (no specific head matters); off-by-one M5a/M5b reconciliation = two views of same decision boundary. H10 ("2-3 narrow IE bands") fully resolved: 1 dominant MLP band at L9 + 4 partial echoes; attention has *zero* narrow IE bands at *any* resolution. The mechanism is construction-and-broadcast, not pull-through-a-specific-head. |
| **H-direction-bidirectional** (M5a-ext, 2026-04-24; revised 2026-04-25) | `v_L10` is a simple bidirectional concept axis where −α suppresses physics-mode back to abstract. | **revised — regime axis within physics-mode** | Exp 1 (textured/ground/both ceiling): −α has no effect → initially framed as "one-way activator". Exp 3 (textured/blank/none moderate baseline, 2026-04-25): −α=40 flips D → B ("stays still") uniformly across (line, textured) × (ball, circle). Both signs of α activate physics-mode; sign selects regime (+kinetic / −static). Baseline D sits *below* the \|α\| threshold, not at one end of the axis. |
| **H-regime** (M5a-derived) | The steering direction is binary "object-ness", not "which physics" — physics regime is label-driven. | **refuted in current form** | Replaced by H-direction-bidirectional's regime-axis interpretation (kinetic vs static is already a regime distinction that the steering sign selects, label-independent at |α|=40). Separately, label *does* select regime in the narrow `line/blank/none × +α=40` case (Exp 2), but not globally — this is now folded into the H7 qualifier. |
| **H-direction-specificity** (§4.6-derived; revisions 2026-04-26 + 2026-04-27; **5-model n=10 sweep 2026-04-28**) | Pixel-space gradient ascent along `v_L` flips PMR; matched-magnitude random directions do not — the regime flip requires directional specificity, not perturbation magnitude. | **supported across 5 architectures × 5 layers at n=10** | **5-model 2026-04-28 layer sweep at ε=0.1**: aggregate random rate 1/250 (5 models × 5 layers × 10 trials = 250 random trials; **24 of 25 (model, layer) random-control cells = 0/10**; only Qwen L10 random hit 1/10 = 10 %, far below the v_unit 10/10 at the same layer — random does not approach v_unit performance anywhere). Qwen all 5 layers ≥ 80 %, LLaVA-Next L20+L25 100 %, LLaVA-1.5 L25 40 %; v_L projection rises consistently for Idefics2 + InternVL3 too (gradient ascent works) but those models don't behaviorally flip — direction-specificity is preserved at the *projection* level even where it doesn't translate to PMR. |
| **H-shortcut** (§4.6-derived; revisions 2026-04-26 + 2026-04-27 afternoon; 5-model n=10 sweep 2026-04-27 night; **2026-04-28 Idefics2 L26-31 deeper-layer disambiguation**) | Shortcut interpretation is encodable in the image itself — pixel-driven. | **supported, architecture-conditional with model-specific shortcut profiles; encoder saturation alone is ruled out, and a projector-design candidate (MLP vs perceiver) emerges from the cross-model pattern — Idefics2's perceiver-resampler is the leading remaining mechanism candidate, not isolated since the 5-model design varies encoder + projector + AnyRes simultaneously (controlled projector-swap test out of scope)** | **§4.6 5-model n=10 sweep (2026-04-28)** + Idefics2 9-layer extension: each architecture has its own shortcut LM layer profile. **Qwen2.5-VL** (SigLIP+Qwen2-7B, AUC 0.99): broad shortcuts at L5/10/15/20/25 (Wilson lower bounds 0.49–0.72). **LLaVA-Next** (CLIP+AnyRes+Mistral-7B, AUC 0.81): L20+L25 (10/10 each). **LLaVA-1.5** (CLIP+Vicuna-7B, AUC 0.73): L25 only (4/10, weaker than n=5 morning suggested). **Idefics2** (SigLIP-SO400M + perceiver-resampler + Mistral-7B, AUC 0.93): **0 clean shortcuts across 9 LM layers (L5/10/15/20/25 + L26/28/30/31, 16-97 % depth)** despite v_L projection rising +28 to +163 cleanly. 2026-04-28 deeper-layer disambiguation rules out "wrong-relative-depth" hypothesis. **Perceiver-resampler is the leading remaining mechanism candidate** (Idefics2 differs from the MLP-projector models on encoder + projector + AnyRes simultaneously, so the 5-model design doesn't isolate projector — controlled projector-swap test would be needed for a clean causal claim). **InternVL3** (InternViT+InternLM3, AUC 0.89): protocol-saturated (baseline_pmr=1.0). Aggregate random rate 1/250 trials across the 25 (model × layer) cells tested (24 of 25 random-control cells = 0/10; the Qwen L10 random 1/10 is the only non-zero cell and is well below the v_unit 10/10 at the same layer — random does not approach v_unit performance anywhere). Pixel-encodability remains an *empirical property of the encoder→LM pipeline*, third architecture-level signature alongside M9 PMR-ceiling and §4.7 decision-stability — but the strict "capacity scales with saturation" claim is downgraded from "law" to "pattern within CLIP/SigLIP+Qwen subset". Insight: `docs/insights/sec4_6_cross_model_revised.md` (+ ko). |

### 1.4 Target models & venue

- **Round 1 (pilot / MVP-full)**: Qwen/Qwen2.5-VL-7B-Instruct — proven loader, 15 GB, 1.0 it/s on H200.
- **ST5 extension** (✅ 2026-04-28, 5-model chain locked): LLaVA-1.5-7B, LLaVA-Next-Mistral-7B, Idefics2-8B (SigLIP-SO400M + Mistral-7B + perceiver-resampler), InternVL3-8B-hf.
- **Venue (revised 2026-04-28, Track B chosen)**: **ICLR 2027 primary** (deadline ~late Sep 2026, ~5 months); **NeurIPS 2027 secondary** (deadline ~mid May 2027). EMNLP 2026 (~June) deferred — venue audience preference for ICLR/NeurIPS reviewer pool. TMLR rolling as final fallback. See `references/submission_plan.md` for the Track B 20-week schedule.
- **Paper framing**: production-VLM **world-model commitment** mechanism, framed at **3 Marr levels** (Computational → PMR; Representational → M3+M4 probes; Mechanistic → M5a+M5b interventions). Robust-not-flashy per user (2026-04-28). Connect to V-JEPA, RT-2, OpenVLA in §1 + §9.

---

## 2. Milestone overview

| # | Milestone | Scope | Status | Completed |
|---|---|---|---|---|
| M0 | Infrastructure scaffold | Package layout, configs, scripts, tests, base docs set | ✅ | 2026-04-24 |
| M1 | **ST1 Pilot** (Qwen2.5-VL-7B) | 240 stim × 2 prompts = 480 inferences; first behavioral S-curve measurement | ✅ | 2026-04-24 |
| M2 | **ST1 MVP-full** (incorporating pilot lessons) | axis C decomposition, axis D expansion, T=0.7, LM hidden-state capture, 2880 inferences (Qwen2.5-VL only). Cross-model extension done in **M6 r7** (2026-04-26) — see below. | ✅ | 2026-04-24 |
| **M6 r7** | **M2 cross-model — 5-model M2-stim apples-to-apples** | LLaVA-Next + Idefics2 + InternVL3 added on M2 stim with LM activation captures (LLaVA-1.5 + Qwen existing). 5-model PMR(_nolabel) ladder: LLaVA-1.5 0.18 / LLaVA-Next 0.79 / Qwen 0.94 / Idefics2 0.97 / InternVL3 0.99. H1 ramp clean only on LLaVA-1.5 (+0.30 range). H2 paired-delta shows 3 distinct patterns (LLaVA = positive, Qwen/Idefics2 = circle override, InternVL3 = ≈0). Per-model v_L10 extracted; saturated models (LLaVA-Next/Idefics2/InternVL3) class-imbalanced (n_neg = 9/5/1 too few for clean v_L10). | ✅ | 2026-04-26 |
| **§4.6 cross-model** | **Pixel-encodability cross-model layer sweep (5 architectures × n=10) + Idefics2 9-layer disambiguation** | 5 rounds: 2026-04-26 morning (Qwen+transfer test) → 2026-04-26 overnight (LLaVA-1.5 layer sweep) → 2026-04-27 afternoon (LLaVA-Next + Qwen layer sweep) → 2026-04-27 night 5-model n=10 chain (Idefics2 + InternVL3 added) → **2026-04-28 Idefics2 L26-31 deeper-layer disambiguation** (new M2 capture at L26/28/30/31, fresh v_L extraction, 80-run sweep). Final picture: Qwen broad (5 shortcut layers, all 80 % +), LLaVA-Next at L20+L25 (10/10), LLaVA-1.5 at L25 only (40 % at n=10), **Idefics2 anomaly resolved** (0 clean shortcuts across 9 layers L5-L31 = 16-97 % depth → wrong-relative-depth falsified, perceiver-resampler bottleneck remaining), InternVL3 untestable (baseline=1.0). H-shortcut → architecture-conditional with **projector design (MLP vs perceiver) as the disambiguating axis**, not encoder saturation alone. | ✅ | 2026-04-28 |
| **§4.8** | **Qwen 7B vs 32B PMR scaling on M2 (open prompt)** | 32B M2 inference (1440 stim × 1 prompt = 16 min wall on H200, single-GPU bf16). **Result: aggregate PMR 0.926 ≈ 7B 0.931** — 5× scaling does not move overall PMR (MechBench-style "scale doesn't fix grounding" supported). Only notable shifts: `abstract_reject` 35× higher in 32B (0.002 → 0.065, mostly on cue=none cells); H2 label gap halves (`ball − circle` 7B +0.071 → 32B +0.010). 32B is more cue-sensitive but the language-prior dominance regime survives. Insight: `docs/insights/sec4_8_pmr_scaling.md` (+ ko). | ✅ | 2026-04-28 |
| M3 | **ST2 — Vision encoder probing** | Vision-block capture (8 layers, 12 GB) + layer-wise linear probes. **Boomerang confirmed**: encoder AUC = 1.0 on every axis; behavioral PMR 0.28-0.95. | ✅ | 2026-04-24 |
| M4 | **ST3 — LM logit lens / layer-wise probe** | LM hidden @ visual tokens AUC 0.94-0.95 across all probed layers; L20 peak. Label prior drives physics margin from L5; object_level effect is 7× smaller. | ✅ | 2026-04-24 |
| **M4 cross-model** | **5-model × 5-layer LM probe AUC** (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 — reuses existing M2 cross-model captures, no new inference) | LM probe AUC ladder aligns with M3 vision AUC ladder: Idefics2 0.995 / Qwen 0.96 / LLaVA-Next 0.78 / LLaVA-1.5 0.76 / InternVL3 untestable (n_abs=1). **Idefics2 LM AUC > vision AUC (0.995 vs 0.93)** — perceiver-resampler does not strip information. Combined with §4.6 Idefics2 0/9 layers shortcut → **information presence ≠ pixel-space shortcut routability** dissociation. H-encoder-saturation gains 2nd downstream signature (LM probe AUC). Insight: `docs/insights/m4_lm_probing_cross_model.md`. | ✅ | 2026-04-28 |
| M5a | **ST4 Phase 1+2 — VTI steering** | Direction extraction + residual-stream injection. **L10 α=40 flips 10/10 D → B** — "physical object-ness" direction causally confirmed. | ✅ | 2026-04-24 |
| **M5a cross-model** | **Runtime steering on 4 testable models** (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2; InternVL3 baseline=1 untestable). Reuses per-model v_L from `m2_extract_per_model_steering.py`. | **3 of 4 testable models flip 10/10**: Qwen L10 α=40 (existing); **LLaVA-Next L20 α=10 / L25 α=15-20** (`line_blank_both` baseline 0/10 → 10/10); **Idefics2 L25 α=20** (`line_blank_none` 0/10 → 10/10, "The tip of the arrow will hit the center of the circle."). LLaVA-1.5 L25 0/10 at every α=0-60 (encoder bottleneck, replicates §4.6 weak-shortcut). **Triangulation with M4 cross-model + §4.6**: Idefics2 LM probe AUC 0.995 + runtime steering 10/10 + §4.6 pixel-flip 0/9 layers → **perceiver-resampler removes pixel-space gradient routability, not LM-side information** (forward pathway works, inverse pixel→v_L pathway blocked). α dynamic range model-specific (Qwen 40, LLaVA-Next 5-15, Idefics2 20). Insight: `docs/insights/m5a_cross_model.md`. Patch: `scripts/06_vti_steering.py` adds `text_model.layers` fallback for Idefics2/Mistral path. | ✅ | 2026-04-28 |
| M5a-ext | **VTI follow-ups (neg α, label swap, bidirectionality recheck)** | Exp 1-2 (2026-04-24): neg α at ceiling + label=ball side-by-side. Exp 3 (2026-04-25): (α × label × obj) grid on moderate baseline. **Key result**: `v_L10` is a regime axis within physics-mode — +α → A (falls), −α → B (stays still), baseline D below threshold. | ✅ | 2026-04-25 |
| M4b | **Label-free prompt — H2 null test** | `open_no_label` variant on M2 stimuli. **Key result**: `ball` ≈ no-label; `circle` suppresses PMR by 6.5 pp. Original H2 reframed: language prior is asymmetric — circle override, not ball enhancement. M4 visual-token capture is prompt-independent (structural artefact). | ✅ | 2026-04-25 |
| M6 r1 | **ST5 round 1 — LLaVA-1.5-7B cross-model** | M2 + M4b protocol on LLaVA-1.5-7B. **Key result**: M4b's "circle suppression" is **Qwen-specific** — LLaVA shows the *original* H2 (ball +47.5 pp, all labels positive vs no-label baseline). New unified hypothesis: language prior is positive across labels; Qwen's visual saturation masked the positive contribution. H7 cross-model replicates (planet GAR << ball GAR in both). LLaVA gives the cleanest H1 S-curve in the project. FC excluded (LLaVA returns "A" for every cell). | ✅ | 2026-04-25 |
| M4c | **Forced-choice label-free** | New `forced_choice_no_label` variant (FC with "the depicted object" antecedent). Qwen reproduces M4b's H2 pattern under FC and adds a planet-suppression effect (option-set bias: orbital regime collapses to D). LLaVA's "A" bias persists under re-template (477/480), confirming model-level pathology. | ✅ | 2026-04-25 |
| M6 r2 | **Cross-model round 2 (3-model + LLaVA captures + FC logit ratio)** | r2a: InternVL3-8B-hf cross-model behavioral; r2b: LLaVA-1.5 activation captures + cross-model M3/M4 probing; r2c: FC first-token logit-ratio scoring on all FC runs. **Key result**: visual-saturation hypothesis fully validated 3-of-3 models; rooted in vision encoder probe AUC (Qwen 0.99, LLaVA 0.73). H-boomerang revised to Qwen-scoped. New **H-encoder-saturation** hypothesis. LLaVA "A" bias is logit-level (not greedy-level). | ✅ | 2026-04-25 |
| **M6 r3** | **Idefics2 vision-encoder probe (closes AUC↔PMR chain)** | 400 M8a stim × 4 SigLIP-SO400M layers (88 s capture), per-layer logistic-regression probe on per-stim PMR. **AUC 0.93 mean (peak 0.948 at layer 9)** — SigLIP cluster confirmed at the encoder-family level (Qwen 0.99 + Idefics2 0.93 vs LLaVA 0.73). H-encoder-saturation chain `encoder family → AUC → PMR → H7` now closed at 3 model points at the mechanism level. | ✅ | 2026-04-25 |
| **M6 r4** | **InternVL3 vision-encoder probe + Qwen/LLaVA M8a re-capture (4-model chain on identical stim, non-CLIP-general)** | M8a inference (1200+400 in 12 min) + 400 stim × 4 InternViT layers (47 s capture) + per-layer logistic-regression probe. **InternVL3 AUC = 0.89 / PMR(_nolabel) = 0.92.** Round also re-captured Qwen + LLaVA on M8a stim for an apples-to-apples 4-model AUC table. **4-point M8a chain: Qwen 0.88/0.84, LLaVA 0.77/0.18, Idefics2 0.93/0.88, InternVL3 0.89/0.92.** 3 distinct non-CLIP encoder families (SigLIP, SigLIP-SO400M, InternViT) cluster at AUC ≥ 0.88; only CLIP-ViT-L falls to 0.77. Paper claim generalizes from "SigLIP saturates" to "non-CLIP encoders saturate; CLIP doesn't (in this sample)". Required vlm_runner fix for InternVL3's vision_tower.encoder.layer (singular) attribute. **Stim-y check (added late round)**: all 4 encoders separate stim-defined factorial cells at AUC = 1.0; encoder discriminability is uniform across families. Reframes the chain to architecture-level (encoder + LM fusion), not encoder-discriminability. | ✅ | 2026-04-25 |
| **M6 r5** | **M8c photo encoder probe (4-model cross-stim)** | InternVL3 M8c inference (180+60 in 2 min) + 4 model × 60 photo captures (~3 min) + behavioral-y + stim-y probes. **Behavioral-y AUC inverts cross-stim**: Qwen 0.88→0.44, LLaVA 0.77→0.86, Idefics2 0.93→0.77, InternVL3 0.89→0.59. **Stim-y AUC stays at 1.0 for all 4 models** on photos (physical-shape vs abstract-shape). Confirms cross-stim that encoder discriminability is uniform; behavioral-y AUC is a "encoder ↔ behavior alignment" measure that varies with each model's per-stim PMR distribution. Final cross-stim confirmation of the architecture-level reframe. | ✅ | 2026-04-25 |
| **M6 r6** | **LLaVA-Next-Mistral 5th model point (2nd CLIP) + cross-stim addendum** | M8a (400 labeled + 400 label-free + 400 stim × 4 layers × 5 tile capture). PMR(_nolabel) = **0.700, 95% CI [0.65, 0.74]**, sitting between LLaVA-1.5 floor [0.14, 0.21] and saturated cluster [0.80, 0.92]. Behavioral-y AUC 0.81; stim-y AUC = 1.0 across all 4 targets. **5-model M8a chain locked**. **Cross-stim** (1440 + 480 M8d + 180 + 60 M8c, ~16 min): M8d PMR 0.625 [0.58, 0.67] preserves mid-band; M8c PMR 0.417 statistically equal to Idefics2 0.417 (photo-collapse generalizes). **H7 cross-stim**: M8a +0.26 (5/5 PASS), M8d −0.05 (CI [−0.10, −0.01], noise floor), M8c +0.02 — same-encoder-family architecture switch attenuates H7 even with PMR headroom (M8d PMR 0.625 well below ceiling). H-encoder-saturation **architecture-level confirmed at 5 model points + 2 CLIP points + 3 stim sources**. H-LM-modulation still suggested-only (two-Mistral H7 ≈ 0 on M8d is multi-axis-confounded). | ✅ | 2026-04-25 |
| **M8a** | **Stimulus diversification — non-circle synthetic shapes** | Square / triangle / hexagon / irregular polygon × line/filled/shaded/textured × bg/cue grid; Qwen + LLaVA, labeled + label-free arms. **Strict pre-registration scoring: Qwen 1/4, LLaVA 4/4** — the asymmetry validates H-encoder-saturation cross-shape. H1 + H7 are unsaturated-only. Triangle (`wedge`) + polygon (`polygon`) exposed as label-design weak points. | ✅ | 2026-04-25 |
| **M8d** | **Stimulus diversification — non-ball physical object categories** | car / person / bird × line/filled/shaded/textured × bg/cue × `(fall, horizontal)` × 5 seeds. **Strict pre-registration scoring: Qwen 0/3 H7 (binary, ceiling), LLaVA 3/3 H7 ✓ — strongest cross-category H7 evidence in the project.** Underneath Qwen ceiling, regime distribution shows 17.5 % static (figurine) / 22.5 % static (statue), validating H7 at the kinetic-vs-static split. H1 fails on both (shape-specific axis). H-encoder-saturation cross-validated cross-category. New `classify_regime` keyword classifier (5.6 % hand-annotation error). | ✅ | 2026-04-25 |
| **M8c** | **Stimulus diversification — real photographs** | 60 photos (12 × {ball, car, person, bird, abstract}) from COCO 2017 + WikiArt. **Key result**: photos REDUCE Qwen PMR(_nolabel) by 18-48 pp across categories — visual-saturation hypothesis refined: behavioral PMR saturation requires both encoder representational confidence AND input-context simplicity. LLaVA H7 partially holds (2/4 binary). LLaVA person photo PMR rises +39 pp vs synthetic (encoder finally recognizes humans). | ✅ | 2026-04-25 |
| **4.5** | **Cross-encoder swap (CLIP / SigLIP / DINOv2)** | Idefics2-8b (SigLIP-SO400M + Mistral-7B) added as third point: PMR(_nolabel) = **0.882** (ceiling, matches Qwen 0.838). LLaVA (CLIP) = 0.175. **H-encoder-saturation causally confirmed at the encoder-family level** — encoder type (SigLIP vs CLIP) drives the PMR ceiling regime regardless of LM (Qwen2-7B / Mistral-7B). | ✅ | 2026-04-25 |
| **4.6** | **Counterfactual stimulus generation via VTI reverse** | Pixel-space gradient ascent on Qwen2.5-VL post-processor `pixel_values` maximizing `<h_L10[visual], v_L10>`. **5/5 v_L10 flips at ε = 0.05; 0/15 random-direction flips** at matched ε = 0.1 — directional specificity falsifies "any perturbation" alternative. Random-control responses surfaced an over-permissive scorer (asymmetric fix verified: 0/20 v_L10 hits new abstract markers). v_L10 is encodable in the image; the M5a shortcut is on the pixel-driven path. | ✅ | 2026-04-26 |
| **4.10** | **Attention visualization UI** | Initial Qwen2.5-VL release (notebook + heatmap overlay) + cross-model extension to all 5 VLMs on the same M8a stim. Last-token attention to visual tokens shows architecture-level differences (Qwen ~17%, LLaVA-1.5 ~7%, Idefics2 ~30%, ...). | ✅ | 2026-04-25 |
| M5b | ST4 Phase 3 — SIP + patching + knockout + SAE | (1) SIP + activation patching on Qwen2.5-VL: 20 SIP pairs × 28 LM layers — sharp L10 boundary, IE=+1.0 at L0-L9, +0.6 at L10-L11, 0 at L14+. (2) Cross-model SIP+patching on LLaVA-1.5 (n=15 × 32 layers): lock-in starts at L20 (62.5% relative depth) vs Qwen L10 (36%) — curve shape replicates, locus shifts. (3) Layer-level attention + MLP knockout (necessity): attention IE=0 at every layer; **L9 MLP IE=+1.0 (uniquely necessary)**; partial ring L8 +0.4 / L10 +0.6 / L11 +0.4 / L14 +0.4. (4) Per-head attention knockout (2026-04-27): 20 stim × 7 layers × 28 heads = 196 (L,h) all IE=0 — confirms attention fully redundant at *both* layer and head resolution. (5) **SAE intervention on Qwen vision encoder L31 (2026-04-27, two rounds)**: 5120-feature SAE trained on 622K visual tokens. **Morning (delta-rank)**: top-20 physics-cue features fully break PMR (0/20); 1 mass-matched random k=20 leaves PMR intact (20/20). **Evening revision (Cohen's d rank + dense k + 3 random)**: feature 3313 (delta-rank 3) drops to ~rank 50 by Cohen's d (high-baseline outlier filter); top-20 has 7/20 turnover; Cohen's d is canonical. Cohen's-d top-30 ablation **breaks all 20 stim (Wilson CI [0.00, 0.16])**; **3 mass-matched random k=30 controls** (mass 72/76/102 % of top-30) **all 20/20 in physics-mode (Wilson CI [0.84, 1.00])**. (6) **Cross-model SAE intervention (2026-04-28)**: per-model SAE retrain at the **actually-consumed** vision-encoder layer (LLaVA `vision_feature_layer=-2` → layer 22; Idefics2 `last_hidden_state` → layer 26; InternVL3 `-1` → layer 23; Qwen layer 31). Top-k ablation (OPEN prompt + PMR scoring). **3 of 5 models break PMR cleanly**: Qwen k=20 (0.4 % of features), Idefics2 k=160 (3.5 %), InternVL3 k=160 (3.9 %). **2 LLaVA-family models NULL** at any k ≤ 160 — encoder-side SAE features absent or too distributed. Random controls all 1.0 (specificity confirmed). Effect concentration tracks M3 vision-encoder probe AUC (Qwen 0.99 > Idefics2 0.93 > InternVL3 0.89 > LLaVA 0.7-0.8). LLaVA-Next M5a positive (10/10 LM-side flip) + M5b NULL → physics-mode commitment routes through LM, not encoder. Triangulation closes: encoder ~30 SAE features (Qwen) → L0-L9 visual tokens → L9 MLP construction → L10 read-out → letter. | complete ✅ (all 6 sub-tasks done) | 2026-04-26..28 |
| M6 r3+ | ST5 round 3+ — encoder counterfactuals + LLaVA-Next | LLaVA-Next, InternVL3 captures, scale variants (Qwen 32B/72B), other VLM families (Pixtral / Phi-V). | optional | — |
| **M9** | **Generalization audit — paper Table 1 (3 models × 3 stim sources, bootstrap CIs)** | Consolidates M8a (5 shapes) + M8d (3 categories) + M8c (5 photo categories) × {Qwen, LLaVA, Idefics2} into 9 (model × stim) cells with 95% bootstrap CIs on mean PMR(_nolabel) and mean H7 delta. **Headlines**: (1) encoder family causally drives synthetic-stim PMR ceiling (CIs fully separate, 0.84–0.89 vs 0.18–0.33); (2) photos compress encoder gap (all 3 models converge to 0.28–0.55); (3) H7 robust only in unsaturated regime (LLaVA M8a + M8d, CIs > 0); (4) LM-modulation of H7 at saturation suggestive only (Idefics2 M8d CI touches 0). | ✅ | 2026-04-25 |
| M7 | Human baseline + paper writing | Prolific 20 raters × 50 stimuli + ICLR/NeurIPS draft (per `references/submission_plan.md`). Human baseline deferred 2026-04-28; re-evaluate at week 14. | deferred | — |
| **M-MP** (Pillar A) | **Multi-prompt cross-task generalization** | Track B Pillar A (`references/submission_plan.md`). 5-model M2 stim × 3 prompts (kinetic prediction / free-form description / meta-categorization). Phase 1 (smoke n=48) + Phase 2 (full n=480 × 3 labels × 3 prompts = 4320/model) ✅ complete 2026-04-28. **Headline: H2 paired-delta (ball−circle) positive in all 5 models × 3 prompts = 15 cells (range +0.006 to +0.344)**; saturation × prompt interaction (saturated → describe most informative; unsaturated → open most informative); 0/2160 unparseable yes/no; Phase 1↔2 ±0.018. Phase 3 (cross-prompt M5a + M5b on Qwen + Idefics2) → next up. | Phase 1+2 ✅, Phase 3 next | 2026-04-28 (P1+P2) |
| **M-PSwap** (Pillar B) | **Controlled projector-swap LoRA on Idefics2** | Track B Pillar B / G3 fix. Replace Idefics2 perceiver-resampler with MLP projector (LoRA rank-32, encoder + LM held fixed). Re-run §4.6 + M5b. **Predicted**: if perceiver-resampler is causal, Idefics2-MLP flips on §4.6 like LLaVA-Next does. 1-day feasibility spike first. Fallback: B2 only + literature-grounded theoretical claim. | planned | week 4–5 |
| **M-LMSwap** (Pillar B) | **Controlled LM-only-swap LoRA (CLIP+Vicuna vs CLIP+Mistral)** | Track B Pillar B / G3 fix. Pair CLIP-ViT-L with Vicuna-7B and Mistral-7B holding encoder fixed. Re-run M5a + M5b. **Predicted**: M5b NULL persists in both LM variants (encoder is the bottleneck). | planned | week 6–7 |
| **M-Marr** (Pillar C) | **Marr-3-level paper restructure** | Track B Pillar C / G4 fix. §6 of `docs/paper/draft_v1.md` reorganized into 3 levels (Computational PMR / Representational M3+M4 / Mechanistic M5a+M5b). §1 + §9 + §10 rewritten with world-model framing. No new experiments. | planned | week 9–11 |

---

## 3. Detailed status

### M0 — Infrastructure scaffold ✅ (2026-04-24)

Completed:
- `src/physical_mode/` modules (config, utils, stimuli, models, inference, metrics, probing scaffold).
- `scripts/0{1,2,3}_*.py` argparse runners.
- `configs/{pilot,mvp_full}.py` — config-as-code.
- `tests/` — 35 tests (stimulus determinism + PMR scoring regression).
- `docs/` (architecture / stimulus_spec / scoring_rubric / run-log / next-steps / insights).
- `notebooks/demo.ipynb` — 32-cell walkthrough with cached outputs.
- `CLAUDE.md`, `README.md`, `pyproject.toml` (cu130 index), `.gitignore`, `uv.lock`.
- Project repo: private https://github.com/namam3gy/physical-mode-activation.

Success criteria (all met):
- `uv sync` succeeds; `uv run python -m pytest` passes.
- `scripts/01_generate_stimuli.py --config configs/pilot.py --limit 10` succeeds.
- Pilot inference + score pipeline runs end-to-end.

### M1 — ST1 Pilot (Qwen2.5-VL-7B) ✅ (2026-04-24)

Run: `uv run python scripts/02_run_inference.py --config configs/pilot.py`.
Output: `outputs/pilot_20260424-072418_2c16efb6/` — 480 predictions, 8 min wall clock.

**Headline findings** (`docs/insights/m1_pilot.md` §2, §3):

| Observation | Number | Implication |
|---|---|---|
| Ground effect | blank 0.49 → ground 0.85 (+36 pp) | **Largest single-factor effect.** Ground is the cheapest, strongest physics trigger. |
| Abstraction endpoints | line 0.58 → textured 0.81 | H1 partially supported; middle two levels tied. |
| Arrow + shadow cue | 1.000 (saturated) | Unmeasurable cell — needs decomposition in MVP-full. |
| Wind cue | 0.513 ≈ none 0.500 | Invisible to the VLM — needs replacement. |
| Open vs forced-choice | PMR 0.80 vs 0.54, abstract_reject 0.00 vs 0.45 | **Language-prior dominance** — H2 strongly supported. |

**Hypothesis scoring**: H1 partial, H2 strong, H3 untested, H4-H6 candidates derived.

**Validated infrastructure properties**:
- `PhysModeVLM` works on Qwen2.5-VL with the generic `AutoModelForImageTextToText` loader. ST5 model swap is a config-only change.
- `predictions.jsonl` streaming flush is crash-safe.
- Among factorial axes, **event_template** has no behavioral output effect → downgrade in MVP-full.

### M2 — ST1 MVP-full ✅ (2026-04-24)

Run: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`.
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/` — 2880 predictions, 55 min wall clock, 5.2 GB of LM activations.

**Success-criteria results** (details in `docs/experiments/m2_mvp_full.md`):

| Criterion | Status |
|---|---|
| Monotone S-curve over object_level (forced-choice) | ✅ 0.583 < 0.647 < 0.711 < 0.714 |
| Open-vs-forced gap at every object_level | ✅ 22-32 pp, positive correlation with abstraction |
| cast_shadow alone > none + 20 pp | ✅ +17.5 pp on average (+23 in the blank-bg condition) |
| RC < 1 cells exist (T>0 verification) | ✅ 103/288 (36 %) cells with RC<1 |
| `outputs/*/activations/` populated | ✅ 480 safetensors, 5 layers, bf16 hidden states |

**New headlines**:
1. Abstraction-axis monotone S-curve cleanly confirmed (H1).
2. Label selects the physics regime — same image with `circle/ball/planet` → static / rolls / orbits the Sun (H7, new).
3. cast_shadow alone = +17.5 pp; arrow is the dominant cue (H6 revised).
4. Open-ended PMR 0.93, abstract_reject 0.002 (3/1440) — language-prior dominance reconfirmed and extended (H4).

**Blocking-issue resolution**:
- The motion-trail primitive ended up unused (the axis-C redesign replaced wind entirely).
- Axis E (scene consistency) was dropped from M2; deferred to a focused mini-experiment in `docs/next_steps.md`.
- The `capture_lm_attentions=False` flag was the right call — disk usage is ⅓ of the alternative (5.2 GB vs 15+ GB with attentions on).

### M2 cross-model addendum (M6 r7) ✅ (2026-04-26)

The original M2 was Qwen-only. The **M6 r7** milestone (2026-04-26)
extends the M2 protocol to 4 more models with LM activation captures —
making the full 5-model M2-stim chain apples-to-apples. Pre-existing
LLaVA-1.5 (M6 r1 + M6 r2b capture) and InternVL3 (M6 r2a, no capture)
are augmented here by LLaVA-Next + Idefics2 + InternVL3 *capture*
re-runs.

**5-model PMR(_nolabel) ladder** (480-stim mean ± 95% CI):

| Model | PMR(_nolabel) | 95% CI |
|---|---:|---|
| LLaVA-1.5 | 0.383 | [0.34, 0.43] |
| LLaVA-Next | 0.790 | [0.75, 0.83] |
| Qwen2.5-VL | 0.938 | [0.92, 0.96] |
| Idefics2 | 0.967 | [0.95, 0.98] |
| InternVL3 | 0.988 | [0.98, 1.00] |

**H1 ramp** (line→textured range): LLaVA-1.5 +0.30 (cleanest);
LLaVA-Next +0.14; Qwen +0.05; Idefics2 +0.09; InternVL3 +0.02.
Confirms unsaturated-only reading.

**H2 paired-delta** (`PMR(label) − PMR(_nolabel)`): three distinct
patterns matching encoder-saturation:
- LLaVA-1.5/LLaVA-Next: all positive (classical H2; ball Δ = +0.475
  / +0.190).
- Qwen/Idefics2: asymmetric — circle / planet Δ < 0 ("circle
  override").
- InternVL3: ≈ 0 (fully saturated).

**Per-model v_L10 extraction** (class-mean diff `mean(h_L10|PMR=1) −
mean(h_L10|PMR=0)`): only LLaVA-1.5 has class-balanced n_neg=105;
LLaVA-Next/Idefics2/InternVL3 have n_neg = 9/5/1 — too saturated on
M2 stim for a clean direction. §4.6 cross-model is therefore feasible
on LLaVA-1.5 only from M2-stim alone.

**Insight doc**: `docs/insights/m2_cross_model.md` (+ ko).
**Figures**: `docs/figures/m2_cross_model_{pmr_ladder,h1_ramp,h2_paired_delta}.png`.

### §4.6 cross-model addendum ✅ (2026-04-26)

Two cross-model §4.6 sub-tests:

**(A) Transfer test** (`scripts/sec4_6_cross_model_transfer.py`):
Qwen's §4.6 synthesized stim fed to 4 non-Qwen models. **Result:
0/140 flips**. LLaVA-1.5/LLaVA-Next/Idefics2 baseline at PMR=0 under
the §4.6 prompt; synth doesn't change that. **InternVL3 negative
transfer**: baseline saturated at PMR=1.0; ε=0.2 / unconstrained
perturbations *drop* synth PMR to 0.6 / 0.2 — large Qwen-derived
perturbations disrupt InternVL3's saturated commitment, don't trigger
or preserve it.

**(B) LLaVA-1.5 per-model gradient ascent**
(`scripts/sec4_6_counterfactual_stim_llava.py`): only model with
class-balanced v_L10 from M2. 5 baseline circle stim × 7 configs ×
200 Adam steps = 35 runs in 9.8 min. **0/5 v_L10 flips at every ε**.
Random controls also 0/5 (consistent with Qwen).

Critical detail: **gradient ascent succeeds at the projection level**
(8 → 150-200 for v_L10 configs, comparable to Qwen's 43-180; random
configs reach only 2-10). The dissociation is between *projection
level* and *behavior level*. LLaVA-1.5's L10 has a "v_L10 direction"
by class-mean diff, but maximizing projection along it doesn't flip
behavioral output the way it does on Qwen.

**Implication**: H-shortcut and H-direction-specificity are
**Qwen-scoped**. Pixel-encodability of the regime axis is
encoder-saturation-specific, parallel to M9's PMR-ceiling and §4.7's
decision-stability ceiling.

**Insight doc**: `docs/insights/sec4_6_cross_model.md` (+ ko).
**Figures**: `docs/figures/sec4_6_counterfactual_stim_{panels,trajectory}_llava.png`.

### M3 — ST2 Vision encoder probing ✅ (2026-04-24)

Run: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_... --output-dir outputs/mvp_full_.../vision_activations --layers 3,7,11,15,19,23,27,31`
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` (480 safetensors, 12 GB) + `probing_vision/*.csv`.
Deep dive: `docs/insights/m3_encoder_boomerang.md`.

**Key findings** (details in `docs/experiments/m3_encoder_probing.md`):

- **Encoder AUC = 1.00 on every factorial axis from layer 3 onward.** The vision encoder perfectly encodes bg / object / cue. No information bottleneck.
- **Behavioral forced-choice PMR ranges 0.28 (cue=none) to 0.95 (both).** The LM gating produces the gap.
- **Controlled no-cue subset (120 stimuli)**: encoder AUC 0.89 vs behavioral PMR 0.28 → the encoder knows which cells will trigger physics-mode but the LM only lets a fraction through.
- **Per-object-level encoder AUC ~0.95 constant while behavior 0.58-0.71**: gap is largest at line (most abstract) at +36 pp — internal-mechanism evidence for H4 (higher abstraction ⇒ stronger language prior).

**Hypothesis updates**:
- H-boomerang (originally §1.4 of the project doc, "encoder knows, decoder doesn't"): **supported (saturated evidence)**.
- H4 and H6 both gain mechanism-level evidence.

**Blocking-issue resolution / wins**:
- `PhysModeVLM.capture()` now has vision hooks (`_resolve_vision_blocks` helper covers Qwen / LLaVA / InternVL).
- Methodological caveat recorded: programmatic stimuli make encoder AUC 1.0 trivial. A photorealistic stimulus extension is needed for validation — see `docs/next_steps.md`.

### M4 — ST3 LM backbone logit lens / layer-wise probing ✅ (2026-04-24)

Run: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`.
Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.{csv,parquet}`.
Deep dive: `docs/insights/m4_logit_lens.md`.

**Key results**:
- LM per-layer probe AUC (forced-choice PMR) = **0.94-0.95 across all layers**, peak L20 = 0.953.
- Logit lens: physics logit > geometry logit from L5 onward because the "ball" label primes the LM.
- Object_level effect (L25 line 3.76 vs textured 4.35, margin +0.6) is **~14 % of the label effect (+4.0 flat shift)**.
- The switching-layer metric is degenerate under label-primed prompts (all = L5) → §4.9 "label-free prompt" promoted to a pre-M5 mini-experiment.

**Boomerang precise location**: vision encoder (0.94-1.0) → LM hidden (0.95) preserves information. ~29 pp accuracy loss at decoding. ST4 intervention priority: LM final layers + logit head.

### M5a — ST4 Phase 1+2 VTI steering ✅ (2026-04-24)

Run:
- Phase 1: inline Python driver, `compute_steering_vectors` from `src/physical_mode/probing/steering.py`.
- Phase 2: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_... --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`.

Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` (vectors) + `steering_experiments/` (intervention predictions). Deep dive: `docs/insights/m5_vti_steering.md`.

**Key results**:
- Steering vectors `v_L = mean(h | PMR=1) − mean(h | PMR=0)`. Norm grows 5× through the LM (L5: 5.9 → L25: 31).
- Projection at L20 aligns with the factorial cue axis (none 22.3 → both 42.7).
- **Layer 10 α=40 injection flips 10/10 `line/blank/none` responses from "D: abstract" to "B: stays still"**. L15/L20/L25 do not flip at the same α.
- The intervention causes an "abstract → physical object" binary shift, not gravity. Goes to "B: stays" rather than "A: falls" → direction is object-ness, not gravity. Consistent with H7 / H-regime.

**Hypothesis updates**:
- H-boomerang: extended + **causally supported**.
- H-locus: **supported (early-mid layer L10)**.
- H-regime (new): **candidate** — steering direction is coarse "object-ness", regime selection is label-driven.

### M5a-ext — VTI follow-ups ✅ (2026-04-24, 2026-04-25)

Run: `uv run python scripts/06_vti_steering.py` with `--output-subdir` flag
to partition sub-experiments within the same M2 output tree.

Output: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/{neg_alpha_textured_ground_both, ball_line_blank_none, bidirectional_recheck_*}/`.
Deep dive: `docs/insights/m5a_ext_bidirection_and_label.md`. Numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.

**Key results**:
- Exp 1 (ceiling): `-α · v_L10` at `textured/ground/both × circle` leaves the
  first-letter distribution at 10/10 A. Interpretation was initially "one-way
  activator"; recheck (Exp 3) revealed this was a ceiling artifact.
- Exp 2 (label swap): `+α=40 · v_L10` at `line/blank/none × ball` flips 10/10
  to A ("falls") vs 10/10 B ("stays still") under label=`circle` (M5a).
  Label-driven regime flip causally demonstrated.
- Exp 3 (bidirectional recheck, 2026-04-25): full (α × label × obj) grid on
  `{line, textured} × blank × none`. **New finding**: `-α=40` flips 10/10 to
  B ("stays still") **uniformly** across all four (obj × label) cells. `v_L10`
  is therefore a regime axis within physics-mode (+α kinetic, −α static), not
  a physics-vs-abstract activator. Baseline D sits below the |α| activation
  threshold, not at one axis endpoint.
- Exp 3 qualifier on H7: +α=40 on `textured/blank/none` gives A regardless of
  label; label-only regime flipping fails as soon as the image carries a
  physical-object signal. Regime is chosen by a joint (image, label, α sign)
  function.

**Hypothesis updates**:
- H-direction-bidirectional (new): **revised 2026-04-25** — regime axis
  interpretation replaces the earlier "one-way activator" framing.
- H-regime: **refuted in its original form** — the label-only regime flip
  does not generalize; subsumed by H-direction-bidirectional + the H7
  qualifier.
- H-locus: **unchanged (reinforced)** — L10 regime-flip holds across all
  four Exp 3 cells.

### M4b — Label-free prompt H2 null test ✅ (2026-04-25)

Run: `uv run python scripts/02_run_inference.py --config configs/label_free.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` followed by `scripts/03_score_and_summarize.py` and `scripts/05_lm_probing.py --sources open_no_label`.

Output: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions + 480 activation safetensors. Deep dive: `docs/insights/m4b_label_free.md`. Numbers: `docs/experiments/m4b_label_free.md`.

**Key results**:
- Paired PMR delta vs label-free baseline (480 matched seeds): ball +0.006,
  planet +0.006, **circle −0.065**. The "ball vs circle" gap reported in M2
  is actually circle suppression, not ball enhancement.
- Per-cell structure: circle suppresses more on abstract images
  (line: −9.2 pp; filled: −4.2 pp); `motion_arrow` cue overrides circle
  suppression entirely (+0.000); `none` cue gives the largest suppression
  (−15.0 pp).
- `line/blank/none` 4-label table cleanly separates label contributions:
  ball (regime shift, kinetic→static), circle (full suppression, PMR 0.40
  → 0.10), planet (+30 pp PMR — only label that genuinely *adds* physics
  over the visual default, due to orbital prior).
- M4 re-run on label-free activations reproduces M2's physics-margin table
  bit-for-bit, confirming visual-token captures are prompt-independent
  (image tokens precede question text under causal attention). The
  collapsed switching-layer at L5 is a structural artefact of the capture
  point, not evidence of label-independent LM commitment.

**Hypothesis updates**:
- H2: **revised** — ball ≈ no-label; circle is a suppressive override.
  Per-label contributions are asymmetric.
- H-boomerang: **reinforced** — visual-token hidden states are
  prompt-independent, so the L5 physics bias is image-only.
- H-locus: **unchanged** — label's behavioral effect localizes downstream
  of visual-token positions, consistent with M5a's L10 efficacy on the
  image-preceding trajectory.
- H4: **refined** — circle suppression strength scales with image
  abstraction, the image-side dual of the abstraction → language-prior-gap
  scaling.

### M6 round 1 — LLaVA-1.5-7B cross-model ✅ (2026-04-25)

Run: `uv run python scripts/02_run_inference.py --config configs/cross_model_llava{,_label_free}.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` (two passes), then `scripts/03_score_and_summarize.py` on each.

Output:
- `outputs/cross_model_llava_20260425-035506_7ff0256b/` — labeled (1440 rows).
- `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/` — label-free (480 rows).

Deep dive: `docs/insights/m6_cross_model_llava.md`. Numbers: `docs/experiments/m6_cross_model_llava.md`.

**Key results**:
- Paired PMR delta vs label-free baseline (LLaVA): ball **+0.475**,
  planet +0.244, circle +0.173 — the original H2 pattern, opposite of
  Qwen's M4b. The H2 reframing was Qwen-specific.
- Visual-saturation hypothesis: Qwen's `PMR(_nolabel)` ≈ 0.93–0.98
  across object levels; LLaVA's is 0.14–0.59. Qwen's labeled run
  cannot show positive label contributions because there's no
  headroom; LLaVA's labeled run shows them clearly.
- H1 (S-curve) cleanest on LLaVA (0.51 → 0.81 across line → textured),
  not visible on Qwen (saturated at 0.93).
- H7 replicates cross-model: `planet GAR << ball/circle GAR` in both
  models, with `planet` routing physics narration to orbital / cosmic
  events ("orbit around the sun", "consumed by a black hole").
- FC excluded: LLaVA returns "A" for every (image, label) FC stimulus
  (12/12 on smoke). Pathological model bias, not addressable via
  prompt template here. Round 2 needs an FC redesign or
  first-letter-token-probability scoring.

**Hypothesis updates**:
- H2: **revised again — visual-saturation hypothesis** unifies M4b and
  M6 within a single statement (positive language-prior contribution
  exists across labels and models; visual saturation masks it).
- H1: **supported, sharper on LLaVA** — recommend canonical figure
  comes from LLaVA's monotone curve.
- H7: **supported, cross-model** — orbital-routing dissociation
  replicates.
- H4 / H-boomerang / H-locus / H-direction-bidirectional:
  **untested cross-model in round 1** — FC failure blocks H4; no
  activation captures on LLaVA blocks the others. Round 2.

### M4c — Forced-choice label-free ✅ (2026-04-25)

Run: `uv run python scripts/02_run_inference.py --config configs/fc_label_free_{qwen,llava}.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` (two passes), then score with `scripts/03_score_and_summarize.py`.

Output:
- `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/` — Qwen, 480 rows.
- `outputs/fc_label_free_llava_20260425-044517_81ae56d5/` — LLaVA, 480 rows (degenerate).

Deep dive: `docs/insights/m4c_fc_label_free.md`. Numbers: `docs/experiments/m4c_fc_label_free.md`.

**Key results**:
- Qwen FC label-free reproduces M4b's H2 reframing under FC: `ball − _nolabel = +0.013` (≈ 0), `circle − _nolabel = −0.208` (stronger than M4b's −0.065), `planet − _nolabel = −0.263` (new — orbital regime collapses to D under FC's gravity-centric option set).
- Qwen open-vs-FC paired delta at no-label: **−0.131** — FC is consistently more conservative; H4 measurable cross-format without label confounding.
- `line/blank/none` under FC: every label condition collapses to D=10/10 (or 9/10 for `_nolabel`). FC's D option is an "abstract sink" pulling all label conditions toward abstract reject at fully ambiguous images.
- LLaVA FC label-free: 477/480 = 99.4 % `A`. Re-templating with "the depicted object" does **not** relax the bias from M6 r1. Confirmed model-level pathology, not prompt-fixable.

**Hypothesis updates**:
- H2: **further reinforced** — Qwen FC reproduces M4b under a different
  prompt format. The "planet suppression" finding adds nuance: per-
  label suppression in Qwen is partly an option-set artefact, supporting
  the visual-saturation framing rather than literal "abstract override"
  claims about specific labels.
- H4: **measurable on Qwen no-label** (paired delta = −0.131); cross-
  model H4 still blocked by LLaVA's FC bias.
- H7: **caveat added** — the regime distinction is only visible under
  prompts that allow narrative latitude. Under FC, all non-gravity
  regimes (orbital, "consumed by black hole", etc.) collapse to D and
  H7 is masked. An extended FC option set would be needed for FC-side H7.
- LLaVA FC pathology: **confirmed** — round-2 idea is to use first-
  token logit ratios instead of greedy argmax.

### M6 round 2 — Cross-model expansion ✅ (2026-04-25)

Three sub-deliverables. Run dirs:

- `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — InternVL3 labeled (1440 rows).
- `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — InternVL3 label-free (480 rows).
- `outputs/cross_model_llava_capture_20260425-054821_65214a5d/` — LLaVA captured (1440 rows + 14 GB activations + probing_vision/ + probing_lm/).

Deep dive: `docs/insights/m6_r2_cross_model.md`. Numbers: `docs/experiments/m6_r2_cross_model.md`.

**Key results**:
- **r2a (InternVL3 behavioral)**: paired delta vs label-free is +0.010 for
  every label — InternVL3's PMR(_nolabel) = 0.99 leaves no headroom.
  3-model paired-delta pattern (Qwen ≈ 0, LLaVA strongly +, InternVL3 ≈ 0)
  matches the encoder-saturation prediction.
- **r2b (LLaVA captures)**: vision encoder AUC ~0.73 (vs Qwen ~0.99); LM
  AUC ~0.75 (flat — no boomerang recovery); behavioral PMR ~0.78. The
  saturation difference between models is rooted in the vision encoder.
- **r2c (FC logit ratio)**: LLaVA's FC bias is at the underlying logit
  level (90% of rows have only `A` surviving top_p=0.95). Greedy → logit-
  ratio rescue fails. For Qwen, logit-argmax is a cleaner FC metric than
  text-PMR (recovers ~14 pp of signal lost to greedy formatting drift).

**Hypothesis updates**:
- H-boomerang: **Qwen-scoped** — encoder-knows / decoder-gates gap exists in Qwen, not in LLaVA (encoder is the bottleneck, not the gate).
- H-encoder-saturation (new): **proposed and supported across 3 model points** — vision encoder probe AUC predicts both `PMR(_nolabel)` and per-label paired delta direction.
- H2: **fully validated under visual-saturation hypothesis** at three model points; the paired-delta pattern matches the encoder-saturation prediction.
- H7: **3-of-3 cross-model** — orbital-routing dissociation universal so far.
- H4: **untested for InternVL3 + LLaVA** — FC excluded (cost) / blocked (LLaVA "A" bias).

### M8a — Non-circle synthetic shapes ✅ (2026-04-25)

**Outcome**: 5 shapes × 4 abstraction × 2 bg × 2 cue × 5 seeds = 400 stimuli; Qwen + LLaVA × labeled + label-free arms = ~3200 inferences in ~43 min on a single (contended) GPU 1.

**Strict pre-registration scoring**:

| Criterion | Qwen | LLaVA |
|---|---|---|
| H1 ramp (PMR(textured)−PMR(line) ≥ 0.05; no inv >0.05) | 3/5 ✗ | 4/5 ✓ |
| H7 (PMR(physical)−PMR(abstract) ≥ 0.05) | 1/5 ✗ | 4/5 ✓ |
| H7-GAR (GAR(physical) ≥ GAR(abstract)) | 1/5 ✗ | 5/5 ✓ |
| Visual-saturation Δ on physical role | 3/5 ✓ borderline | 5/5 ✓ |

**Headline interpretation**: the saturated model fails 3 of 4 criteria; the unsaturated model passes 4 of 4. The asymmetry **is** the cross-shape validation of H-encoder-saturation — predicted, not just consistent. PMR(_nolabel) baseline: Qwen 0.78–0.93 across shapes (ceiling), LLaVA 0.075–0.288 (headroom).

**Notable per-shape findings**:
- Qwen `square` paired-delta -0.20 / -0.28 / -0.21 — clean cross-shape replication of M4b's circle "label-suppresses-physics" effect.
- LLaVA `triangle` paired-delta only +0.125 / +0.10 / +0.10 — `wedge` is a weak physical-object label (PMR(wedge)=0.20 vs PMR(ball/brick/nut/rock)≈0.7). Label-design caveat, not a shape effect.
- LLaVA `polygon abstract` paired-delta -0.05 — only LLaVA paired-delta to go negative. "Polygon" reads as a math term; role taxonomy leaks for shapes without common-vocabulary geometric nouns.

**Roadmap implications**:
- H1 + H7 are now *unsaturated-only*; add this caveat in the paper.
- H-encoder-saturation moves from 3-model correlational to 3-model + 5-shape, predicting the H1/H7 failures.
- Triangle / polygon label-quality issues feed into M8c photo curation and M8d non-ball categories (use stronger physical labels).

**Artifacts**: `docs/insights/m8a_non_circle_shapes.md`, `docs/experiments/m8a_non_circle_shapes.md`, `docs/figures/m8a_{shape_grid,full_scene_samples,pmr_ramp,pmr_by_role,paired_delta}.png`, `notebooks/m8a_non_circle_shapes.ipynb`, `outputs/m8a_*` (4 run dirs).

### M8c — Real photographs ✅ (2026-04-25)

Run: `bash scripts/m8c_run_all.sh` on GPU 0.

Output: 4 run dirs + 60 photos curated from COCO 2017 + WikiArt → 480 inferences in **5 minutes** wall clock.

Stim dir: `inputs/m8c_photos_20260425-162031/` — 60 photos × {ball, car, person, bird, abstract}.

Deep dive: `docs/insights/m8c_real_photos.md`. Numbers: `docs/experiments/m8c_real_photos.md`.

**Key results**:

| category | Qwen synth-textured | Qwen photo | Δ |
|----------|-----:|-----:|-----:|
| ball     | 0.900 | 0.667 | **−0.233** |
| car      | 0.975 | 0.500 | **−0.475** |
| person   | 0.850 | 0.667 | **−0.183** |
| bird     | 0.875 | 0.417 | **−0.458** |

| category | LLaVA synth-textured | LLaVA photo | Δ |
|----------|-----:|-----:|-----:|
| ball     | 0.450 | 0.500 | +0.050 |
| car      | 0.375 | 0.000 | **−0.375** |
| person   | 0.025 | 0.417 | **+0.392** |
| bird     | 0.600 | 0.500 | −0.100 |

**Headline interpretation**: photos REDUCE Qwen PMR(_nolabel) by 18-48 pp across all 4 physical categories. This contradicts the naive "photo-realism saturates the encoder further" prediction. The synthetic-PMR ceiling (M2 / M8a / M8d) is partly driven by *minimality* of synthetic stim — isolated single-object images with explicit motion cues (arrow, cast shadow) maximize the "physics-mode" reading. Real photos add scene context that elicits scene-descriptive responses without firing physics verbs.

**LLaVA person photo PMR rises +39 pp** vs synthetic — the encoder finally recognizes actual humans (vs the stick-figure / textured-person primitive). This is the encoder-recognition effect from M6: when the encoder's visual prior is unsaturated for a synthetic class, real photos can activate the prior more strongly.

**H7 partially holds on photos**: LLaVA 2/4 (ball + bird PASS), Qwen 2/4 (ball + person PASS). M8d synthetic LLaVA H7 was 3/3 → photos add scene-context noise but don't reverse the H7 signal.

**Methodological refinement**: M6 r2 linear-probe AUC measures encoder representational saturation (still valid). Behavioral PMR(_nolabel) saturation is the conjunction of (a) encoder representational saturation AND (b) input-context simplicity enabling minimal "physical-object → next state" reading.

**Code changes**:
- `inference/prompts.py::LABELS_BY_SHAPE`: `ball` triplet (ball, circle, planet) reusing circle's labels for photo-ball; `abstract` triplet (object, drawing, diagram).
- `config.py`: extended `Shape` literal with `ball`, `abstract`. Added `drawing`, `diagram` to `Label` literal.
- `scripts/m8c_curate_photos.py` — COCO + WikiArt curation driver (HF `datasets` package).
- `scripts/m8c_{run_all.sh, analyze.py, figures.py}` + `configs/m8c_*.py`.

**Roadmap implications**:
- Paper claim refinement: "Qwen treats abstract synthetic stim as physical" still holds; M8c reframes as "Qwen treats *minimal* visual-prior stim as physical." Photos provide the counterfactual.
- M8e (cross-source paired analysis) is now well-motivated — same 4 physical categories tested both synthetic (M8a/M8d) and photo (M8c).
- Round-2 curation upgrades: bbox-cropped subsets (less scene noise), penguin/chicken for bird exotic role.
- §4.5 (encoder swap) remains the cleanest causal test of the encoder-AUC piece of the hypothesis (independent of stim simplicity).

**Artifacts**: `docs/insights/m8c_real_photos.md` (+ `_ko.md`), `docs/experiments/m8c_real_photos.md` (+ `_ko.md`), `docs/figures/m8c_{photo_grid,pmr_by_category,paired_synthetic_vs_photo}.png`, `notebooks/m8c_real_photos.ipynb`, `outputs/m8c_summary/`.

### M8d — Non-ball physical-object categories ✅ (2026-04-25)

Run: `bash scripts/m8d_run_all.sh` on GPU 0 (single H200).

Output: 4 run dirs (Qwen labeled / Qwen label-free / LLaVA labeled / LLaVA label-free) → 3840 inferences in **31.9 min** wall clock.

Stim dir: `inputs/m8d_qwen_20260425-151543_19e1fcd0/` — 480 stimuli (3 categories × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds).

Deep dive: `docs/insights/m8d_non_ball_categories.md`. Numbers: `docs/experiments/m8d_non_ball_categories.md`.

**Strict pre-registration scoring**:

| Criterion         | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| Visual-sat. delta | 1/3 (bird) | 2/3 (car, bird; person flips negative) |

**Headline interpretation**:

- **H7 generalizes 3/3 cross-category on LLaVA** — strongest cross-category H7 evidence in the project. car +0.525, person +0.138, bird +0.550 on PMR_regime(physical) − PMR_regime(abstract) over the horizontal subset. The label-selects-regime claim is now category-general.
- **Qwen H7 fails strict (binary, ceiling)** but the regime distribution shows the same pattern at the kinetic-vs-static split: figurine 17.5 % static, statue 22.5 % static (vs ~5 % for physical labels). New methodological finding: **regime distribution rescues H7 signal from binary saturation**.
- **H1 fails on both models** — ceiling on Qwen, non-monotone on LLaVA. The abstraction ramp is a property of the geometric-shape ↔ named-object axis, not a general visual-detail → physics-prior mechanism. M8a established H1-unsaturated-only; M8d narrows further to H1-shape-specific.
- **H-encoder-saturation cross-validated cross-category** — Qwen ceiling on car/person (0.97-1.0), LLaVA range 0.55-0.84.

**Code changes**:
- `stimuli/primitives.py`: 12 new draw functions (car / person / bird × line / filled / shaded / textured).
- `stimuli/scenes.py`: ground-bound shape positioning for `horizontal` events.
- `inference/prompts.py::LABELS_BY_SHAPE`: car / person / bird triplets.
- `metrics/lexicons.py`: `CATEGORY_REGIME_KEYWORDS` + `UNIVERSAL_KINETIC_STEMS` (gravity-fall verbs shared across categories).
- `metrics/pmr.py`: `classify_regime(category, text) → {kinetic, static, abstract, ambiguous}`.
- `configs/m8d_*.py`, `scripts/m8d_*.py`. 123 unit tests pass.

**Classifier validation**: `scripts/m8d_hand_annotate.py --mode score` on 54 stratified rows → **5.6 % error rate** (well below the 15 % paper-ready threshold). 3 mismatches are stem-matching false-positives in "no movement" / "pulled away" patterns.

**Roadmap implications**:
- H7 promoted from "circle-only" to "cross-category" (paper headline).
- H1 narrowed to "geometric-shape ↔ named-object axis" only.
- Regime-distribution becomes a paper methodological contribution (rescues H7 from binary saturation).
- M8c is now strongly motivated — does photo-realism close LLaVA's encoder gap?
- Round-2 improvement: replace `duck` with a flightless bird (penguin / ostrich / chicken) for cleaner H7 exotic role.

**Artifacts**: `docs/insights/m8d_non_ball_categories.md` (+ `_ko`), `docs/experiments/m8d_non_ball_categories.md` (+ `_ko`), `docs/figures/m8d_{shape_grid,full_scene_samples,pmr_ramp,pmr_by_role,paired_delta,regime_distribution}.png`, `notebooks/m8d_non_ball_categories.ipynb`, `outputs/m8d_summary/` (per-model rollups + concatenated annotated parquet).

### M6 r4 — InternVL3 vision-encoder probe (4-model chain) ✅ (2026-04-25)

Deep dive: `docs/insights/m6_r4_internvl3_probe.md`.

**Scope**: Adds the fourth point to the H-encoder-saturation chain — InternVL3-8B (InternViT + InternLM2-7B). Behavior on M8a (12 min for 1200 + 400 inferences) + vision capture (47 s for 400 stim × 4 InternViT layers) + linear probe.

**Headline**: 4-point AUC ↔ behavioral PMR(_nolabel) chain on M8a synthetic stim (all 4 encoder probes computed on M8a stim — apples-to-apples):

```
encoder family            AUC      PMR(_nolabel)
─────────────             ────     ─────────────
SigLIP    (Qwen)          0.88     0.84
SigLIP-SO400M (Idefics2)  0.93     0.88
InternViT (InternVL3)     0.89     0.92     ← M6 r4
CLIP-ViT-L (LLaVA)        0.77     0.18
```

**3 distinct non-CLIP encoder families** (SigLIP, SigLIP-SO400M, InternViT) all reach AUC ≥ 0.88 / PMR ≥ 0.84. **Only CLIP-ViT-L falls below saturation** (0.77 / 0.18). Across 4 LM families (Qwen2-7B, Mistral-7B, InternLM2-7B, Vicuna-7B), encoder family is the unified saturation driver.

**Hypothesis update**: H-encoder-saturation generalizes from "SigLIP saturates" to "non-CLIP encoders saturate; CLIP doesn't (in this sample)". Paper-grade headline: vision-encoder family causally determines synthetic-stim ceiling regime. The nonlinear AUC → PMR mapping (≈0.10 AUC gap → 0.65 PMR gap) is consistent with a saturation threshold around AUC 0.85.

**Implementation note**: required a fix to `_resolve_vision_blocks` to recognize InternVL3's `vision_tower.encoder.layer` (singular, not plural). This round also re-captured Qwen + LLaVA on M8a stim so all 4 AUC numbers come from the same factorial; the M6 r2 numbers (Qwen 0.99 / LLaVA 0.73) were on M2 stim's bimodal line/ground-cells distribution, which is more separable than M8a's wider per-cell PMR range.

**Artifacts**: `docs/insights/m6_r4_internvl3_probe.md` (+ `_ko`), `docs/figures/encoder_chain_4model.png` (paper headline figure), `docs/figures/encoder_swap_internvl3_probe.png`, `outputs/encoder_swap_internvl3_probe/{layer_sweep,by_object_level,by_shape}.csv`, `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`, `scripts/encoder_swap_probe.py` (model-agnostic), `scripts/encoder_swap_probe_summary.py`.

### M6 r3 — Idefics2 vision-encoder probe ✅ (2026-04-25)

Deep dive: `docs/insights/m6_r3_idefics2_probe.md`.

**Scope**: Captures Idefics2 SigLIP-SO400M vision activations on M8a stim (400 stimuli × 4 layers, 88 s wall on GPU 0) and trains layer-wise linear probes for "physics-vs-abstract" using Idefics2's own per-stim labeled-arm PMR as the y target. Closes the AUC ↔ behavioral PMR chain at the third SigLIP point.

**Headline**: Idefics2 vision-encoder probe AUC = **0.93** across layers (peak 0.948 at layer 9). Sits between Qwen SigLIP (M3 / M6 r2: 0.99) and LLaVA CLIP-ViT-L (M6 r2: 0.73). The 3-model AUC ↔ behavioral PMR chain reads:

```
encoder family            AUC      PMR(_nolabel) on M8a
─────────────             ────     ────────────────────
SigLIP    (Qwen)          0.99     0.84
SigLIP-SO400M (Idefics2)  0.93     0.88     ← M6 r3
CLIP-ViT-L (LLaVA)        0.73     0.18
```

**Hypothesis update**: H-encoder-saturation now fully closed at the *mechanism* level: encoder family → encoder probe AUC → behavioral PMR(_nolabel) → H7 measurability, all 4 nodes empirically supported at 3 model points. Updated paper claim: "encoder family causes vision-encoder probe AUC saturation, which causes behavioral PMR(_nolabel) saturation, which gates H7 measurability".

**Caveats**: Per-shape AUC variance is high at this n (`polygon` AUC drops below 0.5 at deep layers — n-imbalance artifact, not a real reversal). InternVL3 captures still pending — natural next move for a 4-point probe table.

**Artifacts**: `docs/insights/m6_r3_idefics2_probe.md` (+ `_ko`), `docs/figures/encoder_swap_idefics2_probe.png`, `outputs/encoder_swap_idefics2_probe/{layer_sweep,by_object_level,by_shape}.csv`, `scripts/encoder_swap_idefics2_probe.py`.

### M9 — Generalization audit (paper Table 1) ✅ (2026-04-25)

Deep dive: `docs/insights/m9_generalization_audit.md`.

**Scope**: Consolidates M8a (5 shapes) + M8d (3 categories) + M8c (5 photo categories) × {Qwen2.5-VL-7B (SigLIP), LLaVA-1.5-7B (CLIP), Idefics2-8b (SigLIP-SO400M)} into a single 9-cell paper Table 1 with **95% bootstrap CIs** (5000 iters) on mean PMR(_nolabel) and mean H7 paired-difference.

**Headlines**:

1. **Encoder family causally drives synthetic-stim PMR(_nolabel) ceiling** — robust. SigLIP CIs [0.800, 0.917] vs CLIP CIs [0.140, 0.371] on M8a + M8d; CIs fully separated.
2. **Photos compress the encoder gap** — robust. All 3 models converge into [0.183, 0.667] on M8c; SigLIP loses its synthetic ceiling. 5× ratio shrinks to ~1.5–2×.
3. **H7 measurability is robust only on LLaVA-on-synthetic** — LLaVA M8a CI [+0.30, +0.42] and M8d CI [+0.25, +0.36] separated from 0; LLaVA M8c CI [−0.03, +0.23] crosses 0 (n=12 underpowered). All Qwen + Idefics2 H7 CIs cross 0 except Idefics2 M8d CI [+0.000, +0.094] which just touches.
4. **LM-modulation of H7 at saturation** — *suggested only*. Idefics2 M8d H7 CI just above 0 vs Qwen M8d H7 CI crosses 0; PASS-rate gap (0.667 vs 0.333) is driven by a single shape (`car`) crossing the strict threshold. Demoted to flagged-future-work; clean test needs same-encoder LM swap.

**Statistical methodology contribution**: replaces PASS/FAIL binarization (M8a, §4.5) with bootstrap CIs on the mean H7 delta. Reveals that the `Qwen 1/5 PASS` pattern was a noise-floor binarization (true mean H7 = 0 with CI crossing 0), not a real "1 out of 5" finding.

**Hypothesis updates**:
- H-encoder-saturation: *strengthened* — paper claim now "encoder family causes synthetic-stim PMR ceiling; saturation gates H7 measurability" (cross-stim bootstrap-validated).
- H7: *clarified scope* — robust only where encoder leaves headroom (LLaVA on synthetic).
- New **H-LM-modulation**: suggested but not defensible from current data.

**Limitations**: M8c n=12/category underpowered for H7; M8d 3 shapes thin for cross-shape variance; no encoder probe AUC for Idefics2 yet.

**Artifacts**: `docs/insights/m9_generalization_audit.md` (+ `_ko`), `docs/figures/m9_summary.png`, `docs/figures/m9_table1_heatmap.png`, `outputs/m9_audit/m9_{table1,summary}.csv`, `scripts/m9_generalization_audit.py`.

### 4.5 Cross-encoder swap — work plan ▶ priority 4 (promoted)

**Motivation**: H-encoder-saturation is currently 3-model correlational (M6 r2). The causal test is to *swap* the encoder while holding everything else constant.

**Sub-tasks**:
1. LLaVA-1.5-7B with SigLIP encoder swap (HF community ports exist: e.g. `google/siglip-base-patch16-224` projector retraining). Or use a from-scratch LLaVA-style training with SigLIP as encoder.
2. Cleanest alternative: take a LLaVA-1.5-derived family that already swapped encoder (e.g. ShareGPT4V, Bunny). Behavioral run only — confirm whether `PMR(_nolabel)` and encoder AUC track together.
3. Stretch: train a minimal projector swap (~few hr GPU) to swap CLIP ↔ SigLIP on LLaVA-1.5.

**Estimated effort**: 4-7 hours (using existing swapped variants; +many hours if training a fresh swap).

### 4.6 Counterfactual stimulus generation via VTI reverse ✅ (2026-04-26)

**Approach**: pixel-space gradient ascent on Qwen2.5-VL post-processor `pixel_values` (T_patches × 1176), maximizing `<mean(h_L10[visual]), v_L10>`. Bypasses the non-differentiable PIL → patch preprocessing while still recovering a viewable RGB via inverse permute + de-norm. New module: `src/physical_mode/synthesis/counterfactual.py`.

**Result (5 baseline circle stim × 7 configs × 200 Adam steps, lr=1e-2)**:

| Config              | n flipped (PMR 0→1) | Mean final projection |
|---------------------|--------------------:|----------------------:|
| `bounded_eps0.05`   |               5 / 5 |                  43.7 |
| `bounded_eps0.1`    |               5 / 5 |                 100.6 |
| `bounded_eps0.2`    |               5 / 5 |                 125.9 |
| `unconstrained`     |               5 / 5 |                 181.1 |
| `control_v_random_*`|              0 / 15 |                 73–85 |

**Headlines**:
1. **5/5 v_L10 flips at ε = 0.05** (pre-registered ≥ 3/5).
2. **0/15 random-direction flips at matched ε = 0.1** — directional specificity falsifies "any pixel perturbation flips PMR." Random directions reach projection magnitudes (73–85) comparable to bounded ε=0.1 (101); behavioral outcome diverges along the *axis*, not the magnitude.
3. **`v_L10` is encodable in the image** — pixel-space change without runtime steering suffices.

**Scorer-fix note**: random controls exposed an over-permissive PMR scorer ("no indication of movement" matched the "mov" stem). Added asymmetric abstract-marker patterns (`remain stationary`, `no indication of mov`, `no indication of motion`). Verified asymmetric: 0/20 v_L10 hits; 14/15 random hits. Headline replicates with the pre-fix scorer.

**Visual character**: ε = 0.05 produces a faint dotted texture visible on close inspection; preserves the abstract circle gestalt; introduces no human-readable physical features. Deep dive: `docs/insights/sec4_6_counterfactual_stim.md` (+ ko).

**Artifacts**: `outputs/sec4_6_counterfactual_20260426-050343/`, `docs/figures/sec4_6_counterfactual_stim_panels.png`, `docs/figures/sec4_6_counterfactual_stim_trajectory.png`.

### 4.10 Attention visualization UI — work plan ▶ priority 6 (promoted)

**Motivation**: Cross-axis (layer × head × visual-token-position) attention maps qualitatively reveal which heads attend to which cues. Useful as paper appendix figure + for finding patching targets.

**Sub-tasks**:
1. Re-run a subset (~20-50 stim) with `capture_lm_attentions=True` on Qwen + LLaVA.
2. Build a notebook UI: select stimulus → layer → head → render attention heatmap overlaid on image, plus attention to label tokens.
3. Curate 5-10 illustrative cells.

**Estimated effort**: 5-7 hours.

### M5b — ST4 Phase 3 (SIP patching + SAE) — work plan

**Sub-tasks**:
1. Re-capture activations with `capture_lm_attentions=True` on a focused subset (~120 stimuli covering single-axis-differ pairs). Disk: ~15 GB additional.
2. Construct Semantic Image Pairs from the M2 factorial — single-axis-differ pairs (e.g. `shaded_ground_none` vs `line_ground_none` on the same seed). Emit `sip_manifest.parquet`.
3. **Activation patching** (TransformerLens or raw PyTorch hooks):
   - Capture clean and corrupted forward passes.
   - Layer sweep: replace each layer's visual-token activations from corrupted to clean; measure PMR-probability recovery (indirect effect).
4. **Attention knockout**: zero out specific heads' visual-to-last-token attention; measure PMR delta.
5. **VTI ablation**: `v_layer = mean(h_clean) − mean(h_corrupted)` (per-pair version of M5a's class-mean approach). Add at test time. Confirm or extend M5a's L10 finding.
6. **SAE** (stretch): train SAE on vision-encoder activations; identify monosemantic "shading" / "ground" features + intervention.

**Headline-claim candidate**: "Knockout of LLaVA layer 19 head 14 drops PMR by 50 pp; preserving only this head while removing all other visual attention preserves the effect" (project doc §3.2 sentence template).

### M6 — ST5 Cross-model sweep

Tasks:
1. `configs/cross_model.py` — list of `model_id`s + same factorial.
2. Modify `scripts/02_run_inference.py` to iterate over a model list.
3. LLaVA-1.5-7B (~13 GB), LLaVA-Next-7B (~14 GB), InternVL2-8B (~16 GB), (stretch) Qwen2-VL-7B (~15 GB). Total download ~60 GB.
4. Behavioral table per model + (where possible) reduced ST3/4 versions.
5. **Prompt steering**: use `system_prompt_override` with `"treat this as an abstract geometric shape"` vs `"treat this as a physical object subject to gravity"` → measure PMR shift.

**Hypothesis**: the ground effect (H5) replicates across all open-source VLMs; the open-vs-forced gap (H4) varies in magnitude across models.

### M7 — Human baseline (optional) + paper

- 20 Prolific raters × 50 stimuli (random subset) × open-ended prompt.
- Per-cell alignment between human PMR and VLM PMR.
- ICLR 2027 (primary, deadline ~late Sep 2026) / NeurIPS 2027 (secondary) draft per `references/submission_plan.md`.

**Status (2026-04-28)**: human baseline deferred per user — re-evaluate at week 14 of Track B schedule once results have matured. Plan: `docs/m7_human_baseline_plan.md`.

---

## 3.X Track B priorities (current open work, 2026-04-28)

**Read order for new sessions**: `references/submission_plan.md` → `references/paper_gaps.md` → this section.

| Pillar | Milestone | Status | Week | Gap fixed |
|---|---|---|---|---|
| A | **M-MP Phase 1+2** Multi-prompt behavioral PMR (5 models × 3 prompts × 480 stim) | ✅ 2026-04-28 | 1–2 | G1 (single-task) — behavioral evidence |
| A | **M-MP Phase 3** Cross-prompt M5a + M5b (Qwen + Idefics2 × 3 prompts) — *causal* test | ✅ 2026-04-28; **Mixed**: generative-task-agnostic for Qwen (open + describe both flip/break) but categorical-blocked (yesno doesn't flip/break); Idefics2 even narrower (kinetic-only). Cross-method M5a-M5b dissociation refines paper claim. | 3 | G1 — causal evidence |
| — | **M5b post-projection SAE** (Qwen) | ✅ 2026-04-28: k=20 break at same threshold as pre-projection — merger preserves physics-mode commitment localization. Cross-model extension optional. | between weeks 2–3 | mechanism resolution (encoder pre/post-projector) |
| B | **M-PSwap** Projector-swap LoRA on Idefics2 (perceiver → MLP) | planned | 4–5 | G3 (n=1 perceiver) |
| B | **M-LMSwap** LM-only-swap LoRA (CLIP + Vicuna vs CLIP + Mistral) | planned | 6–7 | G3 (n=1 perceiver) |
| B | **M-Add6** 6th non-Qwen model (Pixtral / InternVL3.5 / Phi-3.5-V) | optional | 8 | G2 (sparse non-Qwen) |
| C | **M-Marr** Paper §6 Marr-3-level restructure + §1/9/10 world-model framing | planned | 9–11 | G4 (5-sig framing) |
| — | M7 Human baseline | deferred | 14 (reeval) | — |
| Stretch | External benchmark (Physion / CLEVRER subset) | backlog | 13 | G1 (deeper) |

Drop or defer rules per `submission_plan.md` §6:
- If end of week 8 has not produced clean Pillar B results → drop M-PSwap, keep M-LMSwap, lean on M-Add6 + literature.
- If end of week 14 still missing pieces → pivot ICLR 2027 → NeurIPS 2027 (gain 7 months).

---

## 4. Additional ideas not in the original project doc

Extensions that came up during the pilot, or that aren't in `references/project.md` §2.

**Promoted to next-tier priority** (work plans now in §3, see corresponding sections):
- **4.5** Cross-encoder swap — priority 4 after M8a/c/d (causal test of H-encoder-saturation).
- **4.6** Counterfactual stimulus generation via VTI reverse — ✅ 2026-04-26 (5/5 v_L10 flips at ε=0.05; 0/15 random; v_L10 encodable in pixels).
- **4.10** Attention visualization UI — priority 6.

The remainder are still optional / open ideas.

### 4.1 Block stack as a separate "abstract-physical" path

The code (`primitives.py::_draw_block_stack`) exists but the pilot never used it. Blocks are an "abstract geometry + clearly physical" combination, asking a **different question from the circle-ball axis**: "given an abstract shape but a physical configuration (stacking), which way does the VLM go?" → expected: high PMR + low abstract_reject. Useful as a control on the circle-ball axis.

### 4.2 Reverse prompting ✅ (2026-04-25)

What happens to PMR when an `"abstract"` label is attached to a *real*
photograph of a ball? A counterfactual for H2 (language-prior dominance).
**Done 2026-04-25 by reusing existing M8c labeled-arm data** (5 models ×
3 label roles × 4 physical photo categories × 12 seeds = 720/model).
**Headline**: image-prior dominates label-prior on real physical photos —
phys_minus_abs ≤ +0.146 across all 5 models, vs LLaVA-1.5 M8d synthetic
phys_minus_abs +0.306 (label effect halved on photos). **LLaVA-Next phys
− abs = 0.000** on physical photos: calling a real ball `"circle"` does
not lower PMR vs `"ball"`. The image vs label trade-off is the saturation
effect viewed from the input side: rich image → image dominates;
impoverished image → label dominates. Full doc:
`docs/insights/sec4_2_reverse_prompting.md` (+ ko).

### 4.3 Label language switching ✅ (2026-04-26, 5-model)

Does a Korean `"공"` vs an English `"ball"` on the same stimulus produce
different PMR? Qwen2.5-VL is multilingual; cross-model tests whether
this generalizes.

**Done 2026-04-26 on 5 VLMs × M8a circle (n=80 per label per language
per model)**: Qwen-only headline below, cross-model extension after.

Qwen2.5-VL (original):

| Role | EN PMR | KO PMR | Δ |
|------|-------:|-------:|---:|
| ball / 공 | 0.81 | 0.85 | +0.04 |
| circle / 원 | 0.80 | 0.76 | −0.04 |
| planet / 행성 | 0.96 | 0.88 | −0.09 |

Cross-model EN→KO Δ (KO − EN), Korean-aware scorer (added 12 Korean-only
responses that the original English-keyword scorer silently dropped):

| Model | physical | abstract | exotic | mean |Δ| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | +0.04 | −0.04 | −0.09 | 0.06 |
| LLaVA-1.5  | **−0.19** | **+0.13** | +0.01 | 0.11 |
| LLaVA-Next | −0.05 | +0.04 | −0.04 | 0.04 |
| Idefics2   |  0.00 | **+0.11** | −0.05 | 0.05 |
| InternVL3  |  0.00 | −0.03 | −0.03 | 0.02 |

**Headline (5-model)**:
1. **Cross-label ordering preserved 4/5 models** (Qwen, LLaVA-1.5,
   LLaVA-Next, InternVL3). Idefics2 is the exception: KO order
   `공 > 원 > 행성` vs EN `ball > planet > circle` — `행성` rank
   drops below `원`.
2. **LLaVA-1.5 swing largest** (avg |Δ|=0.11; Vicuna LM has weak
   Korean SFT). **InternVL3 swing smallest** (avg |Δ|=0.02; ceiling
   + strong InternLM3 Korean coverage).
3. Original Qwen-only multilingual claim survives, but the cross-model
   picture adds a **language-prior axis**: LM Korean fluency modulates
   how much of the English label prior transfers, independent of the
   vision encoder. Same encoder + different LM → different Korean
   magnitude.

Doc: `docs/insights/sec4_3_korean_vs_english.md` (+ ko).
Figures: `docs/figures/sec4_3_korean_vs_english.png` (Qwen-only) +
`docs/figures/sec4_3_korean_vs_english_cross_model.png` (5-model).

**Japanese extension (2026-04-26, same day)**: same 5-model design with
Japanese labels (ボール / 円 / 惑星) added. Surfaced a different
mechanism than Korean: Qwen2.5-VL keeps the Japanese label 85-91% of
the time (genuine Japanese engagement); LLaVA-1.5 / LLaVA-Next /
InternVL3 mostly translate kanji to English internally; **Idefics2
falls back to Chinese on `惑星` in 19/80 responses** (Mistral-7B has
limited Japanese SFT but recognizes the kanji as Chinese 惑星 = planet).

| Model | physical | abstract | exotic | mean |Δ| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | **+0.13** | 0.00 | −0.01 | 0.05 |
| LLaVA-1.5  | −0.05 | +0.04 | +0.05 | 0.05 |
| LLaVA-Next | −0.03 | +0.10 | +0.04 | 0.05 |
| Idefics2   | −0.01 | +0.06 | +0.05 *| 0.04 |
| InternVL3  |  0.00 | −0.01 | −0.03 | 0.01 |

\* Idefics2 exotic +0.05 comes from Chinese-fallback responses scored
correctly by the new `CHINESE_PHYSICS_VERB_STEMS` lexicon. Without the
fix the apparent Δ would have been **−0.15** — pure scorer artifact.

Cross-language summary:
- Korean tests **language-fluency-bottleneck** (Hangul isolation forces
  engagement; 4/5 ordering preserved; LLaVA-1.5 swing 0.11 measures
  Vicuna-Korean weakness genuinely).
- Japanese tests **kanji-as-bridge** (5/5 ordering preserved within
  bootstrap noise, but via mixed paths: genuine engagement (Qwen),
  internal translation (LLaVA-1.5), or cross-script fallback (Idefics2)).
- LLaVA-1.5 ↓Korean / ≈Japanese asymmetry (0.11 vs 0.05) is *not*
  evidence that Vicuna-Japanese is stronger than Vicuna-Korean — it
  reflects the script's translatability, not LM SFT depth.

**Still open**: Chinese / Spanish; fully target-language prompt (not
just label inserted into English template).

### 4.4 Video frame pair → Michotte-style causality

Give a (t=0, t=1) frame pair where only the object position differs, then ask "launched by X?". Does the Michotte (1946) launching effect appear in VLMs? A 2-image prompt is a proxy that does not require a video model.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2) ⭐ promoted

Hypothesis: "cues that are invisible when the encoder is CLIP can be seen by a DINOv2-based model". Continuation of the Eyes Wide Shut (Tong et al. 2024) MoF proposal. Note: a standalone-encoder comparison is implicitly part of M6 (LLaVA-1.5 with CLIP-ViT-L/14 vs Qwen2.5-VL with SigLIP).

**Status (2026-04-25)**: promoted to next-tier priority — H-encoder-saturation (M6 r2) is currently 3-model correlational; this is the causal counterfactual. Detailed work plan in §3 above.

### 4.6 Activation-based counterfactual stimulus generation ✅ (2026-04-26)

Pixel-space gradient ascent on the Qwen2.5-VL post-processor `pixel_values` (T_patches × 1176, the patch-flattened normalized representation) maximizing `<mean(h_L10[visual]), v_L10>`. Bypasses the non-differentiable PIL → patch preprocessing while still recovering a viewable image via inverse permute + de-norm. **5/5 v_L10 flips at ε = 0.05** (pre-registered ≥ 3/5); **0/15 random-direction flips** at matched ε = 0.1 — directional specificity falsifies "any perturbation flips PMR." `v_L10` is encodable in the image — the M5a shortcut lives on the pixel-driven path, not just at runtime hidden-state injection. Deep dive: `docs/insights/sec4_6_counterfactual_stim.md` (+ ko).

### 4.7 Decision-consistency boundary measurement ✅ (2026-04-26)

Pilot couldn't measure RC because T=0 made it degenerate. Reinterpret M2's
RC (under T=0.7) as **per-axis decision stability**. **Done 2026-04-26
on 5-model M8a label-free**.

**Headline**: `cue_level=both` is the dominant decision stabilizer
(+9–16 pp RC) for the 3 saturated models (Qwen 0.84→1.00, Idefics2
0.88→0.99, InternVL3 0.89→0.98). Inverts/vanishes for LLaVA-1.5 +
LLaVA-Next (CLIP encoders). bg_level=ground is a secondary stabilizer
(+3–8 pp). object_level is the weakest stabilizer.

**Reading**: saturation is not just a behavioral PMR ceiling but also a
**decision-stability ceiling** — non-CLIP models converge to the same
PMR call across all 5 seeds when cues fire. CLIP-based models retain
seed-level variance even under strong cues. Separate signature of the
H-encoder-saturation reframe.

Doc: `docs/insights/sec4_7_rc_per_axis.md` (+ ko). Figure:
`docs/figures/sec4_7_rc_per_axis.png`.

### 4.8 PMR scaling

Per-model PMR for H-class (Qwen2.5-VL-7B/32B/72B) and LLaVA-1.5-7B/13B. Does MechBench (Zhang et al. 2024)'s "scale doesn't help" claim hold for PMR? Strong interpretability implications for H6.

### 4.9 Label-free prompt

`"What do you see? What might happen next?"` — ask the question **without** the word "ball". Measures H2's language-prior contribution as a null-hypothesis test. Easy addition — `prompts.py` `open_no_label` variant.

### 4.10 Attention visualization UI ✅ (2026-04-25)

Captured attentions → interactive heatmap (notebook-based). Per-stimulus,
per-layer, per-head visual-token attention. For the paper appendix figure.

**Done 2026-04-25** (initial release, Qwen2.5-VL only):
- New `configs/attention_viz_qwen.py` (M8a stim, limit=20, layers (5,15,20,25),
  `capture_lm_attentions=True`).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation` auto-switches
  to `"eager"` when capturing attentions (SDPA does not return attention weights).
- `notebooks/attention_viz.ipynb` — 6-section interactive notebook: load
  capture, per-layer heatmap, image overlay, physics-vs-abstract comparison,
  per-head fine structure, attention-entropy aggregate.
- Capture cost: ~30s + ~7 MB per stim.
- Full doc: `docs/insights/sec4_10_attention_viz.md` (+ ko).

Extending to LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 is a follow-up
that would multiply disk cost (~2 GB total for 5 models × 60 records).

### 4.11 H7 follow-up — label-regime category annotation ✅ (2026-04-26)

Systematically validate "label selects the physics regime". **Done
2026-04-26 (5-model M8d, kinetic / static / abstract / ambiguous via
classify_regime)**.

**Headline**: LLaVA-1.5 cleanly selects regime by label (`person × no
label` 40% kinetic + 40% static; `person × physical` 62% kinetic;
`car × abs / silhouette` 28% kinetic + 70% ambiguous). Qwen + Idefics2
+ InternVL3 saturated kinetic everywhere except `person × exotic`
(statue): Qwen ~30% static, **InternVL3 ~65% static** (strongest
single label-driven static commit in the project — saturated-encoder
architectures defer to language when the label uniquely disambiguates).
LLaVA-Next intermediate with a 3-way split on `person × exotic`
(30% kinetic + 25% static + 25% abstract). 5-model gradient is the
granular form of the M9 H7 finding.

Doc: `docs/insights/sec4_11_regime_distribution.md` (+ ko).
Figure: `docs/figures/sec4_11_regime_distribution_5model.png`.

**Still open**: 5-category fine-grained regime (gravity-fall / gravity-
roll / orbital / inertial / static) for M2 circle-only data; M8a 5-shape
extension to classify_regime (would need new keyword sets per shape).

---

## 5. Reference rules during work

**At the start of every new session** (future Claude or user):

1. Read this file (`references/roadmap.md`) first to identify "which milestone are we on?".
2. Check the current milestone's **success criteria** and **blocking issues**.
3. Detailed technical questions: drill down `docs/architecture.md` → `docs/next_steps.md` in order.
4. For the latest experimental results: `docs/experiments/m{N}_*.md` (numbers) + `docs/insights/m{N}_*.md` (per-milestone deep dive).

**Insights file convention**: every major milestone produces one `docs/insights/m{N}_<slug>.md` written in English (with a `_ko.md` Korean translation). Include: link to the raw numbers, interpretation, hypothesis-scorecard updates, and paper implications. Current tree:
- `docs/insights/m1_pilot.md` — M1 pilot (originally `docs/05_insights.md`)
- `docs/insights/m3_encoder_boomerang.md` — M3 encoder boomerang
- `docs/insights/m4_logit_lens.md` — M4 LM logit lens
- `docs/insights/m4b_label_free.md` — M4b label-free prompt H2 null test
- `docs/insights/m4c_fc_label_free.md` — M4c forced-choice label-free (FC version of H2 null test)
- `docs/insights/m5_vti_steering.md` — M5a VTI steering causal intervention
- `docs/insights/m5a_ext_bidirection_and_label.md` — M5a extensions (negative α, label × steering, bidirectionality recheck)
- `docs/insights/m6_cross_model_llava.md` — M6 round 1 (LLaVA-1.5 cross-model H2 + H1 + H7)
- `docs/insights/m6_r2_cross_model.md` — M6 round 2 (InternVL3 behavioral + LLaVA captures + FC logit ratio)
- (M5b, M6 r3+ ... to be added)

**Bilingual file convention**: `references/project.md`, `references/roadmap.md`, every `docs/insights/*.md`, and the various `docs/*.md` reference docs all have an English canonical `*.md` and a Korean translation `*_ko.md`. English is authoritative; if translations drift, English wins. New content → write English first, then translate to Korean.

**Whenever a milestone completes**:

- Update §2 status column (▶ → ✅) and record the completion date.
- Add "verified facts / blocker resolution / new hypothesis" to the milestone's §3 section.
- If a new hypothesis emerges, add an `H*` row to the §1.3 scorecard.
- Add a per-run entry to `docs/experiments/m{N}_<slug>.md` (one file per run, in addition to this file).

**When a hypothesis is refuted or revised**:

- Change the status in §1.3 with a one-line reason.
- Major revisions go in this ROADMAP rather than `references/project.md` (the canonical project doc is read-only spec).

**When a new idea emerges**:

- Add it numbered to §4 immediately. Decide later which milestone to slot it into.

---

## 6. Change log

| Date | Change | Commit |
|---|---|---|
| 2026-04-24 | First draft — M0/M1 complete, M2 prepared | `23171b6` |
| 2026-04-24 | M2 complete: scorecard updates (H1 → supported, H2 → quantified, H4 → supported, H5 → mixed, H6 → supported-revised, H7 new); M3 set as next milestone; H7 follow-up added to §4. | `1d17252` |
| 2026-04-24 | M3 complete: vision encoder probing — boomerang confirmed (encoder AUC = 1.0 / behavioral 0.28-0.95); M4 set as next milestone. | `1205821` |
| 2026-04-24 | M4 complete: LM logit lens + per-layer probe. LM AUC 0.94-0.95 across all layers (peak L20 = 0.953); label drives physics margin from L5; M5 set as next milestone. | `2abdc32` |
| 2026-04-24 | M5a complete (VTI steering): L10 α=40 flips 10/10 of `line/blank/none` from D (abstract) to B (physical-static). "Object-ness" direction causally confirmed. M5b (SIP+SAE) and M6 still to do. | `61ffd29` |
| 2026-04-24 | Repository restructure: `references/`, `docs/{insights,experiments,figures}/` scheme; everything bilingual (English canonical + `_ko.md` translation). | `963e219` |
| 2026-04-24 | M5a-ext Exp 1+2 complete: negative α at ceiling (null result — later found to be a ceiling artifact) + label=ball swap on line/blank/none (clean B→A flip). H-direction-bidirectional added (initially as "one-way activator"), H-regime upgraded to supported. | `9a0ed86` (merge) |
| 2026-04-25 | M5a-ext Exp 3 (bidirectionality recheck on `textured/blank/none` moderate baseline): −α=40 → 10 B uniformly across (line/textured) × (ball/circle). H-direction-bidirectional revised to "regime axis within physics-mode" (+α kinetic, −α static, baseline D below threshold). H-regime refuted in original form and narrowed to an H7 qualifier. | `f8f0fdd` |
| 2026-04-25 | M4b complete: label-free prompt as H2 null test on M2 stimuli. Paired PMR(ball) − PMR(_nolabel) = +0.006 ≈ 0; PMR(circle) − PMR(_nolabel) = −0.065. **H2 revised** — language prior is asymmetric (circle override, not ball enhancement). M4 visual-token capture is prompt-independent (causal-attention artefact); switching-layer collapse is structural. | `e97db16`, `990ddf7` |
| 2026-04-25 | M6 round 1 complete (LLaVA-1.5-7B cross-model): paired PMR delta vs label-free → ball +0.475, planet +0.244, circle +0.173 (all positive). **H2 re-revised — visual-saturation hypothesis**: M4b's "circle suppression only" is Qwen-specific; LLaVA shows the original H2 because its visual prior is unsaturated. H1 S-curve cleanest on LLaVA (0.51 → 0.81). H7 replicates cross-model (planet GAR << ball GAR in both). FC excluded — LLaVA returns "A" for every cell. | `c1b885f` |
| 2026-04-25 | M4c complete (forced-choice label-free): new `forced_choice_no_label` variant. Qwen reproduces M4b under FC (ball ≈ no-label, circle suppresses harder, planet newly suppresses via FC's gravity-centric option set). Qwen open-vs-FC paired delta at no-label = −0.131 (H4 measurable without label confound). LLaVA "A" bias persists under re-template (477/480) — confirmed model-level pathology. | `70dc39c` |
| 2026-04-25 | M6 round 2 complete (3 sub-deliverables): r2a InternVL3 cross-model (paired delta +0.010 for every label, fully saturated), r2b LLaVA-1.5 captures (vision encoder AUC ~0.73, LM AUC ~0.75 — boomerang gap is Qwen-specific because LLaVA encoder is the bottleneck), r2c FC logit-ratio (LLaVA "A" bias is at logit level — 90% rows only A survives top_p, not just greedy). **New H-encoder-saturation hypothesis** anchors the 3-model H2 pattern to vision encoder probe AUC. | `47f4b18` |
| 2026-04-25 | Roadmap re-prioritization: external validity over depth. New milestones **M8a (non-circle synthetic shapes), M8c (real photographs), M8d (non-ball physical-object categories)** added at top priority. **§4.5 (encoder swap), §4.6 (counterfactual stimulus generation), §4.10 (attention viz UI)** promoted to next-tier priority. M5b (SIP+SAE) and M6 r3+ demoted to optional pending M8 + 4.5/6/10 results. New M9 (generalization audit) added as the consolidation milestone after M8 + M6 r3+. | `cfbe5a2` |
| 2026-04-25 | **M8a complete (non-circle synthetic shapes)**: 5 shapes × Qwen + LLaVA, 4 inference configs (labeled + label-free per model), 400 stim, ~43 min. Pre-registered scoring **strict**: Qwen 1/4 PASS (visual-saturation Δ borderline), LLaVA 4/4 PASS. The asymmetry *is* the cross-shape validation of the visual-saturation hypothesis: saturated encoder → ceiling effect → no headroom for ramp/label/gravity-prior to operate; unsaturated encoder → all four measurable. H1 (ramp) and H7 (label-role) revised to **unsaturated-only** (LLaVA-clean, Qwen-suppressed). H-encoder-saturation now cross-shape-validated (was 3-model correlational). Triangle's `wedge` and polygon's `polygon` exposed as label-design weak points; flagged for M8c follow-up. | `a83267c` |
| 2026-04-25 | **M8d complete (non-ball physical-object categories)**: 3 categories (car/person/bird) × 4 abstraction × 2 bg × 2 cue × **2 events** × 5 seeds = 480 stim; Qwen + LLaVA, labeled + label-free arms = 3840 inferences in **31.9 min** on GPU 0. Pre-registered strict: Qwen 0/3 H7 binary (ceiling), LLaVA **3/3 H7 ✓** (car +0.525 / person +0.138 / bird +0.550 PMR_regime physical−abstract). Underneath Qwen ceiling, regime distribution shows figurine 17.5 % static / statue 22.5 % static. H1 fails on both (ramp is shape-axis-specific). H-encoder-saturation cross-category-validated. New `classify_regime` keyword classifier (5.6 % hand-annotation error, well below 15 % threshold). H7 promoted from "circle-only" to "cross-category" (paper headline); regime-distribution becomes a methodological contribution that rescues H7 signal from binary saturation. | `f7d0375` |
| 2026-04-25 | **M8c complete (real photographs)**: 60 photos (12 × {ball, car, person, bird, abstract}) from COCO 2017 + WikiArt; Qwen + LLaVA × labeled + label-free = 480 inferences in **5 min** on GPU 0. **Key finding**: photos REDUCE Qwen PMR(_nolabel) by 18-48 pp across all 4 physical categories — synthetic-stim minimality is a co-factor of behavioral saturation, not just encoder representation. LLaVA H7 partially holds on photos (2/4 binary). LLaVA person photo PMR rises +39 pp vs synthetic (encoder finally recognizes humans). H-encoder-saturation refined to "encoder representational saturation AND input-context simplicity"; M6 r2 linear-probe AUC remains valid as the encoder-saturation marker, but behavioral PMR(_nolabel) is no longer a pure encoder readout. | `c568497` |
| 2026-04-25 | **M8e complete (cross-source paired analysis)**: consolidates M8a + M8d + M8c into single (model × category × source_type) view. Headline figure `m8e_cross_source_heatmap.png` is paper-ready Table 1 candidate. Confirms cross-source PMR shifts: Qwen photos universally lower, LLaVA category-asymmetric. | `87c990c` |
| 2026-04-25 | **§4.5 cross-encoder swap complete (Idefics2)**: Idefics2-8b (SigLIP-SO400M + Mistral-7B) on M8a stim → 1600 inferences in 8 min on GPU 0. **Mean PMR(_nolabel) across 5 shapes: Qwen 0.838 / LLaVA 0.175 / Idefics2 0.882.** Idefics2 patterns identically with Qwen on PMR + H7 (1/5 vs 1/5 strict). LLaVA outlier. **H-encoder-saturation causally confirmed at the encoder-family level** — encoder type (SigLIP vs CLIP) drives PMR ceiling regardless of LM (Qwen2-7B vs Mistral-7B). | `304e927` |
| 2026-04-25 | **§4.5 ext: Idefics2 on M8d + M8c**: 4 additional configs (Idefics2 M8d labeled + label-free, M8c labeled + label-free) → 2160 inferences in 11 min on GPU 0. Idefics2 M8d mean PMR(_nolabel) **0.890** matches Qwen **0.869** (vs LLaVA 0.331); Idefics2 M8c **0.417** vs Qwen **0.550** vs LLaVA **0.283** — all 3 models compress on photos. Cross-stim encoder-swap confirms SigLIP saturates synthetic, CLIP doesn't, all converge on photos. | `3503cd3` |
| 2026-04-25 | **M9 complete (generalization audit / paper Table 1)**: 9 (model × stim) cells × bootstrap CIs (5000 iters). **Robust headlines**: (1) encoder family causes synthetic-stim ceiling (SigLIP CIs [0.80, 0.92] vs CLIP [0.14, 0.37], fully separated); (2) photos compress encoder gap (all 3 → [0.18, 0.67]); (3) H7 robust only LLaVA-on-synthetic. **Suggestive (demoted)**: Idefics2 M8d H7 CI [+0.000, +0.094] just touches 0 — LM-modulation possible but not paper-defensible. Replaces PASS/FAIL binarization with bootstrap CIs — reveals "Qwen 1/5 PASS" pattern was a noise-floor artifact. New **H-LM-modulation** hypothesis flagged. | `6210b13` |
| 2026-04-25 | **M6 r3 complete (Idefics2 vision-encoder probe closes the AUC↔PMR chain)**: 400 M8a stim × 4 SigLIP-SO400M layers captured in 88 s; per-layer logistic-regression probe on per-stim PMR yields **AUC 0.93** (peak 0.948 at layer 9). The 3-point AUC ladder is now **Qwen 0.99 / Idefics2 0.93 / LLaVA 0.73** — SigLIP family clusters at saturation, CLIP at headroom. H-encoder-saturation chain `encoder family → AUC → PMR → H7` empirically grounded at all 4 nodes × 3 model points. | `1a4313a` |
| 2026-04-25 | **M6 r4 complete (InternVL3 InternViT probe → 4-model chain)**: 1200+400 M8a inferences in 12 min + 400 × 4 InternViT layers captured in 47 s + probe → **InternVL3 AUC 0.89 / PMR(_nolabel) 0.92**. 4-point chain: Qwen 0.99/0.84 → LLaVA 0.73/0.18 → Idefics2 0.93/0.88 → InternVL3 0.89/0.92. Three distinct non-CLIP encoder families (SigLIP, SigLIP-SO400M, InternViT) cluster at AUC ≥ 0.89; only CLIP-ViT-L falls below. H-encoder-saturation generalizes to **non-CLIP-general** (was SigLIP-specific in M6 r3). Required `_resolve_vision_blocks` fix for InternVL3's `vision_tower.encoder.layer` (singular) attribute. | `d3840ac` |
| 2026-04-25 | **Apples-to-apples 4-model M8a-stim AUC**: re-captured Qwen + LLaVA on M8a stim (replacing the M2-stim M6 r2 AUC numbers in the 4-model chain). Updated chain: Qwen 0.88/0.84, LLaVA 0.77/0.18, Idefics2 0.93/0.88, InternVL3 0.89/0.92. The M6 r2 0.99/0.73 numbers reflected M2's bimodal line/ground split (more probe-friendly than M8a's wider per-cell PMR range). Headline (non-CLIP cluster ≥ 0.88, CLIP at 0.77) holds in both stim sources; the AUC↔PMR mapping is nonlinear (≈0.10 AUC gap → 0.65 PMR gap) consistent with a saturation threshold near AUC 0.85. | `7e7f101` |
| 2026-04-25 | **M6 r5 complete (M8c photo encoder probe — 4-model cross-stim)**: InternVL3 M8c inference (180+60 in 2 min) + 4 model × 60 photo captures (~3 min) + behavioral-y + stim-y probes. Behavioral-y AUC inverts cross-stim (Qwen 0.88→0.44, LLaVA 0.77→0.86, Idefics2 0.93→0.77, InternVL3 0.89→0.59), but stim-y AUC stays at 1.0 across all 4 models on photos. Confirms cross-stim that encoder discriminability is uniform — final cross-stim confirmation of architecture-level reframe. | `166c053` |
| 2026-04-25 | **M6 r6 complete (LLaVA-Next-Mistral 5th model point, 2nd CLIP)**: LLaVA-v1.6-Mistral-7b on M8a (400 labeled + 400 label-free + 400 stim × 4 layers × 5 tile capture). Mean PMR(_nolabel) **0.700, 95% CI [0.65, 0.74]**, between LLaVA-1.5 floor [0.14, 0.21] and saturated cluster [0.80, 0.92]. Behavioral-y AUC 0.81; stim-y AUC = 1.0 across all 4 targets. **5-model M8a chain locked** (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3). The 2nd CLIP point rules out vision-encoder-family as sole determinant of PMR. Reported as 5th observation, not LM-controlled counterfactual: 4 architectural axes change simultaneously (AnyRes tiling, fusion projector, training, LM family). H-encoder-saturation **architecture-level confirmed at 5 model points + 2 CLIP points**. | `b2434d4` |
| 2026-04-25 | **M6 r6 cross-stim addendum**: LLaVA-Next on M8d + M8c (1620 inferences, ~16 min on GPU 0). M8d PMR 0.625 [0.58, 0.67] preserves mid-band; M8c PMR 0.417 statistically equal to Idefics2 0.417 (photo-collapse generalizes to 5th model). **H7 cross-stim**: M8a +0.26 (5/5 PASS, mid-strong), M8d −0.05 (CI [−0.10, −0.01], noise floor), M8c +0.02. M8d H7 collapse is **not** a saturation effect (PMR 0.625 well below ceiling); same-encoder-family architecture switch attenuates H7 with PMR headroom remaining. H-encoder-saturation reframe holds at 5 models × 3 stim. H-LM-modulation still suggested-only (two-Mistral M8d H7 clustering at ≈0 is multi-axis-confounded per advisor). | `524e32b` |
| 2026-04-26 | **§4.6 complete (VTI-reverse counterfactual stim, Qwen2.5-VL)**: pixel-space gradient ascent on the post-processor `pixel_values` (T_patches × 1176, the patch-flattened normalized representation; bypasses non-differentiable PIL → patch preprocessing) maximizing `<mean(h_L10[visual]), v_L10>`. Sweep: 5 baseline circle stim × {bounded ε ∈ {0.05, 0.1, 0.2}, unconstrained, random unit dir × 3 at ε = 0.1} × 200 Adam steps lr=1e-2 = 35 runs × ~30 min. **Result: 5/5 v_L10 flips PMR 0→1 at ε = 0.05** (pre-registered ≥ 3/5); **0/15 random-direction flips at matched ε = 0.1**. Random-direction final projections (73–85) are comparable in magnitude to bounded ε = 0.1 v_L10 (101) — directional specificity, not magnitude, controls the regime flip. Random-control responses ("no indication of movement") exposed an over-permissive PMR scorer that matched the "mov" stem inside an abstract sentence; added asymmetric abstract-marker patterns (`remain stationary`, `no indication of mov`, `no indication of motion`) — verified asymmetric: 0/20 v_L10 hits, 14/15 random hits; headline replicates with the pre-fix scorer. New module `src/physical_mode/synthesis/counterfactual.py` (gradient_ascent + reconstruct_pil with inverse permute matching Qwen2VLImageProcessor's forward `(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)`). PMR test suite extended from 51 → 54 cases. **`v_L10` is encodable in the image** — the M5a shortcut lives on the pixel-driven path. Deep dive: `docs/insights/sec4_6_counterfactual_stim.md` (+ ko). | `9ec147e` |
| 2026-04-26 | **M6 r7 (M2 cross-model) complete**: extended M2 protocol (5-axis factorial, T=0.7, 480 stim × 3 labels) to LLaVA-Next + Idefics2 + InternVL3 with LM activation captures (Qwen + LLaVA-1.5 existing). Sequential GPU 1 chain: ~22 min for 5 capture runs + 5 label-free runs. **5-model PMR(_nolabel) ladder on M2 stim (apples-to-apples)**: LLaVA-1.5 0.383 [0.34, 0.43] / LLaVA-Next 0.790 [0.75, 0.83] / Qwen 0.938 [0.92, 0.96] / Idefics2 0.967 [0.95, 0.98] / InternVL3 0.988 [0.98, 1.00]. H1 ramp clean only on LLaVA-1.5 (+0.30 range); ceiling on others. **H2 paired-delta shows 3 distinct architecture-conditional patterns** — LLaVA-1.5/LLaVA-Next all positive (classical H2), Qwen/Idefics2 asymmetric (circle/planet < 0; "circle override"), InternVL3 ≈ 0 (saturated). Per-model v_L10 extracted; LLaVA-1.5 alone has class-balanced direction (n_neg=105); LLaVA-Next/Idefics2/InternVL3 have n_neg=9/5/1 — too saturated on M2 for clean v_L10. Insight: `docs/insights/m2_cross_model.md` (+ ko). | `4bd4623` |
| 2026-04-26 | **§4.6 cross-model complete (null — pixel-encodability is encoder-saturation specific)**: (a) Transfer test: Qwen-derived synth stim → 4 non-Qwen models. **0/140 flips** across configs. LLaVA-1.5 / LLaVA-Next / Idefics2 baseline=synth=0; **InternVL3 negative transfer** (baseline 1.0 → unconstrained 0.2 at large ε). (b) LLaVA-1.5 per-model gradient ascent: only model with class-balanced v_L10 from M2. 35 runs in 9.8 min on GPU 1. **0/5 v_L10 flips at every ε**, 0/15 random. **Critical observation**: gradient ascent succeeds at projection level (8 → 150-200 for v_L10, comparable to Qwen 43-180) but PMR doesn't flip — dissociation between *projection level* and *behavior level*. **H-shortcut and H-direction-specificity revised → Qwen-scoped**. Pixel-encodability is encoder-saturation specific, paralleling M9 PMR-ceiling and §4.7 decision-stability ceiling. New module `src/physical_mode/synthesis/counterfactual_llava.py` (standard-CLIP variant). Insight: `docs/insights/sec4_6_cross_model.md` (+ ko). | `ec2aa77` |
| 2026-04-26 (evening) | **H9 + M5b SIP+patching complete**: (a) **H9 cross-model layer-wise probe AUC** — 5 models × 5 captured layers (5/10/15/20/25); Qwen plateaus at AUC 0.97 from L5, LLaVA-1.5 plateaus at 0.77, never crosses 0.85; LLaVA-Next 0.73→0.79; Idefics2 / InternVL3 saturated (n_neg = 5, 1 — fullfit fallback). H9 reframed: not "earlier-vs-later" but "saturated encoders plateau before L5; unsaturated never reach saturated plateau." Insight: `docs/insights/h9_layer_switching.md` (+ ko). (b) **M5b SIP + activation patching on Qwen2.5-VL** — 20 SIP pairs (cue=both clean × cue=none corrupted, M2 stim) × 28 LM layers × ~8 min on H200. **Sharp L10 boundary**: L0-L9 patching → 20/20 physics recovery (IE=+1.0); L10-L11 → 12/20 (0.6); L12 → 6/20 (0.3); L13 → 2/20 (0.1); L14+ → 0/20. Refines M5a (L10 single-causal-layer steering) into L10 = decision-lock-in *boundary*, with information present at every L0-L9. H10 ("2-3 narrow IE bands") supported with revision: one contiguous L0-L9 range. H-locus refined. New module `scripts/m5b_sip_activation_patching.py` (forward-hook visual-token replacement at prefill). Insight: `docs/insights/m5b_sip_activation_patching.md` (+ ko). | (this commit) |
| 2026-04-26 (evening) | **§4.6 cross-model REVISED — pixel-encodability is NOT Qwen-only (overnight followup)**: discovered that the morning's Qwen-scoped finding was a **wrong-layer-choice artifact**. M2 captures had n_neg = 1-9 for saturated models; new M8a captures (3 models × ~50 min on GPU 1) gave n_neg = 100-280, much cleaner. M8a vs M2 v_L cosine analysis: Idefics2 ~0.79 (same direction; class imbalance robust), InternVL3 ~0.7+, LLaVA-Next ~0.4 (moderate). Class imbalance was *not* the issue. **The fix is layer choice**: LLaVA-1.5 §4.6 layer sweep (L5/L15/L20/L25; L10 was prior null) — **L25 yields 5/5 v_L flips at ε=0.2 and 4/5 at ε=0.1**, with random controls 0/15 at every layer. LLaVA-1.5 has 32 LM layers (L25 = 78% depth) vs Qwen 28 (L10 = 36%) — different relative depth. Sample LLaVA-1.5 L25 ε=0.2 response: "The circle will be hit by a ball." (vs baseline "filled in with color"). Insight: `docs/insights/sec4_6_cross_model_revised.md` (+ ko). The prior `sec4_6_cross_model.md` doc has a REVISION NOTICE header pointing here. **Open**: Qwen layer sweep (does Qwen also have a 2nd shortcut layer at L25?), per-model gradient ascent on LLaVA-Next/Idefics2/InternVL3 (need their custom counterfactual_<model>.py). | `165a525` |
| 2026-04-26 (night) | **M5b cross-model SIP + attention/MLP knockout complete**: (a) **Cross-model SIP+patching on LLaVA-1.5** (n=15 × 32 layers, ~12 min on GPU 1) — IE-curve shape replicates Qwen's sharp-then-declining profile, but **decision-lock-in starts at L20 (62.5% relative depth) vs Qwen's L10 (36%)**. Idefics2 / InternVL3 too saturated on M2 open-prompt for SIP construction (n_neg = 0, 0). H-locus reframed: each VLM has its own decision-lock-in layer at a model-specific relative depth; locus exists cross-model but at different positions. Insight: `docs/insights/m5b_sip_cross_model.md` (+ ko). (b) **Attention + MLP knockout (necessity test)** on Qwen2.5-VL: 20 clean SIP stim × 28 LM layers × 2 ablations × ~18 min on GPU 1. **Attention knockout: IE = 0 at every layer L0-L27** — single-layer attention is fully redundant; residual stream + surviving MLPs reconstitute it. **MLP knockout: L9 IE = +1.0 (uniquely necessary, 0/20 retain physics)**, with partial necessity ring L8 +0.4 / L10 +0.6 / L11 +0.4 / L14 +0.4. Triangulation with M5b SIP (sufficiency) + M5a (steering): **L9 MLP *constructs* physics-mode commitment in residual stream; L10 *reads* it via attention**. Off-by-one M5a/M5b reconciliation = two views of same decision boundary — the most localized causal finding in project. H-locus refined twice; H10 ("2-3 narrow IE bands") supported with one dominant L9 MLP band + 4 partial echoes. New module `scripts/m5b_attention_mlp_knockout.py` (prefill-only forward hook zeroing self_attn or mlp output). Insight: `docs/insights/m5b_attention_mlp_knockout.md` (+ ko). | `a96029b` |
| 2026-04-27 | **M5b per-head attention knockout complete (null result, the right kind)**: 20 clean SIP stim × 7 LM layers (L8-L14) × 28 attention heads = **196 (L,h) ablations all show IE = 0** — every single head is individually dispensable; ~36 min on GPU 1. The hook zeroes the slice `x[..., h*head_dim:(h+1)*head_dim]` of the o_proj input at prefill (clean per-Q-head ablation; GQA-safe). Ruling out the "single decision-carrying head" hypothesis at the L8-L14 partial-MLP-necessity zone. **Mechanism crystallized**: L9 MLP *constructs* physics-mode commitment in the residual stream; L10 *reads* it via attention that is genuinely diffuse — no specific head matters. H10 ("2-3 narrow IE bands") fully resolves: 1 dominant L9 MLP band + 4 partial echoes; attention has *zero* narrow IE bands at *any* resolution (single-layer or per-head). H-locus triply-refined. Limitation: single-head only — multi-head combination ablation (top-N visual-attention heads in L9 simultaneously) might reveal cumulative necessity. New module `scripts/m5b_per_head_attention_knockout.py` (forward_pre_hook on o_proj). Insight: `docs/insights/m5b_per_head_attention_knockout.md` (+ ko). | `83a0995` |
| 2026-04-27 | **M5b SAE intervention on Qwen vision encoder complete (positive result, advisor-checked)**: trained 5120-feature SAE (4× expansion, tied weights, input z-score, λ=1.0, 5K steps in 1.1 min on H200) on 622K visual tokens from `vision_hidden_31` (last SigLIP layer, pre-projection, 1280 dim). Final recon = 0.023, 7.3% active per token. Ranked features by `delta = mean(z | physics) − mean(z | abstract)` using mvp_full predictions (310 phys / 19 abs stim). **Causal intervention** (n=20 clean SIP stim × hook on `model.visual.blocks[-1]` that subtracts target features' raw-scale contribution via Bricken et al. trick, avoiding lossy decode round-trip): top_k=1 → 1.0, top_k=5 → 0.6, top_k=10 → 1.0 (non-monotone, unresolved), **top_k=20 → 0.0 (full break, 20/20 abstract; mass=49.23)**. **Initial bottom-of-ranking random control was unmatched on magnitude (mass ≈ 1% of top-20 — L1 penalty kills inactive features); advisor flagged this. Corrected: mass-matched random k=20 (mass=40.97 = 83% of top-20) → 1.0 (20/20 retain physics)** — confirms direction-specificity, not magnitude-driven encoder collapse. Single mass-matched set obtained (heavy-tailed mass distribution prevented 3-set replicate; binary outcome unambiguous). Caveats: feature 3313 (mass=14, ~3× next) is a high-baseline outlier likely "general image content"; future ranking by Cohen's d would filter. Triangulated mechanism closes: input → encoder ~20 physics-cue features → L0-L9 → L9 MLP commitment → L10 read-out → letter. New module `src/physical_mode/sae/{train,feature_id}.py`; drivers `scripts/sae_train.py`, `scripts/sae_intervention.py`. Insight: `docs/insights/m5b_sae_intervention.md` (+ ko). M5b now fully complete (all 5 sub-tasks). | (this commit) |
| 2026-04-28 | **Overnight chain B → A → C complete (5-model §4.6 n=10 + Qwen 32B M2)**: launched at 17:40 KST 2026-04-27, finished 21:42 KST same day (~4 hr wall). (B) Re-ran Qwen + LLaVA-1.5 + LLaVA-Next §4.6 at n=10 stim/layer (`line_blank_none_fall_*` exhausts at 10, so n=50 was infeasible). n=10 tightens Wilson CIs from n=3 [0.44, 1.00] to [0.69, 1.00] for full-flip cells. **n=10 reveals over-confidence of n=3-5 100 % claims**: LLaVA-1.5 L25 4/5 → 4/10 (Wilson [0.17, 0.69]); Qwen L15/L20 3/3 → 8/10. (A) Added Idefics2 + InternVL3 modules (`counterfactual_idefics2.py` 5-tile + pixel_attention_mask, `counterfactual_internvl3.py` single 448×448) + drivers + ran. **Idefics2 anomaly**: 0 clean shortcuts across L5/10/15/20/25 despite v_L projection rising +38 in every run — falsifies "encoder saturation alone causes shortcut breadth". Two candidate explanations (deeper-layer L26-31 untested vs perceiver-resampler bottleneck) unresolved. **InternVL3 protocol-saturated**: `line_blank_none_fall_*` baseline_pmr=1.0; the §4.6 "circle" prompt + InternVL3's saturated InternViT pipeline leaves no abstract-baseline stim in M2 to flip. (C) Qwen 32B on M2 (open, 1440 stim, 16 min wall): aggregate PMR 0.926 ≈ 7B 0.931 — 5× scaling does not help. New §4.8 insight `docs/insights/sec4_8_pmr_scaling.md`; main shift is `abstract_reject` 0.002 → 0.065 and label gap halving (H2 weakened but not dissolved). H-shortcut downgraded from "capacity scales with saturation" to "model-specific shortcut profile within CLIP/SigLIP+Qwen subset; Idefics2 anomaly". 5-model summary script + figure regenerated; 5-panel layout in `docs/figures/sec4_6_cross_model_layer_sweep.png`. New chain script `scripts/overnight_b_a_c.sh`. | (this commit) |
| 2026-04-28 (evening) | **M5b SAE intervention cross-model complete (3 of 5 models break, 2 LLaVA NULL)** — discovered LLaVA `vision_feature_layer=-2` and Idefics2 `last_hidden_state` mean prior cross-model SAEs were trained on the wrong layer (round 1 used `vision_hidden_23` uniformly, but LLaVA discards layer 23, Idefics2 uses layer 26). Round-2 retrain at correct per-model layer: LLaVA-1.5 vis22, LLaVA-Next vis22, Idefics2 vis26 (InternVL3 vis23 already correct). New flags `sae_intervention.py --prompt-mode open --vision-block-idx --test-subset --stimulus-dir`; `sae_train.py --pmr-abs-threshold` for saturated-baseline models; `feature_id.py` chunked encoding to handle 5-tile Idefics2 (2.5 M tokens) without OOM. Per-model OPEN+circle baseline-PMR=1 stim cells: Qwen filled/blank/both, LLaVA-1.5 shaded/ground/cast_shadow, LLaVA-Next shaded/blank/both, Idefics2 + InternVL3 filled/blank/both. **Results**: Qwen k=20 break (0.4 % of 5120 features, original FC); Idefics2 k=160 break (3.5 % of 4608); InternVL3 k=160 break (3.9 % of 4096); **LLaVA-1.5 NULL at any k ≤ 160; LLaVA-Next NULL at any k ≤ 160** (CLIP encoder cluster). 3 mass-matched random controls × 5 models all 1.0 (specificity confirmed). Effect concentration tracks M3 vision-encoder probe AUC (Qwen 0.99 > Idefics2 0.93 > InternVL3 0.89 > LLaVA 0.7-0.8) — second downstream signature of H-encoder-saturation. LLaVA-Next M5a positive (LM-side 10/10 flip) + M5b NULL → physics-mode commitment routes through LM, not encoder, in the LLaVA family. New scripts: `scripts/run_m5b_chain_v2_gpu{0,1}.sh` (parallel GPU chains), `scripts/m5b_sae_intervention_cross_model_summary.py` (5-model aggregator). Insight: `docs/insights/m5b_sae_intervention_cross_model.md`. M5b row promoted to (6) sub-tasks complete. | (this commit) |
| 2026-04-28 (afternoon) | **§4.6 Idefics2 deeper-layer disambiguation (L26-L31) + cross-model SAE training (T1a + T1b from `data_audit_2026-04-28.md`)**. **T1b**: fresh M2 LM activation capture at L26/28/30/31 (Mistral-7B 32 layers, 81-97 % depth) via new `configs/cross_model_idefics2_l26_31.py` (10 min wall). v_L extraction `scripts/sec4_6_idefics2_extract_v_L_l26_31.py` (n_pos=470, n_neg=10 from saturated open-prompt M2). 80-run counterfactual sweep `scripts/sec4_6_idefics2_layer_sweep_unified.py --layers 26,28,30,31` × n=10 × 2 configs (~65 min wall on shared GPU 0). **Result: 0/40 v_unit + 0/40 random with 1 isolated noise hit at L28** (line_blank_none_fall_006: "Appear." → "Move." PMR=1, Wilson [0.0025, 0.40]). v_L projection ascends cleanly at every depth (baseline -10.7 → final +27-30 at L26-30; -72 → +163 at L31). **Wrong-relative-depth hypothesis falsified across 9 LM layers (L5-L31, 16-97 % depth)**; **perceiver-resampler is the leading remaining mechanism candidate** (5-model design doesn't isolate projector cleanly — controlled projector-swap test out of scope). H-shortcut framing now: pixel-encodability is **architecture-conditional with projector design (MLP vs perceiver) as the disambiguating axis** — encoder saturation alone is not sufficient. **T1a**: trained 4 cross-model SAEs (LLaVA-1.5 vis23 4096-feat, LLaVA-Next vis23 4096-feat, Idefics2 vis24 4608-feat, InternVL3 vis23 4096-feat) on existing M3 vision activations (no new captures). InternVL3 SAE first inspection: top feature Cohen's d=0.41 (vs Qwen 0.5+ from M5b) — physics-cue features exist but cross-model separation weaker than Qwen's. SAE → physics features universal? remains for follow-up intervention runs. Updated `scripts/sae_train.py` for multi-tile shape + parquet predictions; updated `scripts/sec4_6_cross_model_layer_summary.py` to concat Idefics2 9-layer data. Insight: `docs/insights/sec4_6_cross_model_revised.md` (Idefics2 anomaly resolution section). | (this commit) |
| 2026-04-27 (afternoon) | **§4.6 cross-model 3-architecture layer sweep complete (Qwen + LLaVA-Next added; LLaVA-1.5 already done overnight)**: closes both open items in `sec4_6_cross_model_revised.md`. (a) **Qwen layer sweep** (`scripts/sec4_6_qwen_layer_sweep_unified.py`, single-process): L5/L15/L20 added to existing L10 (morning) + L25 (smoke). 18 runs in ~10 min on GPU 1. **Result**: L10/15/20/25 all 100 % v_unit flip (3/3 each except L10/L25 5/5); L5 marginal 33 %; random 0 across all layers. Qwen has 4-5 shortcut LM layers, not 1. (b) **LLaVA-Next per-model gradient ascent**: new module `src/physical_mode/synthesis/counterfactual_llava_next.py` handles AnyRes 5-tile pixel_values structure (per-element eps clip, base-tile reconstruction). Driver `scripts/sec4_6_llava_next_layer_sweep_unified.py` runs 30 runs (5 layers × 3 stim × 2 configs) in ~25 min. **Result**: L20+L25 both 3/3 (100 %); L15 borderline (1/3 v_unit matches 1/3 random — noise threshold); L5/L10 null. LLaVA-Next has 2 shortcut layers. (c) **Cross-model summary** (`scripts/sec4_6_cross_model_layer_summary.py`): aggregates Qwen + LLaVA-1.5 + LLaVA-Next into single table + 3-panel figure with Wilson CIs. Architecture-level scaling locked: **the *number* of shortcut LM layers tracks encoder saturation** — LLaVA-1.5 (CLIP, AUC 0.73, 1 layer) < LLaVA-Next (CLIP+AnyRes, AUC 0.81, 2 layers) < Qwen (SigLIP, AUC 0.99, 4-5 layers). H-shortcut promoted from "model-conditional layer" to "architecture-conditional with capacity scaling on encoder saturation"; pixel-encodability is now the **third** architecture-level signature of encoder saturation (joining M9 PMR-ceiling and §4.7 decision-stability ceiling). Limitations 1+2 in `sec4_6_cross_model_revised.md` resolved; Idefics2 + InternVL3 per-model modules remain open (predicted to behave like Qwen given their AUC 0.93 / 0.89). New figure `docs/figures/sec4_6_cross_model_layer_sweep.png`. Insight: `docs/insights/sec4_6_cross_model_revised.md` (+ ko). | (this commit) |
| 2026-04-27 (evening) | **M5b SAE intervention revision (Cohen's d + multi-seed random + dense k-sweep + cluster pivot)**: addresses 3 advisor-flagged weaknesses from the morning run. (a) **Cohen's d ranking**: extended `feature_id.py` to compute pooled-std-normalized delta; new `scripts/sae_rerank_features.py` re-derives `feature_ranking.csv` (now `mean_phys/mean_abs/std_phys/std_abs/pooled_std/delta/cohens_d`). Spearman ρ = 0.98 on full 5120 features but ρ = 0.47 on top-50 by delta — ranking is unstable in the high-delta region; **7/20 top-20 turnover**. Feature 3313 drops from rank 3 (delta) to ~rank 50 (Cohen's d) — high-baseline-noise outlier filter works as designed. (b) **Multi-seed mass-matched random**: `sae_intervention.py` extended with `--rank-by {delta,cohens_d}` + multi-seed loop (seeds 42..91, fallback window [0.5×, 2.5×] if main [0.7×, 2.0×] starves). 3 mass-matched sets at top-30 from seed 42 alone: mass 23.4 / 24.7 / 33.4 (top-30 mass=32.7 → 72/76/102 %) → all 3 sets 20/20 in physics-mode (Wilson CI [0.84, 1.00]). (c) **Dense k-sweep (Cohen's d, k ∈ {1,2,3,5,7,10,15,20,30})**: top-30 → 0/20 (Wilson CI [0.00, 0.16]); top-20 → 2/20; top-15 → 20/20 (full recovery); top-5/7/10 → 12/20. (d) **Stim-cluster pivot resolves the non-monotone**: 8 line_blank stim flip together at k=5/7/10 and recover together at k=15; 12 non-line_blank stim never flip in mid-range. The "0.6" mid-rate is *deterministic and stim-cluster-conditional*, not aggregate-rate noise — implies features at rank 11-15 act as "abstract suppressors" whose removal compensates the line_blank-specific damage. Headline tightens from "single random control 1.0 (Wilson CI [0.84, 1.00])" to "3 random controls × 20 stim = 60 trials all 1.0 (aggregate Wilson CI [0.94, 1.00])"; lower-bound gap from 1.0 narrows 0.16 → 0.06 (~2.7× tighter). Cohen's d is now canonical ranking; raw delta retained for back-compat. New script `scripts/m5b_sae_figures.py` (revised figure with Wilson CIs + cluster pivot at `docs/figures/m5b_sae_intervention_revised.png`). Insight + KO doc updated. Limitations 1/2/3 resolved. | (this commit) |
| 2026-04-28 (planning) | **Track B chosen for ICLR/NeurIPS-grade submission**. New planning docs `references/submission_plan.md` (20-week schedule, 3 pillars: Multi-prompt / Controlled architectural counterfactuals / Marr-3-level framing, ICLR 2027 primary) + `references/paper_gaps.md` (4 weaknesses G1-G4 → fix-track mapping). §1.4 venue updated; §3.X Track B priorities table added; new milestone rows M-MP / M-PSwap / M-LMSwap / M-Marr added. World-model framing chosen as broader insight thread (production VLMs as implicit world models; first step of any world model is "what kind of world am I in?"). 5-signature framing decision: **appendix-relegation over pruning** — keep all 5 measurements organized by Marr level (Computational PMR / Representational M3+M4 / Mechanistic M5a+M5b). Drop counting + Michotte + identity prompts from multi-prompt menu. Strategy memory `paper_strategy.md` updated. | (this commit) |
| 2026-04-28 (M-MP Phase 1+2 complete) | **5-model multi-prompt cross-task generalization Phase 1 + Phase 2 ✅**. Phase 1 stratified smoke (n=48 stim × 3 prompts × 3 labels × 5 models = 2160 inferences in ~15 min). Phase 2 full (n=480 stim × 3 prompts × 3 labels × 5 models = 21,600 inferences in ~3 hr chain on GPU 1). **Headline**: H2 paired-delta `PMR(ball) − PMR(circle)` is positive in **all 5 models × 3 prompts = 15 cells** (range +0.006 to +0.344). Saturation × prompt interaction: saturated models (Qwen/Idefics2/InternVL3) → describe most informative; unsaturated CLIP-cluster models (LLaVA-1.5/LLaVA-Next) → open or yesno most informative. 0/2160 unparseable on `meta_phys_yesno`. Phase 1↔2 within ±0.018 PMR (stratified subset unbiased). New: 2 prompt variants in `prompts.py`, 5 multi_prompt_*.py configs, score_meta_yesno + score_describe + score_for_variant in pmr.py, `m_mp_summarize.py` analysis pipeline, `m_mp_describe_label_helper.py` Claude-rater hand-label gate (5/5 PASS, agreement 0.88-1.00 / κ 0.74-1.00). Phase 3 prep complete: `06_vti_steering.py` and `sae_intervention.py` extended with `--prompt-variant` / `--prompt-mode {fc, open, describe_scene, meta_phys_yesno}` flags. Insight: `docs/experiments/m_mp_phase1.md` + `docs/experiments/m_mp_phase2.md`. **Next**: M5b post-projection SAE (Qwen first) → Phase 3 cross-prompt M5a+M5b causal test. | (this commit) |
| 2026-04-28 (M-MP Phase 3 + post-projection SAE complete) | **M-MP Phase 3 cross-prompt M5a + M5b (Qwen + Idefics2 × 2 new prompts) ✅** + **M5b post-projection SAE on Qwen ✅**. Phase 3 results refine the G1 fix from "task-agnostic physics-mode commitment" to **"generative vs categorical mechanism dissociation"** + **architecture-conditional breadth**: (a) Qwen M5a + M5b both flip/break on `describe_scene` (10/10 / 0/10) but NOT on `meta_phys_yesno` (0/10 / NO break) — physics-mode mechanism is generative-language-specific, not categorical-judgment. (b) Idefics2 M5a + M5b NEITHER flips on `describe_scene` (0/10 / NO break) — Idefics2 mechanism is kinetic-prediction-specific (narrower than Qwen). Cross-method M5a-M5b dissociation with same boundaries = mechanism is real, not method artifact. New scorer additions for `score_describe` lexicon: "impact", "about to (fall|hit|impact|land|drop|bounce)", "in motion" phrases. Insight: `docs/experiments/m_mp_phase3.md`. **Post-projection SAE on Qwen**: 14336-feature SAE trained on `model.model.visual.merger` output (3584-dim LM-space, 480 × 324 = 155K tokens, 6 min train). Top-k Cohen-d ablation: k=20 → 0/20 break; k=5 → intact; random ×3 mass-matched all 1.0. Same k=20 threshold as pre-projection (5120-feature SAE on `vision_hidden_31`) — merger preserves physics-mode commitment localization rather than distributing. New: `scripts/m5b_capture_post_projection.py`, `scripts/sae_intervention.py --hook-target {block, merger}`. Insight: `docs/insights/m5b_post_projection_sae_qwen.md`. | (this commit) |
