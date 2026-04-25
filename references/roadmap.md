# ROADMAP — Physical-Mode Activation

> **Role of this document.** Single source of truth for "where we are now and what's next." When starting a new session, **read this file first**. Update §3 every time a milestone completes. Detailed material is linked out to the relevant doc / code.
>
> - Research philosophy & hypotheses (canonical): `references/project.md`
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
| **H1** | PMR rises S-shaped along the abstraction axis (line → textured); 3D shading and ground introduction produce the largest step-changes. | **supported, unsaturated-only AND shape-axis-only (cross-shape, M8a + M8d)** | M2 (Qwen): monotone across 4 object_levels (0.744 → 0.832) but saturated. M6 (LLaVA-1.5, 2026-04-25): clean S-curve 0.51 → 0.81. **M8a (2026-04-25)**: cross-shape strict scoring — Qwen 3/5 (square/triangle fail; ceiling-effect compression), LLaVA 4/5 (polygon fail at filled→shaded inversion). **M8d (2026-04-25)**: cross-category strict scoring — Qwen 0/3 (ceiling), LLaVA 0/3 (non-monotone). H1 is a property of the geometric-shape ↔ named-object axis: every abstraction level of a car/person/bird is already category-recognizable (line car still has wheels), so visual detail doesn't change the affordance — only surface realism. The ramp is operationally measurable only when the vision encoder is unsaturated AND the input is on the abstract-shape ↔ physical-object axis. |
| **H2** | The "ball" label substantially raises PMR even on line drawings → independent contribution of the language prior. | **fully validated, three-point + encoder-anchored** | Qwen (saturated, M4b): ball/planet ≈ 0, circle = −0.065. LLaVA (unsaturated, M6 r1): ball +0.475, planet +0.244, circle +0.173. InternVL3 (super-saturated, M6 r2a): all labels +0.010 ≈ noise. The 3-model paired-delta pattern matches the encoder-saturation prediction. M6 r2b shows the saturation difference is rooted in the vision encoder probe AUC (Qwen 0.99 vs LLaVA 0.73). M4b's "circle suppression only" pattern is the Qwen-specific symptom of the encoder being already saturated. |
| **H3** | Scene inconsistency degrades RC. | **untested** | Axis E was dropped from M2 (complexity); reserved for a focused mini-experiment. RC infrastructure was validated in M2 (103/288 cells with RC<1). |
| **H4** (pilot-derived) | The open vs forced-choice PMR gap is a stable signature of the **language-prior ↔ visual-evidence** conflict. | **supported — extended** | M2: gap present at every object_level (line 32 pp → textured 22 pp). Higher abstraction ⇒ larger gap — a structural prediction that abstraction weakens visual evidence so the language prior dominates more. Next test: ST5 cross-model. |
| **H5** (pilot-derived) | The single ground line causes a **larger** PMR shift than going from no-ground textured ball to with-ground textured ball. | **mixed** | M2: bg delta (blank 0.67 → scene 0.88 = +21 pp) > object delta (line 0.74 → textured 0.83 = +9 pp). Direction matches; however, scene also surpasses ground. |
| **H6** (pilot-derived) | The arrow+shadow cue saturation is driven entirely by **cast shadow alone**; the arrow is closer to annotation. | **supported (revised)** | M2 decomposition: cast_shadow alone = +17.5 pp above none (Kersten ground-attachment cue confirmed); **but the arrow alone also saturates at 0.96** — partially refutes the "arrow = annotation" sub-claim. Arrow is the dominant cue, shadow is secondary. |
| **H7** (M2-derived) | The label does not toggle PMR — it selects **which physics regime** to apply. | **supported, unsaturated-only AND cross-category (M8a + M8d)** | M2 GAR: ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2: label flip @ +α=40 swaps B vs A on `line/blank/none`. M6 r1 + r2a cross-model: `planet GAR << ball/circle GAR` holds in Qwen (0.32 vs 0.71/0.75), LLaVA-1.5 (0.07 vs 0.36/0.15), and InternVL3 (0.43 vs 0.82/0.79) — circle-only. **M8a (2026-04-25)**: cross-shape role-PMR strict scoring — Qwen 1/5 (only square; rest are -0.10 to +0.075 = ceiling-flat), LLaVA 4/5 (triangle fails at +0.025; `wedge` is a weak physical label, not a shape failure). H7-GAR strict: Qwen 1/5, LLaVA 5/5. Orbital-routing dissociation generalizes only when the encoder is unsaturated. **M8d (2026-04-25)**: cross-category role-PMR strict scoring — LLaVA **3/3** (car +0.525, person +0.138, bird +0.550 on PMR_regime physical−abstract; strongest cross-category H7 evidence in the project). Qwen 0/3 binary (ceiling) but regime distribution shows the same pattern at the kinetic-vs-static split: figurine 17.5 % static, statue 22.5 % static (vs ~5 % static for physical labels). The label-selects-regime claim is now category-general, not circle-specific. |
| **H-boomerang** | Encoder knows, decoder gates: vision encoder linearly separates physics-mode classes even where behavior fails. | **Qwen-scoped (revised)** | Holds in Qwen2.5-VL: M3 encoder AUC ~0.99 at every layer; M4 LM AUC ~0.94 at visual tokens; behavioral PMR ~0.93 — small "encoder knows, decoder mildly gates" gap. M5a: causal intervention at L10 flips behavior. **Refuted in LLaVA-1.5** (M6 r2b): vision encoder AUC ~0.73, LM AUC ~0.75, behavioral ~0.78 — flat through pipeline, encoder is the bottleneck. The boomerang as a phenomenon requires encoder saturation. |
| **H-encoder-saturation** (M6 r2-derived; M8c-refined; §4.5-causal; M9-bootstrap; M6 r4 + apples-to-apples M8a-stim; **stim-y check moves locus to architecture-level (encoder + LM)**; **M6 r6 adds 2nd CLIP point**) | Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the **architecture level** (joint encoder + LM), *not* at encoder representational capacity. **Stim-defined y check (2026-04-25)**: all 5 tested encoders (SigLIP, CLIP-ViT-L ×2, SigLIP-SO400M, InternViT) linearly separate factorial cells at AUC = 1.0 (rendered_vs_line, physics_cell_vs_abstract_cell, within-object-level minimal pairs). Encoder discriminability is uniform; what differs is LM-side consumption of encoder output. The 5-model behavioral PMR ladder (non-CLIP: 0.84–0.92; CLIP-LLaVA-1.5: 0.18; CLIP-LLaVA-Next: 0.70) and the behavioral-y probe AUC (0.77–0.93) reflect each LM's reading of encoder output as physics-mode signal — *downstream-conditional*, not encoder-info. **M9 bootstrap CIs** (3 models × 3 stim sources): non-CLIP CIs [0.80, 0.92] vs CLIP [0.14, 0.37] on synthetic stim — fully separated; on photos all 3 collapse into [0.18, 0.67]. **M6 r6 (2026-04-25)**: 2nd CLIP point (LLaVA-Next, Mistral LM, AnyRes) at PMR 0.70 [0.65, 0.74] rules out CLIP-as-encoder explanation; the LLaVA-1.5 → LLaVA-Next 0.52-PMR jump confounds 4 axes (AnyRes / projector / training / LM family) — consistent with architecture-level reframe but not LM-isolated. | **architecture-level confirmed at 5 model points (3 non-CLIP + 2 CLIP) + bootstrap-validated cross-stim (M9 + M6 r3-r6)** | M6 r2b: Qwen vision AUC 0.99 / behavioral PMR(_nolabel) 0.95; LLaVA-1.5 AUC 0.73 / behavioral 0.38; InternVL3 not captured but behavioral PMR(_nolabel) 0.99 (matches saturation profile). **M8a (2026-04-25)**: cross-shape paired-delta `PMR(physical) − PMR(_nolabel)`: Qwen 5/5 shapes near-zero or negative; LLaVA 5/5 shapes ≥+0.125. **M8d (2026-04-25)**: cross-category paired-delta — Qwen +0.000 / +0.025 / +0.125; LLaVA +0.275 / -0.100 / +0.262. **M9 (2026-04-25)**: 3-model × 3-stim bootstrap CI for mean PMR(_nolabel): non-CLIP [0.800, 0.917] separates from CLIP [0.140, 0.371]; photos converge to [0.183, 0.667]. **M6 r3 (2026-04-25)**: Idefics2 SigLIP-SO400M probe AUC 0.93 (mean 4 layers). **M6 r4 (2026-04-25)**: InternVL3 InternViT probe AUC 0.89 / PMR 0.92 — 4-point chain non-CLIP-generalizes the saturation. **M6 r6 (2026-04-25)**: LLaVA-Next CLIP-ViT-L behavioral-y AUC 0.81, stim-y AUC 1.0, PMR 0.70 [0.65, 0.74] — 5-model chain locked. |
| **H-LM-modulation** (M9-derived, 2026-04-25) | At encoder saturation, LM family may modulate residual H7 measurability — Mistral-7B (Idefics2) shows H7 mean +0.048 [+0.000, +0.094] on M8d vs Qwen2-7B mean +0.008 [−0.033, +0.052] on the same stim. | **suggested only — CIs touch on M8d, fully overlap on M8a/M8c** | M9 bootstrap (5000 iters × 9 cells): Idefics2 M8d H7 CI just touches 0; Qwen M8d H7 CI crosses 0. The 33-pt PASS-rate gap (0.667 vs 0.333) is driven by a single shape (`car`: +0.025 vs +0.094) crossing the strict threshold. Not paper-defensible from current data; needs same-encoder LM swap or 3–5× more shapes. |
| **H-locus** (M4-derived) | The bottleneck is at the LM final layers + decoding head, not earlier. | **supported (early-mid sweet spot)** | M5a: L10 α=40 flips 10/10 abstract → physical responses; later layers do not move. M5a-ext Exp 3: L10 regime-flip (A vs B by α sign) holds in all tested cells. Aligns with the Basu et al. 2024 early-layer constraint-storage finding. |
| **H-direction-bidirectional** (M5a-ext, 2026-04-24; revised 2026-04-25) | `v_L10` is a simple bidirectional concept axis where −α suppresses physics-mode back to abstract. | **revised — regime axis within physics-mode** | Exp 1 (textured/ground/both ceiling): −α has no effect → initially framed as "one-way activator". Exp 3 (textured/blank/none moderate baseline, 2026-04-25): −α=40 flips D → B ("stays still") uniformly across (line, textured) × (ball, circle). Both signs of α activate physics-mode; sign selects regime (+kinetic / −static). Baseline D sits *below* the \|α\| threshold, not at one end of the axis. |
| **H-regime** (M5a-derived) | The steering direction is binary "object-ness", not "which physics" — physics regime is label-driven. | **refuted in current form** | Replaced by H-direction-bidirectional's regime-axis interpretation (kinetic vs static is already a regime distinction that the steering sign selects, label-independent at |α|=40). Separately, label *does* select regime in the narrow `line/blank/none × +α=40` case (Exp 2), but not globally — this is now folded into the H7 qualifier. |

### 1.4 Target models & venue

- **Round 1 (pilot / MVP-full)**: Qwen/Qwen2.5-VL-7B-Instruct — proven loader, 15 GB, 1.0 it/s on H200.
- **ST5 extension**: LLaVA-1.5-7B, LLaVA-Next-7B, InternVL2-8B, (stretch) Qwen2-VL-7B (for layer-index alignment with the models named in `references/project.md` §2.4).
- **Venue**: EMNLP long (grounding-failure / language-prior-dominance angle) primary; NeurIPS main (ST3-4 mechanistic localization) stretch.

---

## 2. Milestone overview

| # | Milestone | Scope | Status | Completed |
|---|---|---|---|---|
| M0 | Infrastructure scaffold | Package layout, configs, scripts, tests, base docs set | ✅ | 2026-04-24 |
| M1 | **ST1 Pilot** (Qwen2.5-VL-7B) | 240 stim × 2 prompts = 480 inferences; first behavioral S-curve measurement | ✅ | 2026-04-24 |
| M2 | **ST1 MVP-full** (incorporating pilot lessons) | axis C decomposition, axis D expansion, T=0.7, LM hidden-state capture, 2880 inferences | ✅ | 2026-04-24 |
| M3 | **ST2 — Vision encoder probing** | Vision-block capture (8 layers, 12 GB) + layer-wise linear probes. **Boomerang confirmed**: encoder AUC = 1.0 on every axis; behavioral PMR 0.28-0.95. | ✅ | 2026-04-24 |
| M4 | **ST3 — LM logit lens / layer-wise probe** | LM hidden @ visual tokens AUC 0.94-0.95 across all probed layers; L20 peak. Label prior drives physics margin from L5; object_level effect is 7× smaller. | ✅ | 2026-04-24 |
| M5a | **ST4 Phase 1+2 — VTI steering** | Direction extraction + residual-stream injection. **L10 α=40 flips 10/10 D → B** — "physical object-ness" direction causally confirmed. | ✅ | 2026-04-24 |
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
| **4.6** | **Counterfactual stimulus generation via SAE / VTI reverse** | Use a learned direction (M5a v_L10 or SAE feature) to gradient-ascent-synthesize a stimulus that maximizes physics-mode in the model's eyes. Adversarial / shortcut-revealing extension to M5a. **PROMOTED from §4.6** to ▶ priority. | ▶ **PRIORITY 5 (next)** | — |
| **4.10** | **Attention visualization UI** | Captured-attention heatmaps × layer × head as an interactive notebook. For paper appendix and qualitative reading. **PROMOTED from §4.10** to ▶ priority. | ▶ **PRIORITY 6 (next)** | — |
| M5b | ST4 Phase 3 — SIP + patching + SAE | Semantic Image Pairs + activation patching (needs attention re-capture) + SAE feature decomposition. | optional | — |
| M6 r3+ | ST5 round 3+ — encoder counterfactuals + LLaVA-Next | LLaVA-Next, InternVL3 captures, scale variants (Qwen 32B/72B), other VLM families (Pixtral / Phi-V). | optional | — |
| **M9** | **Generalization audit — paper Table 1 (3 models × 3 stim sources, bootstrap CIs)** | Consolidates M8a (5 shapes) + M8d (3 categories) + M8c (5 photo categories) × {Qwen, LLaVA, Idefics2} into 9 (model × stim) cells with 95% bootstrap CIs on mean PMR(_nolabel) and mean H7 delta. **Headlines**: (1) encoder family causally drives synthetic-stim PMR ceiling (CIs fully separate, 0.84–0.89 vs 0.18–0.33); (2) photos compress encoder gap (all 3 models converge to 0.28–0.55); (3) H7 robust only in unsaturated regime (LLaVA M8a + M8d, CIs > 0); (4) LM-modulation of H7 at saturation suggestive only (Idefics2 M8d CI touches 0). | ✅ | 2026-04-25 |
| M7 | Human baseline + paper writing | Prolific 20 raters × 50 stimuli + EMNLP/NeurIPS draft | optional | — |

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

### 4.6 Counterfactual stimulus generation via SAE / VTI reverse — work plan ▶ priority 5 (promoted)

**Motivation**: "Adversarial physics-mode" stimulus reveals what the model considers physical. If the synthesized stimulus looks abstract to humans but reads as physical to the model, that is a clean shortcut-interpretation finding.

**Sub-tasks**:
1. Take the M5a steering direction `v_L10` or a learned SAE feature.
2. Gradient-ascent in image space (PIL / torch differentiable) to maximize the projection of the activation onto `v_L10`.
3. Inspect the resulting stimulus visually + measure PMR.

**Estimated effort**: 6-10 hours (image differentiability through Qwen pipeline is non-trivial).

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
- EMNLP long (primary) / NeurIPS main (stretch) draft.

---

## 4. Additional ideas not in the original project doc

Extensions that came up during the pilot, or that aren't in `references/project.md` §2.

**Promoted to next-tier priority** (work plans now in §3, see corresponding sections):
- **4.5** Cross-encoder swap — priority 4 after M8a/c/d (causal test of H-encoder-saturation).
- **4.6** Counterfactual stimulus generation via SAE/VTI reverse — priority 5.
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

### 4.3 Label language switching

Does a Korean `"공"` vs an English `"ball"` on the same stimulus produce different PMR? Qwen2.5-VL is multilingual → measure per-language prior strength. Extends the language-grounding narrative from project doc §3.

### 4.4 Video frame pair → Michotte-style causality

Give a (t=0, t=1) frame pair where only the object position differs, then ask "launched by X?". Does the Michotte (1946) launching effect appear in VLMs? A 2-image prompt is a proxy that does not require a video model.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2) ⭐ promoted

Hypothesis: "cues that are invisible when the encoder is CLIP can be seen by a DINOv2-based model". Continuation of the Eyes Wide Shut (Tong et al. 2024) MoF proposal. Note: a standalone-encoder comparison is implicitly part of M6 (LLaVA-1.5 with CLIP-ViT-L/14 vs Qwen2.5-VL with SigLIP).

**Status (2026-04-25)**: promoted to next-tier priority — H-encoder-saturation (M6 r2) is currently 3-model correlational; this is the causal counterfactual. Detailed work plan in §3 above.

### 4.6 Activation-based counterfactual stimulus generation ⭐ promoted

Use a SAE / VTI steering vector in reverse to gradient-ascent synthesize a stimulus that "maximizes physics-mode in the VLM's eyes". An **adversarial physics-mode prompt** → evidence for shortcut interpretation in open-source VLMs.

**Status (2026-04-25)**: promoted to next-tier priority. The M5a `v_L10` direction is now well characterized (M5a-ext) — reverse-synthesis is the natural extension. Detailed work plan in §3 above.

### 4.7 Decision-consistency boundary measurement

Pilot couldn't measure RC because T=0 made it degenerate. Reinterpret M2's RC (under T=0.7) as **per-axis decision stability**: which cue stabilizes decisions? Expected: ground + shaded + fall → high RC (uniform answer); line + blank + none → mid RC (consistently stationary); borderline cells have low RC.

### 4.8 PMR scaling

Per-model PMR for H-class (Qwen2.5-VL-7B/32B/72B) and LLaVA-1.5-7B/13B. Does MechBench (Zhang et al. 2024)'s "scale doesn't help" claim hold for PMR? Strong interpretability implications for H6.

### 4.9 Label-free prompt

`"What do you see? What might happen next?"` — ask the question **without** the word "ball". Measures H2's language-prior contribution as a null-hypothesis test. Easy addition — `prompts.py` `open_no_label` variant.

### 4.10 Attention visualization UI ⭐ promoted

Captured attentions → interactive heatmap (notebook-based). Per-stimulus, per-layer, per-head visual-token attention. For the paper appendix figure.

**Status (2026-04-25)**: promoted to next-tier priority. Provides a qualitative complement to the per-layer probe AUC numbers and helps target M5b activation patching. Detailed work plan in §3 above.

### 4.11 H7 follow-up — label-regime category annotation

Systematically validate the M2 finding that "label selects the physics regime" (circle → static / ball → rolls / planet → orbits). Hand-annotate or zero-shot-classify open-ended responses into 5 categories (gravity-fall / gravity-roll / orbital / inertial / static) → axis D × category distribution as a confusion matrix. Verify whether the per-label GAR difference (ball 0.79 / planet 0.48) is really a categorical split of "which physics is invoked".

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
