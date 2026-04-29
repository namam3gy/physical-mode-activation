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

The full scorecard with evidence chains lives in **`docs/hypotheses.md`** (extracted 2026-04-29 because the evidence column had grown long). At-a-glance summary:

| ID | Hypothesis (one-line) | Status |
|---|---|---|
| **H1** | PMR rises S-shaped along the abstraction axis | supported, unsaturated-only AND shape-axis-only |
| **H2** | The "ball" label substantially raises PMR (independent contribution of the language prior) | fully validated, three-point + encoder-anchored |
| **H3** | Scene inconsistency degrades RC | untested |
| **H4** | The open vs forced-choice PMR gap is a stable signature of language-prior ↔ visual-evidence conflict | supported (Qwen-only — cross-model FC untested); strengthened under v2 scorer |
| **H5** | A single ground line causes a larger PMR shift than no-ground → with-ground textured ball | mixed |
| **H6** | Arrow+shadow cue saturation is driven entirely by cast shadow alone | supported (revised — arrow is the dominant cue, shadow secondary) |
| **H7** | The label does not toggle PMR — it selects which physics regime to apply | supported, unsaturated-only AND cross-category |
| **H-boomerang** | Encoder knows, decoder gates | Qwen-scoped (revised; refuted in LLaVA-1.5) |
| **H-encoder-saturation** | Behavioral PMR(_nolabel) saturation is determined at the architecture level (joint encoder + LM) | architecture-level confirmed at 5 model points + cross-stim bootstrap |
| **H-LM-modulation** | At encoder saturation, LM family may modulate residual H7 measurability | suggested only — CIs touch on M8d, fully overlap on M8a/M8c |
| **H-locus** | Bottleneck is at LM mid layers, model-specific (Qwen L10 / LLaVA-Next L20-L25 / Idefics2 L25) | supported, triply-refined and cross-model confirmed |
| **H-direction-bidirectional** | `v_L10` is a regime axis within physics-mode (+α kinetic, −α static) | revised — was "bidirectional concept axis"; baseline D sits below the \|α\| threshold |
| **H-regime** | The steering direction is binary "object-ness", not "which physics" | refuted in current form (folded into H7 qualifier) |
| **H-direction-specificity** | Pixel-space gradient ascent along `v_L` flips PMR; matched-magnitude random does not | supported across 5 architectures × 5 layers at n=10 |
| **H-shortcut** | Shortcut interpretation is encodable in the image itself (pixel-driven) | supported, architecture-conditional; perceiver-resampler is leading remaining mechanism candidate |

Drill into evidence: `docs/hypotheses.md`. When a hypothesis status flips or new evidence lands, update both `docs/hypotheses.md` (full evidence) and this table (one-line status), and add a `docs/CHANGELOG.md` entry.

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
| **M-MP** (Pillar A) | **Multi-prompt cross-task generalization** | Track B Pillar A (`references/submission_plan.md`). **Behavioral coverage = 5-model × 4-prompt** (Phase 1 + Phase 2 + audit follow-up MCQ Phase 1 smoke + Phase 2 full = 28,800 inferences total). **Causal coverage = 2-model × 3 new prompts** (Qwen on `describe_scene` + `meta_phys_yesno` + `meta_phys_mcq`; Idefics2 on `describe_scene` + `meta_phys_yesno` + cell-2 SAE). **Behavioral headline (5-model × 4-prompt, audit follow-up 2026-04-28 evening)**: H2 paired-delta positive in **19/20 (model × prompt) cells**; **Qwen × MCQ is the exception** (Δ = −0.050, label-image-mismatch interaction). 100% parse rates on all 7200 MCQ inferences. Saturation × prompt interaction holds (saturated → describe most informative; unsaturated → open most informative). **Causal headline (refined again 2026-04-28 evening, post-MCQ + 2nd-cell)**: (i) **Cross-method split at Qwen × MCQ** (was: single unified gen-vs-cat boundary): at the audit-pinned cells, M5a method nulls (matches yesno: 0/10 flip vs describe + open 10/10) but M5b method breaks (matches describe: 10/10 break under top-k=20 SAE ablation; random retains baseline). The unified framing splits — M5a-side reading remains "categorical-task blocks steering" (matches yesno + MCQ); M5b-side reading is that yes/no is the **single (n=1) categorical prompt where ablation does NOT break**, described as **yes/no-prompt-specific** until more categorical-binary prompts confirm whether the load-bearing axis is "format" or "this specific prompt". (ii) **Idefics2 single-cell finding partial-lift**: 2nd-cell test on `textured/ground/cast_shadow ball` finds baseline already suspended ("in the air") rather than kinetic — top-k SAE ablation no-op. Provides **specificity evidence** (SAE features are kinetic-verb-encoding, not arbitrary perturbation) but does NOT lift cell-1 framing-shift claim to architecture-level. 3rd-cell with kinetic baseline left for Pillar B follow-up. (iii) **Cross-method agreement recount**: was 1+3 asymmetry (1 positive/positive + 3 null/null); now 5 cells with mixed picture: Qwen × describe (M5a+ M5b+ co-fire), Qwen × MCQ (M5a− M5b+ split — NEW), Qwen × yesno (M5a− M5b− shared null), Idefics2 × describe (M5a− M5b−), Idefics2 × yesno (M5a− M5b−). The split cell (Qwen MCQ) is the audit follow-up's primary contribution. Phases 1+2+3 + audit MCQ + 2nd-cell all ✅. | ✅ behavioral 5-model × 4-prompt, ✅ causal 2-model refined | 2026-04-28 (all phases) |
| **M-PSwap** (Pillar B) | **Controlled projector-swap LoRA on Idefics2** | Track B Pillar B / G3 fix. Replace Idefics2 perceiver-resampler with MLP projector (LoRA rank-32, encoder + LM held fixed). Re-run §4.6 + M5b. **Predicted**: if perceiver-resampler is causal, Idefics2-MLP flips on §4.6 like LLaVA-Next does. **Status (2026-04-29)**: feasibility spike done — perceiver-bypass FAILS (`38302ec` / `10bafd3`), full LoRA training required. LoRA infra built (`d35d512` / `69634e7`): `src/physical_mode/lora/{idefics2_mlp_resampler.py, load_swapped.py}` + `scripts/m_pswap_{train,smoke,regression_eval,post_training,diagnose_nan,repro_nan_batch,discriminator}.py`. Smoke (50 step) PASS. Full training **NaN-blocked at step 1000** (run `outputs/mpswap_run_20260429-033238/step1000`); diagnostic suite (`m_pswap_diagnose_nan_v2.py`) WIP. Fallback: B2 only + literature-grounded theoretical claim. | **infra built / training NaN-blocked WIP** | week 4–5 |
| **M-LMSwap** (Pillar B) | **Controlled LM-only-swap LoRA (CLIP+Vicuna vs CLIP+Mistral)** | Track B Pillar B / G3 fix. Pair CLIP-ViT-L with Vicuna-7B and Mistral-7B holding encoder fixed. Re-run M5a + M5b. **Predicted**: M5b NULL persists in both LM variants (encoder is the bottleneck). | planned | week 6–7 |
| **M-Marr** (Pillar C) | **Marr-3-level paper restructure** | Track B Pillar C / G4 fix. §6 of `docs/paper/draft_v1.md` reorganized into 3 levels (Computational PMR / Representational M3+M4 / Mechanistic M5a+M5b). §1 + §9 + §10 rewritten with world-model framing. No new experiments. | planned | week 9–11 |

---

## 3. Detailed status

Per-milestone deep dives live in `docs/insights/m{N}_*.md` — this section keeps a one-line headline + insight pointer per ✅ milestone, plus brief plans for any milestone that is still forward-looking. When a milestone completes, write the full insight in `docs/insights/`, add a `docs/CHANGELOG.md` entry, then add (or update) the row here.

### Completed milestones

| Milestone | Date | Headline | Insight |
|---|---|---|---|
| **M0** Infrastructure scaffold | 2026-04-24 | `src/physical_mode/` package, scripts, configs, tests, docs scaffold; `uv sync` + pytest pass. | (no insight — code-only) |
| **M1** ST1 Pilot (Qwen2.5-VL-7B) | 2026-04-24 | 480 inferences; ground effect = +36 pp (largest single-factor); language-prior dominance under FC; H1 partial, H2 strong. | `m1_pilot.md` |
| **M2** ST1 MVP-full (Qwen) | 2026-04-24 | 5-axis factorial × 480 stim × 3 labels × 2 prompts; 2880 inferences; H1 monotone but saturated; H7 promoted to "label selects regime"; LM hidden-state capture for M4. | (folded into `m4_logit_lens.md`, `m4b_label_free.md`) |
| **M3** ST2 Vision-encoder probing | 2026-04-24 | 8-layer SigLIP capture × per-stim PMR probes; **encoder boomerang** — encoder AUC ≈ 1.0 every layer despite behavioral PMR 0.28-0.95 on Qwen. | `m3_encoder_boomerang.md` |
| **M4** ST3 LM logit lens / per-layer probe | 2026-04-24 | LM AUC 0.94-0.95 across all layers (peak L20 = 0.953); label drives physics-margin from L5; M5 set as next. | `m4_logit_lens.md` |
| **M4b** Label-free prompt — H2 null test | 2026-04-25 | Paired PMR(ball) − PMR(_nolabel) ≈ 0; circle = −0.065 → **H2 revised** (Qwen-asymmetric, "circle override"); M4 visual-token capture is prompt-independent. | `m4b_label_free.md` |
| **M4c** Forced-choice label-free | 2026-04-25 | Qwen reproduces M4b under FC; open-vs-FC paired delta at no-label = −0.131 (label-free H4 measurable). LLaVA "A"-bias is logit-level pathology. | `m4c_fc_label_free.md` |
| **M5a** ST4 Phase 1+2 VTI steering | 2026-04-24 | L10 α=40 flips 10/10 of `line/blank/none` D→B — "object-ness" direction causally confirmed. | `m5_vti_steering.md` |
| **M5a-ext** VTI follow-ups (neg α, label swap, bidirectionality) | 2026-04-24/25 | Exp 1-2 ceiling artifact + label=ball B→A flip; **Exp 3** reframes `v_L10` as a regime axis (+α kinetic, −α static, baseline D below threshold). | `m5a_ext_bidirection_and_label.md` |
| **M6 r1** Cross-model — LLaVA-1.5-7B | 2026-04-25 | Paired PMR delta vs label-free → ball +0.475, planet +0.244, circle +0.173 (all positive); H2 re-revised under **visual-saturation hypothesis**. | `m6_cross_model_llava.md` |
| **M6 r2** 3-model + LLaVA captures + FC logit ratio | 2026-04-25 | InternVL3 saturated; LLaVA encoder AUC ~0.73 vs LM AUC ~0.75 — bottleneck is encoder. **New H-encoder-saturation hypothesis**. | `m6_r2_cross_model.md` |
| **M6 r3** Idefics2 vision-encoder probe | 2026-04-25 | SigLIP-SO400M AUC 0.93 (peak L9 = 0.948) — closes 3-point AUC ↔ PMR chain. | `m6_r3_idefics2_probe.md` |
| **M6 r4** InternVL3 InternViT probe → 4-model chain | 2026-04-25 | InternVL3 AUC 0.89 / PMR(_nolabel) 0.92; non-CLIP cluster ≥ 0.88, CLIP at 0.77 (apples-to-apples M8a-stim). | `m6_r4_internvl3_probe.md` |
| **M6 r5** M8c photo encoder probe — 4-model cross-stim | 2026-04-25 | Behavioral-y AUC inverts cross-stim; stim-y AUC stays 1.0 — encoder discriminability is uniform, architecture-level reframe. | `m6_r5_m8c_photo_probe.md` |
| **M6 r6** LLaVA-Next-Mistral 5th model point (2nd CLIP) | 2026-04-25 | PMR 0.700 [0.65, 0.74] between LLaVA-1.5 floor and saturated cluster; 5-model M8a chain locked. | `m6_r6_llava_next.md` |
| **M6 r7** M2 cross-model — 5-model M2-stim apples-to-apples | 2026-04-26 | 5-model PMR(_nolabel) ladder 0.18 → 0.99; H2 paired-delta 3 patterns; per-model `v_L10` extracted. | `m2_cross_model.md` |
| **M8a** Non-circle synthetic shapes | 2026-04-25 | 5 shapes × Qwen + LLaVA strict scoring: Qwen 1/4 PASS, LLaVA 4/4 PASS — H1/H7 unsaturated-only. | `m8a_non_circle_shapes.md` |
| **M8c** Real photographs | 2026-04-25 | 60 photos; **photos REDUCE Qwen PMR(_nolabel) by 18-48 pp** — synthetic-stim minimality is a co-factor of saturation. | `m8c_real_photos.md` |
| **M8d** Non-ball physical-object categories | 2026-04-25 | car/person/bird × abstraction × bg × cue: LLaVA 3/3 H7 ✓ (strongest cross-category H7 evidence); Qwen 0/3 (ceiling). | `m8d_non_ball_categories.md` |
| **M8e** Cross-source paired analysis | 2026-04-25 | M8a + M8d + M8c consolidated heatmap (paper Table 1 candidate); cross-source PMR shift confirmed. | `m8e_cross_source.md` |
| **§4.5** Cross-encoder swap (Idefics2) | 2026-04-25 | Idefics2-8b on M8a: Qwen 0.838 / LLaVA 0.175 / Idefics2 0.882 — encoder type drives PMR ceiling regardless of LM. **H-encoder-saturation causally confirmed at encoder-family level.** | `encoder_swap_idefics2.md` |
| **M9** Generalization audit (paper Table 1) | 2026-04-25 | 9 (model × stim) cells × bootstrap CIs; non-CLIP [0.80, 0.92] vs CLIP [0.14, 0.37] fully separated on synthetic; photos compress to [0.18, 0.67]. | `m9_generalization_audit.md` |
| **§4.6** Counterfactual stim via VTI reverse | 2026-04-26 | 5/5 v_L10 flips at ε=0.05; 0/15 random-direction at matched ε=0.1 — **`v_L10` encodable in the image**. | `sec4_6_counterfactual_stim.md` |
| **§4.6 cross-model** Pixel-encodability layer sweep | 2026-04-26 → 2026-04-28 | 5-model n=10 chain locked; **Idefics2 anomaly resolved across L5-L31** — perceiver-resampler is leading remaining mechanism candidate. | `sec4_6_cross_model_revised.md` |
| **§4.7** Decision-stability boundary | 2026-04-26 | `cue_level=both` is the dominant decision stabilizer; ceiling effect on saturated models. | `sec4_7_rc_per_axis.md` |
| **§4.8** Qwen 7B vs 32B PMR scaling | 2026-04-28 | Aggregate PMR 0.926 ≈ 7B 0.931 — 5× scaling does not move overall PMR; only `abstract_reject` (0.002 → 0.065) and label gap (halved) shift. | `sec4_8_pmr_scaling.md` |
| **§4.10** Attention visualization UI (5-model) | 2026-04-25 | Last-token attention to visual tokens varies architecture-level (Qwen 17%, LLaVA-1.5 7%, Idefics2 30%, ...); paper appendix. | `sec4_10_attention_viz.md` |
| **§4.11** H7 follow-up — regime distribution | 2026-04-26 | LLaVA-Next intermediate 3-way split on `person × exotic`; 5-model gradient is granular form of M9 H7 finding. | `sec4_11_regime_distribution.md` |
| **M5b** ST4 Phase 3 — SIP + patching + knockout + SAE | 2026-04-26..28 | (1) Qwen SIP+patching: sharp L10 boundary; (2) LLaVA-1.5 SIP locks at L20; (3) attention/MLP knockout: **L9 MLP uniquely necessary** (IE=+1.0); (4) per-head knockout: 196 (L,h) all IE=0; (5) SAE intervention: **top-20 vision-encoder physics-cue features → 0/20 retain physics**, mass-matched random retains. Round-2 cross-model SAE: Qwen + Idefics2 + InternVL3 break; LLaVA-1.5 + LLaVA-Next NULL — encoder-vs-LM mechanism dissociation locked. M5b fully complete (5+1 sub-tasks). | `m5b_sip_activation_patching.md`, `m5b_attention_mlp_knockout.md`, `m5b_per_head_attention_knockout.md`, `m5b_sae_intervention.md`, `m5b_sip_cross_model.md`, `m5b_sae_intervention_cross_model.md`, `m5b_post_projection_sae_qwen.md` |
| **M-MP** Multi-prompt cross-task generalization (Pillar A) | 2026-04-28 | Phase 1+2+3 + audit follow-up MCQ + Idefics2 2nd-cell. Behavioral 19/20 (model × prompt) cells positive; causal **cross-method split at Qwen × MCQ** (M5a− M5b+). | `m_mp_phase3_followup_2026-04-28.md` (+ Phase 1+2+3 experiments + audit) |

### Forward-looking (still to do)

- **M-PSwap** (Pillar B) — projector-swap LoRA on Idefics2 (perceiver → MLP). Status: infra built, training NaN-blocked at step 1000, diagnostic suite WIP. See §3.X row + `submission_plan.md` §B + `paper_gaps.md` G3.
- **M-LMSwap** (Pillar B) — LM-only-swap LoRA (Vicuna-7B vs Mistral-7B on CLIP+ViT-L). Planned, week 5-6.
- **M-Marr** (Pillar C) — paper §6 restructure into 3 Marr levels. Paper-side, week 7-8.
- **M7** — human baseline (Prolific 20 raters × 50 stim) + ICLR 2027 / NeurIPS 2027 draft. Deferred to week 14 of Track B schedule. Plan: `docs/m7_human_baseline_plan.md`.
- **§4.4** — Michotte 2-frame causality (needs 2-image prompt support; out of scope for now).
- **H3 Scene-inconsistency × RC** — focused mini-experiment, untouched since M2 dropped Axis E.

---

## 3.X Track B priorities (current open work, 2026-04-28)

**Read order for new sessions**: `references/submission_plan.md` → `references/paper_gaps.md` → this section.

| Pillar | Milestone | Status | Week | Gap fixed |
|---|---|---|---|---|
| A | **M-MP Phase 1+2** Multi-prompt behavioral PMR (5 models × 3 prompts × 480 stim) | ✅ 2026-04-28 | 1–2 | G1 (single-task) — behavioral evidence |
| A | **M-MP Phase 3** Cross-prompt M5a + M5b (Qwen + Idefics2 × 3 prompts; new prompts = `describe_scene` + `meta_phys_yesno`, existing baseline = `open`) — *causal* test | ✅ 2026-04-28; **Mixed (Qwen-specific gen-vs-cat)**: Qwen (open + describe) M5a + M5b both flip/break, Qwen yesno doesn't — generative-prompt-conserved + categorical-blocked **within Qwen**. Idefics2 describe and yesno BOTH null/null — Idefics2 cannot dissociate generative-vs-categorical from these data. **Idefics2 single-cell finding**: on shaded/ground/both ball, top-k SAE ablation shifts framing kinetic→suspended ("falling" → "in the air"); **10/10 stim replicates at k=160/320/500** with 1 unique intervention text, random 10/10 retains kinetic — single-cell finding, not architecture-level; 2nd-cell test required to lift. **Audit caveats (2026-04-28)**: (a) cross-method M5a-M5b "agreement" rests on **1 positive/positive cell + 3 null/null cells**, not 4 independent positive confirmations; (b) gen-vs-cat dissociation is Qwen-specific from current data. | 3 | G1 — causal evidence |
| — | **M5b post-projection SAE** (Qwen) | ✅ 2026-04-28: k=20 break at same threshold as pre-projection — merger preserves physics-mode commitment localization. Cross-model extension optional. | between weeks 2–3 | mechanism resolution (encoder pre/post-projector) |
| B | **M-PSwap** Projector-swap LoRA on Idefics2 (perceiver → MLP) | **infra built; training NaN-blocked WIP (2026-04-29)** — feasibility spike confirmed bypass-only fails; full LoRA training pipeline runs through step 1000 then NaNs (run `outputs/mpswap_run_20260429-033238/step1000`); diagnostic suite WIP | 4–5 | G3 (n=1 perceiver) |
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

The full chronological changelog (project-level changes — milestones, hypothesis flips, re-prioritizations, load-bearing infra) lives in **`docs/CHANGELOG.md`**. New entries go there, not here.

When updating, append a row at the bottom of `docs/CHANGELOG.md` with the date, a one-paragraph summary, and the commit hash. Cross-link the relevant `docs/insights/m{N}_*.md` and update §3's milestone table here if a milestone completed.
