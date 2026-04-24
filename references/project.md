# Triggers for VLM "Physical Mode" Switching: Literature Review and Experimental Design

The user's research question — **under what visual conditions does an abstract shape (a circle) get processed by a VLM as a physical object (a ball)?** — sits at the intersection of three sparsely-studied areas in the current literature (parametric abstract–concrete axis manipulation, next-state prediction in open-source VLMs, and mechanistic analysis of physical-mode triggers), and **no direct prior work exists**. Pixels-to-Principles (Ballout et al., 2025) probes intuitive-physics representations across the same three open-source VLM families (InternVL 2.5, Qwen2.5-VL, LLaVA-OneVision) but uses only photorealistic stimuli and does not manipulate the abstraction level. This report (1) surveys the literature in the three sub-areas, (2) lays out concrete sub-tasks that would scale to a NeurIPS / EMNLP-grade paper, and (3) discusses positioning and risk.

---

## Part 1. Literature Survey

### 1.1 VLM physics-reasoning benchmarks (area a)

The literature is split between **abstract 2D shape physics (pre-VLM era)** and **photorealistic 3D / video physics (current VLM era)**, with **no work that parametrically varies the abstraction level within the same physical scenario**.

**Recent VLM physics benchmarks (2023–2026)**

| Paper | Year/venue | Stimulus modality | Task | Models evaluated | Key finding |
|---|---|---|---|---|---|
| **PhysBench** (Chow et al.) | ICLR 2025, arXiv:2501.16411 | photo + synthetic video+image, 10 002 items | 4-domain QA (property/relation/scene/**dynamics**) | 75 VLMs (GPT-4o, Gemini, LLaVA, Qwen-VL, InternVL all variants) | VLMs are strong on commonsense but **fail on dynamics**; errors split between perception and knowledge |
| **Pixels to Principles** (Ballout, Jassim, Bruni) | arXiv:2507.16572, 2025 | GRASP + IntPhys2 video | plausibility judgment + **intermediate embedding probing** | **InternVL2.5, Qwen2.5-VL, LLaVA-OneVision**, Gemini | **Vision encoder captures physics cues but the LLM does not use them**; vision-language misalignment is the bottleneck. t-SNE clusters collapse after the projector |
| **VLM4D** (Zhou et al.) | ICCV 2025, arXiv:2508.02095 | real + synthetic video | translation/rotation/continuity MCQ | GPT-4o 62 % vs human 98.8 %; open-source has an even larger gap | a huge VLM gap on motion reasoning |
| **MechBench / "Probing Mechanical Reasoning in Large VLMs"** (Zhang et al.) | arXiv:2410.00318, 2024 | static images of mechanical systems (155 experiments) | stability / lever / inertia / fluid QA | 26 VLMs (all major open-source) | **scaling does not help** — implies architectural limits |
| **PhysGame** (Cao et al.) | arXiv:2412.01800, 2024 | gameplay video (glitches) | physics-violation detection MCQ | LLaVA-Next-Video, Qwen2-VL, InternVL-Video, etc. | open-source is well behind closed-source |
| **GRASP** (Jassim et al.) | IJCAI 2024, arXiv:2311.09048 | Unity-synthesized video | permanence / continuity / gravity plausibility | Video-LLaMA, VideoChat, Video-ChatGPT, LLaVA, Qwen-VL | video-MLLMs are near-chance overall |
| **IntPhys 2** (Bordes et al., FAIR) | arXiv:2506.09849, 2025 | complex synthetic video | VoE (permanence / invariance / spatio-temporal continuity / solidity) | frontier MLLMs | most ≈ 50 % |

**Classic (abstract-shape) benchmarks (pre-VLM)**

Bakhtin et al. 2019's **PHYRE** (NeurIPS), Ates et al. 2022's **CRAFT** (Findings of ACL), Patel et al. 2022's **CRIPP-VQA** (EMNLP), and Yi et al. 2020's **CLEVRER** (ICLR) all built physics events from Box2D/CLEVR-style abstract 2D shapes, but **none have been re-evaluated systematically on modern VLMs**.

**Static object-property VLM work**: Gao et al. 2023's **PhysObjects / PG-InstructBLIP** (arXiv:2309.02561) annotates household items with material/fragility/mass, but is restricted to **static-property QA** rather than next-state prediction. Liu et al. 2024's **PhysGen** (ECCV) shows in image→video generation that VLMs *can* extract physical parameters from a static image, but it does not address when this mode is activated or deactivated.

### 1.2 Abstract vs concrete visual perception (area b)

VLMs **systematically fail on pure geometric shapes**, and multiple independent studies converge on **perception** (not reasoning) being where the failure occurs.

- **"Vision Language Models Are Blind"** (Rahmanzadehgervi, Bolton, Taesiri, Nguyen, ACCV 2024 Oral, arXiv:2407.06581) — BlindTest with 7 tasks (overlapping circles, line crossings, counting Olympic rings). GPT-4o / Gemini-1.5-Pro / Claude-3.5 average 58 %. Crucially, **encoder probing shows the information is present** — an "encoder knows, decoder doesn't" dissociation. Direct template for the present project.
- **"Vision Language Models Are Biased"** (Vo et al., ICLR 2026 accepted, arXiv:2505.23941) — counterfactual counting (4-stripe Adidas, 5-leg dog). ~100 % on canonical instances vs ~17 % on counterfactuals. **VLMs project memorized text priors rather than reading the pixels**. The strongest analog precedent for our "projecting ball-properties onto a circle" hypothesis.
- **MARVEL** (Jiang et al., NeurIPS 2024, arXiv:2404.13591) and **"Curious Case of Nonverbal Abstract Reasoning with MLLMs"** (Ahrabian et al., COLM 2024) — Raven's PM / abstract figures. Large MLLM gaps; perception dominates reasoning as the bottleneck.
- **Bongard series**: Bongard-LOGO (NeurIPS 2020), Bongard-HOI (CVPR 2022), **Bongard-OpenWorld** (ICLR 2024, arXiv:2310.10207) — VLM 64 % vs human 91 %; perception is the main bottleneck.
- **Sketch-photo domain gap**: Sain et al. 2023 (CVPR), Efthymiadis et al. 2024 (ECCV) quantify how the CLIP family systematically under-represents sketches. **CLIPasso** (Vinker et al., SIGGRAPH 2022) shows CLIP can assign meaning to abstract sketches when guided — bidirectional mapping exists.
- Others: **IconQA** (NeurIPS D&B 2021), **MathVista** (ICLR 2024), **MMMU** (CVPR 2024), **MultiStAR** (arXiv:2505.21850) all report a consistent VLM weakness on abstract diagrams.

### 1.3 Human intuitive physics — cognitive-science foundation (area c)

**Spelke's core knowledge** (Spelke 1990, *Cog Sci*; Spelke & Kinzler 2007, *Dev Sci*): infants parse objects via cohesion / boundedness / rigidity / no action at a distance / continuity. **Kellman & Spelke (1983, *Cog Psych*)**: **common motion** rather than closure or good form is the primary cue for object unity.

**Violation-of-Expectation**: Baillargeon, Spelke & Wasserman (1985, *Cognition*) drawbridge paradigm; Margoni, Surian & Baillargeon (2024, *Psych Review*) modern review.

**Michotte's launching effect** (1946 / 1963, *Perception of Causality*): contact + precise spatio-temporal contingency immediately triggers the perception of causation between two squares. Leslie & Keeble (1987, *Cognition*) show the effect at 6 months. Bechlivanidis et al. (2025, *Royal Soc Open Sci*) registered replication.

**Heider-Simmel animacy** (1944, *Am J Psych*): simple motion of triangles and circles is spontaneously perceived as agentive. **Scholl & Tremoulet (2000, *TiCS*)**: appropriate kinematics (self-propulsion, inertia violation, goal-directed contingency) trigger fast, automatic, irresistible animacy / causality percepts.

**3D and object cues**: **Ramachandran (1988, *Nature*)** shape-from-shading + light-from-above prior produce immediate 3D convex sphere percepts (flips to concave at 180° rotation); **Kersten, Mamassian & Knill (1997, *Perception*)** cast shadows attach an object to the ground and impose a gravity frame; **Gibson (1979)** ecological approach — surfaces, ground plane, texture gradients, optic flow are the core physical cues.

**Intuitive physics = simulation**: **Battaglia, Hamrick & Tenenbaum (2013, *PNAS*)** "Intuitive Physics Engine" — humans use an approximate probabilistic physics simulator. **Sanborn, Mansinghka & Griffiths (2013, *Psych Review*)** "noisy Newton". **Ullman et al. (2017, *TiCS*)** "game engine in the head". **Smith, Hamrick et al. (2024, MIT Press)** the latest canonical review, "Intuitive physics as probabilistic inference".

**Bridges to AI**: Smith et al. (2019, NeurIPS) **ADEPT** — deep perception + probabilistic physics + particle filter. **Piloto et al. (2022, *Nature Hum Behav*)** **PLATO** — object-centric representations are essential to VoE effects. **Yildirim et al. (2024, *Nature Hum Behav*)** — 3D shape perception integrates with intuitive physics.

**Synthesized stimulus-strength hierarchy (static images, weakest → strongest)**: closed contour (baseline) → line-art-style realistic cue → occlusion → perspective / ground → photo-grade material / texture → cast shadow + ground contact → **3D shape-from-shading** (the strongest static cue). For dynamic stimuli, Michotte contact and gravitational ballistics likely beat 3D shading.

### 1.4 VLM mechanistic interpretability (area d)

**Vision-encoder decomposition**: **Gandelsman, Efros & Steinhardt (ICLR 2024 Oral)** "Interpreting CLIP's Image Representation via Text-Based Decomposition" — decomposes CLIP-ViT outputs into patches × layers × heads, automatically labels each head's subspace via TextSpan (finds shape / colour / counting / location specialised heads). **Balasubramanian et al. (NeurIPS 2024)** extends this to DINO / DeiT.

**Sparse autoencoders**: **Pach, Karthik et al. (NeurIPS 2025, arXiv:2504.02821)** "Sparse Autoencoders Learn Monosemantic Features in VLMs" — train SAEs on CLIP activations, **causally manipulate LLaVA outputs by intervening on CLIP features without touching the LLM**.

**LLaVA internal visual-token processing**: **Neo, Ong, Torr, Geva, Krueger, Barez (arXiv:2410.07149, ICLR 2025)** "Towards Interpreting Visual Information Processing in VLMs" — logit lens + attention knockout at visual-token positions. Findings: (1) object-specific tokens are spatially localized; their removal drops object-ID accuracy by >70 %; (2) layers 1-10 do broad context processing, **specific object features are extracted in layers 15-24 (of LLaVA-1.5's 32 layers)**; in Qwen2-VL-2B the peak layer is ~25/29. **The methodological blueprint for this project**.

**Causal tracing**: **Basu, Grayson et al. (NeurIPS 2024, arXiv:2406.04236)** "Understanding Information Storage and Transfer in MLLMs" — MultimodalCausalTrace. **In LLaVA, constraint-satisfaction information is stored in the early MLP / self-attention of layers 1-4** (vs the mid-layer MLP in text-only LLMs). A small number of visual tokens are responsible for transmitting image information.

**Symmetric Image Pairs**: **Golovanevsky, Rudman, Palit, Singh, Eickhoff (NAACL 2025, arXiv:2406.16320)** "What Do VLMs NOTICE?" — introduces **Semantic Image Pairs** to avoid the hallucination artifacts of Gaussian-noise corruption (Zhang & Nanda 2024). **LLaVA self-attention has no image-grounding head, only outlier suppression** (in contrast to BLIP's cross-attention) — an architecturally important finding.

**Layer-wise information flow**: **Kaduri, Bagon, Dekel (arXiv:2411.17491)** "What's in the Image?" — cross-modal flow concentrates in **the middle ~25 % of layers**. **Jiang et al. (2024)** "two-stage process": visual enrichment → semantic refinement.

**Hidden-state steering**: **Liu, Ye, Zou (ICLR 2025, arXiv:2410.15778)** VTI — compute per-layer shift vector from (hallucinated, grounded) pairs, add at test time. **Li et al. (ICML 2025, arXiv:2502.03628)** VISTA — "Hidden Life of Tokens": progressive visual information loss, early excitation, hidden genuine tokens.

**Encoder–decoder dissociation**: **Zhang, Unell et al. (NeurIPS 2024, arXiv:2405.18415)** "Why are VLMs Bad at Image Classification?" — classification information exists in LLaVA's latent space but **the LM does not use it**. **Tong et al. (CVPR 2024)** "Eyes Wide Shut" — CLIP-blind pairs and the MMVP benchmark, MoF (CLIP+DINOv2 interleave).

**Visual-token redundancy**: **Chen et al. (ECCV 2024 Oral)** FastV — "after layer 2 the image is worth half its tokens"; deep-layer visual-token attention is extremely inefficient. Followups: **SparseVLM** (ICML 2025), LLaVA-PruMerge.

**Meta-review**: **Yiming Liu, Zhang & Yeung-Levy (ICLR 2025 Blogpost Track)** "Mechanistic Interpretability Meets VLMs" — documents that the current toolset has barely been applied to physics concepts.

### 1.5 Controlled-stimulus studies (area e)

Representative minimal-pair work:
- **What'sUp** (Kamath, Hessel, Chang, EMNLP 2023, arXiv:2310.19785) — varies only on/under/left/right with the object held fixed. BLIP-VQAv2 56 % vs human 99 %.
- **Winoground** (Thrush et al., CVPR 2022) — caption pairs with the same words in different order + two images. All VLMs are below chance on group score.
- **VALSE** (Parcalabescu et al., ACL 2022) — foil-based, six-language phenomena.
- **CLEVR** (Johnson et al., CVPR 2017) — Blender-synthesized; controls colour / shape / material / size / count. Extended by CLEVR-Hyp, Super-CLEVR, CLEVRSkills (NeurIPS 2024).
- **WHOOPS!** (Bitton-Guetta et al., ICCV 2023) — commonsense-violation Midjourney images. BLIP2-XXL explanation 27 % vs human 95 %.
- **HallusionBench** (Guan et al., CVPR 2024) — visual illusions and counterfactual edits.
- **NaturalBench** (Li et al., NeurIPS 2024 D&B) — 10 k paired VQA, alternating answers force vision use.

### 1.6 Physical-trigger visual cues (area f)

- **Geirhos et al. (ICLR 2019 Oral)** "ImageNet-trained CNNs are biased towards texture" — origin of the cue-conflict methodology.
- **Gavrikov, Lin, Bethge, Keuper (arXiv:2403.09193, 2024)** "Are VLMs Texture or Shape Biased and Can We Steer Them?" — **VLMs are more shape-biased than their vision backbone**; multimodal fusion changes cue preference; language prompts can steer the bias but never reach human 96 % shape bias. **Direct precedent for this project**.
- **"Shape and Texture Recognition in Large VLMs"** (arXiv:2503.23062, 2025) — LVLMs heavily rely on semantic features; fail on abstract 2D shapes that lack class associations.
- **Garrido et al. (arXiv:2502.11831, 2025)** **V-JEPA intuitive physics** — a representation-space prediction objective makes intuitive physics emerge (IntPhys 98 %); **pixel prediction and MLLMs are near-random**. The strongest current architecture-level evidence on "what triggers physics mode".
- **Bi, Yamins, Fan et al. (*Nature Comms*, 2025)** — DNN feature spaces encode soft-body physical judgments in alignment with humans.
- **Peters & Kriegeskorte (*Nature Hum Behav*, 2021)** — review of object-based representations; DNNs lack grouping / amodal completion.
- **"Pixels to Principles"** (Ballout et al., 2025) — already cited; demonstrates the vision-language misalignment bottleneck on the three target open-source VLMs but **does not manipulate the abstraction level**.

### 1.7 Gap analysis summary

This project sits at the **intersection of three sparse areas**:
1. **(abstraction manipulation × VLM)** — Geirhos-style cue-conflict exists but only for **classification**, not physics.
2. **(next-state prediction × open-source VLM)** — PhysBench / VLM4D / GRASP use only photorealistic stimuli; Pixels-to-Principles probes but only for plausibility judgment.
3. **(mechanistic / probing × physics)** — essentially only Pixels-to-Principles; a rich methodological toolbox (NOTICE's SIP, SAE, activation patching, logit lens) exists but has not been applied to physics-object triggers.

**No prior work** parametrically varies a single physical event (e.g. falling) across **line drawing → shading → photographic** to identify a VLM's physics-mode "threshold" and localize it at the mechanistic level.

---

## Part 2. Research Plan and Sub-tasks

### 2.1 Overall paper narrative

**Central claim**: open-source VLMs switch between two modes for visual input — (M1) abstract geometric-shape recognition, (M2) physical-object reasoning. The switch is (a) triggered by a specific subset of visual cues and (b) mediated by identifiable layers, heads, and latent directions inside the model. This work systematically generates stimuli following the cognitive-science cue hierarchy, measures behavioral thresholds, then localizes the internal mechanism via probing and causal intervention.

**Five sub-tasks**: (§1) build a controlled stimulus set PhysCue + measure behavioral thresholds; (§2) linearly separate "physical-ness" in the vision encoder; (§3) layer-wise emergence analysis in the LLM backbone via logit lens; (§4) causal localization via attention / activation patching; (§5) cross-model generalization and text-prompt steering.

### 2.2 Sub-task 1 — PhysCue dataset and behavioral thresholds

**Task definition**. Input: controlled static images. Output: next-state prediction extracted from a free-form text response (e.g. "What happens next to the object?" → "falls" / "rolls" / "stays" / "moves sideways"). Metrics: (1) **Physics-Mode Priming Rate (PMR)** — binary rate of physics-verb presence in the response; (2) **Gravity-Align Rate (GAR)** — fraction predicting downward fall when a ground is present; (3) **Response Consistency (RC)** — agreement across multiple renderings of the same scenario.

**Why this is informative**. A behavioral threshold curve is the baseline for every subsequent mechanistic analysis. The key contribution is generating **a switching function along a stimulus axis**, not a single accuracy number.

**Stimulus design (factorial 2 × 2 × 3 × 3 × 3 = 108 conditions × 50 scenarios per condition)**:
- **Axis A (abstraction level, 5 levels)**: line circle → coloured circle → shaded sphere (light-from-above gradient) → textured ball (leather pattern) → photographic ball.
- **Axis B (background, 3 levels)**: empty → single horizon line (ground) → full landscape (room / outdoor).
- **Axis C (context cue, 3 levels)**: none → wind lines (lateral motion) → trajectory arrow / cast shadow.
- **Axis D (object category label, 3 levels)**: "circle" vs "ball" vs "planet" (controlled at prompt-time).
- **Axis E (scene consistency)**: consistent vs inconsistent (e.g. a photo ball on a line-drawn background).

Five physics event templates per condition (fall, roll-on-slope, wall-bounce, hover, horizontal-motion).

**Stimulus generation**: programmatic (matplotlib / PIL / Blender) + Midjourney / SD for the photoreal conditions; WHOOPS-style double annotation for quality control.

**Prompts**. "The image shows {object}. Describe what will happen to the {object} in the next moment." plus a forced-choice version ("Which will happen next: (A) falls down, (B) stays still, (C) moves sideways, (D) this is just an abstract shape and it doesn't move").

**Hypothesis H1**: PMR rises **S-shaped** with abstraction; introducing 3D shading and introducing the ground produce the largest step-changes. H2: the "ball" label increases PMR substantially even on line drawings, demonstrating an **independent contribution of the language prior**. H3: scene inconsistency degrades RC.

**Difficulty / feasibility**. Low–medium. Stimulus generation and VLM querying are standard. Scale: 108 × 50 × 3 VLMs × 2 prompts ≈ 32 k queries (2-3 days of open-source inference).

### 2.3 Sub-task 2 — Linear separation of "physical-ness" in the vision encoder

**Task definition**. Pass each PhysCue image through CLIP-ViT-L/14 (LLaVA-1.5), SigLIP (LLaVA-OneVision / Qwen2-VL), and InternViT-6B (InternVL2); extract patch and CLS activations. Train **linear probes targeting the behavioral PMR labels** from Sub-task 1 (layer-wise, token-wise). Metrics: probe AUC, accuracy; spatial token distribution.

**Why this is informative**. Tests Pixels-to-Principles' claim ("vision encoder captures physics cues but the LLM doesn't use them") across the abstraction axis. If the probe achieves high AUC throughout the vision encoder while VLM behavior fails to follow, this **explicitly localizes a decoding bottleneck**.

**Extensions**:
- **Gandelsman decomposition**: per-attention-head "physical-ness" contribution (auto-labelled with TextSpan). Identifies candidate "support plane", "3D shading", "motion blur" heads.
- **Pach et al. 2025 SAE**: train an SAE on CLIP patch tokens; extract monosemantic features that activate along the abstraction axis.
- **Encoder–decoder dissociation index (EDI)**: EDI = probe AUC − downstream accuracy. Large EDI quantifies "knowledge present but not used".

**Predicted findings / hypotheses**. H4: the shading (3D) cue is linearly separable in vision-encoder layers ~20+; ground presence appears earlier (~layers 10–15). H5: probe AUC is S-shaped along the abstraction axis but with a steeper slope than the behavioral S-curve → **a decoder-side boomerang**. H6: a specific head (CLIP-ViT around L22 H7, distinct from the previously-known shape-preferring heads) specializes in physical cues.

**Difficulty / feasibility**. Medium. Probes are simple, but headwise TextSpan is a CLIP-only method and would need the Balasubramanian adaptation for SigLIP / InternViT.

### 2.4 Sub-task 3 — Layer-wise emergence of physics concepts in the LLM backbone (logit lens + cross-layer probing)

**Task definition**. On LLaVA-1.5-7B, LLaVA-Next-7B, Qwen2-VL-7B, and InternVL2-8B, apply logit lens to the **hidden states at visual-token positions** + per-layer linear probes ("physical object?" binary, "gravity direction" 4-way, "next motion verb" 5-way). Follow Neo et al. 2024's recipe exactly.

**Why this is informative**. Answers "when does physics mode arise" **at the layer level**. Tests whether physics information is stored in early layers (1-4) per Basu et al. 2024, or emerges in mid-layers (15-24) per Neo et al.

**Analyses**:
- **Logit lens across layers**: track the trajectory of physics verbs ("fall", "roll", "bounce", "sit") vs geometry nouns ("circle", "shape", "line") in the unembedding projection at each layer.
- **Per-layer probe on a "physics-mode" binary label**: a 2D heatmap (layer × token position) is the "spatio-temporal map of physics concept emergence".
- **Per-abstraction-level analysis**: cosine distance / CKA between hidden states of shaded sphere vs line circle, layer by layer.
- **Cross-model comparison**: contrast the per-layer maps of LLaVA-1.5 (simple MLP projector) vs LLaVA-Next (extra-resolution tokens) vs Qwen2-VL (dynamic resolution + M-RoPE) vs InternVL2 (pixel shuffle).

**Predicted findings**. H7: line circles project to "circle" / "shape" in the logit lens, with physics verbs not arising until late layers. H8: shaded sphere + ground co-elevates the logits of "ball" and "fall" in mid-layers (~L15-20). H9: Qwen2-VL / InternVL2 switch to physics mode at **earlier layers** than LLaVA-1.5 (thanks to a larger vision encoder and a more sophisticated projector).

**Difficulty / feasibility**. Medium. Hidden-state hook code is standard. Neo et al.'s method also applies to SigLIP / InternViT-based VLMs (already validated on Qwen2-VL).

### 2.5 Sub-task 4 — Causal localization (Semantic Image Pairs + attention patching + steering)

**Task definition**. Construct **Semantic Image Pairs (SIP)** from PhysCue — each pair differs along a single cue axis (e.g. shading present/absent, ground present/absent). Perform **activation patching** between a "clean" image (which elicits a physics response) and a "corrupted" image (which doesn't), following Golovanevsky et al. NAACL 2025's NOTICE recipe. Metric: **indirect effect (IE)** of the patched layer/head — i.e. recovery of the clean response probability.

**Why this is informative**. Probing and logit lens are *correlational*, not causal. This sub-task identifies "the components that are actually necessary for the shading cue to trigger physics mode". The SIP method avoids the hallucination artifact of Gaussian-noise corruption (Zhang & Nanda 2024).

**Intervention types**:
- **Visual token patching**: replace the corrupted image's object visual tokens with the clean image's tokens; measure layer-wise IE.
- **Attention knockout**: zero out attention between visual tokens ↔ last token at specific heads / layers. Rank physics-head candidates.
- **MLP replacement**: restore each layer's MLP output to the clean-image value.
- **Steering vector intervention (VTI-style)**: compute a residual-stream shift vector from (physics-mode response, non-physics response) pairs. Test whether adding this vector at test time can "force" line circles into physics mode.
- **SAE intervention (Pach et al. recipe)**: identify "shading" / "ground plane" directions in CLIP SAE features, amplify or suppress, measure LLM-output change.

**Predicted findings**. H10: 2-3 narrow layer/head ranges (mid-layers, matching Kaduri et al.'s middle 25 %) show large IE. H11: a steering vector can induce physics mode in line circles → "physical-ness is localized as a linear direction inside the LLM". H12: LLaVA, consistent with Golovanevsky et al., lacks visual-grounding self-attention heads; physical triggers will instead **rely on the MLP path**.

**Difficulty / feasibility**. Medium–high. Patching code in TransformerLens / nnsight. SAE training needs additional compute (a few A100 days). The most headline-eligible result.

### 2.6 Sub-task 5 — Cross-model comparison + interaction with prompt / image steering

**Task definition**. Run reduced versions of Sub-tasks 1-4 on 5-7 models: LLaVA-1.5-7B/13B, LLaVA-Next-7B, Qwen-VL, Qwen2-VL-7B, InternVL2-8B/26B, (if possible) Llama-3.2-Vision. Additionally, Gavrikov et al. 2024-style **prompt steering**: measure PMR shift between "treat this as an abstract geometric shape" vs "treat this as a physical object subject to gravity".

**Why this is informative**. Establishes the generality vs model-specificity of the findings. Prompt steering tests whether **image cues and language cues induce physics mode independently or interactively**.

**Predicted findings**. H13: the "ball" prompt raises PMR on line circles substantially but doesn't reach the level of shaded sphere + ground → visual cues retain an independent contribution. H14: models with a larger projector (InternVL2 26B's pixel shuffle) rely more on image cues. H15: Qwen2-VL's M-RoPE captures richer spatial cues → ground-cue effect is larger than in LLaVA.

**Difficulty / feasibility**. Low–medium (reuses the already-built pipeline).

### 2.7 Minimum viable vs ambitious version

**Minimum viable (EMNLP short or workshop)**: Sub-task 1 (axis A + B only, 54 conditions) + Sub-task 2 (CLIP / SigLIP probing only) + Sub-task 3's logit lens (LLaVA-1.5 only). 1 person-month of compute.

**Standard full paper (EMNLP long)**: Sub-tasks 1-3 + Sub-task 4's SIP patching (steering / SAE excluded) + Sub-task 5's 2-model comparison. 3-4 person-months.

**Ambitious (NeurIPS)**: all 5 sub-tasks + SAE-based physics-feature discovery + ≥ 5-VLM comparison + human baseline (Prolific, 50 PhysCue samples, 20 raters). 6 months + 2 people.

### 2.8 Cross-sub-task narrative

§1 observes a behavioral switching curve → §2 shows that the curve already exists in the vision encoder → §3 maps when it emerges inside the LLM → §4 identifies the components that causally produce it → §5 establishes model / prompt generality. The three "whys" are answered in turn: *what triggers it (§1)*, *where it is localized (§2-3)*, *how it can be intervened on (§4-5)*.

---

## Part 3. Positioning, headlines, risks

### 3.1 NeurIPS vs EMNLP framing

**NeurIPS (ML / interpretability angle)**: a title like "Localizing Physical-Mode Activation in Vision-Language Models". Treat Spelke / Michotte cognitive-science as background; foreground the **mechanistic findings** (SAE features, patching IE curves, steering vectors). Track: NeurIPS main (Interpretability, Evaluation & Analysis). Figure 1 = PhysCue stimulus grid + layer-wise probe heatmap. Sub-task 4 results as the headline claim.

**EMNLP (language / grounding angle)**: a title like "When Does a Circle Become a Ball? Probing Physical-Object Reasoning Triggers in Vision-Language Models". Foreground the image cue × language label interaction (Sub-task 5); frame as a **failure mode of vision–language grounding** (a direct successor to Pixels-to-Principles). Track: EMNLP Interpretability and Analysis of Models for NLP, or Resources and Evaluation. Gavrikov et al. 2024 as the central interlocutor.

### 3.2 Headline-result candidates

1. **"S-shaped switching curve"**: open-source VLMs do not gradually transition across the 5 abstraction levels — they show a **sharp phase transition** when shading + ground are simultaneously present. Replicated across 3 models.
2. **"Encoder-decoder boomerang"**: vision-encoder probing achieves 67 % AUC at separating "physicalness" even on line circles, while behavioral PMR stays at 8 % → the failure is on the decoder side.
3. **"Physics head localization"**: attention knockout of LLaVA-1.5's layer 19 head 14 (illustrative) drops physics-mode PMR by 50 pp; keeping the same head active while removing all other visual attention preserves the effect → **causal necessity of a small number of heads**.
4. **"Physics steering vector"**: a residual-stream direction computed from (ball, circle) pairs, when injected at layer 15, induces a physics response on line-circle images with 70 % probability. The physical counterpart of shape-texture steering (Gavrikov et al.).

The most viral headline is the combination of 2 + 3: "VLMs see a circle as a ball but don't say so — and we can force them to".

### 3.3 Risks and alternative questions

**R1: the model shows no clear switching behavior (only monotone increase)**. Fallback: (a) analyze cross-model differences in switch *shape*; (b) reframe as alignment with a human baseline ("do humans and VLMs respond to the same cues?" — a cog-sci alignment paper). Vo et al. 2025 has a similar reframing precedent.

**R2: even the vision encoder fails to separate via probing**. This is itself a major finding — "CLIP / SigLIP do not encode physics cues at all" (extending Eyes Wide Shut). If DINOv2 gives the opposite result, directly connects to the MoF proposal.

**R3: patching shows distributed effects (no concentrated heads)**. Fallback: pivot to SAE-based feature-level findings. Identifying monosemantic "support" / "gravity" features is itself an interesting result.

**R4: the results are open-source-VLM-specific (do not generalize to GPT-4V)**. Mitigation: include a small closed-source behavioral comparison (Sub-task 1 only) so a scaling claim is possible. Pixels-to-Principles includes Gemini for this reason.

**R5: stimulus quality issues (synthetic vs photo distinction reads as a domain shift to the VLM)**. Apply the WHOOPS lesson — Bitton-Guetta et al. already showed accuracy differences came from commonsense, not synthetic artifacts. Apply the same validation to PhysCue (3 same-scene matched rendering styles, domain-shift control).

**R6: linguistic contamination of stimuli (the prompt drives too strongly)**. Mitigation: forced-choice vs open-ended contrast; minimize prompt signal ("What do you see? What happens next?").

### 3.4 Novelty assessment

**No overlapping prior work** at the three-area intersection. Differentiation from the closest competitors:

- vs **Pixels to Principles (Ballout et al. 2025)**: same three open-source VLMs, same probing approach. Differentiator — they do plausibility judgments + photo stimuli + t-SNE-level analysis. We do next-state prediction + parametric abstraction axis + causal patching / steering. **The natural successor that asks *why* the misalignment occurs and *which* cues exacerbate it**.
- vs **"VLMs are Blind" (Rahmanzadehgervi et al. 2024)**: shows VLM failures on abstract shapes + encoder probing. Differentiator — no physics task, no steering / patching. We bridge the "Blind" and "Physics" areas.
- vs **"Are VLMs Shape or Texture Biased" (Gavrikov et al. 2024)**: direct precedent for the prompt-steering methodology. Differentiator — targets physics mode. We extend their method to a **new dimension (physical-ness)**.
- vs **MechBench (Zhang et al. 2024)**: a mechanical-reasoning VLM benchmark. Differentiator — benchmark only; no mechanistic analysis; no abstraction manipulation.

**Venue recommendation**. **EMNLP 2026 long** (estimated May 2026 deadline) is the best fit — the interpretability track is growing and Pixels-to-Principles, NOTICE, BlindTest are all in the NAACL/EMNLP/ACCV family. **NeurIPS 2026** (May deadline) is a candidate for a parallel D&B-track submission of PhysCue alone. **ICLR 2027** is feasible if the mechanistic results are strong.

---

## Conclusion

The user's research question lies precisely at the **intersection of three gaps** in the current VLM literature — abstract–concrete stimulus manipulation, next-state prediction in open-source VLMs, and mechanistic localization of physical triggers. The cognitive-science cue hierarchy (3D shading > shadow + ground > material > perspective > motion blur) provides a **principled scaffold** for stimulus design, and the recently-mature mechanistic-interpretability toolset (Neo et al. logit lens + attention knockout, Golovanevsky SIP, Pach SAE, VTI / VISTA steering) means the **methodological lift is essentially zero**. Pixels-to-Principles' recent vision-language misalignment finding is the ideal launchpad, and Gavrikov et al.'s shape-texture steering paradigm provides a natural extension into "physics-mode steering". The minimum viable experiment (1 month) is defensible as an EMNLP short, while the ambitious version (6 months) is competitive at NeurIPS main track. Even for the largest risk — if no clean switching is observed — fallbacks based on the monotone change itself and a human-baseline comparison provide a stable return on the research investment.
