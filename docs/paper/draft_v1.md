---
target: EMNLP long primary / NeurIPS main stretch
date: 2026-04-26
status: outline + abstract + intro draft (M7 v1)
authors: namam3gy
---

# When Does a VLM See a Ball Instead of a Circle? — Architecture-Level Determinants of Physics-Mode Reading in Open-Source Vision-Language Models

## Abstract (250-word target)

Open-source vision-language models (VLMs) often describe a black circle on
a white background as a "ball that will fall," even when the image carries
no physical cue beyond the circle itself. This shortcut from minimal
abstract geometry to physical interpretation has been observed
anecdotally but never localized. We measure when and how it happens
across **5 production-grade open-source VLMs** (Qwen2.5-VL,
LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3) on **three stimulus sources**
(synthetic shapes, photo-categorical, and real photographs) using a
**rule-based physics-mode reading rate (PMR)** with bootstrap confidence
intervals.

We make three paper-grade claims. **First**, behavioral PMR saturation on
synthetic stim is determined at the **architecture level (joint encoder +
LM)** — not at the encoder representational level. Every encoder
linearly separates physics-vs-abstract factorial cells at AUC = 1.0, yet
behavioral PMR ranges 0.18 → 0.92 on identical stim across the 5 models.
The 2-CLIP-point comparison (LLaVA-1.5 PMR 0.18 vs LLaVA-Next 0.70) rules
out vision-encoder-family as the sole driver. **Second**, the shortcut
is **causally localized along two architecturally-distinct routes**.
LM-side runtime steering (`+α · v_L`) flips behavior in 3 of 4 testable
models at appropriate α (Qwen L10, LLaVA-Next L20+L25, Idefics2 L25,
all 10/10 stim flip). Encoder-side SAE feature ablation breaks PMR in 3
of 5 models (Qwen k=40, Idefics2 k=160, InternVL3 k=160) while
mass-matched random controls hold; the 2 LLaVA models are encoder-NULL
but LLaVA-Next is LM-positive — physics-mode commitment routes through
*LM-side direction* in the CLIP cluster and through *both* encoder-side
features and LM-side direction in the non-CLIP cluster. The same v_L
acts as a regime axis (+α → kinetic, −α → static), not a binary toggle.
**Third**, `v_L10` is **encodable in the image itself**: a small
pixel-space gradient ascent (ε = 0.05 in L∞) flips PMR on 5/5 baseline
circles, while matched-magnitude random directions flip 0/15. The
shortcut path can be "spelled" in pixels — directional specificity,
not perturbation magnitude, drives the regime flip.

These results contribute a clean architecture-vs-representation
disambiguation to the open-source VLM grounding-failure literature
and provide a reusable causal-localization recipe for shortcut-style
behaviors at layer-resolution granularity.

## 1. Introduction

(Target length: 1.5-2 pages.)

### 1.1 The shortcut problem (motivation)

Open-source VLMs trained on web-scale image-text data exhibit a curious
behavior: a minimal synthetic stimulus (e.g., a black circle on white)
elicits physics-mode language ("the ball will fall," "it rolls
downward") even when the prompt is open-ended and the image carries no
physical cue (no ground line, no shading, no texture, no contextual
arrow or shadow). The model commits to a *physical-object* reading of
something a human would read as *abstract geometry*. This is a
shortcut: the visual evidence is consistent with multiple readings, but
the model collapses to one.

Anecdotally documented in red-teaming reports and benchmarks like
*Eyes Wide Shut* (Tong et al., 2024) and grounding-failure analyses,
the shortcut has not been *localized* — i.e., where in the model's
forward pass the abstract→physical transition happens, what
representational axis encodes it, and what input properties activate
it have remained open. Existing causal-interpretability work on
language-only LMs (e.g., Basu et al., 2024 on constraint storage,
Neo et al., 2024 on switching layers) has not been ported to the
vision-language setting beyond a handful of anecdotal probes. And
existing shortcut-analysis tools (linear probes, SAE features) have
not been combined with **synthesis-side counterfactuals** that test
whether the shortcut is encodable in the image.

### 1.2 Contributions

We localize the abstract→physical shortcut along three independent
dimensions, each yielding a paper-grade claim:

1. **Cross-architectural quantification** (§4-§5). 5 production-grade
   open-source VLMs × 3 stim sources × bootstrap CIs reveal that the
   shortcut is determined at the **architecture level** — not at
   encoder representational capacity. Every encoder linearly separates
   physics-vs-abstract factorial cells at AUC = 1.0; behavioral PMR
   ranges 0.18 → 0.92. The 2-CLIP-point comparison (LLaVA-1.5 vs.
   LLaVA-Next) is the cleanest disconfirmer.

2. **Causal localization** (§6). Adding `+α · v_L10` at LM layer 10
   over visual-token positions flips Qwen2.5-VL's behavior with α=40
   (10/10 stim). The same direction is bidirectional (+α kinetic,
   −α static) within physics-mode — a regime axis, not a binary
   toggle. **Cross-model M5a** (§6.2): 3 of 4 testable models flip
   PMR 10/10 at appropriate α (Qwen L10, LLaVA-Next L20+L25,
   Idefics2 L25). LLaVA-1.5 NULL. **Cross-model M5b SAE
   intervention** (§6.4): ablating the top-k physics-cue features at
   the actually-consumed encoder layer breaks PMR cleanly in 3 of 5
   models (Qwen k=20, Idefics2 k=160, InternVL3 k=160) while
   mass-matched random controls hold at 1.0. **The 2 LLaVA models
   are NULL on M5b but POSITIVE on M5a (LLaVA-Next)** — physics-mode
   commitment in the CLIP cluster routes through LM-side direction,
   not encoder-side localized features. Together: a 3-cluster
   architectural decomposition of *where* the shortcut lives.

3. **Pixel encodability with model-conditional shortcut layer**
   (§7; revised 2026-04-26 evening). On Qwen2.5-VL, gradient-ascent on
   post-processor `pixel_values` to maximize `<h_L10, v_L10>` produces
   PMR flips with **5/5 success at ε = 0.05** (L∞-bounded). Random
   unit-direction controls at matched magnitude flip 0/15. Cross-model
   layer sweep on LLaVA-1.5 (L5/L10/L15/L20/L25) reveals **L25 admits
   5/5 v_L flips at ε = 0.2 and 4/5 at ε = 0.1**, with random controls
   0/15 at every tested layer. Pixel-encodability is *not* Qwen-only;
   each model has its own shortcut layer at a different relative LM
   depth (Qwen L10 ≈ 36%, LLaVA-1.5 L25 ≈ 78%). The earlier "Qwen-
   scoped" reading was a wrong-layer-choice artifact. Sample LLaVA-1.5
   L25 ε=0.2 synth response: "The circle will be hit by a ball."
   (baseline: "filled in with color").

A fourth claim — that real photographs **compress** the encoder gap
across all 5 models (paper Table 1) — provides external validity for
the architecture-level reframe and reveals that synthetic-stim
saturation is a co-product of encoder representation *and* input-
context simplicity.

### 1.3 Roadmap

§2 reviews related work. §3 introduces our stimulus design and
metrics. §4 reports the cross-model behavioral PMR ladder with
bootstrap CIs (M6 + M8a + M9; M1-M2 are Qwen-only single-model
runs that establish the protocol and surface H7). §5 reports the
encoder-vs-LM disambiguation (M3 Qwen-only + M6 r2-r6 cross-model
+ §4.5 + M8c + M4 cross-model LM logit-lens AUC ladder). §6
reports causal localization: M5a runtime steering (Qwen + cross-
model 3 of 4 testable), M5a-ext regime axis, and M5b SAE
encoder-side intervention cross-model (3 of 5 models break PMR;
2 LLaVA NULL with the M5a-positive + M5b-NULL dissociation in
LLaVA-Next). §7 reports the pixel-encodability result with the
5-model n=10 layer sweep + Idefics2 9-layer disambiguation that
falsifies wrong-relative-depth and points at perceiver-resampler.
§8 discusses external validity (M8a/d cross-model + multilingual +
decision-stability + §4.8 Qwen 7B vs 32B PMR scaling). §9 catalogs
limitations and remaining open questions including the un-tested
controlled projector-swap and LM-only counterfactual.

## 2. Related Work

(Target length: 1 page. Outline only — flesh out in revision pass.)

### 2.1 Shortcut learning in VLMs
- *Eyes Wide Shut* (Tong et al., 2024): visual primitives that VLMs
  miss → MoF (Mixture-of-Features) proposal as a remedy.
- Vision-language grounding failures: Liu et al., Zhang et al., …
- Language-prior dominance in VLM benchmarks: anecdotally documented,
  not localized.

### 2.2 Linear probing & encoder analysis in VLMs
- Linear probes on encoder activations (Alain & Bengio, Belrose et al.)
- Cross-encoder swap experiments: SigLIP / CLIP / DINOv2 comparisons
  (Tong et al., …)
- Encoder probe AUC vs behavior: existing work usually correlational;
  we add a 5-model bootstrap-CI ladder + LM-side counterfactual.

### 2.3 Causal localization in language models
- Activation patching / SIP (Wang et al., Conmy et al.)
- Logit lens (nostalgebraist), tuned lens (Belrose et al.)
- Constraint-information storage (Basu et al., 2024) — early-layer
  intervention findings we replicate at the visual-token positions.
- VTI steering vectors / class-mean directions (Burns et al., …)

### 2.4 Adversarial / counterfactual stimulus generation
- Standard adversarial perturbations (Goodfellow et al., Madry et al.)
  — minimax-loss optimization in pixel space.
- Activation-targeting feature visualization (Olah et al.)
- Our §4.6 uses class-mean direction as the optimization target,
  closer to *targeted feature visualization* than to standard
  adversarial perturbation.

## 3. Methods

(Target length: 2 pages.)

### 3.1 Stimulus design

We use four stimulus sources, three of which are cross-model and one
of which (M2) is single-model:

**M2 synthetic factorial — Qwen2.5-VL only** (2880 inferences = 480
stim × 6 prompt-variants × 1 model). Single-shape (circle), 5 axes:
- `object_level` ∈ {line, filled, shaded, textured} — abstraction axis
- `bg_level` ∈ {blank, ground, scene} — context axis
- `cue_level` ∈ {none, cast_shadow, motion_arrow, both} — physics-cue axis
- `event` ∈ {fall, rise, horizontal} — direction axis
- `seed` ∈ {1..N} — randomization

The fine-grained axis decomposition (cast_shadow vs motion_arrow,
ground vs scene, FC vs open-ended at every cell) is M2-specific and
Qwen-only. Cross-model generalization of M2's headline findings
(H1 ramp, H7 label-selects-regime) lives in M8a (5-shape × 5-model)
and M8d (3-category × 2-model). M2's protocol was partially replicated
on LLaVA-1.5 in M6 r1 (M2 stim + label-free prompt).

**M8a non-circle shapes — cross-model** (5 shapes × 5 models): replace
the disk with square / triangle / hexagon / irregular polygon at every
level. Reduced factorial (object_level × bg × cue × seed, fall event
only) = 80 stim per shape × 5 shapes × 5 models for the canonical run.

**M8d non-ball categories — Qwen + LLaVA** (3 categories × 2 models):
replace the ball with car / person / bird (event-axis doubled to 2 to
include `horizontal` natural motion). 480 stim total per arm.

**M8c real photographs — 5 models on subset** (60 photos total): 12
photographs each of {ball, car, person, bird, abstract} from COCO
2017 + WikiArt; covered by all 5 models for the cross-stim check.

### 3.2 Models

Five production-grade open-source VLMs, all loaded via the generic
`AutoModelForImageTextToText` / `AutoProcessor` interface
(transformers ≥ 4.45):

| Model | Vision encoder | LM | Image strategy |
|---|---|---|---|
| Qwen2.5-VL-7B | SigLIP | Qwen2-7B | dynamic 504×504 |
| LLaVA-1.5-7B | CLIP-ViT-L/14 | Vicuna-7B | 336×336 fixed |
| LLaVA-Next-7B | CLIP-ViT-L/14 | Mistral-7B | AnyRes 5-tile |
| Idefics2-8B | SigLIP-SO400M | Mistral-7B | 384×384 |
| InternVL3-8B | InternViT-300M | InternLM3-8B | 448×448 dynamic |

### 3.3 Metrics

**PMR (physics-mode reading rate)**: fraction of model responses that
describe the next-state in physical terms (e.g., "the ball falls,"
"it rolls"). Rule-based scorer with multilingual stems
(English / Korean / Japanese / Chinese), gated by abstract markers
("won't move," "remain stationary," "no indication of motion") to
avoid false positives. Hand-validated 5-6 % disagreement vs.
hand-annotation.

**PMR variants**:
- `PMR(_nolabel)`: open-ended prompt without label cue ("What do you
  see? What might happen next?"). Direct measure of joint
  encoder+LM tendency.
- `PMR(_physical)`: prompt with role-physical label ("the ball...",
  "the car...").
- `PMR(_abstract)`: prompt with role-abstract label ("the circle...",
  "the silhouette...").

**GAR (gravity-align rate)**: fraction of physics-mode responses that
describe downward motion (subset of PMR with direction).

**RC (response consistency)**: fraction of (model, stim) pairs where
T=0.7 sampling produces the same PMR call across N seeds.

**Bootstrap CI**: 95 % percentile (2.5–97.5) over 5000 iterations,
with predictions resampled within each (shape × role) cell.

### 3.4 Activation capture and probing

For each model we hook the vision tower's transformer blocks at
selected layers (3, 7, 11, 15, 19, 23, 27, 31 in the encoder; 5, 10,
15, 20, 25 in the LM) and capture per-stim mean-pooled hidden
states + per-token states at visual-token positions. Linear logistic-
regression probes over the captured states give per-layer AUC against
two label types: `behavioral-y` (the per-stim PMR call) and `stim-y`
(the factorial-cell label).

### 3.5 Causal intervention (M5a / M5a-ext / §4.6)

**Class-mean steering vector**:
```
v_L = mean_sid (mean_token h_L[sid] | PMR(sid)=1)
    − mean_sid (mean_token h_L[sid] | PMR(sid)=0)
v_unit_L = v_L / ||v_L||₂
```
Intervention: forward hook on `model.model.language_model.layers[L]`
adds `α · v_unit_L` to all output hidden_states (uniformly over
token positions). α ∈ {0, 5, 10, 20, 40, −5, −10, −20, −40}. T=0
(deterministic).

**Pixel-space gradient ascent** (§4.6): same target
`<mean(h_L10[visual]), v_L10>`, but optimize on the post-processor
`pixel_values` tensor (Qwen2.5-VL: shape `(T_patches, 1176)` where
1176 = 2·3·14·14). float32 leaf cast to bf16 in the forward pass;
the vision tower → projector → LM 0..10 path is end-to-end
differentiable. Adam, lr=1e-2, n_steps=200. L∞-bounded on
`pv_leaf − pv_initial` ∈ {±0.05, ±0.1, ±0.2} or unconstrained.

## 4. Behavioral findings — cross-model PMR ladder

(Target: 1.5 pages with 1-2 figures.)

### 4.1 The PMR(_nolabel) ladder

![Figure 1: 5-model × 3-stim PMR(_nolabel) ladder with bootstrap CIs](../figures/session_5model_cross_stim_pmr.png)

*Figure 1.* Mean PMR(_nolabel) ± 95% bootstrap CI per
(model × stim source). Encoder-family split is clean on synthetic
stim (M8a, M8d) and collapses on real photos (M8c).

Across the 5 models on M8a synthetic shapes:

| Model | PMR(_nolabel) | 95 % CI |
|---|---|---|
| Qwen2.5-VL | 0.838 | [0.79, 0.88] |
| LLaVA-1.5 | 0.175 | [0.14, 0.21] |
| LLaVA-Next | 0.700 | [0.65, 0.74] |
| Idefics2 | 0.882 | [0.84, 0.92] |
| InternVL3 | 0.917 | [0.88, 0.95] |

Three clusters: **saturated** (Qwen / Idefics2 / InternVL3 ≥ 0.84),
**mid-band** (LLaVA-Next 0.70), **floor** (LLaVA-1.5 0.18). CIs are
fully separated between the three clusters.

### 4.2 H1 (abstraction ramp) — unsaturated-only

![Figure 6: M8a abstraction ramp per shape per model](../figures/m8a_pmr_ramp.png)

*Figure 6.* PMR(line / filled / shaded / textured) per (shape × model)
on M8a. LLaVA-1.5 shows clean monotone ramps; Qwen / Idefics2 /
InternVL3 are at ceiling and the ramp is invisible.

LLaVA-1.5 (the unsaturated model) shows a clean monotone S-curve
(line 0.45 → textured 0.78). Qwen / Idefics2 / InternVL3 are at
ceiling and the ramp is invisible. Strict pre-registered scoring
(M8a) on 5 shapes: Qwen 1/4 PASS, LLaVA 4/4 PASS — the asymmetry
*is* the cross-shape validation of the architecture-level reframe.

### 4.3 H7 (label selects regime) — cross-category replication

![Figure 7: M8d paired-delta per (model × category × label_role)](../figures/m8d_paired_delta.png)

*Figure 7.* PMR_regime(physical) − PMR_regime(abstract) per
(model × category) on M8d. LLaVA shows positive deltas in all 3
categories; Qwen is flat (ceiling).

Cross-category strict scoring (M8d):
- LLaVA: 3/3 PASS (car +0.525, person +0.138, bird +0.550 on
  PMR_regime physical−abstract).
- Qwen: 0/3 binary (ceiling-flat) but regime distribution shows
  figurine 17.5% static, statue 22.5% static — the label-selects-
  regime claim is now category-general, not circle-specific.

### 4.4 Photo collapse (M8c)

![Figure 8: M8c synthetic vs photo paired comparison](../figures/m8c_paired_synthetic_vs_photo.png)

*Figure 8.* Per-category mean PMR(_nolabel) on synthetic vs photo
stim, paired across (model × category). Photos compress the encoder-
family gap and reduce Qwen PMR by 18-48 pp.

Real photographs reduce Qwen PMR(_nolabel) by 18-48 pp across
categories. All 3 tested models converge to PMR [0.18, 0.67] on
photos. The encoder gap that was clean on synthetic stim collapses
on rich photos — synthetic-stim minimality is a co-factor of
behavioral saturation, not just encoder representation.

## 5. Encoder vs LM disambiguation

(Target: 1.5 pages.)

### 5.1 Vision encoder probes — uniform discriminability

![Figure 2: 5-model encoder probe AUC chain](../figures/encoder_chain_5model.png)

*Figure 2.* Layer-sweep AUC of a logistic regression probe trained
on per-stim PMR labels at each encoder layer. Non-CLIP encoders
(SigLIP, SigLIP-SO400M, InternViT) cluster at AUC ≥ 0.88; only
CLIP-ViT-L falls below.

5-model encoder probe AUC chain (M6 r2-r6 apples-to-apples on M8a
stim):

| Model | Encoder | AUC | Behavioral PMR(_nolabel) |
|---|---|---|---|
| Qwen2.5-VL | SigLIP | 0.88 | 0.84 |
| LLaVA-1.5 | CLIP-ViT-L | 0.77 | 0.18 |
| LLaVA-Next | CLIP-ViT-L | — | 0.70 |
| Idefics2 | SigLIP-SO400M | 0.93 | 0.88 |
| InternVL3 | InternViT | 0.89 | 0.92 |

The encoder discriminability gap (0.77 vs 0.93) is much smaller than
the PMR gap (0.18 vs 0.92). On a stim-defined y target, every
encoder hits AUC = 1.0 — the information is uniform across encoders.

### 5.2 The 2-CLIP-point insight

LLaVA-1.5 (CLIP-ViT-L + Vicuna): PMR(_nolabel) = 0.18.
LLaVA-Next (CLIP-ViT-L + Mistral + AnyRes tiling): PMR(_nolabel) =
0.70. Same encoder family, 0.52-PMR jump. This rules out
vision-encoder family as the sole determinant.

(The jump is 4-axis-confounded — AnyRes tiling, fusion projector,
training, LM family — so we cannot isolate the LM-only contribution
from this comparison alone. We flag this as a future-work LM-swap
counterfactual.)

### 5.3 §4.5 cross-encoder swap

![Figure 9: §4.5 cross-encoder swap heatmap (Idefics2 vs Qwen vs LLaVA)](../figures/encoder_swap_heatmap.png)

*Figure 9.* PMR(_nolabel) per (model × shape) on M8a. Idefics2
(SigLIP-SO400M + Mistral) clusters with Qwen (SigLIP + Qwen2);
LLaVA (CLIP-ViT-L + Vicuna) is the outlier.

Idefics2-8B (SigLIP-SO400M + Mistral-7B) provides a causal
counterfactual at the encoder-family level. Patterns identically
with Qwen on PMR + H7. With LLaVA at 0.18 (CLIP + Vicuna) and
Idefics2 at 0.88 (SigLIP-SO400M + Mistral), the encoder type drives
PMR ceiling regardless of LM (Qwen2-7B vs Mistral-7B).

### 5.4 LM logit-lens cross-model — second downstream signature (M4)

The encoder-saturation chain extends downstream into the LM. M4
cross-model trains a logistic-regression probe on per-stim mean PMR
(phys ≥ 0.667 / abs ≤ 0.333) using the visual-token mean hidden
state at LM layers L5/L10/L15/L20/L25, with 5-fold StratifiedKFold.
Reusing existing M2 cross-model captures (no new inference), the
5-model AUC ladder is:

| Model | L5 | L10 | L15 | L20 | L25 | M3 vision AUC |
|---|--:|--:|--:|--:|--:|--:|
| Idefics2-8B | **0.995** | **0.995** | **0.995** | **0.995** | **0.995** | 0.93 |
| Qwen2.5-VL-7B | 0.965 | 0.965 | 0.962 | 0.959 | 0.957 | 0.99 |
| LLaVA-Next-Mistral-7B | 0.732 | 0.762 | 0.751 | 0.786 | 0.791 | 0.81 |
| LLaVA-1.5-7B | 0.758 | 0.760 | 0.762 | 0.763 | 0.768 | 0.73 |
| InternVL3-8B | NaN | NaN | NaN | NaN | NaN | 0.89 |

Three findings: (1) the LM AUC ladder aligns with the M3 vision AUC
ladder — encoder-saturation propagates downstream as "amount of
PMR-relevant information at visual-token positions in LM hidden
states." (2) **Idefics2 LM AUC (0.995) > Idefics2 vision AUC (0.93)**:
the perceiver-resampler does not strip the physics-mode signal — if
anything, it concentrates the signal on the compressed 320-token
budget, raising the LM-side probe AUC above the pre-compression vision
encoder AUC. (3) Combined with the §4.6 Idefics2 0/9 layers shortcut
result, this triangulates as **"information presence ≠ pixel-space
shortcut routability"** — the LM has the physics-mode signal at high
quality, yet pixel-space gradient ascent cannot find a perturbation
that flips PMR. The bottleneck is on the inverse (pixel-side) pathway,
not the forward (encoder → LM) pathway. InternVL3 untestable
(n_neg = 1 → probe degenerate).

### 5.5 The architecture-level reframe

Reading: behavioral PMR(_nolabel) saturation on synthetic stim is
determined at the **architecture level (joint encoder + LM)**, not
at encoder representational capacity alone. Stim-defined AUC = 1.0
across all encoders; behavioral-y AUC and PMR vary 0.18-0.92. The
PMR ladder reflects each LM's reading of encoder output as
"physics-mode signal" — downstream-conditional, not encoder-info.

## 6. Causal localization — M5a (LM-side) + M5b (encoder-side) cross-model

(Target: 2-2.5 pages with the L10 plot, the cross-model M5a table,
and the M5b SAE drop curve figure.)

### 6.1 v_L direction extraction

For each captured layer L, we compute the class-mean difference
direction over M2's labeled subset (n_pos = 312, n_neg = 168).
Direction norm grows 5× through the LM (5.9 at L5 to 31 at L25).

### 6.2 Causal intervention sweep

![Figure 3: The stim being steered](../figures/01_line_blank_none.png)

*Figure 3.* The `line/blank/none` baseline that v_L10 flips at L10
α=40. PMR(_nolabel) on this stim ≈ 0 across all 5 models.

10 baseline `line/blank/none` stim × 4 layers × 5 α values = 200
inferences with forced-choice prompt and label = "circle" (baseline
PMR ≈ 0).

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10 D | 10 D | 10 D | 10 D | **10 B** |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

L10 α=40 flips 10/10 from D ("This is an abstract shape...") to B
("It stays still — the circle appears to be floating or suspended
in space without any external force..."). No other layer moves at
the same α.

### 6.2b M5a runtime steering — cross-model (3 of 4 testable models flip)

The M5a forward-hook protocol extends to the 5-model M8a chain by
re-extracting per-model v_L from M2 cross-model captures. For each
testable model, we identify a baseline-PMR=0 stim cell (per-model
abstract baseline), inject `α · v_L_unit` into the chosen LM decoder
layer, and use the OPEN prompt with PMR scorer.

| Model | LM layer | α sweet spot | PMR flip rate | Stim cell (baseline) |
|---|---|---:|---:|---|
| Qwen2.5-VL-7B | L10 | 40 | **10/10** | line/blank/none × circle (0) |
| **LLaVA-Next-Mistral-7B** | **L20** | **10** | **10/10** | **line_blank_both × circle (0)** |
| **LLaVA-Next-Mistral-7B** | **L25** | **15-20** | **10/10** | **line_blank_both × circle (0)** |
| **Idefics2-8B** | **L25** | **20** | **10/10** | **line_blank_none × circle (0)** |
| LLaVA-1.5-7B | L25 | (any 0-60) | **0/10** | line_blank_none × circle (0) |
| InternVL3-8B | — | — | (untestable) | n/a (saturated, baseline=1) |

Coherent physics-mode responses (paper-quality, not token degeneracy):

- LLaVA-Next L25 α=15: "The ball will bounce up."
- LLaVA-Next L20 α=10: "The ball will roll down the ramp."
- **Idefics2 L25 α=20: "The tip of the arrow will hit the center of the circle."**
- Qwen L10 α=40 (M5a original): "The ball is falling down due to gravity."

LLaVA-1.5 NULL at every α (responses change semantically — "filled with
color" → "in the center" → "on the floor" — but the PMR scorer never
catches a motion stem) replicates the §4.6 weak-shortcut finding
(L25 4/10 at n=10) at the LM-side intervention.

**The Idefics2 result resolves the §4.6 perceiver-resampler hypothesis
ambiguity** by triangulating with M4 LM probe and §4.6 pixel ascent:

| Test on Idefics2 | Result | Implication |
|---|---|---|
| M4 LM probe AUC | 0.995 across L5-L25 | Information *reaches* the LM at high quality |
| **M5a runtime steering** | **L25 α=20 → 10/10 flip** | Forward-direction `v_L` is *operative* in the LM |
| §4.6 pixel-space gradient ascent (L5-L31) | 0/90 v_unit + 0/90 random | Inverse pathway lacks routability |

**Refined hypothesis**: the perceiver-resampler does not strip the
physics-mode signal — the LM has it (M4 0.995) and forward-hook
steering exploits it (M5a 10/10). What the perceiver-resampler removes
is the *pixel-space gradient route* that selectively hits this
direction from the input side. The bottleneck is on the **inverse**
(pixels → v_L direction) pathway, not the **forward** (v_L direction
→ LM commitment) pathway.

At sweet-spot α, all 10 stim seeds at any single (layer, α) produce
essentially identical responses (Idefics2 L25 α=20: 10/10 stim →
"The tip of the arrow will hit the center of the circle.", unique=1;
LLaVA-Next L25 α=15: 10/10 → "The ball will bounce up.", unique=1).
This **regime-attractor** behavior — a deterministic physics-mode
attractor regardless of stimulus — is consistent with M5a-ext's
regime-axis finding: at sufficient α, the regime is forcibly selected
and visual content tracking lives at moderate α. The cross-model 10/10
flip is causal evidence of **regime control**, not of stim-conditioned
physics interpretation; the §4.6 pixel-space gradient ascent and M5b
SAE knockout provide the stim-conditioned signatures (see §6.4).

### 6.3 v_L10 is a regime axis (M5a-ext)

Initial reading framed v_L10 as a one-way "physical object-ness"
activator. Three follow-up experiments revised this:

**Exp 1** (textured/ground/both, near-ceiling baseline): −α has no
effect (ceiling artifact).

**Exp 2** (line/blank/none × +α=40, label = ball): the model
switches B (static) → A (rolls / falls). Label selects regime when
the steering direction is active.

**Exp 3** (textured/blank/none, moderate baseline): −α=40 flips
D → B uniformly across (line, textured) × (ball, circle).

Revised reading: both signs of α activate physics-mode. Sign
selects regime: +α kinetic / −α static. Baseline D sits below the
|α| threshold, not at one end of the axis. **v_L10 is a regime axis
within physics-mode**, not a binary object-ness toggle.

### 6.4 Encoder-side intervention (M5b SAE) — cross-model

Sections 6.1–6.3 establish that injecting `α · v_L` at LM layer L
flips behavior. M5b asks the complementary question: are the
**encoder-side** features that feed into v_L causally bound? If so,
ablating top SAE physics-cue features at the consumed vision-encoder
layer should break PMR while mass-matched random ablations should not.

**Setup.** For each architecture, we (1) train a 4×-overcomplete sparse
autoencoder on `vision_hidden_<L>` activations, where L is the layer
the LM actually consumes (Qwen2.5-VL: layer 31 = last; LLaVA family:
layer 22 per `vision_feature_layer=-2`; Idefics2: layer 26 = last;
InternVL3: layer 23 = last per `vision_feature_layer=-1`). (2) Rank
features by Cohen's d on the per-stim mean PMR split (phys vs abs).
(3) For each test stim, ablate the top-k features via the Bricken
et al. trick (subtract feature contributions from the encoder
hidden state), then run the OPEN prompt and PMR-score the response.
Random controls use 3 mass-matched feature sets drawn from the
high-mass non-top-k pool.

**Result.** 3 of 5 models break PMR cleanly with selectivity vs random.

| Model | layer | k_break | random rate | encoder cluster |
|---|---:|---:|---:|---|
| Qwen2.5-VL-7B | 31 | 20 (0.4 % features) | 1.0 | non-CLIP, AUC 0.99 |
| Idefics2-8B | 26 | 160 (3.5 %) | 1.0 | non-CLIP, AUC 0.93 |
| InternVL3-8B-hf | 23 | 160 (3.9 %) | 1.0 | non-CLIP, AUC 0.89 |
| LLaVA-1.5-7B | 22 | NULL (no break ≤ 160) | 1.0 | CLIP, AUC 0.73 |
| LLaVA-Next-7B | 22 | NULL (no break ≤ 160) | 1.0 | CLIP, AUC 0.81 |

The break threshold scales inversely with M3 vision-encoder probe AUC:
the more discriminative the encoder, the more concentrated its
physics-cue features, and the smaller the k needed to break PMR.

**The CLIP NULL is mechanistically informative.** LLaVA-Next shows
*positive* M5a steering (L20+L25 10/10 flip at LM side) — the LM-side
direction is operative. But its encoder-side SAE intervention is NULL
at any tested k. This is a **dissociation**: LLaVA-family physics-mode
commitment routes through LM-side residual-stream direction, not
through encoder-side localized features. The non-CLIP cluster
(Qwen / Idefics2 / InternVL3) routes through both — encoder-side
features and LM-side direction are jointly causal.

This is the **second downstream signature** of H-encoder-saturation
(after M3 probe AUC and §4.6 pixel-encodability) — M5b SAE intervention
sorts the same 3-cluster decomposition. Combined with the M5a × M4 ×
§4.6 triangulation for Idefics2 (LM has signal, forward-hook works,
but pixel→v_L route blocked), the architectural picture is:

> Physics-mode commitment requires a *direction* in the LM residual
> stream (M5a, all non-encoder-saturated models flip 10/10 with
> appropriate α). What varies cross-architecturally is whether the
> *encoder localizes* this direction in extractable features (M5b
> POSITIVE in non-CLIP) and whether *pixel-space gradient* can route
> to it (§4.6 POSITIVE in MLP-projector models). The CLIP encoder +
> projector pipeline supports neither encoder-side localization nor
> pixel-space routability, but the LM-side direction is still
> causally operative when injected directly.

## 7. Pixel encodability — §4.6

(Target: 1.5 pages with the §4.6 panel + trajectory figures.)

### 7.1 The reverse direction

M5a establishes that adding `α · v_L10` at LM L10 over visual tokens
steers behavior. §4.6 asks the inverse question: **can a small
perturbation in pixel space make the model itself project onto
v_L10 without runtime steering?**

If yes, the shortcut is *encodable* in the image — the model is
extracting v_L10-directional information from pixel-level features,
not from runtime intervention.

### 7.2 Method

Optimize Qwen2.5-VL's post-processor `pixel_values` tensor (T × 1176
where 1176 = 2·3·14·14) to maximize
`⟨mean(h_L10[visual]), v_L10⟩`. Adam (lr=1e-2, n_steps=200).
float32 leaf cast to bf16 in the forward pass; the vision tower →
projector → LM 0..10 path is end-to-end differentiable (Phase 1
gate confirmed gradient max_abs = 13.75 with no NaNs).

L∞-bounded on `pv_leaf − pv_initial`: ε ∈ {0.05, 0.1, 0.2} or
unconstrained.

### 7.3 Configurations + result

![Figure 4: §4.6 4-panel canonical seed](../figures/sec4_6_counterfactual_stim_panels.png)

*Figure 4.* baseline → v_L10 ε=0.05 → v_L10 ε=0.1 → v_L10 unconstrained
on a single seed. The abstract circle gestalt is preserved across all
bounded conditions; ε=0.05 is a low-amplitude texture visible only on
close inspection.

5 baseline circle stim × 7 configurations × 200 steps = 35 runs.

| Config | n flipped (PMR 0→1) | Mean final projection |
|---|---|---|
| bounded ε=0.05 (v_L10) | **5 / 5** | 43.7 |
| bounded ε=0.10 (v_L10) | **5 / 5** | 100.6 |
| bounded ε=0.20 (v_L10) | **5 / 5** | 125.9 |
| unconstrained (v_L10) | **5 / 5** | 181.1 |
| control random unit dir × 3 @ ε=0.10 | **0 / 15** | 73-85 |

![Figure 5: §4.6 projection trajectories per config](../figures/sec4_6_counterfactual_stim_trajectory.png)

*Figure 5.* Mean projection trajectory ± std over 5 seeds per
config. Random directions (dashed) reach final projections ≈ 73-85;
bounded ε=0.1 v_L10 reaches ≈ 101 — same order of magnitude, but
behavioral outcome diverges (5 flips vs 0 flips).

5/5 v_L10 flips at the smallest ε; 0/15 random-direction flips at
matched magnitude. **Directional specificity, not magnitude,
controls the regime flip.** Sample synthesized response: "The
circle will continue to fall downward due to gravity." Sample
random-control response: "The circle will remain stationary as
there is no indication of movement..."

### 7.4 Visual character of the perturbation

ε = 0.05 produces a low-amplitude pattern visible on close
inspection — a faint dotted texture overlaid on the white
background — but the abstract circle gestalt is preserved. A casual
viewer would still describe the image as "a black circle on white"; the
perturbation does **not** introduce gravity cues, ground lines,
shadows, or any physically suggestive features that a human would
read.

The relevant claim is: **the model can be flipped by a perturbation
that does not introduce human-readable physical content**. We do not
claim the perturbation is imperceptible.

### 7.5 Implication: shortcut on the pixel path

Combined with M5a (runtime steering), §4.6 places v_L10 on the
**shortcut path**: it is a direction the vision encoder + projector
can write into from pixel-level features alone, the LM reads out
from at L10, and the behavioral consequence (PMR) follows from the
projection magnitude along this specific axis. The random-direction
controls falsify "any sufficient perturbation flips PMR" and
isolate v_L10.

### 7.6 Cross-model §4.6 — pixel-encodability is architecture-conditional

The §4.6 protocol extends to the 5-model M8a chain by writing a
per-architecture counterfactual generator (Qwen patch-flattened,
LLaVA standard CLIP, LLaVA-Next 5-tile AnyRes with per-element clip,
Idefics2 5-tile + pixel_attention_mask, InternVL3 single 448×448).
For each model, we compute a per-model `v_L_unit` from M2 cross-model
captures and run gradient ascent at ε=0.1 across LM layers
{L5, L10, L15, L20, L25} × n=10 baseline-PMR=0 stim per (model, layer).
For Idefics2 we add 4 deeper layers (L26, L28, L30, L31) — total 9
layers spanning 16 % to 97 % relative LM depth.

![Figure 6: §4.6 cross-model layer sweep](../figures/sec4_6_cross_model_layer_sweep.png)

*Figure 6.* Per-model PMR flip rate at ε=0.1 across LM layers. Wilson
95 % CI on Bernoulli rate. Each architecture has its own shortcut
layer profile.

| Model | Encoder + projector | Shortcut layers (≥ 50 %) |
|---|---|---|
| Qwen2.5-VL-7B | SigLIP + MLP | L5, L10, L15, L20, L25 (broad) |
| LLaVA-Next-Mistral-7B | CLIP-ViT-L + AnyRes 5-tile + MLP | L20, L25 (10/10 each) |
| LLaVA-1.5-7B | CLIP-ViT-L + MLP | L25 only (4/10 at n=10) |
| **Idefics2-8B** | **SigLIP-SO400M + perceiver-resampler** | **0 of 9 layers (L5-L31)** |
| InternVL3-8B-hf | InternViT + MLP | (untestable, baseline=1.0) |

**The Idefics2 0-of-9 result falsifies the "wrong-relative-depth"
explanation** (the LLaVA-1.5 → LLaVA-Next inference that LLaVA-1.5's
shortcut is at greater relative LM depth than Qwen's). Idefics2's v_L
projection ascends cleanly at every depth (baseline -10.7 → final
+27 to +30 at L26-L30; -72 → +163 at L31), confirming gradient ascent
*works at the projection level* — yet 0 PMR flips behaviorally.

Across the 5 models × 5 layers (25 cells) plus the Idefics2 4 deeper
layers, the **aggregate random-direction flip rate is 1/250 trials**
(24 of 25 random-control cells = 0/10; only Qwen L10 random has 1/10,
far below v_unit 10/10 at the same layer). Direction-specificity is
preserved at the projection level even where it does not translate
to behavioral PMR flip.

**Refined H-shortcut**: pixel-encodability is **architecture-
conditional**. Encoder saturation alone (M3 vision AUC) is necessary
but not sufficient — Idefics2's encoder probe AUC (0.93) is between
Qwen's 0.99 and LLaVA-Next's 0.81, yet Idefics2 has 0 shortcut layers
while LLaVA-Next has 2. The disambiguating axis is **projector
design**: Idefics2 is the only perceiver-resampler model in the chain;
Qwen + LLaVA-1.5 + LLaVA-Next + InternVL3 all use MLP projectors of
varying widths. The 5-model design does not isolate this axis — a
controlled projector-swap (same encoder + LM, perceiver ↔ MLP only)
would be the rigorous test.

Triangulation with M4 LM probe and M5a runtime steering on Idefics2
(see §5.4 and §6.2b) supports the refined hypothesis: **perceiver-
resampler removes pixel-space gradient routability, not the LM-side
information**. The LM has the physics-mode signal at AUC 0.995 + the
direction is operative under runtime injection (10/10 flip), yet the
inverse pixel→v_L pathway lacks selectivity. The bottleneck is on the
**inverse** (pixels → v_L) side, not the **forward** (v_L → LM)
side. We promote this to "leading remaining mechanism candidate"
without claiming isolation; the controlled projector-swap remains
future work.

## 8. External validity & secondary findings

### 8.1 Multilingual labels (§4.3)

![Figure 10: §4.3 5-model Korean vs English label PMR](../figures/sec4_3_korean_vs_english_cross_model.png)

*Figure 10.* Per-model EN vs KO PMR per role (physical / abstract /
exotic). Cross-label ordering preserved 4/5 models; LLaVA-1.5 has
the largest swing.

Korean (공/원/행성) and Japanese (ボール/円/惑星) labels on M8a
circle stim. Cross-label ordering preserved 4/5 models on Korean.
Mechanisms differ:
- Qwen genuinely engages Japanese-as-Japanese (label-echo 85-91%).
- LLaVA-1.5 translates kanji to English internally.
- Idefics2 falls back to Chinese on 惑星 in 24% of responses
  (cross-script bypass, scorer extended).

### 8.2 Decision-stability ceiling (§4.7)

![Figure 11: §4.7 5-model per-axis RC](../figures/sec4_7_rc_per_axis.png)

*Figure 11.* Per-axis (object_level / bg_level / cue_level)
response-consistency RC at T=0.7. Non-CLIP models converge to RC ≈ 1.0
when cues fire; CLIP-based models retain seed-level variance.

Saturated models (Qwen / Idefics2 / InternVL3) converge to the same
PMR call across 5 seeds when cues fire. CLIP-based models
(LLaVA-1.5 / LLaVA-Next) retain seed-level variance even under
strong cues. Saturation is not just a behavioral PMR ceiling but
also a **decision-stability ceiling** — separate signature of the
architecture-level reframe.

### 8.3 Categorical regime distribution (§4.11)

![Figure 12: §4.11 5-model M8d regime distribution](../figures/sec4_11_regime_distribution_5model.png)

*Figure 12.* Stacked-bar fractions of (kinetic / static / abstract /
ambiguous) per (model × category × label_role). LLaVA-1.5 most
regime-discriminative; InternVL3 person × exotic (statue) shows
the largest single label-driven static commit (65% static).

Granular form of M9's H7 finding. InternVL3 person × exotic
(statue): PMR drops 0.800 → 0.481, 65% static — strongest single
label-driven static commit in the project. Categorical view reveals
the *kind* of commitment, not just whether the model commits.

### 8.4 PMR scaling — Qwen 7B vs 32B (§4.8)

We test whether 5× scaling moves the architecture-level PMR ceiling.
Qwen2.5-VL-32B on the same M2 stim (480 stim × 3 labels × OPEN prompt)
gives aggregate PMR 0.926 vs 7B's 0.931 — a difference within noise.

| Cell | 7B PMR | 32B PMR | Δ |
|---|--:|--:|--:|
| Aggregate | 0.931 | 0.926 | −0.005 |
| cue=both | 0.978 | 0.972 | −0.006 |
| cue=cast_shadow | 0.957 | 0.945 | −0.012 |
| cue=motion_arrow | 0.992 | 0.987 | −0.005 |
| **cue=none (weakest)** | **0.797** | **0.711** | **−0.086** |

Two scaling-dependent shifts surface only at the weak-cue end:
abstract_reject jumps 35× (0.002 → 0.065), and the H2 label gap
(`ball − circle`) halves (+0.071 → +0.010). 32B is more cue-sensitive
on the 5% of cells where the cue is weakest, but the language-prior
dominance regime survives. **Scaling moves the within-cluster floor
on weak cues; it does not move the architecture-level ceiling on
strong cues.** Consistent with MechBench-style "scale doesn't fix
grounding" findings — the SigLIP+Qwen2 architecture cluster is the
unit that determines ceiling, not parameter count within the cluster.

Qwen 72B (~144 GB at bf16, dual-GPU or quantization required) is
predicted to land near 32B by the 7B↔32B null pattern; it is
deferred from v1 paper scope as confirmatory.

### 8.5 Cross-model attention to visual tokens (§4.10)

![Figure 13: §4.10 cross-model attention to visual tokens](../figures/session_attention_cross_model.png)

*Figure 13.* Per-layer attention from the last text token to visual
tokens, per model. Visual attention peaks at mid-layers; absolute
allocation varies architecturally (Qwen ~17%, LLaVA-1.5 ~7%,
Idefics2 ~30%) despite all models receiving 79-98% visual tokens.

Last-token attention to visual tokens varies architecturally despite
all 5 LMs receiving 79–98% visual input tokens: Qwen ~17%, LLaVA-1.5
~7%, Idefics2 ~30%. Visual attention peaks at mid-layers in all
models. Cross-model architecture difference, not encoder
difference.

## 9. Discussion + limitations

### 9.1 What we showed

The architecture-level reframe is the cleanest finding, and it is
manifested as a **5-fold downstream signature redundancy** —
behavioral PMR ceiling (M9), decision-stability ceiling (§4.7),
pixel-encodability (§4.6), LM logit-lens probe AUC (M4 cross-model),
and encoder-side SAE feature ablation break threshold (M5b cross-model)
all sort the 5 architectures into the same 3-cluster decomposition
(High-saturation Qwen / Mid-saturation Idefics2-InternVL3 /
Low-saturation LLaVA family). A single architectural property is
expressed redundantly across five distinct measurement modalities.

The causal-localization layer extends from Qwen-only to a cross-model
mechanistic story: M5a runtime steering flips PMR 10/10 in 3 of 4
testable models (Qwen L10 / LLaVA-Next L20+L25 / Idefics2 L25), and
M5b SAE intervention at the actually-consumed encoder layer breaks
PMR cleanly in 3 of 5 models (Qwen k=40, Idefics2 k=160, InternVL3
k=160). The two LLaVA models are M5b-NULL but LLaVA-Next is
M5a-positive — this dissociation specifies that **the CLIP-cluster
shortcut routes through LM-side residual-stream direction, not
through encoder-side localized features**, while the non-CLIP cluster
routes through both. The 2-CLIP-point insight (LLaVA-1.5 vs LLaVA-Next)
directly disconfirms an "encoder-determines-everything" story, and
the §4.6 pixel-encodability result shows the shortcut path is
encodable in the input image itself for the non-CLIP architectures.

### 9.2 What we didn't show

- No clean LM-only counterfactual (the LLaVA-1.5 → LLaVA-Next jump
  is 4-axis-confounded between AnyRes tiling, projector design,
  training corpus, and LM family). The M5a-positive + M5b-NULL
  dissociation in LLaVA-Next is consistent with an LM-side routing
  story but does not isolate it; controlled LM-swap is future work.
- No controlled projector-swap test isolating perceiver-resampler.
  The Idefics2 §4.6 result (0/9 layers shortcut despite v_L
  projection ascending +28 to +163) combined with the M4 LM probe
  AUC 0.995 and M5a 10/10 flip points at perceiver-resampler as the
  leading remaining mechanism candidate, but the 5-model design
  varies encoder + projector + AnyRes simultaneously. A controlled
  swap (same encoder + same LM, perceiver ↔ MLP only) would be
  needed for full isolation.
- v_L is a 1-d class-mean axis; multi-axis SAE decomposition or
  non-linear feature analysis would reveal additional steering
  directions that the PMR-based v_L misses. M5b cross-model
  intervention validates that *some* encoder-side feature
  decomposition is causally bound, but the structure of the SAE
  feature manifold is not characterized.
- Human baseline (Prolific): 20 raters × 50 stim is budgeted but
  not yet collected. Without it, we cannot directly compare the
  VLM saturation pattern to human judgment.

### 9.3 Limitations of the experimental setup

- **Single-task**: "next-state-prediction prompt" is the only
  behavioral readout. Other shortcut-style behaviors (counting,
  spatial reasoning, causality) are not tested.
- **Programmatic stim** makes encoder AUC = 1.0 trivial. M8c photos
  partially address this; richer photo distributions are open.
- **Adversarial stim is not naturalistic**: §4.6's synthesized noise
  pattern is visible. The result demonstrates the *existence* of a
  pixel-driven flip channel, not that this channel is engaged on
  natural images.
- **Per-model stim cell selection in M5b**: per-model OPEN+circle
  baseline-PMR=1 cell selection is necessary because each model
  has a different saturated/unsaturated PMR distribution; this
  precludes a single fully-cross-model stim comparison and limits
  the interpretation to per-model causal claims.
- **Scaling axis tested only on Qwen family**: §4.8 7B vs 32B
  shows scale doesn't fix grounding within the SigLIP+Qwen2
  architecture cluster, but cross-family scaling (e.g., Llama4
  Vision 8B vs 80B) is not tested.

### 9.4 Open questions

- Does the 2-CLIP-point gap shrink when we control for AnyRes
  tiling, projector, training, and LM family separately? Specifically:
  does an LM-only swap (CLIP-ViT-L + Vicuna → CLIP-ViT-L + Mistral)
  reproduce the LLaVA-1.5 → LLaVA-Next 0.52 PMR jump?
- Does perceiver-resampler block pixel-space gradient routability
  in a controlled projector-swap experiment, or is the Idefics2
  §4.6 0/9 result driven by some other architectural factor?
- What is the structure of the SAE feature manifold at the
  consumed encoder layer? Are physics-cue features
  one-dimensional, or do they form a multi-dimensional submanifold?
  Multi-axis SAE intervention is the natural follow-up to M5b.
- Do the 5 architecture-level signatures cohere on tasks beyond
  "next-state prediction"? A second task readout (e.g., counting
  or spatial relations) would test the generality of the cluster
  decomposition.

## 10. Conclusion

We localized the abstract→physical shortcut in 5 production-grade
open-source VLMs along three independent dimensions: cross-architectural
quantification (5 models × 3 stim sources × bootstrap CIs), cross-model
causal localization (LM-side runtime steering flips PMR 10/10 in 3 of 4
testable models; encoder-side SAE feature ablation breaks PMR in 3 of 5
models, with the LLaVA-Next M5a-positive + M5b-NULL dissociation
isolating LM-side direction as the CLIP-cluster routing path), and
pixel-space encodability (directional specificity, not perturbation
magnitude, drives flips for non-CLIP architectures; perceiver-resampler
is the leading candidate for blocking pixel-space gradient routability
in Idefics2). The architecture-level reframe is expressed redundantly
across five distinct downstream signatures (PMR ceiling, decision-
stability ceiling, pixel-encodability, LM logit-lens AUC, encoder-side
SAE intervention threshold), each sorting the 5 architectures into the
same 3-cluster decomposition. The 2-CLIP-point insight (LLaVA-1.5 vs
LLaVA-Next) directly disconfirms an encoder-determines-everything
reading and points to the joint **encoder + LM + projector** as the
architectural unit that determines shortcut behavior in VLMs.

---

## Appendix A — Full stimulus design

(See `docs/stimulus_spec.md` for the complete factorial.)

## Appendix B — PMR scoring rubric

(See `docs/scoring_rubric.md`. Multilingual stems: English / Korean /
Japanese / Chinese.)

## Appendix C — Reproducibility

All code, configs, stimulus generators, and inference pipelines:
github.com/namam3gy/physical-mode-activation. Per-milestone insight
deep dives: `docs/insights/m{1,3,4,5,6,8a,8c,8d,9}_*.md`,
`docs/insights/sec4_{2,3,5,6,7,10,11}_*.md`. Full roadmap:
`references/roadmap.md`.

## Appendix D — Hypothesis scorecard

Final state of H1-H7 + named H- hypotheses:

| ID | Status | Key evidence |
|---|---|---|
| H1 (S-curve abstraction ramp) | supported, unsaturated-only | M2 (Qwen, saturated): monotone 0.74→0.83 but compressed. M6 r1 (LLaVA): clean S-curve 0.51→0.81. M8a: Qwen 3/5 / LLaVA 4/5 strict. |
| H2 (label-prior independent contribution) | validated, encoder-anchored | M4b (Qwen) + M6 r1 (LLaVA) + M6 r2a (InternVL3); revised by encoder-saturation lens. |
| H4 (open-vs-FC gap signature) | supported, scorer-strengthened (Qwen-only at M2; cross-model FC untested) | M2 (Qwen, v1 scorer): 22-32 pp gap at every object_level. **2026-04-28 scorer audit**: paired open-vs-FC delta on no-label widens −0.131 → −0.180 under v2 scorer (no-motion patterns). Direction preserved, gap larger. ST5 cross-model FC is open (LLaVA "A" greedy bias prevented uniform protocol). |
| H5 (ground line shift > visual diff) | mixed (Qwen-only) | M2 (Qwen): bg +21 pp > object +9 pp. Scene also wins. Cross-model untested. |
| H6 (cast shadow drives saturation) | supported, revised (Qwen-only) | M2 (Qwen): cast_shadow alone +17.5 pp; arrow alone also saturates → "arrow = annotation" sub-claim refuted. Cross-model untested. |
| H7 (label selects regime) | validated, cross-category | M2 GAR (Qwen): ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2. M6 r1+r2a circle-only cross-model. M8d cross-category: LLaVA 3/3, Qwen 0/3 binary (regime distribution PASS). |
| H-boomerang (encoder knows / decoder gates) | Qwen-scoped | M3 (Qwen): encoder AUC ≈ 1.0 vs behavioral 0.28-0.95. Refuted on LLaVA-1.5 (encoder is the bottleneck). |
| H-encoder-saturation | **architecture-level confirmed (5 model × 3 stim × 5 downstream signatures)** | M9 5-model bootstrap CIs (PMR ceiling) + §4.7 (decision-stability ceiling) + §4.6 cross-model 9-layer Idefics2 (pixel-encodability) + M4 cross-model (LM logit-lens AUC ladder) + M5b cross-model round 2 (encoder-side SAE intervention break threshold). All 5 signatures sort the 5 architectures into the same 3-cluster decomposition. |
| H-LM-modulation | suggested only (LLaVA-Next M5a-positive + M5b-NULL dissociation) | M9 Idefics2 M8d H7 CI just touches 0; no clean LM-only counterfactual. **2026-04-28 cross-model M5b**: LLaVA-Next has positive M5a steering (LM-side L20+L25 10/10 flip) but NULL M5b SAE encoder-side intervention — consistent with LM-side direction routing the CLIP-cluster commitment, but does not isolate from 4-axis confound. |
| H-locus (mid-LM L_decision) | **supported, cross-model with model-specific layer** | M5a (Qwen): L10 α=40 flips 10/10. **2026-04-28 M5a cross-model**: 3 of 4 testable models confirm with model-specific decision layer — Qwen L10 (36% depth), LLaVA-Next L20+L25 (62-78% depth), Idefics2 L25 (78% depth). LLaVA-1.5 NULL at every α (encoder bottleneck). InternVL3 untestable (saturated baseline=1). M5b SIP+patching cross-model (LLaVA-1.5): lock-in starts at L20 (62.5% relative depth). |
| H-direction-bidirectional | supported (Qwen-only) | M5a-ext Exp 3 (Qwen): −α flips D→B at L10. Cross-model not tested for −α (would need per-model regime axis confirmation). |
| H-direction-specificity | **supported across 5 architectures × 5 layers (n=10 each)** | §4.6 5-model n=10 layer sweep + Idefics2 9-layer (L5-L31): 24 of 25 random-control cells = 0/10 (only Qwen L10 random has 1/10, far below v_unit 10/10). Aggregate random rate 1/250 across the 25 (model × layer) cells. Direction-specificity preserved at the projection level for all 5 models even where it does not translate to behavioral PMR flip (e.g., Idefics2 v_L projection rises +28 to +163 cleanly across 9 layers despite 0/9 PMR flips). |
| H-shortcut (pixel-encodable) | **supported, architecture-conditional** | §4.6 5-model n=10 layer sweep: each architecture has its own shortcut layer profile. Qwen broad (5 shortcut layers ≥ 80%), LLaVA-Next L20+L25 (10/10 each), LLaVA-1.5 L25 only (4/10 at n=10), **Idefics2 9-layer 0 shortcuts (L5-L31, 16-97% relative depth)**. Wrong-relative-depth hypothesis falsified by 9-layer Idefics2 evidence; **perceiver-resampler is the leading remaining mechanism candidate** (M4 LM AUC 0.995 + M5a 10/10 + §4.6 0/9 triangulation: forward pathway works, inverse pixel-space gradient routability blocked). InternVL3 protocol-saturated (baseline=1.0). |
| H-encoder-localized features (M5b cross-model) | **architecture-conditional — 3 of 5 models supported, 2 LLaVA NULL** | M5b cross-model round 2 (per-model SAE retrain at actually-consumed layer): Qwen k=40 (0.78% of features), Idefics2 k=160 (3.5%), InternVL3 k=160 (3.9%) all break PMR with mass-matched random controls at 1.0 specificity. LLaVA-1.5 + LLaVA-Next NULL at any k ≤ 160 (LLaVA-1.5 NULL extended to k=800 = 19.5% of features). Effect concentration scales inversely with M3 vision encoder probe AUC. The **CLIP-cluster M5a-positive + M5b-NULL dissociation** isolates LM-side direction as the routing path for shortcut commitment in CLIP-based VLMs. |

## Appendix E — Software stack

- transformers 4.45+
- pytorch 2.4+ on CUDA 13.0 (H200)
- python-pptx 1.0.2 (review PPT)
- All Python code formatted with black; markdown bilingual via project
  rule §6 (English canonical + `*_ko.md`).
