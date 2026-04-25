# H-encoder-saturation — paper-ready synthesis (5-model)

**Status**: 5-model M8a chain complete (Qwen, LLaVA-1.5, LLaVA-Next, Idefics2,
InternVL3) as of 2026-04-25. Section 4 of the paper is locked at 5 model points.

## One-line claim

Open-source VLM behavioral physics-mode reading (PMR) on minimal synthetic
stim is determined at the **architecture level** (encoder + LM fusion), not
at encoder representational capacity. All tested vision encoders linearly
separate physics-vs-abstract stim categories at AUC = 1.0; the per-architecture
behavioral PMR(_nolabel) ladder reflects how the LM consumes encoder output as
a physics-mode signal — non-CLIP architectures saturate, CLIP-LLaVA-Vicuna
does not, on synthetic minimal stim.

## Evidence chain (in narrative order)

### 1. Behavioral PMR ladder (5-model M8a, n=400 each)

| Model       | Encoder         | LM           | M8a PMR(_nolabel) | 95% CI            |
|-------------|-----------------|--------------|------------------:|-------------------|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | **0.838**         | [0.800, 0.872]    |
| LLaVA-1.5   | CLIP-ViT-L/14   | Vicuna-7B    | **0.175**         | [0.140, 0.212]    |
| LLaVA-Next  | CLIP-ViT-L/14   | Mistral-7B   | **0.700**         | [0.653, 0.743]    |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | **0.882**         | [0.850, 0.912]    |
| InternVL3   | InternViT       | InternLM2-7B | **0.917**         | [0.890, 0.943]    |

3 non-CLIP models saturate at PMR ~0.84–0.92. LLaVA-1.5 at 0.18.
LLaVA-Next adds a 5th model point on M8a — same encoder family as
LLaVA-1.5 (CLIP-ViT-L) but with **multi-axis architectural difference**
(AnyRes multi-tile splitting, different fusion projector, different
training recipe, plus Mistral-7B LM). Its CI [0.65, 0.74] is wholly
below the saturated cluster (Qwen [0.80,0.87] / Idefics2 [0.85,0.91] /
InternVL3 ≈0.92) and wholly above LLaVA-1.5's [0.14, 0.21]. It is *not*
a clean LM-controlled encoder swap; for that we'd need the same
architecture with only the LM swapped. We report this as a 5th
observation, not a counterfactual: PMR has moved 0.18 → 0.70 across
4 simultaneously-changing axes (encoder fusion, image tiling, training
data + recipe, LM family).

### 2. M9 cross-stim bootstrap CIs (synthetic vs photos)

| stim | model       | mean PMR(_nolabel) | 95% bootstrap CI |
|------|-------------|-------------------:|-------------------|
| M8a  | Qwen        | 0.838              | [0.800, 0.872]   |
| M8a  | LLaVA-1.5   | 0.175              | [0.140, 0.212]   |
| M8a  | LLaVA-Next  | 0.700              | [0.653, 0.743]   |
| M8a  | Idefics2    | 0.882              | [0.850, 0.912]   |
| M8a  | InternVL3   | 0.917              | [0.890, 0.943]   |
| M8d  | Qwen        | 0.869              | [0.840, 0.898]   |
| M8d  | LLaVA-1.5   | 0.331              | [0.294, 0.371]   |
| M8d  | LLaVA-Next  | 0.625              | [0.583, 0.667]   |
| M8d  | Idefics2    | 0.890              | [0.862, 0.917]   |
| M8c  | Qwen        | 0.550              | [0.433, 0.667]   |
| M8c  | LLaVA-1.5   | 0.283              | [0.183, 0.383]   |
| M8c  | LLaVA-Next  | 0.417              | [0.300, 0.533]   |
| M8c  | Idefics2    | 0.417              | [0.317, 0.517]   |

On synthetic M8a, the 4 PMR clusters separate cleanly: LLaVA-1.5 floor
[0.14, 0.21] → LLaVA-Next mid-band [0.65, 0.74] → saturated non-CLIP cluster
[0.80, 0.92]. The same-encoder-family (CLIP-ViT-L) jump from LLaVA-1.5 to
LLaVA-Next is **0.52 PMR units across 4 simultaneously-confounded axes**;
this is *consistent with* but *not isolated to* LM modulation. On M8d the
same architecture-stratified ordering holds: LLaVA-Next 0.625 [0.58, 0.67]
sits between LLaVA-1.5 0.331 [0.29, 0.37] and the saturated cluster
[0.84, 0.92]. On photos (M8c, all 4 models), the encoder gap collapses
into [0.18, 0.67] — LLaVA-Next M8c PMR 0.417 is *statistically
indistinguishable* from Idefics2 M8c 0.417, **photos compress the
encoder gap for the 5th model too**. M8c finding generalizes.

### 3. Vision-encoder probe AUC — apples-to-apples M8a (5 models, M8a stim)

| Model       | Encoder         | LM           | M8a behavioral-y AUC | M8a stim-y AUC |
|-------------|-----------------|--------------|---------------------:|---------------:|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | 0.880                | **1.000**      |
| LLaVA-1.5   | CLIP-ViT-L      | Vicuna-7B    | 0.771                | **1.000**      |
| LLaVA-Next  | CLIP-ViT-L      | Mistral-7B   | 0.809                | **1.000**      |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | 0.926                | **1.000**      |
| InternVL3   | InternViT       | InternLM2-7B | 0.886                | **1.000**      |

Behavioral-y AUC (each model's own PMR as target) ranges 0.77–0.93 — looks
like an encoder-family pattern. **But stim-defined y AUC is 1.0 for all 5
encoders**: every encoder linearly separates factorial cells perfectly across
4 stim-y targets (rendered_vs_line, physics_cell_vs_abstract_cell,
within_line_context, within_textured_context). Encoder discriminability is
**uniform across families** — including the 2nd CLIP point (LLaVA-Next),
which rules out CLIP-as-encoder explanations for the LLaVA-1.5 PMR floor.

### 4. Cross-stim probe — M8c photos (n=60)

| Model       | M8c PMR(_nolabel) | M8c behavioral-y AUC | M8c stim-y AUC |
|-------------|------------------:|---------------------:|---------------:|
| Qwen2.5-VL  | 0.550             | 0.582                | **1.000**      |
| LLaVA-1.5   | 0.283             | 0.785                | **0.988**      |
| Idefics2    | 0.417             | 0.745                | **0.992**      |
| InternVL3   | 0.533             | 0.661                | **0.996**      |

**Behavioral-y AUC inverts cross-stim** (M8a → M8c): Qwen 0.88→0.58,
Idefics2 0.93→0.75, InternVL3 0.89→0.66; LLaVA 0.77→0.79 (stable). Encoder-
behavior alignment differs by stim.

**Stim-y AUC stays at 1.0** — encoder discriminability is also stim-invariant.

## Mechanism (revised)

The pre-stim-y-check version of H-encoder-saturation framed the mechanism as
"encoder family → encoder probe AUC → behavioral PMR → H7 measurability".
The stim-y check forces a refinement:

```
encoder family + LM family
       ↓
joint architecture (encoder + LM fusion)
       ↓
LM-side reading of encoder output as physics-mode signal
       ↓
behavioral PMR(_nolabel) saturated vs unsaturated
       ↓
H7 measurability gating
```

Encoder representational capacity is uniform; behavioral PMR is determined
by the joint encoder+LM system. The "encoder probe AUC with behavioral y"
is a *downstream-conditional* measure — it reflects how well encoder
representation aligns with the *behavioral* PMR distribution, not encoder
discriminability per se.

## Hypothesis status

- **H-encoder-saturation** — *architecture-level confirmed cross-stim*.
  5 model points (3 non-CLIP + 2 CLIP) × 2 stim sources × 2 y modes;
  reframed from "encoder family" to "joint encoder+LM architecture."
  LLaVA-Next adds a 2nd CLIP point with multi-axis architectural
  difference from LLaVA-1.5 (not a clean LM swap).
- **H-LM-modulation** (M9-derived) — *suggested only, still*. Idefics2
  M8d H7 CI [+0.000, +0.094] just touches 0; LLaVA-Next M8d H7 CI
  [−0.102, −0.006] symmetrically excludes 0 by ~0.005. **Both are in the
  noise floor** under the M9 bootstrap framework. Two-Mistral clustering
  at H7 ≈ 0 on M8d is suggestive but multi-axis-confounded (encoder
  family, image pipeline, projector, training all differ between the
  two). Not paper-defensible.
- **H7** (label-selects-regime): unsaturated-only on LLaVA-1.5 was the
  cleanest signal in the project (M8d +0.31). LLaVA-Next preserves H7 on
  M8a (+0.26, 5/5 PASS) but **removes the M8d H7 signal entirely**
  (-0.05, CI just below 0). PMR is well below ceiling on M8d for
  LLaVA-Next (0.625), so this is not a saturation effect — the
  architectural change broke H7 even with measurement headroom. H7
  strength is not preserved across same-encoder-family architecture
  changes.

## Limitations

1. ~~n=1 CLIP point~~ → addressed by LLaVA-Next (5th model). The 0.52
   PMR jump LLaVA-1.5 → LLaVA-Next is consistent with the architecture-
   level reframe but **confounded across 4 axes**: AnyRes multi-tile
   image splitting, fusion projector, training data + recipe, LM
   family (Vicuna → Mistral).
2. **Same-encoder LM swap** would still be the cleanest counterfactual.
   No tested pair holds encoder + image pipeline + projector + training
   constant while varying only the LM. With LLaVA-Next, the smallest
   change vs LLaVA-1.5 is "encoder + 4 architecture axes," not "encoder
   + LM only."
3. **n=12 photos per category on M8c** is underpowered for H7 detection.
4. **Synthetic stim factorial is M8a-style** — line/blank/none vs
   textured/ground/both. Real-world stim distributions are more varied.

## Roadmap

- §4.5 + M9 + M6 r3 + r4 + r5 + LLaVA-Next = paper Section 4 complete.
- M5b (SIP+SAE) for layer-level mechanism evidence — round 7.
- M7 paper draft.

## Artifacts (consolidated)

- `docs/insights/m8c_real_photos.md` (M8c photo behavioral)
- `docs/insights/m9_generalization_audit.md` (paper Table 1 with bootstrap)
- `docs/insights/encoder_swap_idefics2.md` (§4.5)
- `docs/insights/m6_r3_idefics2_probe.md` (§4.5 ext probe)
- `docs/insights/m6_r4_internvl3_probe.md` (4-model + stim-y check)
- `docs/insights/m6_r5_m8c_photo_probe.md` (cross-stim probe)
- `docs/insights/m6_r6_llava_next.md` (5th model, 2nd CLIP — LLaVA-Next, multi-axis confound)
- `notebooks/encoder_saturation_chain.ipynb` (reproduction)
- `docs/figures/encoder_chain_5model.png` (paper headline figure — supersedes 4model)
- `docs/figures/encoder_chain_4model.png` (frozen 4-model snapshot, kept for r3/r4/r5 docs)
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
- `outputs/m9_audit/m9_table1.csv` and `m9_summary.csv`
