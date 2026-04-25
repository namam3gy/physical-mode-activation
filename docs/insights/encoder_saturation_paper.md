# H-encoder-saturation — paper-ready synthesis (placeholder, fills in with LLaVA-Next)

**Status**: In-progress as of 2026-04-25.

## One-line claim

Open-source VLM behavioral physics-mode reading (PMR) on minimal synthetic
stim is determined at the **architecture level** (encoder + LM fusion), not
at encoder representational capacity. All tested vision encoders linearly
separate physics-vs-abstract stim categories at AUC = 1.0; the per-architecture
behavioral PMR(_nolabel) ladder reflects how the LM consumes encoder output as
a physics-mode signal — non-CLIP architectures saturate, CLIP-LLaVA-Vicuna
does not, on synthetic minimal stim.

## Evidence chain (in narrative order)

### 1. Behavioral PMR ladder (4-model M8a, n=400 each)

| Model       | Encoder         | LM           | M8a PMR(_nolabel) |
|-------------|-----------------|--------------|------------------:|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | **0.838**         |
| LLaVA-1.5   | CLIP-ViT-L/14   | Vicuna-7B    | **0.175**         |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | **0.882**         |
| InternVL3   | InternViT       | InternLM2-7B | **0.918**         |
| LLaVA-Next  | CLIP-ViT-L/14   | Mistral-7B   | **TBD**           |

3 non-CLIP models saturate at PMR ~0.84–0.92. LLaVA-1.5 at 0.18.
LLaVA-Next is the matched-LM CLIP comparison (CLIP + Mistral, pairs with
Idefics2's SigLIP-SO400M + Mistral). Pending.

### 2. M9 cross-stim bootstrap CIs (synthetic vs photos)

| stim | model    | mean PMR(_nolabel) | 95% bootstrap CI |
|------|----------|-------------------:|-------------------|
| M8a  | Qwen     | 0.838              | [0.800, 0.872]   |
| M8a  | LLaVA    | 0.175              | [0.140, 0.212]   |
| M8a  | Idefics2 | 0.882              | [0.850, 0.912]   |
| M8d  | Qwen     | 0.869              | [0.840, 0.898]   |
| M8d  | LLaVA    | 0.331              | [0.294, 0.371]   |
| M8d  | Idefics2 | 0.890              | [0.862, 0.917]   |
| M8c  | Qwen     | 0.550              | [0.433, 0.667]   |
| M8c  | LLaVA    | 0.283              | [0.183, 0.383]   |
| M8c  | Idefics2 | 0.417              | [0.317, 0.517]   |

On synthetic stim (M8a + M8d), non-CLIP CIs (0.80–0.92) and CLIP-LLaVA CIs
(0.14–0.37) fully separate. On photos (M8c), all 3 collapse into [0.18, 0.67]
— **photos compress the encoder gap** (M8c finding).

### 3. Vision-encoder probe AUC — apples-to-apples M8a (4 models, M8a stim)

| Model       | Encoder         | LM           | M8a behavioral-y AUC | M8a stim-y AUC |
|-------------|-----------------|--------------|---------------------:|---------------:|
| Qwen2.5-VL  | SigLIP          | Qwen2-7B     | 0.880                | **1.000**      |
| LLaVA-1.5   | CLIP-ViT-L      | Vicuna-7B    | 0.771                | **1.000**      |
| Idefics2    | SigLIP-SO400M   | Mistral-7B   | 0.926                | **1.000**      |
| InternVL3   | InternViT       | InternLM2-7B | 0.886                | **1.000**      |
| LLaVA-Next  | CLIP-ViT-L      | Mistral-7B   | TBD                  | TBD            |

Behavioral-y AUC (each model's own PMR as target) ranges 0.77–0.93 — looks
like an encoder-family pattern. **But stim-defined y AUC is 1.0 for all 4
encoders**: every encoder linearly separates factorial cells perfectly across
4 stim-y targets (rendered_vs_line, physics_cell_vs_abstract_cell,
within_line_context, within_textured_context). Encoder discriminability is
**uniform across families**.

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
  4 model points × 2 stim sources × 2 y modes; reframe at architecture level.
  LLaVA-Next adds the matched-LM CLIP point.
- **H-LM-modulation** (M9-derived) — *suggested only*. Idefics2 vs Qwen
  M8d H7 CI just touches 0; not paper-defensible from current data.
- Pending hypothesis test: with LLaVA-Next data, does CLIP+Mistral behave
  like CLIP+Vicuna (LLaVA-1.5, low PMR) or like SigLIP-SO400M+Mistral
  (Idefics2, saturated)?

## Limitations

1. ~~n=1 CLIP point~~ → addressed by LLaVA-Next.
2. **Same-encoder LM swap** would still be the cleanest counterfactual.
   No tested model has both CLIP+Qwen2 and SigLIP+Vicuna paired, e.g.
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
- (TBD) `docs/insights/m6_r6_llava_next.md` (matched-LM CLIP point)
- `notebooks/encoder_saturation_chain.ipynb` (reproduction)
- `docs/figures/encoder_chain_4model.png` (paper headline figure)
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
- `outputs/m9_audit/m9_table1.csv` and `m9_summary.csv`
