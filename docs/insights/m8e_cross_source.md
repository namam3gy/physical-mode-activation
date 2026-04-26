# M8e — Cross-source paired analysis (synthetic vs photo)

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H1** — PMR rises in an S-shape along the abstraction axis (line → filled → shaded → textured); ground introduction adds the largest single jump.
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **H-encoder-saturation** — Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the architecture level (joint encoder + LM), not encoder representational capacity alone.
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim diversification — real photographs (60 photos × 5 categories from COCO + WikiArt). Photos REDUCE Qwen PMR(_nolabel) 18-48 pp.
> - **M8d** — Stim diversification — non-ball physical-object categories (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M8e** — Cross-source paired analysis (M8a + M8d + M8c consolidated). Model × category × source_type heatmap is the paper Table 1 candidate.
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.
> - **M6 r2** — ST5 round 2 — InternVL3 super-saturated, LLaVA captures expose CLIP-encoder bottleneck, FC logit ratio confirms LLaVA "A" bias is logit-level.

**Status**: Complete 2026-04-25.

## Motivation

M8a (circle textured), M8d (car / person / bird textured), and M8c
(real photos) ran the same prompt protocol on the same 4 physical
categories with two different stim types — synthetic-textured and real
photographs. M8e consolidates the three runs into a single
(model × category × source_type) view, producing the paper-ready
Table-1 / heatmap.

This is an analysis-only milestone (no new inference).

## Method

For each (model, category, source_type) cell, compute:
1. `PMR(_nolabel)` baseline — does the encoder + LM treat this stim as
   physical without any label?
2. H7 paired-difference `PMR(physical) − PMR(abstract)` on the labeled
   arm.
3. Photo − synthetic delta of both quantities.

Sources:
- **synthetic-textured ball** = M8a `circle / textured` slice.
- **synthetic-textured car / person / bird** = M8d `<cat> / textured`
  slice.
- **photo ball / car / person / bird** = M8c `<cat>` slice.
- **abstract** is not in the synthetic baseline (no synthetic abstract
  texture); reported as M8c-only.

The driver script is `scripts/m8e_cross_source.py --out-dir
outputs/m8e_summary`. CSVs written:
`m8e_synth_pmr_nolabel.csv`, `m8e_photo_pmr_nolabel.csv`,
`m8e_paired_delta.csv`, `m8e_h7_cross_source.csv`.

## Results

### PMR(_nolabel) cross-source paired delta

| category | Qwen synth | Qwen photo | Δ (photo−synth) | LLaVA synth | LLaVA photo | Δ |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|
| ball     | 0.900 | 0.667 | **−0.233** | 0.450 | 0.500 | +0.050 |
| car      | 0.975 | 0.500 | **−0.475** | 0.375 | 0.000 | **−0.375** |
| person   | 0.850 | 0.667 | −0.183 | 0.025 | 0.417 | **+0.392** |
| bird     | 0.875 | 0.417 | **−0.458** | 0.600 | 0.500 | −0.100 |

**Qwen photos universally lower PMR(_nolabel)** (range −18 to −48 pp).
**LLaVA shifts in both directions**: car/bird down, person up
(encoder finally recognizes humans), ball flat.

### H7 cross-source paired delta

| category | Qwen H7 synth | Qwen H7 photo | photo − synth | LLaVA H7 synth | LLaVA H7 photo | photo − synth |
|----------|-----:|-----:|-----:|-----:|-----:|-----:|
| ball     | 0.000 | +0.083 | +0.083 | +0.200 | +0.167 | −0.033 |
| car      | +0.025 | −0.167 | **−0.193** | **+0.650** | 0.000 | **−0.650** |
| person   | −0.050 | **+0.500** | **+0.550** | −0.075 | −0.250 | −0.175 |
| bird     | +0.075 | 0.000 | −0.075 | +0.525 | **+0.667** | +0.142 |

**LLaVA car H7 collapses on photos** (+0.65 → 0.00 = -0.65). Both car
photos with `car` and `silhouette` labels produce nearly identical low
PMR (0.083 vs 0.083).

**Qwen person H7 amplifies on photos** (-0.05 → +0.50 = +0.55).
`stick figure` label on synthetic stick figure produces high physics
(model interprets stick figure as a human walking); `stick figure`
label on a real person photo produces "this is a stick figure drawing,
no motion" (abstract-leaning).

**LLaVA bird H7 strengthens on photos** (+0.525 → +0.667). The
`silhouette` label on a real bird photo is more strongly suppressive
than on a synthetic black-bird silhouette.

### Cross-source pattern by model

**Qwen** — saturated on synthetic, less saturated on photos:
- Synthetic baselines 0.85-0.97 (ceiling).
- Photo baselines 0.42-0.67 (substantial drops).
- H7 measurability *increases* on photos for Qwen — person and car
  show photo-only H7 signal while synthetic was at ceiling.

**LLaVA** — encoder-recognition asymmetric across categories:
- Synthetic person 0.03 → photo person 0.42 (encoder finally sees a
  human; synthetic stick figure unrecognized).
- Synthetic car 0.38 → photo car 0.00 (street photos described as
  scenes, not single cars).
- Synthetic bird 0.60 → photo bird 0.50 (slight compression).
- Synthetic ball 0.45 → photo ball 0.50 (flat).

The asymmetry tells us LLaVA's CLIP encoder has uneven priors across
categories: humans well-represented from photo data, single-object
context-free cars under-represented.

## Headline interpretation

**M8e consolidates a counter-intuitive finding from M8c.** The naive
prediction was: photo-realism would saturate the encoder more, pushing
behavioral PMR closer to ceiling. In reality:

1. **Photos do NOT saturate Qwen's encoder more** — they LOWER PMR
   universally because real photos have scene context that elicits
   descriptive responses rather than motion predictions.
2. **LLaVA's photo PMR shifts depend on encoder recognition**: a
   well-represented category (people in photos) gains; a context-rich
   category (cars in street scenes) loses.
3. **H7 measurability is partially recovered on photos for Qwen**
   because the binary PMR ceiling is broken; categorical regime
   selection becomes visible at the binary level (Qwen person H7
   photo +0.50).

This refines H-encoder-saturation:
- M6 r2 finding: `vision encoder probe AUC` predicts `synthetic
  PMR(_nolabel)` well (Qwen 0.99 ↔ 0.95 / LLaVA 0.73 ↔ 0.38).
- M8e finding: `synthetic PMR(_nolabel)` is also driven by stim
  simplicity. Photo PMR(_nolabel) is the *encoder-recognition-and-
  context-handling* readout; synthetic PMR(_nolabel) is the *encoder-
  recognition-with-minimal-context-distraction* readout.

Both quantities are valid; they measure different things.

## Headline figure

![m8e_cross_source_heatmap](../figures/m8e_cross_source_heatmap.png)

`docs/figures/m8e_cross_source_heatmap.png` — three panels:
1. Synthetic-textured PMR(_nolabel) per (model × category).
2. Photo PMR(_nolabel) per (model × category).
3. H7 photo − H7 synthetic delta per (model × category).

This figure is paper-ready as Table 1 / Figure 1 for the cross-source
section of the paper.

## Hypothesis updates

- **H-encoder-saturation** — *unchanged from M8c-refined version*.
  M8e provides the consolidated table-form view but no new conceptual
  update.
- **H7** — *cross-source nuance added*. H7 is **(a) shape-axis-only**
  (M8d), **(b) unsaturated-only** (M8a), AND **(c) source-type
  dependent**. The strongest H7 result remains LLaVA M8d 3/3 (synthetic
  car/person/bird textured); photos add scene-context noise and
  category-specific encoder-recognition asymmetries.
- **H1** — *no new test in M8e* (no abstraction-axis variation).

## Roadmap implications

1. **Paper Table 1 candidate**: `m8e_cross_source_heatmap.png` as the
   "external validity" figure showing the cross-(model × category ×
   source_type) view.
2. **§4.5 (encoder swap)** is now even more sharply motivated. M8c
   showed that *behavioral* PMR(_nolabel) is partly stim-simplicity;
   the *encoder-AUC* piece of H-encoder-saturation can only be
   isolated by swapping encoders.
3. **Round-2 photo curation**: bbox-cropped subsets to remove scene
   context and put a single car/person/bird in the frame. This would
   close the photo PMR gap to the synthetic PMR (testing whether the
   gap is purely stim-simplicity).
4. **Prompt redesign**: the current prompt asks "what will happen
   next?" — a scene-descriptive response is also valid. A more
   physics-focused prompt ("what physical force is acting on the
   foreground object?") might recover the synthetic PMR levels on
   photos. M9 could test this.

## Artifacts

- `outputs/m8e_summary/` — per-CSV: synth_pmr_nolabel,
  photo_pmr_nolabel, paired_delta, h7_cross_source.
![3-panel heatmap](../figures/m8e_cross_source_heatmap.png)
- `notebooks/m8e_cross_source.ipynb` — cell-by-cell reproduction.
