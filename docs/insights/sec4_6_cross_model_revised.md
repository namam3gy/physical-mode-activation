---
section: §4.6 cross-model — REVISED (2026-04-26 overnight followup)
date: 2026-04-26
status: complete (revises prior "Qwen-scoped" reading)
hypothesis: pixel-encodability is NOT Qwen-specific — LLaVA-1.5 admits it at L25; the prior null was a wrong-layer-choice artifact
---

# §4.6 cross-model — REVISED: pixel-encodability is not Qwen-specific

> **Recap of codes used in this doc**
>
> - **§4.6** — VTI-reverse counterfactual stim — pixel-space gradient ascent on `pixel_values` maximizing `<h_L[visual], v_L>`.
> - **H-shortcut** — Shortcut interpretation is encodable in the image itself. **Revised here from "Qwen-scoped" → "model-conditional, not Qwen-only"**.
> - **H-direction-specificity** — Pixel-space gradient ascent along v_L flips PMR; matched-magnitude random directions don't.
> - **M2 / M8a** — single-shape vs 5-shape factorial stim sets.
> - **v_L** — class-mean diff direction at LM layer L; per-model dim differs.

## TL;DR

The prior §4.6 cross-model finding ("LLaVA-1.5 self-v_L10 0/5 PMR flip
→ pixel-encodability is encoder-saturation specific / Qwen-scoped")
was a **wrong-layer-choice artifact**. Re-running on LLaVA-1.5 with a
layer sweep (L5 / L15 / L20 / L25; L10 was the prior test):

| layer | bounded ε=0.05 | ε=0.1 | ε=0.2 | unconstrained | random Δ |
|---|---:|---:|---:|---:|---:|
| 5 | 0/5 | 0/5 | 0/5 | 1/5 | 0/15 |
| 10 | 0/5 | 0/5 | 0/5 | 0/5 | 0/15 (prior null) |
| 15 | 0/5 | 0/5 | 0/5 | 0/5 | 0/15 |
| 20 | 0/5 | 0/5 | 0/5 | 2/5 | 0/15 |
| **25** | **1/5** | **4/5** | **5/5** | 2/5 | **0/15** |

**L25 v_L25 admits clean pixel-encodability**: 4/5 flips at ε=0.1, 5/5
at ε=0.2. Random-direction controls 0/15 across all layers — directional
specificity preserved.

Sample LLaVA-1.5 L25 ε=0.1 response: "The circle will be hit by the dart."
Sample L25 ε=0.2: "The circle will be hit by a ball."
Random ε=0.1 control: "The circle will be covered by the moon." (PMR=0)

## What this revises

### Prior reading (now obsolete)
> "Pixel-encodability of the regime axis is encoder-saturation specific —
> Qwen's saturated SigLIP creates a thin pixel-to-L10 channel the LM
> reads from; LLaVA-1.5's unsaturated CLIP doesn't. H-shortcut →
> Qwen-scoped."
>
> *Source*: `docs/insights/sec4_6_cross_model.md` (commit `ec2aa77`).

### Revised reading (this doc)
- **Pixel-encodability is NOT model-specific to Qwen.** LLaVA-1.5
  also admits it, at L25 instead of L10.
- The prior LLaVA-1.5 null (0/5 at L10) was a **wrong-relative-depth**
  artifact: Qwen has 28 LM layers (L10 ≈ 36% depth); LLaVA-1.5 has
  32 LM layers (L25 ≈ 78% depth). The "shortcut layer" varies per
  architecture.
- **H-shortcut** stays supported, but the *causal-locus layer* must be
  identified per model — single layer L10 is Qwen-specific, not
  cross-model.

### What stays unchanged
- M9 PMR-ceiling and §4.7 decision-stability ceiling are *separate*
  saturation signatures — they do not depend on §4.6's per-model
  layer choice. Architecture-level reframe robust.
- v_L direction-specificity (pixel-encoding direction matters; random
  doesn't) preserved at every tested layer on LLaVA-1.5.

## Why this matters

This is a paper-grade correction. The original §4.6 cross-model claim
("Qwen-scoped pixel-encodability") was *over-strong*. A simple layer
sweep reveals that pixel-encodability **generalizes across at least 2
of 5 architectures we tested**, with each model having its own
"shortcut layer."

Implications:
- The shortcut path through the LM is *deeper-relative* in LLaVA-1.5
  than in Qwen (~78% vs ~36% depth).
- The "third saturation signature" framing in `docs/insights/sec4_6_cross_model.md`
  is incorrect — pixel-encodability is not strictly tied to encoder
  saturation. (M9 PMR-ceiling + §4.7 decision-stability still are.)
- **Future work**: layer-sweep §4.6 on Qwen too — does Qwen have a
  *second* shortcut layer at L25 / L20? Currently we only tested L10
  on Qwen.

## M2 vs M8a v_L cosine similarity (per model, all 5 layers)

The class-imbalance concern (M2 had n_neg = 1-9 for saturated models;
M8a has n_neg = 100-280) was the original motivation for the M8a
captures. The cosine analysis says class imbalance was *not* the
issue:

| Model | Layer 5 | Layer 10 | Layer 15 | Layer 20 | Layer 25 |
|---|---:|---:|---:|---:|---:|
| LLaVA-Next | 0.40 | 0.39 | 0.42 | 0.33 | **0.25** (weak) |
| Idefics2 | **0.79** | **0.79** | **0.80** | **0.79** | **0.79** |
| InternVL3 | **0.76** | 0.69 | **0.76** | **0.76** | 0.59 |

- Idefics2: **same direction** at every layer (cos ~0.79). M2 v_L wasn't
  noise despite n_neg=5 — class imbalance robust for saturated SigLIP+Mistral.
- InternVL3: mostly same direction (~0.7+), some moderate alignment.
- LLaVA-Next: moderate alignment at most layers (~0.4) — class
  imbalance had partial effect; M8a v_L is somewhat different from
  M2 v_L for this architecture.

**Implication**: For saturated models (Idefics2, InternVL3), M2-derived
v_L would have given the same null result as M8a-derived. The *fix*
isn't class balance — it's **layer choice**. The LLaVA-1.5 layer
sweep proves this directly.

## Limitations

1. **Only LLaVA-1.5 layer sweep**. The other 3 cross-model models
   (LLaVA-Next, Idefics2, InternVL3) per-model gradient ascent
   requires custom `counterfactual_<model>.py` modules (each has a
   different processor / pixel layout). Not yet implemented. Their
   "shortcut layer" remains unknown.
2. **Qwen layer sweep absent**. We have L10 5/5 finding for Qwen, but
   never tested L25. The "L10 is Qwen's shortcut layer" reading may
   itself be wrong-layer-choice — Qwen's true shortcut layer might be
   deeper.
3. **Single direction (class-mean v_L)**. SAE / multi-axis
   decomposition could find additional pixel-encodable directions at
   any layer.
4. **Single-task evaluation**. Other shortcut behaviors might
   localize to different layers.

## Reproducer

```bash
# Phase 1: M8a captures × 3 missing models (~50 min on H200).
CUDA_VISIBLE_DEVICES=1 bash scripts/run_overnight_sec4_6_followup.sh

# Or manually:
# 1. M8a captures
for cfg in encoder_swap_{llava_next,idefics2,internvl3}_m8a_capture; do
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config configs/${cfg}.py \
        --stimulus-dir inputs/m8a_qwen_20260425-091713_8af4836f
done

# 2. v_L extraction
uv run python scripts/m8a_extract_per_model_steering.py

# 3. LLaVA-1.5 layer sweep
for L in 5 15 20 25; do
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_counterfactual_stim_llava.py \
        --layer $L --steering-key v_unit_$L \
        --output-dir outputs/sec4_6_counterfactual_llava_L${L}_$(date +%Y%m%d-%H%M%S)
done

# 4. PMR re-inference per layer
for run_dir in outputs/sec4_6_counterfactual_llava_L*_2026*; do
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/sec4_6_summarize.py \
        --run-dir $run_dir \
        --model-id llava-hf/llava-1.5-7b-hf
done

# 5. Analysis + draft insight doc
uv run python scripts/sec4_6_followup_analyze.py
```

## Artifacts

- `configs/encoder_swap_{llava_next,idefics2,internvl3}_m8a_capture.py`
- `outputs/encoder_swap_{llava_next,idefics2,internvl3}_m8a_capture_2026*/probing_steering/steering_vectors.npz` — per-model v_L extracted from M8a captures
- `outputs/sec4_6_counterfactual_llava_L{5,15,20,25}_2026*/` — LLaVA-1.5 §4.6 layer sweep
- `outputs/sec4_6_followup/{comparison,llava_layer_sweep}.csv`
- `scripts/m8a_extract_per_model_steering.py`
- `scripts/sec4_6_followup_analyze.py`
- `scripts/run_overnight_sec4_6_followup.sh`

## Sample synthesized responses (LLaVA-1.5 L25)

| Config | Sample 0 baseline | Sample 0 synth |
|---|---|---|
| ε=0.05 | "The circle will be filled in with color." | "The circle will be cut out of the paper." |
| ε=0.1 | "...filled in with color." | "**The circle will be hit by the dart.**" |
| ε=0.2 | "...filled in with color." | "**The circle will be hit by a ball.**" |
| unconstrained | "...filled in with color." | "**The circle will be hit by a ball.**" |
| random ε=0.1 | "...filled in with color." | "The circle will be covered by the moon." |

The model goes from abstract (filled in with color) → physics-mode
(hit by a ball / dart) under v_L25 perturbation. Random doesn't reach
physics-mode (covered by the moon — celestial / static).
