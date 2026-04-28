---
section: M5a VTI causal steering — cross-model
date: 2026-04-28
status: complete (4 testable models — Qwen + LLaVA-1.5 + LLaVA-Next + Idefics2; InternVL3 untestable)
hypothesis: forward-hook injection of α · v_L causally flips PMR cross-model. v_L direction extracted from M2 cross-model captures (m2_extract_per_model_steering.py).
---

# M5a VTI causal steering — cross-model

> Builds on M5a Qwen-only causal steering (2026-04-24, L10 α=40 → 10/10 flip)
> by reusing per-model v_L extracted in M6 r7 cross-model captures
> (`m2_extract_per_model_steering.py`). Forward-hook is the same as
> Qwen original — `output[0] += α · v_L_unit` at the chosen LM
> decoder layer.

## TL;DR

Runtime steering with model-specific v_L direction **flips PMR
0 → 1 in 3 of 4 testable models** at appropriately scaled α. This
extends M5a's "causal localization" claim from Qwen-only to a
cross-model architecture-conditional finding.

| Model | Layer | α sweet spot | PMR flip rate | Stim cell (baseline) |
|---|---|---|---|---|
| Qwen2.5-VL-7B | L10 | 40 | **10/10** | line/blank/none × circle (0) |
| **LLaVA-Next-Mistral-7B** | **L20** | **10** | **10/10** | **line_blank_both × circle (0)** |
| **LLaVA-Next-Mistral-7B** | **L25** | **15-20** | **10/10** | **line_blank_both × circle (0)** |
| **Idefics2-8B** | **L25** | **20** | **10/10** | **line_blank_none × circle (0)** |
| **Idefics2-8B** | **L20** | **20** | **10/10** | **line_blank_none × circle (0)** |
| LLaVA-1.5-7B | L25 | (any 0-60) | **0/10** | line_blank_none × circle (0) |
| InternVL3-8B | — | — | (untestable, baseline=1) | n/a |

**Coherent physics responses** (paper-quality, not token-degeneracy):

- LLaVA-Next L25 α=15: "The ball will bounce up."
- LLaVA-Next L20 α=10: "The ball will roll down the ramp."
- **Idefics2 L25 α=20: "The tip of the arrow will hit the center of the circle."**
- Qwen L10 α=40 (M5a original): "The ball is falling down due to gravity."

LLaVA-1.5's null at every α (responses change semantically — "filled
with color" → "in the center" → "on the floor" — but PMR scorer never
catches a motion stem) replicates the §4.6 weak-shortcut finding.

## Headline finding — Idefics2 paper-changing result

The Idefics2 result **resolves the §4.6 perceiver-resampler hypothesis
ambiguity**. Triangulating three measurements:

| Test on Idefics2 | Result | Implication |
|---|---|---|
| M4 LM probe AUC (visual-token mean hidden state) | 0.995 across L5-L25 | Information *reaches* LM at high quality |
| **M5a runtime steering** (forward-hook + α·v_L) | **L25 α=20 → 10/10 flip** | Forward-direction `v_L` is *operative* in LM |
| §4.6 pixel-space gradient ascent (L5-L31) | 0/90 v_unit + 0/90 random | Inverse-direction (pixel→LM) **does not have selective routability for v_L** |

**Refined H-shortcut + perceiver-resampler hypothesis**:

> The perceiver-resampler does not strip the physics-mode signal —
> the LM has it (M4 AUC 0.995) and forward-hook steering exploits it
> (M5a 10/10). What the perceiver-resampler removes is the
> *pixel-space gradient route* that selectively hits this direction
> from the input side. The bottleneck is on the *inverse* (pixels →
> v_L direction) pathway, not the *forward* (v_L direction → LM
> commitment) pathway.

This refinement is paper-grade and reframes the §4.6 cross-model
result more carefully. The previous claim was "perceiver-resampler
strips v_L-aligned information"; the refined claim is "**perceiver-
resampler removes pixel-space gradient routability**", which is
consistent with all three signatures.

## Per-model detail

### LLaVA-Next-Mistral-7B (B from data audit) — POSITIVE

**Two stim cells tried**:

(1) `line_blank_none_fall × circle` (LLaVA-Next baseline PMR 0.5,
    not abstract enough): runtime steering at α=10-20 gave PMR=1.0
    on `L20+L25` but baseline is also 1.0 — not a clean flip
    measurement. Useful only for sanity-check.

(2) `line_blank_both_fall × circle` (LLaVA-Next baseline PMR 0.3,
    abstract-baseline candidate): **clean 0→10/10 flip** at L20
    α=5-15 and L25 α=5-20. α=20+ at L20 hits "thrown up in the air"
    (physics but PMR scorer false-negative on "thrown up"). α=40+ at
    both layers degenerates to "rock rock rock" or "ball ball ball"
    (Mistral's saturation pattern under large residual perturbation).

LLaVA-Next has **wider α dynamic range than LLaVA-1.5 but narrower
than Qwen**: 5-15 sweet spot vs Qwen's 10-60.

### Idefics2-8B (C from data audit) — UNEXPECTED POSITIVE

`line_blank_none_fall × circle` (Idefics2 baseline PMR 0.0 — perfect
abstract baseline since `planet`/`ball` are saturated but `circle`
sits at the abstract floor for this saturated SigLIP-SO400M architecture).

**L25 α=5-10 produces no flip** (still abstract: "circle will grow
larger"); **α=20 produces a coherent physics commit**: "The tip of
the arrow will hit the center of the circle." (10/10 stim, all
identical or near-identical).

**L20 α=10 partial flip (7/10)**: half "circle will be filled with
grey" (abstract) and half "circle will be hit by an arrow" (physics).
α=20 becomes 10/10 with some token degeneracy ("The tip of tip of
tip of tip..."), and α=40+ becomes pure repetition ("tip tip tip
tip..." for all 10 stim).

The α=20 L25 case is the clean evidence — coherent physics-mode
sentence, deterministic across stim, baseline 0/10 → flipped 10/10.

### LLaVA-1.5-7B — NULL (replicates §4.6)

`line_blank_none_fall × circle`, L25 (the §4.6 shortcut layer for
LLaVA-1.5), α=0/10/20/40/60: **0/10 flip at every α**.

Responses do change semantically with α:
- α=0: "The circle will be filled with a color."
- α=10: "The circle will be filled with color."
- α=20: "The circle will be in the center of the image."
- α=40: "The circle will be in the center of the image."
- α=60: "The circle will be on the floor."

The α=60 response "on the floor" is *closer* to physics-mode framing
(location implies gravity), but the PMR scorer requires a motion stem
("falls", "rolls", "bounces", "moves") which none of these contain.

This **replicates the §4.6 weak-shortcut finding for LLaVA-1.5 L25
(4/10 pixel-space flip)**: the runtime steering is a *stronger* test
that LLaVA-1.5's CLIP-encoder pipeline doesn't have a physics-mode
direction localized at the LM single-position residual, even though
the encoder→LM channel can be partially shortcut'd from pixel-space.

### InternVL3-8B — UNTESTABLE

InternVL3's M2 baseline at `line/blank/none × circle` is PMR=1.0 (saturated
to physics). No abstract baseline → cannot measure PMR flip. Same
protocol-saturation as §4.6 InternVL3.

## Cross-cutting interpretation

The cross-model M5a result combined with M4 LM probe + §4.6 pixel
sweep produces **three triangulating signatures of the encoder→LM
pipeline**:

| Model | M4 probe AUC | M5a flip rate | §4.6 flip rate | Shortcut profile |
|---|---|---|---|---|
| Qwen2.5-VL | 0.96 | 1.0 (L10) | 0.8-1.0 (5 layers) | Broad pixel-encodable + clean LM direction |
| Idefics2 | **0.995** | **1.0 (L25)** | **0** | LM has signal + LM direction works + pixel route blocked |
| LLaVA-Next | 0.79 | 1.0 (L20+L25) | 1.0 (L20+L25) | LM weaker but L20+L25 fully shortcut + steerable |
| LLaVA-1.5 | 0.76 | 0 (L25) | 0.4 (L25) | LM weak + steering null + partial pixel-flip |
| InternVL3 | n/a | n/a | n/a | Protocol-saturated |

The Idefics2 row is the cleanest disambiguation: **all three
signatures separable**. M4 says info is in the LM. M5a says LM
direction is operative. §4.6 says pixel→LM doesn't have a selectivity
gradient. The architectural feature that flips between Idefics2
(perceiver) and LLaVA-Next (MLP+AnyRes) is the projector design,
and the §4.6 vs M5a *dissociation* localizes the bottleneck to the
*inverse* (pixel-side) pathway.

## Regime-attractor at sweet-spot α

M5a steering at sweet-spot α flips the LM into a **deterministic
physics-mode attractor** — the model commits to physics-mode regardless
of visual content. Across the 10 seeds of `line_blank_none × circle`
(Idefics2) / `line_blank_both × circle` (LLaVA-Next), responses at any
single (layer, α) are essentially **identical**:

- Idefics2 L25 α=20: 10/10 stim → "The tip of the arrow will hit the center of the circle." (unique=1)
- LLaVA-Next L25 α=15: 10/10 stim → "The ball will bounce up." (unique=1)
- Qwen L10 α=40 (M5a original): 10/10 stim → "The ball is falling down due to gravity." (unique=1)

This is consistent with M5a-ext's regime-axis finding: at sufficient
α, the regime is forcibly selected and visual content tracking lives
at moderate α. The cross-model 10/10 flip is therefore causal evidence
of **regime control** (the LM enters a physics-mode attractor regardless
of stimulus) rather than **stim-conditioned physics interpretation**
(the LM uses the stimulus to determine which physics commitment to
make). The §4.6 pixel-space gradient ascent and M5b SAE knockout
provide the stim-conditioned signatures; M5a steering is the
complementary regime-axis evidence.

## Limitations

1. **α dynamic range model-specific** — picking the sweet spot
   requires sweep. Qwen 40, LLaVA-Next 5-15, Idefics2 20.
   At α > sweet spot, all 3 models eventually exhibit token
   degeneracy ("rock rock rock", "tip tip tip"). PMR scorer can be
   tricked by such repetition matching motion stems → α=40+ rates
   are not pure flip measurements.

2. **Baseline cell varies** — LLaVA-Next has no `line_blank_none ×
   circle` abstract-baseline (PMR 0.5), but `line_blank_both × circle`
   does (0.3). Per-model baseline cell selection is noted in the
   per-model section. Not a confound for the headline (each model's
   own baseline → flip rate is a per-model causal claim, not a
   cross-model PMR comparison).

3. **InternVL3 untestable** under M2 abstract-baseline. Different
   prompt or stim category needed (same as §4.6 InternVL3).

4. **Single layer per model** for the headline (Qwen L10 / LLaVA-Next
   L20+L25 / Idefics2 L25). A full per-model layer sweep would
   strengthen the locus claim but is incremental at this point —
   §4.6 already provides the layer-level architecture-conditional
   profile.

5. **v_L is class-mean diff direction** — same as M5a Qwen original.
   SAE / multi-axis decomposition could reveal additional steering
   axes that PMR-based v_L misses.

## Reproducer

```bash
# LLaVA-1.5 (null baseline)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_llava_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none --label circle \
    --layers 25 --alphas 0,10,20,40,60 \
    --model-id llava-hf/llava-1.5-7b-hf \
    --output-subdir m5a_cross_llava15_l25 --prompt-variant open

# LLaVA-Next L20+L25
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_llava_next_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/both --label circle \
    --layers 20,25 --alphas 0,5,10,15,20 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --output-subdir m5a_cross_llava_next_lbb --prompt-variant open

# Idefics2 L20+L25
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_idefics2_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none --label circle \
    --layers 20,25 --alphas 0,5,10,20,40,60 \
    --model-id HuggingFaceM4/idefics2-8b \
    --output-subdir m5a_cross_idefics2_lbn --prompt-variant open
```

## Artifacts

- `scripts/06_vti_steering.py` — patched 2026-04-28 to support
  `model.text_model.layers` (Idefics2/Mistral path) in addition to
  `model.language_model.layers` (Qwen/LLaVA path).
- `outputs/cross_model_llava_capture_*/steering_experiments/m5a_cross_llava15_l25/` — null
- `outputs/cross_model_llava_next_capture_*/steering_experiments/m5a_cross_llava_next_lbb/` — positive (L20+L25)
- `outputs/cross_model_idefics2_capture_*/steering_experiments/m5a_cross_idefics2_lbn/` — positive (L20+L25)
- Same dirs for the earlier LLaVA-Next `line_blank_none` sanity-check
  (`m5a_cross_llava_next_l20_l25`).
