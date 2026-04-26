---
section: M5b — cross-model SIP + activation patching (Qwen vs LLaVA-1.5)
date: 2026-04-26
status: complete (Qwen-only baseline + LLaVA-1.5 cross-model — Idefics2 / InternVL3 skipped due to class imbalance)
hypothesis: per-model decision-lock-in layer differs in absolute *and* relative depth — Qwen L10 (36%) vs LLaVA-1.5 L20 (62%)
---

# M5b — cross-model SIP + activation patching: Qwen vs LLaVA-1.5

> **Recap of codes**
>
> - **M5b** — ST4 Phase 3: SIP + activation patching to identify causal LM layers.
> - **SIP** — Semantic Image Pairs (Golovanevsky et al. NAACL 2025): paired stim differing only on a single visual cue. Clean = elicits physics; corrupted = elicits abstract.
> - **IE** (Indirect Effect) — Δ P(physics-mode) from patching: P(physics | patched corrupted) − P(physics | corrupted baseline).
> - **§4.6 revised** — LLaVA-1.5 L25 admits pixel-space gradient ascent at ε=0.2 (5/5 flips). Qwen's L10 is its analog. The two models have different shortcut layers.
> - **H9** — saturated encoders plateau before L5; unsaturated never reach saturated plateau in captured range.

## Question

The Qwen-only M5b finding ("L0-L9 patching → 100% physics recovery,
L10-L11 → 60%, L14+ → 0%") identifies L10 as Qwen's decision-lock-in
boundary. Does this generalize? Specifically:

- Does the *shape* of the IE × layer curve replicate cross-model?
- Does the *position* of the decision-lock-in layer correspond to a
  fixed *relative* depth, or does it differ per architecture?

We answer with LLaVA-1.5 SIP+patching using its M2 captures
(n_pos=375, n_neg=105 — adequate class balance). LLaVA-Next /
Idefics2 / InternVL3 had only 2 / 0 / 0 clean SIP candidates each
(saturated open-prompt PMR), so they're skipped this round.

## Method

For each model:
1. SIP from open-prompt M2 captures, label="ball": pair cue=both
   seeds (clean: PMR=1) with cue=none seeds (corrupted: PMR=0) by
   index. Filter strict (clean PMR=1 AND corrupted PMR=0).
2. For each pair: cache clean's per-layer hidden state at visual-
   token positions; baseline corrupted; per target_L ∈ [0..N_layers],
   patched corrupted (forward hook on `layers[target_L]` at prefill,
   replace visual-token h with cached clean values).
3. IE per layer = Δ P(physics-mode) = (patched PMR rate) − (baseline
   PMR rate).

Generic LM-layer resolver via `model.model.language_model.layers`
(works for Qwen, LLaVA, InternVL); fallback to `model.model.text_model.layers`
(Idefics2). LLaVA-Next AnyRes might give visual-token count mismatch
between clean/corrupted (both pairs were dropped).

## Result

### Qwen2.5-VL (n=20 pairs, 28 LM layers)

![Qwen IE per layer](../figures/m5b_sip_per_layer_ie.png)

- L0-L9: IE = +1.0 (100% recovery on all 20 pairs)
- L10-L11: IE = +0.6 (60%)
- L12-L13: IE = 0.3 / 0.1
- L14+: IE = 0.0

**Decision-lock-in starts at L10** (36% relative depth).

### LLaVA-1.5 (n=15 pairs, 32 LM layers)

![LLaVA-1.5 IE per layer](../figures/m5b_sip_cross_model_llava15_per_layer_ie.png)

- L0-L19: IE = +0.40 (15/15 → 15/15 PMR=1; baseline 9/15 PMR=1 +
  6/15 flipped from PMR=0)
- L20-L23: IE = +0.27 to +0.33 (gradual decline)
- L24-L28: IE = +0.13
- L29-L30: IE = −0.07 (slight negative — patching late-layer hurts)
- L31: IE = 0.0

**Decision-lock-in starts at L20** (62.5% relative depth).

The corrupted SIP partly drifted toward PMR=1 at re-inference time
(M2 capture-time PMR=0 → re-inference PMR=1 for 9/15 pairs), so the
"true SIP signal" is on the 6/15 genuinely-corrupted re-inference
pairs. For these, patching at L0-L19 recovers physics-mode in 6/6;
patching at L20+ progressively fails.

## Headlines

1. **Curve shape replicates cross-model**: both models show a sharp-
   then-declining IE × layer profile — full IE in early layers,
   declining transition zone, zero IE at the end of the LM. The
   *shape* is the same.

2. **Decision-lock-in layer differs in *both* absolute and relative
   depth**:
   - Qwen2.5-VL: L10 (36% relative)
   - LLaVA-1.5: L20 (62% relative)
   - LLaVA-1.5's commitment locks in at *deeper* relative depth than
     Qwen's. Consistent with §4.6 cross-model revised: LLaVA-1.5's
     "shortcut layer" is L25 (78% of LM depth), not L10's analog.

3. **The Qwen-LLaVA gap connects to encoder saturation** (H9): Qwen
   has a saturated encoder feeding the LM cleanly-separable physics-
   mode information. The LM commits early (L10) because the input
   is unambiguous. LLaVA-1.5's CLIP-encoder gives noisier signal;
   the LM has to integrate over more layers (L0-L19) before
   committing.

4. **Random-baseline check**: LLaVA-1.5 IE at L29-L30 is slightly
   *negative* (−0.07), which is unusual. Plausibly: late-layer
   patching disturbs the "decision tree" too aggressively at this
   depth, occasionally pushing physics-mode → abstract. Worth
   replicating with larger n.

## Cross-architecture generalization assessment

**Validated for 2/5 models**: the SIP+patching IE-curve shape
generalizes from Qwen to LLaVA-1.5 with the locus shifted in relative
depth.

**Skipped for 3/5**: LLaVA-Next (2 clean SIP candidates), Idefics2
(0), InternVL3 (0). These are too saturated on open-prompt M2 stim
for SIP construction. Either re-run with M8a or photo stim where
they have more PMR=0 cells, or use forced-choice protocol on these
models.

## Connection to other findings

- **§4.6 cross-model revised**: LLaVA-1.5 L25 admits pixel-encoding
  (78% relative). M5b finds LLaVA-1.5 lock-in starts at L20 (62.5%).
  L25 is in the lock-in zone — the model is *partially committed*
  there, hence pixel perturbation can still flip behavior with more
  effort (ε=0.2 vs Qwen's ε=0.05). The two findings are mutually
  consistent.

- **H-locus** (M4-derived): "bottleneck at LM mid layers (L10
  specifically)" is now properly *Qwen-specific*. The general
  formulation: each VLM has its own "decision-lock-in" layer at
  some relative depth (Qwen 36%, LLaVA-1.5 62%). The locus exists
  cross-model but at different positions.

- **H-encoder-saturation** (M6 r2 / M9): Qwen's earlier lock-in
  matches its earlier encoder-side saturation (encoder probe AUC
  ~0.99 from L3). LLaVA-1.5's later lock-in matches its slower
  build-up (encoder probe AUC ~0.73; LM probe plateau 0.77). The
  two-cluster pattern from H9 maps onto the M5b lock-in layer
  difference.

## Limitations

1. **Only 2 of 5 models tested** for cross-model SIP+patching;
   LLaVA-Next / Idefics2 / InternVL3 had insufficient n_neg on
   open-prompt M2 stim. Need harder stim source.

2. **Single intervention type** (visual-token full-replacement
   patching). Attention knockout, MLP replacement, SAE intervention
   are still open per research plan §2.5.

3. **No head-level resolution**. Each layer's patching replaces
   *all* visual-token hidden states; per-head IE requires
   layer-internal patching.

4. **n=15 LLaVA-1.5 is small**. Replicate with larger n_pairs from
   richer stim sources (M8a / M8c).

5. **Re-inference drift**: 9/15 LLaVA-1.5 corrupted SIP samples
   gave PMR=1 at re-inference time even without patching. The
   SIP-construction PMR=0 was over a label-collapse aggregate; per-
   label baseline drift is a known M5a-style caveat.

## Reproducer

```bash
# Qwen (already done; outputs/m5b_sip/per_layer_ie.csv).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_activation_patching.py \
    --n-pairs 20 --device cuda:0

# LLaVA-1.5 (this round).
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_cross_model.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --capture-pattern "cross_model_llava_capture_*" \
    --label ball --n-pairs 15 --model-tag llava15 --device cuda:0
```

## Artifacts

- `scripts/m5b_sip_cross_model.py` — generic per-model SIP+patching
  (handles Qwen / LLaVA / InternVL via shared `model.model.language_model.layers`
  path; Idefics2 via `model.model.text_model.layers`).
- `outputs/m5b_sip_cross_model/llava15_per_pair_results.csv`,
  `_per_layer_ie.csv`, `_manifest.csv`.
- `docs/figures/m5b_sip_cross_model_llava15_per_layer_ie.png`.

## Follow-ups (research plan §2.5 remaining)

1. **Attention knockout (per-layer-head)**: Qwen-only, focus on
   L0-L13 transition zone. Identify the 1-3 heads carrying the
   visual-token → text decision.
2. **MLP replacement**: similar layer-sweep but only MLP output
   patched.
3. **SAE intervention** (Pach et al. 2025 recipe): train SAE on
   Qwen's vision-encoder activations; identify monosemantic
   "physics-cue" features.
4. **Multi-axis SIP with matched seeds**: regenerate stim where
   bg_level / object_level toggle independently with same seed for
   cleaner per-axis IE analysis.
5. **Cross-model SIP for saturated models**: re-do M2 captures with
   FC prompt (not just open) on LLaVA-Next / Idefics2 / InternVL3
   for n_neg headroom.
