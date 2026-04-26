---
section: M5b — SIP + LM activation patching (Qwen2.5-VL)
date: 2026-04-26
status: complete (n=20 SIP pairs × 28 LM layers)
hypothesis: H10 (research plan §2.5) — "2-3 narrow layer/head ranges show large IE" → SUPPORTED with refinement: full recovery in L0-L9 (early-mid block); decline at L10-L13; zero at L14+
---

# M5b — Semantic Image Pairs + activation patching (Qwen2.5-VL)

> **Recap of codes used in this doc**
>
> - **M2** — 480-stim 5-axis factorial; FC prompt produces letter-first responses.
> - **M5a** — VTI steering: adding +α·v_L10 at LM L10 over visual tokens flips line/blank/none from "stays still" → physics-mode. Single layer L10 with α=40 was the only causal-intervention layer.
> - **M5b** — Phase 3 of ST4 (this doc): activation patching to identify causally necessary layers.
> - **SIP** (Semantic Image Pairs, Golovanevsky et al., NAACL 2025) — minimal-pair stim that differ on a single visual cue.
> - **IE** (Indirect Effect) — Δ P(physics-mode) from patching: P(physics | patched corrupted) − P(physics | corrupted baseline).
> - **H-locus** (M4-derived) — the bottleneck is at LM mid layers (L10).
> - **H10** (research plan §2.5) — "2-3 narrow layer/head ranges show large IE."

## Question

M5a's runtime VTI steering found that **only L10** in Qwen2.5-VL admits
α=40 intervention to flip behavior. The natural follow-up: is L10 the
*sole* causal layer, or is it the decision-lock-in *boundary* with
upstream layers also carrying the cue information? Activation patching
between SIP — where the only difference is presence of physics cue —
distinguishes these.

## Method

**SIP construction** (`build_sip_manifest`):
- M2 forced-choice predictions (Qwen2.5-VL, 1440 inferences).
- For each (object_level, bg_level, event_template), pair cue=both
  seeds (clean: physics-mode response, abs_rate=0) with cue=none
  seeds (corrupted: abstract response, abs_rate=1) by index.
- Filter: `clean abs_rate=0 AND corr abs_rate=1` (strictly).
- Yields 32 candidate pairs; first n=20 used.

**Patching** (`scripts/m5b_sip_activation_patching.py`):
- For each pair (clean_sid, corrupted_sid):
  1. **Cache**: forward(clean_pil) with `output_hidden_states=True`,
     extract h_L at visual-token positions for L ∈ [0..27].
  2. **Baseline**: forward(corrupted_pil) → first-letter response.
  3. **Patched**: for each target_L ∈ [0..27]: register forward hook
     on `model.model.language_model.layers[target_L]` that, on prefill
     pass only (matched seq_len), replaces output hidden state at
     visual-token positions with cached clean values; generate text;
     parse first letter (A/B/C → physics, D → abstract).
- IE per layer = (patched physics rate) − (baseline physics rate).

Total: n=20 × (1 cache + 1 baseline + 28 patched) = 600 forward
passes; ~8.3 min on H200.

## Result

![M5b SIP per-layer IE — Qwen2.5-VL activation patching](../figures/m5b_sip_per_layer_ie.png)

| Layer | IE | Patched physics rate (n=20) |
|---:|---:|---:|
| L0-L9 | **+1.0** | 20/20 (full recovery) |
| L10 | +0.6 | 12/20 |
| L11 | +0.6 | 12/20 |
| L12 | +0.3 | 6/20 |
| L13 | +0.1 | 2/20 |
| L14-L27 | 0.0 | 0/20 |

Baseline corrupted physics rate: 0/20 (all "D — abstract"). Clean
control: 20/20 physics (by SIP construction).

## Headlines

1. **Sharp L10 boundary.** Patching corrupted's hidden state at any
   layer L0-L9 with clean's hidden state recovers physics-mode in
   100% of pairs. Patching at L10-L11 recovers 60%. Beyond L13,
   patching has no effect.

2. **L10 is the "decision lock-in" layer.** This refines M5a's
   reading: M5a found L10 the *only* layer where +α·v_L10 steering
   flips behavior. Activation patching shows L0-L9 *all* carry
   sufficient cue information — but by L10-L11 the LM is committing
   to its decision, and by L14+ the decision is fully baked.

3. **Information flow direction**: visual-cue information enters at
   L0 and propagates through the early-mid block (L0-L9), is read by
   the text-side decision token at L10-L13, and after L14 the
   decision is no longer overridable by visual-token replacement.

4. **n=20 with 100% baseline-corrupted = abstract** is a clean
   experimental setup. The 0% → 100% IE jump at L0-L9 patching is
   fully consistent across (line, filled, shaded, textured) ×
   (blank, ground, scene) × fall.

## Connection to existing hypotheses

- **H-locus** (M4-derived): "the bottleneck is at LM mid layers (L10
  specifically)" → **refined**. L10 is the *boundary* where decision
  lock-in begins; the *information* exists at every L0-L9. The
  "single causal layer" reading from M5a holds for additive steering
  but not for full-state patching.

- **H10** (research plan §2.5): "2-3 narrow layer/head ranges show
  large IE" → **partially supported with revision**. We see *one
  contiguous range* (L0-L9) with full IE rather than 2-3 narrow
  bands. But the L10-L13 decline + L14+ zero matches the broader
  shape Kaduri et al. 2024 reported ("middle ~25% of layers carry
  cross-modal flow" — for Qwen 28-layer LM, L7-L21 is the middle
  50%, our L0-L13 is the lower-middle).

- **M5a's L10 specificity**: now interpretable as a lower bound on
  effective steering. Pre-L10 is too early for additive steering
  (representation hasn't crystallized into the v_L10 direction yet);
  post-L10 is too late (decision committed). Patching is more
  permissive because it transplants the *full* representation.

- **Basu et al. 2024**: "constraint-satisfaction information stored
  in early MLP / self-attention of layers 1-4 in LLaVA" — broadly
  consistent. We show the information persists through L0-L9 in Qwen.
  Their finer-grained MLP-vs-attention split is open for our setup.

## Limitations

1. **Single model (Qwen2.5-VL).** Cross-model SIP+patching needs
   per-model image-token resolution + hook adaptation. LLaVA-1.5,
   LLaVA-Next (AnyRes), Idefics2, InternVL3 all open.

2. **Single intervention type (full visual-token replacement).** Plan
   §2.5 mentions: visual token patching ✓ (this), attention knockout
   ✗, MLP replacement ✗, steering vector intervention ✓ (M5a),
   SAE intervention ✗. Three of five types still open.

3. **n=20 SIP pairs.** All from cue-axis (none vs both). Axis-A
   (object_level: line vs textured) and Axis-B (bg_level: blank vs
   ground) SIP not built — different seeds confound those.

4. **Single-axis cue effect only.** The "physics-mode information"
   we measure is specifically what a cast_shadow + motion_arrow vs
   none distinction encodes. Other cue distinctions (e.g., shading
   intensity) might localize to different layer ranges.

5. **No head-level resolution.** Each patching replaces the entire
   layer's visual-token hidden state. Per-head IE requires
   layer-internal patching (attention output decomposition).

## Reproducer

```bash
# Run on GPU 1 (free); ~10 min on H200.
CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_activation_patching.py \
    --n-pairs 20 --device cuda:0
```

## Artifacts

- `scripts/m5b_sip_activation_patching.py` — driver.
- `outputs/m5b_sip/manifest.csv` — 20 SIP pairs (cue=both clean × cue=none corrupted).
- `outputs/m5b_sip/per_pair_results.csv` — per-pair × per-layer first-letter responses.
- `outputs/m5b_sip/per_layer_ie.csv` — aggregated IE per layer.
- `docs/figures/m5b_sip_per_layer_ie.png` — IE × layer plot.

## Open follow-ups

1. **Cross-model SIP+patching**. LLaVA-1.5 (32 layers, hook
   `model.language_model.model.layers[L]`) + Idefics2 + InternVL3
   patching to find each model's "L10 equivalent". H9 already
   showed LM probe AUC plateaus differ; M5b would test causal
   intervention layer.
2. **Attention knockout**. For specific (layer, head) pairs in the
   L10-L13 transition zone, knock out attention from visual tokens
   to last token; measure IE.
3. **Multi-axis SIP**. Generate matched-seed stim where bg_level
   toggles independently — needs new stim render run.
4. **Head ranking**. Per-(layer, head) IE map. Identify the 1-3
   heads in L8-L11 that carry the cue → text decision.
