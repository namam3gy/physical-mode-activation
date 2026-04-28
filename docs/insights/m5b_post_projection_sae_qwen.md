# M5b Post-projection SAE on Qwen2.5-VL (2026-04-28)

> **Status**: ✅ Qwen-only complete. Cross-model extension is optional follow-up.
> **Track B reference**: `references/roadmap.md` §3.X (M5b post-projection row),
> per-user instruction 2026-04-28 to run between M-MP Phase 2 and Phase 3.

## Question

Does the **projector / merger** that maps vision-encoder output (1280-dim
patches) to LM embedding space (3584-dim visual tokens) **preserve** or
**distribute** the physics-mode commitment that was localized in the
encoder-side SAE?

The original M5b SAE was trained on `vision_hidden_31` (last SigLIP block,
**pre-projection**, 1280-dim) and produced a clean k=20 → PMR break with
mass-matched random controls retaining at 1.0. This round trains a
**post-projection** SAE on the merger output (3584-dim LM-space tokens)
and asks whether the same physics-cue features survive the projection.

## Setup

- **Model**: Qwen/Qwen2.5-VL-7B-Instruct.
- **Hook point**: `model.model.visual.merger` (`Qwen2_5_VLPatchMerger`,
  outputs n_visual_tokens × 3584 in LM-space).
- **Stim**: 480 stim from M2 mvp_full (`mvp_full_20260424-093926_e9d79da3`).
- **Capture**: 480 × 324 visual tokens × 3584 dim ≈ 155K tokens for
  SAE training (4× fewer than the pre-projection 622K tokens since the
  merger groups 4 patches → 1 token).
- **SAE**: 14336 features (4× expansion of 3584), tied weights, λ=1.0,
  5K Adam steps. Trained 6.0 min on H200.
- **Intervention**: same protocol as original M5b — top-k Cohen's-d
  ablation via Bricken et al. raw-scale subtraction. n=20 stim, k ∈
  {5, 10, 20, 50, 100, 200}.
- **Random controls**: 3 mass-matched sets at k=20 (the break threshold).

## Results

### SAE training

Final reconstruction = 0.0136, L1 = 0.0251, alive features = 100%,
active per token = 2.36%. Training quality matches pre-projection SAE.

### Top-10 features by Cohen's d

| feature_idx | mean_phys | mean_abs | cohens_d |
|---|---|---|---|
| 4998  | 3.44 | 0.35 | **0.71** |
| 10471 | 2.09 | 0.09 | 0.63 |
| 8121  | 4.78 | 0.32 | 0.60 |
| 7733  | 5.65 | 1.11 | 0.50 |
| 246   | 3.00 | 0.42 | 0.45 |
| 10200 | 5.07 | 0.67 | 0.38 |
| 1819  | 4.98 | 0.97 | 0.38 |
| 3100  | 2.51 | 0.40 | 0.38 |
| 3523  | 0.68 | 0.23 | 0.35 |
| 442   | 1.27 | 0.42 | 0.34 |

**Cohen's d top features are LOWER than pre-projection** (pre's top
features had Cohen's d > 1.0). This signals broader feature
distribution post-projection — but a smaller absolute number of
features is still required to break PMR (see intervention below).

### Intervention (n=20 stim, FC prompt with --label circle)

| k | top-k Cohen's d ablation | Random ×3 (mass-matched) |
|---|---|---|
| 5 | 1.00 (intact) | — |
| 10 | 0.60 (transition) | — |
| **20** | **0.00 (full break)** | 1.00 / 1.00 / 1.00 |
| 50 | 0.00 (sustained) | — |
| 100 | 0.00 | — |
| 200 | 0.00 | — |

**Critical**: post-projection SAE **breaks PMR at the same k=20 threshold**
as the pre-projection SAE. Mass-matched random controls (3 sets) all
remain at 1.0, ruling out magnitude-driven collapse.

## Interpretation

### Pre vs post comparison

| | Pre-projection (vision_hidden_31) | Post-projection (merger output) |
|---|---|---|
| Feature space dim | 1280 | 3584 (LM space) |
| SAE features | 5120 (4× expansion) | 14336 (4× expansion) |
| Top-feature Cohen's d | > 1.0 | 0.71 |
| k=20 break | ✅ | ✅ |
| Random k=20 | 1.0 (1 set) | 1.0 (3 sets, tighter CI) |
| % features needed | 20/5120 = 0.39% | 20/14336 = 0.14% |

### Headline

**The Qwen merger preserves physics-mode commitment in a localized form.**
The same threshold (k=20 features) breaks PMR at both pre- and
post-projection. Post-projection features are **more concentrated by
percentage** (0.14% vs 0.39% of total features), but **less individually
discriminating** by Cohen's d (top d ≈ 0.71 vs > 1.0).

### Why it matters

1. **Confirms the original M5b localization is not an encoder artifact**:
   the same number of physics-cue features (k=20) is required at both
   sides of the projector — the merger doesn't strip the commitment, but
   doesn't aggregate it into fewer features either.
2. **Refines the "L9 MLP construction" picture**: the L9 MLP in the LM
   reads physics-mode from a small number (≈20) of LM-space visual tokens
   that already carry the commitment. The encoder's `vision_hidden_31`
   features are projected through the merger into a parallel set of ≈20
   post-projection features.
3. **Architecture-level implication**: the merger is a **lossy projector**
   in dim (1280 → 3584 with 4-patch grouping = effectively 1280×4 → 3584
   per group, a 5120 → 3584 compression). Despite the dim compression, the
   physics-mode signal is preserved — suggesting the projector learned to
   route physics-mode features cleanly into LM space.

## Limitations

- **Cross-model extension untested**: this round is Qwen-only. The same
  hook point (`model.model.visual.merger`) does not exist on LLaVA family
  or InternVL3 (different projector architectures). Testing whether the
  same finding holds across architectures would require per-model
  projector identification + capture + SAE training (~5 hours, optional
  follow-up per user 2026-04-28).
- **k-sweep not as dense as original M5b**: only k ∈ {5,10,20,50,100,200}
  vs original's {1,2,3,5,7,10,15,20,30}. The transition at k=10 → 0.6 is
  consistent with the pre-projection's non-monotone behavior at mid-k.
  Could be tightened with k=15 + cluster pivot analysis if interesting.

## Files

- New script: `scripts/m5b_capture_post_projection.py` (480-stim merger
  capture in 42 sec on H200).
- New script edit: `scripts/sae_intervention.py` adds `--hook-target {block, merger}`.
- New SAE: `outputs/sae/qwen_post_proj_14336/` (sae.pt + feature_ranking.csv).
- Activations: `outputs/post_projection_qwen/` (480 safetensors, 155K total tokens).
- Intervention results: `outputs/sae_intervention/qwen_post_proj_14336_cohens_d/results.csv`.
