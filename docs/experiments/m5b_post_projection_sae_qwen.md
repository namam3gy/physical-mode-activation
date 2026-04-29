# M5b — post-projection SAE (Qwen merger) (run log, 2026-04-28)

## Question

Does the projector / merger that maps Qwen's vision-encoder output (1280-dim patches, pre-projection) to LM embedding space (3584-dim visual tokens, post-projection) **preserve** or **distribute** the physics-mode commitment localized in the encoder-side SAE?

## Capture

- **Command**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_capture_post_projection.py --stim-dir inputs/mvp_full_20260424-093926_e9d79da3 --device cuda:0`
- **Hook point**: `model.model.visual.merger` (`Qwen2_5_VLPatchMerger`, outputs n_visual_tokens × 3584 in LM-space).
- **Stim**: 480 stim from M2 mvp_full (`mvp_full_20260424-093926_e9d79da3`).
- **N visual tokens**: 480 × 324 ≈ 155K post-projection tokens (4× fewer than the 622K pre-projection tokens — merger groups 4 patches into 1 LM-space token).
- **Output**: `outputs/sae/qwen_post_proj_14336/activations.pt` and matching predictions parquet.

## SAE training

- **Command**: `scripts/sae_train.py` with `--n-features 14336 --n-steps 5000 --l1-lambda 1.0 --tag qwen_post_proj_14336`.
- **Architecture**: 14336 features (4× expansion of 3584), tied weights.
- **Wall clock**: 6.0 min on H200.
- **Final metrics**: recon = 0.0136, L1 = 0.0251, 100 % features alive, 2.36 % active per token.

## Intervention (FC mode, `--label circle`, n=20 stim)

- **Command**: `scripts/sae_intervention.py --sae-dir outputs/sae/qwen_post_proj_14336 --hook-target merger --rank-by cohens_d --top-k-list 5,10,20,50,100,200 --random-controls 3 --n-stim 20`
- **Output**: `outputs/sae_intervention/qwen_post_proj_14336_cohens_d_<ts>/results.csv`
- **Hook**: post-projection (LM-space) merger output replacement using Bricken-style subtraction.

### Result table

| k | top-k Cohen's d ablation | 3 mass-matched random controls |
|---|---|---|
| 5 | 1.00 (intact) | — |
| 10 | 0.60 (transition) | — |
| **20** | **0.00 (full break)** | 1.00 / 1.00 / 1.00 |
| 50 | 0.00 (sustained) | — |
| 100 | 0.00 | — |
| 200 | 0.00 | — |

## Comparison to pre-projection SAE (`m5b_sae_intervention.md`)

| Metric | Pre-projection (vis31, 5120 feat) | Post-projection (merger, 14336 feat) |
|---|---|---|
| Top-feature Cohen's d | > 1.0 | 0.71 |
| **k for full PMR break** | **20** | **20** ✓ |
| Random k=20 control | 1.0 (1 set) | 1.0 (3 sets, tighter CI) |
| % features needed | 20/5120 = 0.39 % | 20/14336 = 0.14 % |

## Headline

**Qwen's merger preserves physics-mode commitment in a localized form**: same k=20 break threshold pre- and post-projection. Post-projection features are **more concentrated by percentage** (0.14 % vs 0.39 %) but **less individually discriminating** by Cohen's d (top d ≈ 0.71 vs > 1.0). The merger doesn't dilute the commitment — it routes it through a smaller proportional subset of LM-space directions.

## Deep dive

`docs/insights/m5b_post_projection_sae_qwen.md`.
