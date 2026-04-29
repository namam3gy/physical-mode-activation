# M5b — SAE intervention cross-model (5-model run log, 2026-04-28)

## Setup

- **Wall clock**: ~3 hr chain on GPU 0 + GPU 1 (parallel) — see `scripts/run_m5b_chain_v2_gpu{0,1}.sh`.
- **Methodological correction (vs round 1)**: round-1 used `vision_hidden_23` uniformly, which is the **last layer** for Qwen / InternVL3 but **NOT** the layer LLaVA-family / Idefics2 actually consume. Round-2 retrains SAEs at the per-model actually-consumed layer.

### Per-model layer mapping

| Model | Encoder | n_layers | Consumed layer | SAE trained |
|---|---|---:|---|---:|
| Qwen2.5-VL-7B | Qwen2_5_VLVisionTower | 32 | last (Qwen convention) | 31 ✓ |
| LLaVA-1.5-7B | CLIP-ViT-L/14 | 24 | layer 22 (`vision_feature_layer=-2`) | 22 ✓ |
| LLaVA-Next-7B | CLIP-ViT-L/14 | 24 | layer 22 (`vision_feature_layer=-2`) | 22 ✓ |
| Idefics2-8B | SigLIP-SO400M | 27 | layer 26 (`last_hidden_state` post-LN) | 26 ✓ |
| InternVL3-8B-hf | InternViT-300M | 24 | layer 23 (`vision_feature_layer=-1`) | 23 ✓ |

## Per-model commands (representative for Qwen)

```bash
# 1. Train SAE at consumed layer
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_train.py \
    --activations-dir outputs/.../vision_activations \
    --layer-key vision_hidden_31 --n-features 5120 --n-steps 5000 \
    --tag qwen_vis31_5120

# 2. Re-rank by Cohen's d
uv run python scripts/sae_rerank_features.py --sae-dir outputs/sae/qwen_vis31_5120 ...

# 3. Top-k ablation, OPEN prompt, n=20 stim, baseline-PMR≈1 cell
uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/qwen_vis31_5120 \
    --prompt-mode open \
    --vision-block-idx 31 \
    --stimulus-dir inputs/mvp_full_... \
    --test-subset filled/blank/both \
    --top-k-list 5,10,20,40,80,160 \
    --rank-by cohens_d --random-controls 3 --n-stim 20
```

(LLaVA-1.5/Next use `vision_feature_layer=-2` → vis22; Idefics2 vis26; InternVL3 vis23. Per-model OPEN-prompt baseline-PMR=1 cell varies: Qwen filled/blank/both, LLaVA-1.5 shaded/ground/cast_shadow, LLaVA-Next shaded/blank/both, Idefics2 + InternVL3 filled/blank/both.)

## Output dirs

- `outputs/sae/{qwen_vis31_5120, llava15_vis22_4096, llava_next_vis22_4096, idefics2_vis26_4608, internvl3_vis23_4096}/`
- `outputs/sae_intervention/{model}_vis{layer}_{prompt}_{ts}/`
- 5-model summary: `scripts/m5b_sae_intervention_cross_model_summary.py`
- Figure: `docs/figures/m5b_sae_intervention_cross_model.png`

## Headline

| Model | Layer | top-k for break | Random control | Architecture cluster |
|---|---:|---:|---:|---|
| Qwen2.5-VL-7B | 31 | **40** (0.78 % of 5120 features) | 1.0 | non-CLIP, high-saturation |
| Idefics2-8B | 26 | **160** (3.5 % of 4608) | 1.0 | non-CLIP, perceiver-resampler |
| InternVL3-8B | 23 | **160** (3.9 % of 4096) | 1.0 | non-CLIP, MLP projector |
| LLaVA-1.5-7B | 22 | **NULL** (≤ 800 = 20 %) | 1.0 | CLIP, low-saturation |
| LLaVA-Next-7B | 22 | **NULL** (≤ 160 = 4 %) | 1.0 | CLIP, low-saturation |

(Qwen original FC-mode result was k=20 break; under uniform OPEN prompt it's k=40. Same direction, slightly slower — preserved.)

## Headlines

1. **3 of 5 models break PMR cleanly under encoder-side SAE feature ablation**: Qwen / Idefics2 / InternVL3.
2. **Effect concentration tracks M3 vision-encoder probe AUC**: Qwen 0.99 (k=40, most concentrated) > Idefics2 0.93 (k=160) > InternVL3 0.89 (k=160) > LLaVA-1.5 0.73 (NULL) — higher discriminability → more localized SAE features.
3. **CLIP-based LLaVA models NULL** at any k ≤ 160 (LLaVA-Next) / k ≤ 800 (LLaVA-1.5). LLaVA-Next M5a-positive (LM-side L20+L25 10/10 flip) + M5b NULL → **physics-mode commitment routes through LM, not encoder, in the LLaVA family**. Mechanistic encoder-vs-LM dissociation.
4. All random-feature controls (3 mass-matched sets × 5 models) preserve PMR at 1.0 — direction-specificity confirmed cross-model.

## Deep dive

`docs/insights/m5b_sae_intervention_cross_model.md`.
