# M2 — cross-model M2-stim apples-to-apples (M6 r7) (run log, 2026-04-26)

- **Configs**: `configs/cross_model_{llava_next,idefics2,internvl3}{,_label_free}.py` (+ pre-existing Qwen + LLaVA-1.5 captures from earlier rounds).
- **Driver**: `scripts/run_m2_cross_model_chain.sh` (sequential: 5 capture runs + 5 label-free runs).
- **Wall clock**: ~22 min total on H200 GPU 1.
- **Stim**: M2 mvp_full (480 stim × 3 labels × 1 prompt = 1440 inferences/model + 480 label-free; T=0.7).
- **Captures**: LM hidden states at L=(5,10,15,20,25) + vision-encoder activations.

## Commands

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_m2_cross_model_chain.sh
uv run python scripts/m2_extract_per_model_steering.py     # per-model v_L
uv run python scripts/m2_cross_model_analyze.py            # PMR / H1 / H2 + figures
```

## Output dirs

- `outputs/cross_model_{llava_next,idefics2,internvl3}_capture_*/`
- `outputs/m2_cross_model_summary/{per_label_pmr,per_object_level_pmr,h2_paired_delta}.csv`
- `outputs/cross_model_*_capture_*/probing_steering/steering_vectors.npz`
- Figures: `docs/figures/m2_cross_model_{pmr_ladder,h1_ramp,h2_paired_delta}.png`

## 5-model PMR(\_nolabel) ladder (M2 stim, n=480)

| Model | PMR(_nolabel) | 95% CI |
|---|---:|---|
| LLaVA-1.5 | 0.383 | [0.34, 0.43] |
| LLaVA-Next | 0.790 | [0.75, 0.83] |
| Qwen2.5-VL | 0.938 | [0.92, 0.96] |
| Idefics2 | 0.967 | [0.95, 0.98] |
| InternVL3 | 0.988 | [0.98, 1.00] |

## H1 ramp (line → textured PMR range)

LLaVA-1.5 +0.30 (cleanest) > LLaVA-Next +0.14 > Idefics2 +0.09 > Qwen +0.05 > InternVL3 +0.02. Confirms **unsaturated-only reading**.

## H2 paired-delta (3 distinct architecture-conditional patterns)

- LLaVA-1.5 / LLaVA-Next: all positive (classical H2; ball Δ = +0.475 / +0.190).
- Qwen / Idefics2: asymmetric — circle / planet Δ < 0 ("circle override").
- InternVL3: ≈ 0 (fully saturated).

## v_L10 extraction class balance

Only LLaVA-1.5 has class-balanced n_neg=105. LLaVA-Next / Idefics2 / InternVL3 have n_neg = 9 / 5 / 1 — too saturated on M2 for clean v_L10 (M8a-stim re-extraction was needed for §4.6 cross-model — see `sec4_6_cross_model.md`).

## Deep dive

`docs/insights/m2_cross_model.md`.
