# M4 — LM logit-lens probe AUC cross-model (run log, 2026-04-28)

- **Command**: `uv run python scripts/m4_lm_probing_cross_model.py`
- **Output**: `outputs/m4_lm_probing_cross_model/probe_auc.csv` (5 models × 5 layers = 25 rows)
- **Figure**: `docs/figures/m4_lm_probing_cross_model.png` (line plot per model)
- **Setup**: analysis-only — reuses existing M2 cross-model captures (LM hidden states at L=5/10/15/20/25 captured during M6 r7).
- **No new GPU run**.
- **Deep dive**: `docs/insights/m4_lm_probing_cross_model.md`.

## Method

Per-(model × layer) train a logistic-regression probe on `LM_hidden[visual-token positions]` to predict per-stim `PMR_open ∈ {0, 1}`. Report AUC. 80/20 train/test split on the 480-stim cohort × 3 labels = 1440 examples per model.

## Result

| Model | Vision encoder probe AUC (M3) | LM probe AUC (this round) |
|---|---:|---:|
| Qwen2.5-VL-7B | 0.99 | 0.96 |
| **Idefics2-8B** | 0.93 | **0.995** ← higher than vision side |
| LLaVA-Next-7B | 0.81 | 0.78 |
| LLaVA-1.5-7B | 0.73 | 0.76 |
| InternVL3-8B | 0.89 | untestable (n_neg=1) |

## Headlines

1. **LM-probe AUC ladder aligns with M3 vision-encoder probe AUC ladder** — second downstream signature of H-encoder-saturation (after M9 PMR ceiling and §4.7 decision-stability ceiling).
2. **Idefics2 LM AUC > vision AUC** (0.995 vs 0.93) — perceiver-resampler does NOT strip information; the LM has full access to physics-mode signal.
3. Combined with §4.6 Idefics2 0/9 layers shortcut (no pixel-encodability), this dissociates **information presence ≠ pixel-space gradient routability**. Forward path from encoder to LM is intact for Idefics2, but the inverse (pixel → v_L) pathway is blocked by perceiver-resampler. Same dissociation closed by M5a runtime steering on Idefics2 (10/10 LM-side flip at L25 α=20).

## Reading

H-encoder-saturation now has 2 mechanism-level downstream signatures: M3 vision-encoder probe AUC + this M4 LM probe AUC. Idefics2's anomalous "LM > vision AUC" is the leading signal that perceiver-resampler is the relevant remaining axis (pending controlled M-PSwap test).
