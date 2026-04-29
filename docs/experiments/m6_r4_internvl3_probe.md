# M6 r4 — InternVL3 vision-encoder probe + apples-to-apples 4-model M8a (run log, 2026-04-25)

- **Configs**: `configs/encoder_swap_internvl3{,_label_free}.py`
- **Inference command**: `uv run python scripts/02_run_inference.py --config configs/encoder_swap_internvl3.py` + label-free.
- **Capture + probe**: `uv run python scripts/encoder_swap_probe.py --model-name internvl3 ...`
- **Cross-model summary**: `uv run python scripts/encoder_swap_probe_summary.py`
- **Wall clock**:
  - InternVL3 M8a inference: ~8 min labeled + ~4 min label-free = 12 min on GPU 0
  - InternVL3 vision capture: 47 s (400 stim × 4 InternViT layers)
  - Re-capture of Qwen + LLaVA on M8a stim (apples-to-apples): a few minutes
- **Output dirs**:
  - `outputs/encoder_swap_internvl3_*/predictions.{jsonl,parquet,csv}`
  - `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
  - Figure: `docs/figures/encoder_chain_4model.png` (paper headline)
- **Required fix**: `_resolve_vision_blocks` to recognize InternVL3's `vision_tower.encoder.layer` (singular, not plural).
- **Deep dive**: `docs/insights/m6_r4_internvl3_probe.md`.

## 4-model AUC ↔ PMR chain (M8a stim, apples-to-apples)

| Encoder family | Probe AUC | PMR(_nolabel) |
|---|---:|---:|
| SigLIP (Qwen) | 0.88 | 0.84 |
| SigLIP-SO400M (Idefics2) | 0.93 | 0.88 |
| **InternViT (InternVL3)** | **0.89** | **0.92** |
| CLIP-ViT-L (LLaVA-1.5) | 0.77 | 0.18 |

3 distinct **non-CLIP encoder families** (SigLIP / SigLIP-SO400M / InternViT) all reach AUC ≥ 0.88 / PMR ≥ 0.84. **Only CLIP-ViT-L falls below saturation** (0.77 / 0.18). Across 4 LM families (Qwen2-7B, Mistral-7B, InternLM2-7B, Vicuna-7B), **encoder family is the unified saturation driver**.

## Stim-y check (added late round)

All 4 encoders separate stim-defined factorial cells at AUC = 1.0 — encoder discriminability is uniform across families. Reframes the chain to architecture-level (encoder + LM fusion), not encoder-discriminability alone.

## Headline

H-encoder-saturation generalizes from "SigLIP saturates" to "non-CLIP encoders saturate; CLIP doesn't (in this sample)". The nonlinear AUC → PMR mapping (≈0.10 AUC gap → 0.65 PMR gap) is consistent with a saturation threshold around AUC 0.85.
