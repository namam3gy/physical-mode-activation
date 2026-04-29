# M6 r3 — Idefics2 vision-encoder probe (run log, 2026-04-25)

- **Capture command**: `uv run python scripts/encoder_swap_idefics2_probe.py` (capture + probe combined driver, original 1-model version).
- **Output**:
  - `outputs/encoder_swap_idefics2_vision_activations/*.safetensors` (4 layers × 400 stim × ~88 s capture)
  - `outputs/encoder_swap_idefics2_probe/{layer_sweep,by_object_level,by_shape}.csv`
  - Figure: `docs/figures/encoder_swap_idefics2_probe.png` (2 panels: layer-sweep + by-object-level)
- **Wall clock**: 88 s capture on GPU 0 + ~30 s probe.
- **Stim**: M8a Qwen stim dir (400 stim × 4 SigLIP-SO400M layers).
- **Probe**: layer-wise logistic-regression on per-stim PMR.
- **Deep dive**: `docs/insights/m6_r3_idefics2_probe.md`.

## Headline

| Layer (of 27) | Probe AUC |
|---:|---:|
| 9 | **0.948** (peak) |
| (mean across 4 layers) | **0.93** |

3-point AUC ↔ behavioral PMR(_nolabel) chain at this round:

| Encoder family | AUC | PMR(_nolabel) on M8a |
|---|---:|---:|
| SigLIP (Qwen, M3) | 0.99 | 0.84 |
| **SigLIP-SO400M (Idefics2)** | **0.93** | **0.88** |
| CLIP-ViT-L (LLaVA, M6 r2) | 0.73 | 0.18 |

H-encoder-saturation chain `encoder family → AUC → PMR → H7` empirically grounded at all 4 nodes × 3 model points. Per-shape AUC variance is high at this n (`polygon` AUC drops below 0.5 at deep layers — n-imbalance artifact, not real reversal).
