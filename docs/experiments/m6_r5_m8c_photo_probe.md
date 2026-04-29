# M6 r5 — M8c photo encoder probe (4-model cross-stim) (run log, 2026-04-25)

- **Configs**: `configs/encoder_swap_internvl3_m8c{,_label_free}.py` + pre-existing 3-model M8c configs.
- **Pipeline**:
  - InternVL3 M8c inference: 180 + 60 = 240 inferences in 2 min on GPU 0.
  - 4 model × 60 photo captures: ~3 min total.
  - Probe: `scripts/encoder_swap_probe.py --model-name <model>` × 4 + `_stim_y.py` for stim-y AUC.
- **Output dirs**:
  - `outputs/encoder_swap_internvl3_m8c_*/predictions.{jsonl,parquet,csv}`
  - `outputs/encoder_swap_{qwen,llava,idefics2,internvl3}_m8c_vision_activations/*.safetensors`
  - `outputs/encoder_swap_{...}_m8c_probe/{layer_sweep,by_object_level,by_shape}.csv`
  - `outputs/encoder_swap_{...}_m8c_probe_stim_y/layer_sweep_stim_y_physical_shape_vs_abstract_shape.csv`
- **Deep dive**: `docs/insights/m6_r5_m8c_photo_probe.md`.

## 4-model behavioral-y AUC inversion (synthetic M8a → photo M8c)

| Model | M8a behavioral-y AUC | M8c behavioral-y AUC | M8c stim-y AUC |
|---|---:|---:|---:|
| Qwen2.5-VL | 0.88 | **0.44** | 1.00 |
| LLaVA-1.5 | 0.77 | **0.86** | 1.00 |
| Idefics2 | 0.93 | **0.77** | 1.00 |
| InternVL3 | 0.89 | **0.59** | 1.00 |

## Headline

Behavioral-y AUC inverts cross-stim: Qwen drops 44 pp, LLaVA rises 9 pp, Idefics2 drops 16 pp, InternVL3 drops 30 pp. **Stim-y AUC stays at 1.0 across all 4 models on photos** — encoder discriminability is uniform across stim sources.

This confirms cross-stim that **encoder discriminability is uniform**; behavioral-y AUC is a "encoder ↔ behavior alignment" measure that varies with each model's per-stim PMR distribution. Final cross-stim confirmation of the architecture-level reframe (M9).
