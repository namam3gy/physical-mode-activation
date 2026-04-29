# M6 r6 — LLaVA-Next-Mistral 5th model point (run log, 2026-04-25)

- **Configs**: `configs/encoder_swap_llava_next{,_label_free,_m8a_capture,_m8c,_m8c_label_free,_m8d,_m8d_label_free}.py`
- **M8a inference + capture**:
  ```bash
  uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next.py
  uv run python scripts/02_run_inference.py --config configs/encoder_swap_llava_next_label_free.py
  uv run python scripts/encoder_swap_capture.py \
      --stimulus-dir inputs/m8a_qwen_<ts> \
      --output-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
      --layers 5,11,17,23 \
      --model-id llava-hf/llava-v1.6-mistral-7b-hf
  uv run python scripts/encoder_swap_probe.py \
      --model-name llava_next \
      --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
      --predictions outputs/encoder_swap_llava_next_m8a_<ts>/predictions.parquet \
      --out-dir outputs/encoder_swap_llava_next_m8a_probe \
      --layers 5,11,17,23 --behavioral-pmr 0.700
  uv run python scripts/encoder_swap_probe_stim_y.py \
      --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
      --stim-dir inputs/m8a_qwen_<ts> --out-dir outputs/encoder_swap_llava_next_m8a_probe_stim_y
  ```
- **Stim**: M8a (400 labeled + 400 label-free + 400 stim × 4 layers × 5 tile capture). Cross-stim addendum runs M8d + M8c (1620 inferences in ~16 min).
- **Wall clock**: M8a ~10 min on GPU 0; M8d + M8c ~16 min.
- **Deep dive**: `docs/insights/m6_r6_llava_next.md`.

## 5-model M8a chain locked

| Model | Encoder family | Probe AUC | PMR(_nolabel) | 95% CI |
|---|---|---:|---:|---|
| Qwen2.5-VL-7B | SigLIP (1 CLIP) | 0.88 | 0.84 | [0.80, 0.88] |
| LLaVA-1.5-7B | CLIP-ViT-L | 0.77 | 0.175 | [0.14, 0.21] |
| Idefics2-8B | SigLIP-SO400M | 0.93 | 0.88 | [0.85, 0.91] |
| InternVL3-8B | InternViT | 0.89 | 0.92 | [0.89, 0.95] |
| **LLaVA-Next-Mistral-7B** (2nd CLIP) | CLIP-ViT-L + AnyRes | **0.81** | **0.700** | **[0.65, 0.74]** |

LLaVA-Next sits between LLaVA-1.5 floor and saturated cluster — rules out CLIP-as-encoder-alone explanation. The 0.18 → 0.70 PMR jump from LLaVA-1.5 → LLaVA-Next confounds 4 axes (AnyRes / projector / training / LM family) but is consistent with the architecture-level reframe.

## Cross-stim H7 result

| Stim | M8a | M8d | M8c |
|---|---:|---:|---:|
| LLaVA-Next H7 (PMR_phys − PMR_abs) | +0.26 | −0.05 (CI [−0.10, −0.01]) | +0.02 |
| PMR(_nolabel) | 0.700 | 0.625 | 0.417 |

H7 collapses on M8d *with PMR headroom remaining* (0.625, well below ceiling) — same-encoder-family architecture switch attenuates H7 beyond ceiling effects. Suggestive but multi-axis-confounded for LM-modulation.

## Headline

H-encoder-saturation **architecture-level confirmed at 5 model points + 2 CLIP points + 3 stim sources**. M9 paper Table 1 is locked.
