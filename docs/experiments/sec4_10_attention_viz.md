# §4.10 — Attention visualization captures (run log, 2026-04-25)

## Setup

- **Configs**: `configs/attention_viz_{qwen,llava,llava_next,idefics2,internvl3}.py`
  - Each: `limit=20` stim from M8a × 3 labels = 60 capture records, `capture_lm_attentions=True`, `layers=(5, 15, 20, 25)`.
- **Code**: `src/physical_mode/models/vlm_runner.py` — `attn_implementation` auto-switches to `"eager"` when capturing attentions (SDPA does not return attention weights).
- **Output dirs**: `outputs/attention_viz_<model>_<ts>/` containing predictions + 20 safetensors capture files.
- **Notebook**: `notebooks/attention_viz.ipynb` (6-section interactive: load capture / per-layer heatmap / image overlay / phys-vs-abs comparison / per-head fine structure / attention-entropy aggregate).
- **Deep dive**: `docs/insights/sec4_10_attention_viz.md`.

## Per-model capture cost

| Model | Per-stim wall | Per-stim disk |
|---|---:|---:|
| Qwen2.5-VL-7B | ~30 s | 7 MB |
| LLaVA-1.5-7B | similar | 21 MB |
| LLaVA-Next-7B | similar | **480 MB** (AnyRes 5-tile balloons attention tensors) |
| Idefics2-8B | ~30 s | 10 MB |
| InternVL3-8B | similar | 5 MB |

(Total disk for 5 models × 20 stim × 3 labels: ~30 GB, dominated by LLaVA-Next.)

## Saved tensor schema (per `<sample_id>.safetensors`)

| Key | Shape | Dtype |
|---|---|---|
| `attentions_<L>` | (n_heads, n_query, n_key) | fp16 |
| `image_token_mask` | (seq_len,) | bool |
| `last_token_idx` | scalar | int |

## Headline (last-token attention to visual tokens, 5-model M8a)

- Qwen ~17 %
- LLaVA-1.5 ~7 %
- Idefics2 ~30 % (highest — perceiver-resampler aggregates 4900 patches into 64 latents which the LM attends densely)
- LLaVA-Next ~8 %
- InternVL3 ~12 %

Architecture-level differences in how much LM "looks at" visual tokens; useful as paper-appendix figure.
