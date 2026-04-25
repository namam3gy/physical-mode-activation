---
milestone: M6 r6
date: 2026-04-25
status: complete
hypothesis: H-encoder-saturation (architecture-level reframe)
---

# M6 r6 — LLaVA-Next-Mistral 5th model point (2nd CLIP)

## Summary

LLaVA-v1.6-Mistral-7b adds a **5th model point** to the encoder-saturation
chain. It uses the same vision encoder family as LLaVA-1.5 (CLIP-ViT-L/14)
but pairs it with Mistral-7B instead of Vicuna-7B — plus AnyRes multi-tile
image splitting, a different fusion projector, and a different training
data + recipe. Behavioral PMR(_nolabel) on M8a = **0.700, 95% CI [0.65,
0.74]**, sitting cleanly between the LLaVA-1.5 floor [0.14, 0.21] and the
saturated non-CLIP cluster [0.80, 0.92].

The 2nd CLIP point is **not** a clean "LM swap" counterfactual — four
architectural axes change at once between LLaVA-1.5 and LLaVA-Next.
We report it as a 5th observation that **rules out vision-encoder-family
as the sole determinant of behavioral PMR**, not as a causal isolation
of LM modulation.

## Numbers

### Behavioral PMR(_nolabel) on M8a

| Model       | Encoder       | LM           | M8a PMR | 95% CI         |
|-------------|---------------|--------------|--------:|----------------|
| Qwen2.5-VL  | SigLIP        | Qwen2-7B     | 0.838   | [0.800, 0.872] |
| LLaVA-1.5   | CLIP-ViT-L    | Vicuna-7B    | 0.175   | [0.140, 0.212] |
| **LLaVA-Next** | **CLIP-ViT-L** | **Mistral-7B** | **0.700** | **[0.653, 0.743]** |
| Idefics2    | SigLIP-SO400M | Mistral-7B   | 0.882   | [0.850, 0.912] |
| InternVL3   | InternViT     | InternLM2-7B | 0.917   | [0.890, 0.943] |

CIs are 5000-iter prediction-level bootstrap, resampled within each shape
(as in M9).

### Vision-encoder probe AUC on M8a

| Model       | behavioral-y AUC (deepest layer) | stim-y AUC (4 targets) |
|-------------|--------------------------------:|------------------------:|
| Qwen2.5-VL  | 0.880                            | 1.000                   |
| LLaVA-1.5   | 0.771                            | 1.000                   |
| **LLaVA-Next** | **0.809**                     | **1.000**               |
| Idefics2    | 0.926                            | 1.000                   |
| InternVL3   | 0.886                            | 1.000                   |

LLaVA-Next behavioral-y AUC at layer 23 = 0.809 — between LLaVA-1.5 (0.77)
and the saturated cluster (0.88–0.93), like the PMR ordering. Stim-y AUC
is 1.0 on all 4 targets (rendered_vs_line, physics_cell_vs_abstract_cell,
within_line_context, within_textured_context). **Same finding as the other
4 models**: the encoder linearly separates physics-cell from abstract-cell
factorial cells perfectly. CLIP-ViT-L is not an encoder bottleneck.

## Implication for H-encoder-saturation

The 2nd CLIP point closes the most obvious encoder-side counter-hypothesis:
"maybe CLIP just can't represent the physics-vs-abstract distinction." It
can — same encoder, different downstream architecture, PMR moves
0.18 → 0.70. Whatever drives the PMR floor in LLaVA-1.5 is **not at the
vision encoder**.

The remaining unknown is **which of the 4 confounded axes** (AnyRes tiling,
fusion projector, training, LM family) carries the load. Disambiguation
would require a same-architecture LM-only swap, which no released model
provides. Status: H-encoder-saturation is confirmed at the
**joint-architecture** level; the LM-modulation hypothesis is suggested
by Idefics2 (SigLIP-SO400M+Mistral PMR 0.88) vs LLaVA-1.5
(CLIP-ViT-L+Vicuna 0.18) but cannot be isolated from the available data.

## How to read the multi-axis confound

LLaVA-1.5 → LLaVA-Next changes:
1. **AnyRes multi-tile image splitting** (5 tiles vs 1) — each image is
   processed at higher effective resolution. Visual capture confirms 5×577
   patch tokens.
2. **Fusion projector** — linear projector → MLP, with different parameter
   initialization and training.
3. **Training data + recipe** — LLaVA-Next was trained on 760k examples
   vs LLaVA-1.5's 158k, with a different mix (more reasoning + chart QA).
4. **LM family** — Vicuna-7B → Mistral-7B-Instruct.

Any single one (or interaction) could drive the 0.52-PMR jump. We do not
make a "the LM did it" claim from this row.

## Reproducer

Configs:
- `configs/encoder_swap_llava_next.py` — labeled-arm M8a inference
- `configs/encoder_swap_llava_next_label_free.py` — open-prompt arm

Inference (labeled + label-free):
```bash
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next.py
uv run python scripts/02_run_inference.py \
    --config configs/encoder_swap_llava_next_label_free.py
```

Capture (uses M8a stim from Qwen run):
```bash
uv run python scripts/04_capture_vision.py \
    --stimulus-dir inputs/m8a_qwen_<ts> \
    --output-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --layers 5,11,17,23 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --torch-dtype bfloat16
```

Probes:
```bash
# Behavioral-y
uv run python scripts/encoder_swap_probe.py \
    --model-name llava_next \
    --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
    --predictions outputs/encoder_swap_llava_next_m8a_<ts>/predictions.parquet \
    --out-dir outputs/encoder_swap_llava_next_m8a_probe \
    --layers 5,11,17,23 \
    --behavioral-pmr 0.700

# Stim-y (4 targets)
for target in rendered_vs_line physics_cell_vs_abstract_cell \
              within_line_context within_textured_context; do
    uv run python scripts/encoder_swap_probe_stim_y.py \
        --vision-dir outputs/encoder_swap_llava_next_m8a_vision_activations \
        --stim-dir inputs/m8a_qwen_<ts> \
        --layers 5,11,17,23 \
        --target $target \
        --out-dir outputs/encoder_swap_llava_next_m8a_probe_stim_y
done
```

Summary regeneration (5-model figure):
```bash
uv run python scripts/encoder_swap_probe_summary.py \
    --idefics2 outputs/encoder_swap_idefics2_probe \
    --internvl3 outputs/encoder_swap_internvl3_probe \
    --llava-next outputs/encoder_swap_llava_next_m8a_probe \
    --internvl3-pmr 0.917 \
    --llava-next-pmr 0.700 \
    --out-dir outputs/encoder_swap_probe_summary
```

Outputs `docs/figures/encoder_chain_5model.png` (supersedes the 4-model
figure used in r3/r4/r5 insight docs).

## Artifacts

- `outputs/encoder_swap_llava_next_m8a_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_label_free_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/encoder_swap_llava_next_m8a_vision_activations/*.safetensors` (400 stim × 4 layers)
- `outputs/encoder_swap_llava_next_m8a_probe/{layer_sweep,by_object_level}.csv`
- `outputs/encoder_swap_llava_next_m8a_probe_stim_y/layer_sweep_stim_y_*.csv`
- `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`
- `docs/figures/encoder_chain_5model.png`
