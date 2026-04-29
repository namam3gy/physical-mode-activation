# M5a — runtime steering cross-model (run log, 2026-04-28)

## Setup

- **Per-model commands** (all OPEN prompt, `--prompt-variant open`):

```bash
# Qwen baseline (already validated; M5a)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d \
    --test-subset line/blank/none --label circle \
    --layers 10 --alphas 0,10,20,40 \
    --output-subdir m5a_qwen_l10  # 10/10 flip at α=40

# LLaVA-1.5
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_llava_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none --label circle \
    --layers 25 --alphas 0,10,20,40,60 \
    --model-id llava-hf/llava-1.5-7b-hf \
    --output-subdir m5a_cross_llava15_l25 --prompt-variant open

# LLaVA-Next
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_llava_next_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/both --label circle \
    --layers 20,25 --alphas 0,5,10,15,20 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --output-subdir m5a_cross_llava_next_lbb --prompt-variant open

# Idefics2
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir outputs/cross_model_idefics2_capture_<ts> \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset line/blank/none --label circle \
    --layers 20,25 --alphas 0,5,10,20,40,60 \
    --model-id HuggingFaceM4/idefics2-8b \
    --output-subdir m5a_cross_idefics2_lbn --prompt-variant open
```

(InternVL3 not testable — open-prompt baseline saturated at PMR=1.)

- **Patch**: `scripts/06_vti_steering.py` adds `text_model.layers` fallback for Idefics2/Mistral path.
- **Output dirs**: `outputs/cross_model_<model>_capture_*/steering_experiments/m5a_cross_<model>_<subset>/`
- **Deep dive**: `docs/insights/m5a_cross_model.md`.

## Headline

| Model | LM layer | α | Test cell | Flip rate | Sample response |
|---|---:|---:|---|---:|---|
| Qwen2.5-VL | 10 | 40 | line/blank/none × circle | **10/10** | "B: stays still" |
| LLaVA-Next | 20 | 10 | line/blank/both × circle | **10/10** | (kinetic phys-mode) |
| LLaVA-Next | 25 | 15-20 | line/blank/both × circle | **10/10** | (kinetic phys-mode) |
| **Idefics2** | 25 | 20 | line/blank/none × circle | **10/10** | "The tip of the arrow will hit the center of the circle." |
| LLaVA-1.5 | 25 | 0-60 | line/blank/none × circle | **0/10** | (encoder bottleneck — replicates §4.6 NULL) |

**3 of 4 testable models flip 10/10**.

## α dynamic range model-specific

- Qwen 40
- LLaVA-Next 5-15
- Idefics2 20

## Triangulation with M4 + §4.6

**Idefics2**: M4 LM probe AUC 0.995 + M5a runtime 10/10 + §4.6 pixel-flip 0/9 layers → **perceiver-resampler removes pixel-space gradient routability, not LM-side information**. Forward pathway works (LM has the info, runtime steering can flip), inverse pixel→v_L pathway blocked.

**LLaVA-Next** M5a positive (LM-side L20+L25 10/10 flip) + M5b SAE NULL → **physics-mode commitment routes through LM, not encoder, in the LLaVA family** (encoder-vs-LM mechanistic dissociation).

**LLaVA-1.5** M5a NULL + M5b SAE NULL → encoder + LM both bottlenecked (low AUC across pipeline).

## Causal localization extension

Paper contribution 2 (causal localization via runtime steering) extended Qwen-only → 3-model cross-model.
