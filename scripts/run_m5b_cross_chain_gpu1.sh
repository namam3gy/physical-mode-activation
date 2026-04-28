#!/usr/bin/env bash
# GPU 1 chain: LLaVA-Next capture (already started) → SAE train → intervention.
# Capture is already running on GPU 1 (started separately); this script waits then continues.
set -euo pipefail

LOG_DIR="outputs/sae_intervention"
mkdir -p "$LOG_DIR"

echo "=== Wait for LLaVA-Next capture (GPU 1) to finish ==="
until [ -f outputs/cross_model_llava_next_l22_capture/vision_activations/textured_scene_none_fall_009.safetensors ]; do
    sleep 30
done
echo "Capture done at $(date +%H:%M:%S)"
echo "Files: $(ls outputs/cross_model_llava_next_l22_capture/vision_activations/ | wc -l)"

echo
echo "=== Train LLaVA-Next SAE on layer 22 ==="
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_train.py \
    --activations-dir outputs/cross_model_llava_next_l22_capture/vision_activations \
    --predictions outputs/cross_model_llava_next_capture_20260426-110246_621a66ff/predictions_with_pmr.parquet \
    --layer-key vision_hidden_22 \
    --n-features 4096 \
    --tag llava_next_vis22_4096 \
    > "$LOG_DIR/llava_next_l22_train.log" 2>&1
echo "Train done at $(date +%H:%M:%S)"

echo
echo "=== Run LLaVA-Next intervention on layer 22 ==="
CUDA_VISIBLE_DEVICES=1 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/llava_next_vis22_4096 \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    --vision-block-idx 22 \
    --prompt-mode open \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset shaded/blank/both \
    --top-k-list 5,10,20,40,80,160 \
    --random-controls 3 \
    --n-stim 20 \
    --rank-by cohens_d \
    --tag llava_next_vis22_4096_full \
    > "$LOG_DIR/llava_next_l22_intervention.log" 2>&1
echo "Intervention done at $(date +%H:%M:%S)"

echo "=== GPU 1 chain complete ==="
