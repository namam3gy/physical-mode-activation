#!/usr/bin/env bash
# GPU 0 chain v2: Idefics2 train (looser threshold) → Idefics2 intervention →
#                 InternVL3 intervention.
set -euo pipefail

LOG_DIR="outputs/sae_intervention"
mkdir -p "$LOG_DIR"

echo "=== Train Idefics2 SAE on layer 26 (--pmr-abs-threshold 0.5) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sae_train.py \
    --activations-dir outputs/cross_model_idefics2_l26_capture/vision_activations \
    --predictions outputs/cross_model_idefics2_capture_20260426-111434_49ac35be/predictions_with_pmr.parquet \
    --layer-key vision_hidden_26 \
    --n-features 4608 \
    --pmr-abs-threshold 0.5 \
    --tag idefics2_vis26_4608 \
    > "$LOG_DIR/idefics2_l26_train.log" 2>&1
echo "Idefics2 SAE train done at $(date +%H:%M:%S)"

echo
echo "=== Run Idefics2 intervention on layer 26 ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/idefics2_vis26_4608 \
    --model-id HuggingFaceM4/idefics2-8b \
    --vision-block-idx 26 \
    --prompt-mode open \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset filled/blank/both \
    --top-k-list 5,10,20,40,80,160 \
    --random-controls 3 \
    --n-stim 20 \
    --rank-by cohens_d \
    --tag idefics2_vis26_4608_full \
    > "$LOG_DIR/idefics2_l26_intervention.log" 2>&1
echo "Idefics2 intervention done at $(date +%H:%M:%S)"

echo
echo "=== Run InternVL3 intervention on layer 23 (already correct layer) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/internvl3_vis23_4096 \
    --model-id OpenGVLab/InternVL3-8B-hf \
    --vision-block-idx 23 \
    --prompt-mode open \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    --test-subset filled/blank/both \
    --top-k-list 5,10,20,40,80,160 \
    --random-controls 3 \
    --n-stim 20 \
    --rank-by cohens_d \
    --tag internvl3_vis23_4096_full \
    > "$LOG_DIR/internvl3_l23_intervention.log" 2>&1
echo "InternVL3 intervention done at $(date +%H:%M:%S)"

echo "=== GPU 0 chain v2 complete ==="
