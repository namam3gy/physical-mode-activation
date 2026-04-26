#!/bin/bash
# M2 cross-model run chain — 3 capture runs + 2 label-free runs.
# Runs sequentially on whichever GPU CUDA_VISIBLE_DEVICES selects.
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_m2_cross_model_chain.sh

set -euo pipefail

STIM=inputs/mvp_full_20260424-093926_e9d79da3
echo "===== M2 cross-model chain start: $(date) ====="

for cfg in \
    configs/cross_model_llava_next.py \
    configs/cross_model_llava_next_label_free.py \
    configs/cross_model_idefics2.py \
    configs/cross_model_idefics2_label_free.py \
    configs/cross_model_internvl3_capture.py
do
    echo "----- $(date) — $cfg -----"
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir "$STIM"
    echo "----- $(date) — $cfg DONE -----"
done

echo "===== M2 cross-model chain done: $(date) ====="
