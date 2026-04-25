#!/bin/bash
# Run all four M8c inference configs in sequence on GPU 0.

set -euo pipefail

cd "$(dirname "$0")/.."

M8C_DIR=$(ls -td inputs/m8c_photos_* | head -1)
echo "===== M8C_DIR: $M8C_DIR ====="

LOG=outputs/m8c_run_all.log
: > "$LOG"

run() {
    local cfg=$1
    echo "----- $(date -u +%H:%M:%S) starting $cfg -----" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
        --config "$cfg" --stimulus-dir "$M8C_DIR" 2>&1 | tee -a "$LOG"
    echo "----- $(date -u +%H:%M:%S) done with $cfg -----" | tee -a "$LOG"
}

run configs/m8c_qwen.py
run configs/m8c_qwen_label_free.py
run configs/m8c_llava.py
run configs/m8c_llava_label_free.py

echo "===== ALL DONE $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
