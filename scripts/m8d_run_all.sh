#!/bin/bash
# Run all four M8d inference configs in sequence on GPU 0.
# Each model is loaded fresh per config (small overhead, simpler memory profile).

set -euo pipefail

cd "$(dirname "$0")/.."

M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
echo "===== M8D_DIR: $M8D_DIR ====="

LOG=outputs/m8d_run_all.log
: > "$LOG"

run() {
    local cfg=$1
    echo "----- $(date -u +%H:%M:%S) starting $cfg -----" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
        --config "$cfg" --stimulus-dir "$M8D_DIR" 2>&1 | tee -a "$LOG"
    echo "----- $(date -u +%H:%M:%S) done with $cfg -----" | tee -a "$LOG"
}

run configs/m8d_qwen.py
run configs/m8d_qwen_label_free.py
run configs/m8d_llava.py
run configs/m8d_llava_label_free.py

echo "===== ALL DONE $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
