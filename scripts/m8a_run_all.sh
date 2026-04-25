#!/bin/bash
# Run all four M8a inference configs in sequence on GPU 1.
# Each model is loaded fresh per config (small overhead, simpler memory profile).

set -euo pipefail

cd "$(dirname "$0")/.."

M8A_DIR=$(ls -td inputs/m8a_qwen_* | head -1)
echo "===== M8A_DIR: $M8A_DIR ====="

LOG=outputs/m8a_run_all.log
: > "$LOG"

run() {
    local cfg=$1
    echo "----- $(date -u +%H:%M:%S) starting $cfg -----" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "$cfg" --stimulus-dir "$M8A_DIR" 2>&1 | tee -a "$LOG"
    echo "----- $(date -u +%H:%M:%S) done with $cfg -----" | tee -a "$LOG"
}

run configs/m8a_qwen.py
run configs/m8a_qwen_label_free.py
run configs/m8a_llava.py
run configs/m8a_llava_label_free.py

echo "===== ALL DONE $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
