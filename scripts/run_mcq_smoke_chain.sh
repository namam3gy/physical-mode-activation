#!/usr/bin/env bash
# M-MP MCQ Phase 1 stratified smoke (audit follow-up) — 5 models on GPU 1.
# Stratified subset: 48 cells × 1 seed × 3 labels × 1 prompt = 144 inferences per model.
# Expected wall-clock per model: ~1-3 min. Total chain: ~10-15 min.
# Pre-committed parse-rate threshold (advisor 2026-04-28): ≥85% parseable per model.

set -euo pipefail

cd "$(dirname "$0")/.."

STIM_DIR="inputs/m_mp_smoke_strat"

for cfg in mcq_qwen mcq_llava mcq_llava_next mcq_idefics2 mcq_internvl3; do
    echo "=== ${cfg} smoke (chain) ==="
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "configs/${cfg}.py" \
        --stimulus-dir "${STIM_DIR}" 2>&1 | tail -3
    echo
done
echo "Chain done."
