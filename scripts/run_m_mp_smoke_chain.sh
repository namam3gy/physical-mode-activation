#!/usr/bin/env bash
# M-MP Phase 1 stratified smoke — runs the 3 remaining models sequentially on GPU 1.
# Stratified subset: 48 cells × 1 seed × 3 labels × 3 prompts = 432 inferences per model.
# Expected wall-clock per model: LLaVA-1.5 ~3 min, LLaVA-Next ~6 min, InternVL3 ~3 min.

set -euo pipefail

cd "$(dirname "$0")/.."

STIM_DIR="inputs/m_mp_smoke_strat"

for cfg in multi_prompt_llava multi_prompt_llava_next multi_prompt_internvl3; do
    echo "=== ${cfg} smoke (chain) ==="
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "configs/${cfg}.py" \
        --stimulus-dir "${STIM_DIR}" 2>&1 | tail -3
    echo
done
echo "Chain done."
