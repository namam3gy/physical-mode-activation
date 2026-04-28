#!/usr/bin/env bash
# M-MP Phase 2 — Full multi-prompt PMR chain (5 models × 4320 inferences each).
# Uses M2 mvp_full stim (480 stim × 3 labels × 3 prompts).
# Estimated wall-clock per model on GPU 1:
#   Qwen ~17 min / LLaVA-1.5 ~12 min / LLaVA-Next ~50 min (5-tile AnyRes)
#   Idefics2 ~20 min / InternVL3 ~30 min. Total chain ~130 min ≈ 2.2 hr.

set -euo pipefail

cd "$(dirname "$0")/.."

STIM_DIR="inputs/mvp_full_20260424-093926_e9d79da3"

# Order: cheaper models first (LLaVA-1.5 is fastest), required-pair priority intact.
# If anything fails mid-chain, we still have the upstream models' outputs.
for cfg in multi_prompt_llava multi_prompt_qwen multi_prompt_idefics2 multi_prompt_internvl3 multi_prompt_llava_next; do
    echo "=== $(date -Iseconds) ${cfg} Phase 2 (full 480 stim) ==="
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "configs/${cfg}.py" \
        --stimulus-dir "${STIM_DIR}" 2>&1 | tail -3
    echo
done
echo "=== $(date -Iseconds) Phase 2 chain done. ==="
