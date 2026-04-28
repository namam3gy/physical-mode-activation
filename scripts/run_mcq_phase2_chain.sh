#!/usr/bin/env bash
# M-MP MCQ Phase 2 — Full MCQ-prompt PMR chain (5 models × 1440 inferences each).
# Uses M2 mvp_full stim (480 stim × 3 labels × 1 prompt = mcq).
# Estimated wall-clock per model on GPU 1 (single prompt, no captures):
#   Qwen ~6 min / LLaVA-1.5 ~4 min / LLaVA-Next ~17 min (5-tile AnyRes)
#   Idefics2 ~7 min / InternVL3 ~10 min. Total chain ~45 min ≈ 0.75 hr.
# Runs only after Phase 1 smoke parse-rate gate passes (≥85% per model).

set -euo pipefail

cd "$(dirname "$0")/.."

STIM_DIR="inputs/mvp_full_20260424-093926_e9d79da3"

for cfg in mcq_llava mcq_qwen mcq_idefics2 mcq_internvl3 mcq_llava_next; do
    echo "=== $(date -Iseconds) ${cfg} Phase 2 (full 480 stim) ==="
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "configs/${cfg}.py" \
        --stimulus-dir "${STIM_DIR}" 2>&1 | tail -3
    echo
done
echo "=== $(date -Iseconds) MCQ Phase 2 chain done. ==="
