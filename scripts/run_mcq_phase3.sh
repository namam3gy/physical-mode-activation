#!/usr/bin/env bash
# 1a-4: Qwen MCQ Phase 3 — M5a steering + M5b SAE intervention.
# Pinned cells (audit): M5a = line/blank/none × circle × α=40
#                       M5b = shaded/ground/both × ball × Cohen's-d top-20.
# Same cells as Phase 3 yesno + describe runs → apples-to-apples.
# Expected wall: ~5 min M5a + ~10 min M5b = ~15 min total on GPU 0.
set -euo pipefail
cd "$(dirname "$0")/.."

M2_RUN_DIR="outputs/mvp_full_20260424-094103_8ae1fa3d"
M2_STIM_DIR="inputs/mvp_full_20260424-093926_e9d79da3"

echo "=== $(date -Iseconds) MCQ Phase 3 M5a (Qwen L10 α=40, line/blank/none circle) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/06_vti_steering.py \
    --run-dir "${M2_RUN_DIR}" \
    --stimulus-dir "${M2_STIM_DIR}" \
    --test-subset line/blank/none \
    --label circle \
    --prompt-variant meta_phys_mcq \
    --layers 10 \
    --alphas 0,40 \
    --temperature 0.0 \
    --output-subdir mcq_audit_l10_a40 2>&1 | tail -10

echo
echo "=== $(date -Iseconds) MCQ Phase 3 M5b (Qwen vision31 top-20, shaded/ground/both ball) ==="
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sae_intervention.py \
    --sae-dir outputs/sae/qwen_vis31_5120 \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    --vision-block-idx -1 \
    --prompt-mode meta_phys_mcq \
    --stimulus-dir "${M2_STIM_DIR}" \
    --test-subset shaded/ground/both \
    --label ball \
    --top-k-list 20 \
    --random-controls 3 \
    --n-stim 10 \
    --rank-by cohens_d \
    --tag qwen_vis31_5120_mcq_audit 2>&1 | tail -10

echo
echo "=== $(date -Iseconds) MCQ Phase 3 done. ==="
