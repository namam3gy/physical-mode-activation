#!/usr/bin/env bash
# Overnight chain: B (3-model §4.6 n=10 layer sweep) → A (Idefics2 + InternVL3
# §4.6 sweeps) → C (Qwen 32B M2 inference).
#
# Each stage runs sequentially with `&&` so a failure stops the chain.
# Outputs go to dated dirs. PMR scoring happens after each model sweep.
# Cross-model summary regenerated after all sweeps.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/overnight_b_a_c.sh > /tmp/overnight.log 2>&1 &
#
# Note: n=10 (not n=50) because line_blank_none_fall_*.png provides only
# 10 unique abstract baselines in M2. Wilson CI tightening goes from
# n=3 [0.44, 1.00] to n=10 [0.69, 1.00] for the v_unit side.

set -e
set -o pipefail

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation

TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="outputs/overnight_${TS}"
mkdir -p "${LOG_DIR}"

echo "=== Overnight chain started at $(date) ===" | tee -a "${LOG_DIR}/master.log"

# ============================================================================
# B: 3-model n=10 §4.6 layer sweep at L5/L10/L15/L20/L25 + PMR scoring.
# ============================================================================

# B1. Qwen 2.5-VL 7B sweep — n=10.
echo "[$(date)] B1 Qwen sweep (n=10)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_qwen_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_qwen_layer_sweep_n10_${TS} \
    > "${LOG_DIR}/b1_qwen_sweep.log" 2>&1
echo "[$(date)] B1 Qwen scoring..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_qwen_layer_sweep_n10_${TS} \
    --model-id Qwen/Qwen2.5-VL-7B-Instruct \
    > "${LOG_DIR}/b1_qwen_score.log" 2>&1

# B2. LLaVA-1.5 7B sweep — n=10.
echo "[$(date)] B2 LLaVA-1.5 sweep (n=10)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_llava15_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_llava15_layer_sweep_n10_${TS} \
    > "${LOG_DIR}/b2_llava15_sweep.log" 2>&1
echo "[$(date)] B2 LLaVA-1.5 scoring..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_llava15_layer_sweep_n10_${TS} \
    --model-id llava-hf/llava-1.5-7b-hf \
    > "${LOG_DIR}/b2_llava15_score.log" 2>&1

# B3. LLaVA-Next Mistral sweep — n=10.
echo "[$(date)] B3 LLaVA-Next sweep (n=10)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_llava_next_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_llava_next_layer_sweep_n10_${TS} \
    > "${LOG_DIR}/b3_llava_next_sweep.log" 2>&1
echo "[$(date)] B3 LLaVA-Next scoring..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_llava_next_layer_sweep_n10_${TS} \
    --model-id llava-hf/llava-v1.6-mistral-7b-hf \
    > "${LOG_DIR}/b3_llava_next_score.log" 2>&1

echo "[$(date)] B done." | tee -a "${LOG_DIR}/master.log"

# ============================================================================
# A: Idefics2 + InternVL3 §4.6 layer sweeps + PMR scoring.
# ============================================================================

# A1. Idefics2 sweep — n=10.
echo "[$(date)] A1 Idefics2 sweep (n=10)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_idefics2_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_idefics2_layer_sweep_n10_${TS} \
    > "${LOG_DIR}/a1_idefics2_sweep.log" 2>&1
echo "[$(date)] A1 Idefics2 scoring..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_idefics2_layer_sweep_n10_${TS} \
    --model-id HuggingFaceM4/idefics2-8b \
    > "${LOG_DIR}/a1_idefics2_score.log" 2>&1

# A2. InternVL3 sweep — n=10.
echo "[$(date)] A2 InternVL3 sweep (n=10)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_internvl3_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_internvl3_layer_sweep_n10_${TS} \
    > "${LOG_DIR}/a2_internvl3_sweep.log" 2>&1
echo "[$(date)] A2 InternVL3 scoring..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_internvl3_layer_sweep_n10_${TS} \
    --model-id OpenGVLab/InternVL3-8B-hf \
    > "${LOG_DIR}/a2_internvl3_score.log" 2>&1

echo "[$(date)] A done." | tee -a "${LOG_DIR}/master.log"

# ============================================================================
# Cross-model summary regen (now 5 architectures).
# Note: scripts/sec4_6_cross_model_layer_summary.py currently only loads
# Qwen + LLaVA-1.5 + LLaVA-Next directories. After A finishes, we have
# Idefics2 + InternVL3 too — the summary script needs to be extended to
# include them. For now we just regenerate the 3-model summary; manual
# extension of the summary script can happen the next day.
# ============================================================================

echo "[$(date)] Regen cross-model summary (3-model only; 5-model after manual update)..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/sec4_6_cross_model_layer_summary.py \
    > "${LOG_DIR}/cross_model_summary.log" 2>&1 || true

# ============================================================================
# C: Qwen 2.5-VL 32B M2 inference (scaling test).
# ============================================================================

echo "[$(date)] C Qwen 32B M2 inference..." | tee -a "${LOG_DIR}/master.log"
uv run python -u scripts/02_run_inference.py \
    --config configs/m2_qwen_32b.py \
    --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
    > "${LOG_DIR}/c_qwen32b_inference.log" 2>&1

echo "[$(date)] C done. Scoring..." | tee -a "${LOG_DIR}/master.log"
QWEN32B_RUN=$(ls -dt outputs/m2_qwen_32b_*/ 2>/dev/null | head -1)
if [ -n "${QWEN32B_RUN}" ]; then
    uv run python -u scripts/03_score_and_summarize.py \
        --run-dir "${QWEN32B_RUN}" \
        > "${LOG_DIR}/c_qwen32b_score.log" 2>&1 || true
fi

echo "=== Overnight chain finished at $(date) ===" | tee -a "${LOG_DIR}/master.log"
