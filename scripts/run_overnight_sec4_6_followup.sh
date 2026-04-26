#!/bin/bash
# Overnight chain (~3-4 hr GPU) for §4.6 cross-model proper test follow-up.
#
# Phase 1: M8a LM captures × 3 models (LLaVA-Next, Idefics2, InternVL3) so
#          per-model v_L has clean class balance (M2 had n_neg=1-9 saturated).
# Phase 2: Extract per-model v_L from each M8a capture.
# Phase 3: LLaVA-1.5 layer sweep §4.6 (L5, L15, L20, L25; L10 already done).
#          Tests whether the §4.6 LLaVA-1.5 null was a wrong-layer-choice
#          artifact.
# Phase 4: Generate analysis + insight doc draft.
#
# Each phase logs to outputs/overnight_sec4_6_followup_<phase>.log and
# checks for prior success before re-running. Robust to single-phase failure.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_overnight_sec4_6_followup.sh

set +e  # do NOT exit on individual command failure — chain through

LOG_BASE=outputs/overnight_sec4_6_followup
M8A_STIM=inputs/m8a_qwen_20260425-091713_8af4836f
echo "===== Overnight chain start: $(date) ====="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
mkdir -p outputs

# ---- Phase 1: M8a LM captures × 3 models ----
echo
echo "===== Phase 1: M8a LM captures (3 models, ~2 hr) ====="
for cfg in \
    configs/encoder_swap_llava_next_m8a_capture.py \
    configs/encoder_swap_idefics2_m8a_capture.py \
    configs/encoder_swap_internvl3_m8a_capture.py
do
    echo "----- $(date) — $cfg -----"
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir "$M8A_STIM" 2>&1 | tee -a "${LOG_BASE}_phase1.log"
    echo "----- $(date) — $cfg DONE (rc=$?) -----"
done

# ---- Phase 2: per-model v_L extraction from M8a captures ----
echo
echo "===== Phase 2: v_L extraction from M8a captures ====="
uv run python scripts/m8a_extract_per_model_steering.py 2>&1 | tee "${LOG_BASE}_phase2.log"

# ---- Phase 3: LLaVA-1.5 layer sweep §4.6 (L5, L15, L20, L25) ----
echo
echo "===== Phase 3: LLaVA-1.5 §4.6 layer sweep (L5, L15, L20, L25; L10 done) ====="
TS=$(date +%Y%m%d-%H%M%S)
for layer in 5 15 20 25
do
    echo "----- $(date) — LLaVA-1.5 §4.6 L${layer} -----"
    OUT_DIR="outputs/sec4_6_counterfactual_llava_L${layer}_${TS}"
    uv run python scripts/sec4_6_counterfactual_stim_llava.py \
        --layer "$layer" \
        --steering-key "v_unit_${layer}" \
        --output-dir "$OUT_DIR" 2>&1 | tee -a "${LOG_BASE}_phase3.log"
    echo "----- $(date) — L${layer} DONE -----"

    echo "----- $(date) — LLaVA-1.5 §4.6 L${layer} summarize (PMR re-inference) -----"
    uv run python scripts/sec4_6_summarize.py \
        --run-dir "$OUT_DIR" \
        --model-id "llava-hf/llava-1.5-7b-hf" 2>&1 | tee -a "${LOG_BASE}_phase3.log"
done

# ---- Phase 4: Analysis + insight doc draft ----
echo
echo "===== Phase 4: M2 vs M8a v_L comparison + LLaVA-1.5 layer-sweep summary ====="
uv run python scripts/sec4_6_followup_analyze.py 2>&1 | tee "${LOG_BASE}_phase4.log"

echo
echo "===== Overnight chain done: $(date) ====="
echo "Logs: ${LOG_BASE}_phase{1,2,3,4}.log"
echo "Insight doc draft: docs/insights/sec4_6_cross_model_m8a_followup.md"
