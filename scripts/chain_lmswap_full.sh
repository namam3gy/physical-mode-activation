#!/usr/bin/env bash
# Auto-chain: M-LMSwap Variant A Stage 2 → regression eval → branch
#
# PASS branch:
#   B Stage 1 (cuda:0, 17K) → B Stage 2 (cuda:0, 21K) → B regression eval → done
#
# FAIL branch (fallback queue, all on cuda:0):
#   1. LLaVA-1.5 post-projection SAE (capture → train → rerank → ablation eval)
#   2. Qwen2.5-VL 32B M2 label-free (§4.8 H2 cross-scale)
#   3. Pixtral 12B M8a (M-Add6 / G2 sparse non-Qwen)
#
# Each fallback is independent; if one fails the next still runs.
#
# Usage:
#   nohup bash scripts/chain_lmswap_full.sh > outputs/chain_lmswap_full.log 2>&1 &

set -u  # error on unset vars; do NOT set -e (we want to keep going on fallback fails)

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_lmswap_full.log
mkdir -p outputs

A_STAGE2_PID=512819
A_STAGE2_DIR="outputs/lmswap_run_a_stage2_20260430-105555"
GATE_DIR=outputs/lmswap_a_regression_eval

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_step() {
    local label="$1"; shift
    log "---- BEGIN: $label ----"
    log "  cmd: $*"
    "$@"
    local rc=$?
    log "---- END: $label (rc=$rc) ----"
    return $rc
}

#--- 1. Wait for Variant A Stage 2 to finish ---
log "Polling A Stage 2 PID $A_STAGE2_PID every 60s..."
while kill -0 $A_STAGE2_PID 2>/dev/null; do
    sleep 60
done
log "A Stage 2 PID $A_STAGE2_PID gone."

# Buffer for GPU memory release after training process ends.
sleep 30

# Resolve final ckpt (highest step number)
A_CKPT=$(ls -d "$A_STAGE2_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
if [ -z "$A_CKPT" ]; then
    log "ERROR: no checkpoint found in $A_STAGE2_DIR — abort chain"
    exit 1
fi
log "Final A ckpt: $A_CKPT"

#--- 2. Regression eval (gate to Variant B) ---
run_step "Variant A regression eval" \
    uv run python scripts/m_lmswap_regression_eval.py \
        --ckpt "$A_CKPT" --variant A \
        --output-dir "$GATE_DIR" \
        --device "$DEVICE"
EVAL_RC=$?

if [ "$EVAL_RC" -eq 0 ]; then
    log "==== GATE PASS — proceeding to Variant B ===="

    #--- 3a. Variant B Stage 1 ---
    B_S1_DIR="outputs/lmswap_run_b_stage1_$(date +%Y%m%d-%H%M%S)"
    log "Launching B Stage 1 -> $B_S1_DIR (cuda:0, 17K steps)"
    run_step "B Stage 1 train" \
        uv run python scripts/m_lmswap_train.py \
            --variant B --stage 1 \
            --output-dir "$B_S1_DIR" \
            --max-steps 17000 \
            --device "$DEVICE"
    B_S1_RC=$?

    if [ "$B_S1_RC" -ne 0 ]; then
        log "ERROR: B Stage 1 failed (rc=$B_S1_RC) — aborting B chain, falling back"
    else
        B_S1_CKPT=$(ls -d "$B_S1_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
        if [ -z "$B_S1_CKPT" ]; then
            log "ERROR: no B stage 1 checkpoint found in $B_S1_DIR — aborting B chain"
        else
            log "B Stage 1 final ckpt: $B_S1_CKPT"

            #--- 3b. Variant B Stage 2 ---
            B_S2_DIR="outputs/lmswap_run_b_stage2_$(date +%Y%m%d-%H%M%S)"
            log "Launching B Stage 2 from $B_S1_CKPT -> $B_S2_DIR (cuda:0, 21K steps)"
            run_step "B Stage 2 train" \
                uv run python scripts/m_lmswap_train.py \
                    --variant B --stage 2 \
                    --output-dir "$B_S2_DIR" \
                    --stage1-ckpt "$B_S1_CKPT" \
                    --max-steps 21000 \
                    --device "$DEVICE"
            B_S2_RC=$?

            if [ "$B_S2_RC" -eq 0 ]; then
                B_S2_CKPT=$(ls -d "$B_S2_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
                if [ -n "$B_S2_CKPT" ]; then
                    log "B Stage 2 final ckpt: $B_S2_CKPT"
                    run_step "B regression eval" \
                        uv run python scripts/m_lmswap_regression_eval.py \
                            --ckpt "$B_S2_CKPT" --variant B \
                            --output-dir outputs/lmswap_b_regression_eval \
                            --device "$DEVICE"
                fi
            fi
        fi
    fi

    log "==== Variant B chain complete ===="
    exit 0
fi

log "==== GATE FAIL — entering fallback queue ===="

#--- Fallback 1: LLaVA-1.5 post-projection SAE ---
FB1_DIR=outputs/post_projection_llava15
SAE_TAG=llava15_post_proj_5120
log "[Fallback 1] LLaVA-1.5 post-projection SAE intervention"

run_step "FB1.1 capture" \
    uv run python scripts/m5b_capture_post_projection_llava.py \
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
        --output-dir "$FB1_DIR" \
        --model-id llava-hf/llava-1.5-7b-hf \
        --device "$DEVICE"

if [ $? -eq 0 ]; then
    run_step "FB1.2 train SAE" \
        uv run python scripts/sae_train.py \
            --activations-dir "$FB1_DIR" \
            --predictions outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/predictions_scored.parquet \
            --layer-key post_projection_visual \
            --pmr-abs-threshold 0.5 \
            --n-features 5120 \
            --tag "$SAE_TAG" \
            --device "$DEVICE"

    if [ $? -eq 0 ]; then
        run_step "FB1.3 rerank Cohen's d" \
            uv run python scripts/sae_rerank_features.py \
                --sae-dir "outputs/sae/$SAE_TAG"

        run_step "FB1.4 top-k ablation eval" \
            uv run python scripts/sae_intervention.py \
                --sae-dir "outputs/sae/$SAE_TAG" \
                --layer-key post_projection_visual \
                --hook-target merger \
                --top-k-list 20,40,80,160 \
                --rank-by cohens_d \
                --random-controls 3 \
                --n-stim 10 \
                --model-id llava-hf/llava-1.5-7b-hf \
                --prompt-mode open \
                --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
                --test-subset filled/blank/both \
                --label ball \
                --device "$DEVICE" \
                --tag "${SAE_TAG}_intervention"
    fi
fi

#--- Fallback 2: Qwen 32B M2 label-free ---
log "[Fallback 2] Qwen2.5-VL 32B M2 label-free (§4.8 cross-scale H2)"
run_step "FB2.1 generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/m2_qwen_32b_label_free.py
run_step "FB2.2 inference" \
    uv run python scripts/02_run_inference.py --config configs/m2_qwen_32b_label_free.py
run_step "FB2.3 score" \
    bash -c 'set -e; latest=$(ls -td outputs/m2_qwen_32b_label_free_* 2>/dev/null | head -1); [ -n "$latest" ] && uv run python scripts/03_score_and_summarize.py --run-dir "$latest"'

#--- Fallback 3: Pixtral M8a ---
log "[Fallback 3] Pixtral 12B M8a (M-Add6 / G2)"
run_step "FB3.1 generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/m_add6_pixtral_m8a.py
run_step "FB3.2 inference" \
    uv run python scripts/02_run_inference.py --config configs/m_add6_pixtral_m8a.py
run_step "FB3.3 score" \
    bash -c 'set -e; latest=$(ls -td outputs/m_add6_pixtral_m8a_* 2>/dev/null | head -1); [ -n "$latest" ] && uv run python scripts/03_score_and_summarize.py --run-dir "$latest"'

log "==== Fallback queue DONE ===="
