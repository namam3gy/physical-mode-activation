#!/usr/bin/env bash
# Chain — M-LMSwap Variant B Stage 2 retry (post Mistral double-BOS fix).
#
# Context: B Stage 2 first attempt (outputs/lmswap_run_b_stage2_20260502-062306)
# NaN'd at step 443. Root cause: format_chat[B] had a literal `<s>` prefix,
# which combined with the processor's add_special_tokens=True default produced
# double-BOS [1, 1, ...] in input_ids — an OOD pattern that destabilizes
# Mistral-LoRA training. Fix landed in m_lmswap_train.py: drop the literal
# `<s>` for B (let tokenizer auto-prepend BOS), keeping the trailing `</s>`.
#
# Stage 1 ckpt is reused as-is — Stage 1 also went through the same collate
# path but the MLP-only LR=1e-3 / no LoRA combo is far less sensitive than
# Stage 2's LoRA r=32 α=64 LR=2e-4 setup. (Stage 1 final loss ~1.85,
# trajectory smooth; safe to keep.)
#
# Sequence (single H200, cuda:0, ~12h wall):
#   1. B Stage 2 — 21K steps from outputs/lmswap_run_b_stage1_20260501-220544/step17000.
#   2. B regression eval — 480-stim M2 set, 3 gates.
#
# Usage:
#   nohup bash scripts/chain_lmswap_b_s2_retry.sh > outputs/chain_lmswap_b_s2_retry.log 2>&1 &

set -u

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_lmswap_b_s2_retry.log
mkdir -p outputs

B_S1_CKPT="outputs/lmswap_run_b_stage1_20260501-220544/step17000"

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

if [ ! -d "$B_S1_CKPT" ]; then
    log "ERROR: missing S1 ckpt at $B_S1_CKPT — abort"
    exit 1
fi
log "Using S1 ckpt: $B_S1_CKPT"

#============ 1. B Stage 2 ============
B_S2_DIR="outputs/lmswap_run_b_stage2_$(date +%Y%m%d-%H%M%S)"
log "==== B Stage 2 from $B_S1_CKPT -> $B_S2_DIR (cuda:0, 21K steps) ===="
run_step "B Stage 2 train" \
    uv run python scripts/m_lmswap_train.py \
        --variant B --stage 2 \
        --output-dir "$B_S2_DIR" \
        --stage1-ckpt "$B_S1_CKPT" \
        --max-steps 21000 \
        --device "$DEVICE"
B_S2_RC=$?

if [ "$B_S2_RC" -ne 0 ]; then
    log "ERROR: B Stage 2 failed (rc=$B_S2_RC) — abort"
    exit 1
fi

B_S2_CKPT=$(ls -d "$B_S2_DIR"/step* 2>/dev/null | sort -V | tail -1)
if [ -z "$B_S2_CKPT" ]; then
    log "ERROR: no S2 checkpoint in $B_S2_DIR — abort"
    exit 1
fi
log "S2 final ckpt: $B_S2_CKPT"

sleep 30

#============ 2. Regression eval ============
log "==== B regression eval (480-stim M2, 3 gates) ===="
run_step "B regression eval" \
    uv run python scripts/m_lmswap_regression_eval.py \
        --ckpt "$B_S2_CKPT" --variant B \
        --output-dir outputs/lmswap_b_regression_eval \
        --device "$DEVICE"

log "==== Chain LMSwap-B-S2-retry DONE ===="
