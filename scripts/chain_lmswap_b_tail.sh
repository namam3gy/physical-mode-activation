#!/usr/bin/env bash
# Tail-chain for the in-flight chain_lmswap_b.sh — replaces the original chain
# bash because the original has a `sort -t p -k2 -n` bug that picks step9000
# instead of step17000 as the final S1 ckpt (verified empirically). This
# tail polls the running S1 python process (PID 1381060) until it exits,
# then resolves the correct ckpt with `sort -V` and continues to S2 + eval.
#
# Safe to run concurrently with the original chain only briefly — but the
# expected workflow is:
#   1. Launch this tail (it blocks polling on PID 1381060).
#   2. SIGKILL the original chain bash (1381044). Python child (1381060)
#      reparents to init and keeps training — no progress lost.
#   3. This tail picks up when python exits, runs S2 + eval cleanly.
#
# Usage:
#   nohup bash scripts/chain_lmswap_b_tail.sh > outputs/chain_lmswap_b_tail.log 2>&1 &

set -u

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_lmswap_b_tail.log
mkdir -p outputs

B_S1_DIR="outputs/lmswap_run_b_stage1_20260501-220544"
S1_PYTHON_PID=1381060  # python3 child of uv child of bash chain

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

#--- 1. Wait for S1 python to finish ---
log "Polling S1 python PID $S1_PYTHON_PID every 60s..."
while kill -0 $S1_PYTHON_PID 2>/dev/null; do
    sleep 60
done
log "S1 python PID $S1_PYTHON_PID gone."

# GPU memory release buffer
sleep 30

#--- 2. Resolve correct final S1 ckpt with version sort ---
B_S1_CKPT=$(ls -d "$B_S1_DIR"/step* 2>/dev/null | sort -V | tail -1)
if [ -z "$B_S1_CKPT" ]; then
    log "ERROR: no S1 checkpoint in $B_S1_DIR — abort"
    exit 1
fi
log "S1 final ckpt (sort -V): $B_S1_CKPT"

# Sanity check: the final ckpt should be step17000 (max-steps).
expected_max="$B_S1_DIR/step17000"
if [ "$B_S1_CKPT" != "$expected_max" ]; then
    log "WARN: final ckpt $B_S1_CKPT != expected $expected_max; continuing anyway"
fi

#--- 3. S2 train ---
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
log "S2 final ckpt (sort -V): $B_S2_CKPT"

sleep 30

#--- 4. Regression eval ---
log "==== B regression eval (480-stim M2, 3 gates) ===="
run_step "B regression eval" \
    uv run python scripts/m_lmswap_regression_eval.py \
        --ckpt "$B_S2_CKPT" --variant B \
        --output-dir outputs/lmswap_b_regression_eval \
        --device "$DEVICE"

log "==== Chain LMSwap-B tail DONE ===="
