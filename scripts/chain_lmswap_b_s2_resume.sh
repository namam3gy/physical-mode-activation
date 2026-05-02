#!/usr/bin/env bash
# Resume chain for B Stage 2 after pod restart.
#
# Context: B S2 3rd attempt (post-EOS-fix, dir lmswap_run_b_stage2_20260502-152743)
# was running healthy at step ~2200 / 21000 when pod restart was requested.
# Last saved ckpt: step2000 (or step3000 if you waited 30 min).
#
# This script resumes from the LAST saved ckpt in the existing S2 dir,
# continues training to step 21000, then runs regression eval.
#
# Usage:
#   nohup bash scripts/chain_lmswap_b_s2_resume.sh > outputs/chain_lmswap_b_s2_resume.log 2>&1 &

set -u

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
export CUDA_VISIBLE_DEVICES=3

# DEVICE=cuda:0 maps to physical GPU 3 because CUDA_VISIBLE_DEVICES filters
# the device list — the process only sees one GPU and indexes it as 0.
DEVICE=cuda:0
LOG=outputs/chain_lmswap_b_s2_resume.log
mkdir -p outputs

S2_DIR="outputs/lmswap_run_b_stage2_20260502-152743"

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

# Find latest ckpt with sort -V
RESUME_CKPT=$(ls -d "$S2_DIR"/step* 2>/dev/null | sort -V | tail -1)
if [ -z "$RESUME_CKPT" ]; then
    log "ERROR: no ckpt found in $S2_DIR — abort"
    exit 1
fi

# Extract step number from path
STEP_OFFSET=$(basename "$RESUME_CKPT" | sed 's/^step//')
log "Resume ckpt: $RESUME_CKPT (step_offset=$STEP_OFFSET)"

#============ 1. Resume B Stage 2 ============
log "==== B Stage 2 resume from $RESUME_CKPT -> $S2_DIR (cuda:0, target step 21000) ===="
run_step "B Stage 2 train (resume)" \
    uv run python scripts/m_lmswap_train.py \
        --variant B --stage 2 \
        --output-dir "$S2_DIR" \
        --resume-from "$RESUME_CKPT" \
        --step-offset "$STEP_OFFSET" \
        --max-steps 21000 \
        --device "$DEVICE"
B_S2_RC=$?

if [ "$B_S2_RC" -ne 0 ]; then
    log "ERROR: B Stage 2 resume failed (rc=$B_S2_RC) — abort"
    exit 1
fi

B_S2_CKPT=$(ls -d "$S2_DIR"/step* 2>/dev/null | sort -V | tail -1)
log "S2 final ckpt: $B_S2_CKPT"

sleep 30

#============ 2. Regression eval ============
log "==== B regression eval (480-stim M2, 3 gates) ===="
run_step "B regression eval" \
    uv run python scripts/m_lmswap_regression_eval.py \
        --ckpt "$B_S2_CKPT" --variant B \
        --output-dir outputs/lmswap_b_regression_eval \
        --device "$DEVICE"

log "==== Chain LMSwap-B-S2-resume DONE ===="
