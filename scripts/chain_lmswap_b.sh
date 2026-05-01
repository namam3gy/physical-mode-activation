#!/usr/bin/env bash
# Chain — M-LMSwap Variant B (CLIP+Mistral-7B-Instruct-v0.2) Stage 1 → Stage 2 → regression eval.
#
# Decision: A1 (gate override + B). Variant A's step21000 line/blank/none
# baseline matches LLaVA-1.5's abstract floor (PMR=0.000), so per-cell
# A↔B Δ-PMR is the natural read even though aggregate PMR (0.869) is
# above the [0.03, 0.50] gate. See docs/insights/lmswap_a_recipe_drift.md
# "Update — step21000 result" + "Updated decision".
#
# Sequence (single H200, cuda:0, ~24h wall):
#   1. B Stage 1 — MLP-only on LCS-558K, 17K steps, lr=1e-3 (~12h).
#   2. B Stage 2 — MLP+LoRA(r=32, α=64, q/k/v/o_proj) on LLaVA-Instruct-665K,
#      21K steps, lr=2e-4 (~12h). Loads B Stage 1 MLP weights at start.
#   3. B regression eval — 480-stim M2 set, 3 gates (sanity + PMR_nolabel + line baseline).
#
# Output dirs are timestamped:
#   outputs/lmswap_run_b_stage1_<ts>/step{1000..17000}
#   outputs/lmswap_run_b_stage2_<ts>/step{1000..21000}
#   outputs/lmswap_b_regression_eval/{summary.json,regression_eval.jsonl}
#
# Usage:
#   nohup bash scripts/chain_lmswap_b.sh > outputs/chain_lmswap_b.log 2>&1 &

set -u  # error on unset vars; do NOT set -e — we want to keep going even if a stage partially completes

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_lmswap_b.log
mkdir -p outputs

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

#============ 1. B Stage 1 ============
B_S1_DIR="outputs/lmswap_run_b_stage1_$(date +%Y%m%d-%H%M%S)"
log "==== B Stage 1 -> $B_S1_DIR (cuda:0, 17K steps) ===="
run_step "B Stage 1 train" \
    uv run python scripts/m_lmswap_train.py \
        --variant B --stage 1 \
        --output-dir "$B_S1_DIR" \
        --max-steps 17000 \
        --device "$DEVICE"
B_S1_RC=$?

if [ "$B_S1_RC" -ne 0 ]; then
    log "ERROR: B Stage 1 failed (rc=$B_S1_RC) — abort chain"
    exit 1
fi

B_S1_CKPT=$(ls -d "$B_S1_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
if [ -z "$B_S1_CKPT" ]; then
    log "ERROR: no B Stage 1 checkpoint in $B_S1_DIR — abort chain"
    exit 1
fi
log "B Stage 1 final ckpt: $B_S1_CKPT"

# Buffer for GPU memory release between training processes.
sleep 30

#============ 2. B Stage 2 ============
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
    log "ERROR: B Stage 2 failed (rc=$B_S2_RC) — chain partial; run eval manually if a useful intermediate ckpt exists"
    exit 1
fi

B_S2_CKPT=$(ls -d "$B_S2_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
if [ -z "$B_S2_CKPT" ]; then
    log "ERROR: no B Stage 2 checkpoint in $B_S2_DIR — abort chain"
    exit 1
fi
log "B Stage 2 final ckpt: $B_S2_CKPT"

sleep 30

#============ 3. Regression eval ============
log "==== B regression eval (480-stim M2, 3 gates) ===="
run_step "B regression eval" \
    uv run python scripts/m_lmswap_regression_eval.py \
        --ckpt "$B_S2_CKPT" --variant B \
        --output-dir outputs/lmswap_b_regression_eval \
        --device "$DEVICE"

log "==== Chain LMSwap-B DONE ===="
