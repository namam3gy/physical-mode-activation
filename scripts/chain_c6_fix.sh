#!/usr/bin/env bash
# C6 fix — re-run llava/llava_next/idefics2/internvl3 for both C6-orig and
# C6-alt-A using --stimulus-dir override (the original chain was missing this,
# causing the non-qwen configs to fail because they each have a unique
# run_name and no per-model stim dir was generated).
#
# Polls for current C6 chain to finish first, then runs the fix.
#
# Usage:
#   nohup bash scripts/chain_c6_fix.sh > outputs/chain_c6_fix.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_c6_fix.log
mkdir -p outputs

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() {
    local label="$1"; shift
    log "BEGIN $label"
    log "  cmd: $*"
    "$@"
    local rc=$?
    log "END   $label rc=$rc"
    return $rc
}

# Wait for chain_c6.sh to finish.
log "Polling for chain_c6.sh / 02_run_inference / Qwen subtype to finish..."
while pgrep -fa "chain_c6.sh|02_run_inference.*m8d_subtype_qwen" >/dev/null 2>&1; do
    sleep 30
done
log "Original C6 chain finished. Starting fix."
sleep 10

# Resolve shared stim dirs (one per variant).
EXTRA_STIM=$(ls -td inputs/m8d_extra_qwen_2026* 2>/dev/null | head -1)
SUBTYPE_STIM=$(ls -td inputs/m8d_subtype_qwen_2026* 2>/dev/null | head -1)
log "Using extra stim:   $EXTRA_STIM"
log "Using subtype stim: $SUBTYPE_STIM"

declare -a NON_QWEN=(llava llava_next idefics2 internvl3)

#============ C6-orig fix: 4 non-Qwen models with --stimulus-dir ============
log "==== C6-orig fix — 4 non-Qwen models ===="
for tag in "${NON_QWEN[@]}"; do
    run "C6-orig fix $tag" \
        uv run python scripts/02_run_inference.py \
            --config configs/m8d_extra_${tag}.py \
            --stimulus-dir "$EXTRA_STIM"
    run "C6-orig fix $tag score" \
        bash -c "set -e; latest=\$(ls -td outputs/m8d_extra_${tag}_2026* 2>/dev/null | head -1); [ -n \"\$latest\" ] && uv run python scripts/03_score_and_summarize.py --run-dir \"\$latest\""
done

#============ C6-alt-A fix: 4 non-Qwen models ============
log "==== C6-alt-A fix — 4 non-Qwen models ===="
for tag in "${NON_QWEN[@]}"; do
    run "C6-alt-A fix $tag" \
        uv run python scripts/02_run_inference.py \
            --config configs/m8d_subtype_${tag}.py \
            --stimulus-dir "$SUBTYPE_STIM"
    run "C6-alt-A fix $tag score" \
        bash -c "set -e; latest=\$(ls -td outputs/m8d_subtype_${tag}_2026* 2>/dev/null | head -1); [ -n \"\$latest\" ] && uv run python scripts/03_score_and_summarize.py --run-dir \"\$latest\""
done

log "==== Chain C6 fix DONE ===="
