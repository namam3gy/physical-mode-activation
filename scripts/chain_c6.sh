#!/usr/bin/env bash
# Chain C6 — orig + alt-A.
#
# C6-orig: 5 models × {boat, fish, plant} × M8d factorial (480 stim × 3 labels = 1440 inferences each).
# C6-alt-A: 5 models × {car, person, bird} × M8d factorial × 5 labels (subtype/style added) = 2400 inferences each.
#
# Total ~2-3 h GPU sequential.
#
# Polls C3 chain (chain_c3_n30.sh) before starting via /proc state check.
#
# Usage:
#   nohup bash scripts/chain_c6.sh > outputs/chain_c6.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_c6.log
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

# Wait for C3 chain (and any other GPU 0 work) to clear.
log "Polling for GPU 0 idle (waiting for C3 to finish)..."
while true; do
    if pgrep -fa "sec4_6_.*layer_sweep_unified.py" >/dev/null 2>&1; then
        sleep 60
        continue
    fi
    used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used_mb" -lt 2000 ]; then
        log "GPU 0 idle (used=${used_mb} MB). Starting C6."
        break
    fi
    sleep 60
done

#============ Generate stim once per variant (shared by 5 models) ============
log "==== C6 generate stim ===="

run "C6-orig generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/m8d_extra_qwen.py
run "C6-alt-A generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/m8d_subtype_qwen.py

# Resolve stim dirs.
EXTRA_STIM=$(ls -td inputs/m8d_extra_qwen_2026* 2>/dev/null | head -1)
SUBTYPE_STIM=$(ls -td inputs/m8d_subtype_qwen_2026* 2>/dev/null | head -1)
log "Extra stim:   $EXTRA_STIM"
log "Subtype stim: $SUBTYPE_STIM"

#============ C6-orig: 5 models × boat/fish/plant ============
log "==== C6-orig — 5 models × boat/fish/plant ===="

declare -a MODELS=(qwen llava llava_next idefics2 internvl3)
for tag in "${MODELS[@]}"; do
    run "C6-orig $tag inference" \
        uv run python scripts/02_run_inference.py --config configs/m8d_extra_${tag}.py
    run "C6-orig $tag score" \
        bash -c "set -e; latest=\$(ls -td outputs/m8d_extra_${tag}_2026* 2>/dev/null | head -1); [ -n \"\$latest\" ] && uv run python scripts/03_score_and_summarize.py --run-dir \"\$latest\""
done
log "==== C6-orig done ===="

#============ C6-alt-A: 5 models × car/person/bird × subtype+style labels ============
log "==== C6-alt-A — 5 models × subtype+style labels ===="

for tag in "${MODELS[@]}"; do
    run "C6-alt-A $tag inference" \
        uv run python scripts/02_run_inference.py --config configs/m8d_subtype_${tag}.py
    run "C6-alt-A $tag score" \
        bash -c "set -e; latest=\$(ls -td outputs/m8d_subtype_${tag}_2026* 2>/dev/null | head -1); [ -n \"\$latest\" ] && uv run python scripts/03_score_and_summarize.py --run-dir \"\$latest\""
done
log "==== C6-alt-A done ===="

log "==== Chain C6 DONE ===="
