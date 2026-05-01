#!/usr/bin/env bash
# C3 — §4.6 cross-model layer sweep at n=30 (focused on positive layers).
#
# Existing n=10 results:
#   Qwen:       L5/10/15/20/25 broad (8-10/10 each)
#   LLaVA-Next: L20+L25 (10/10), L15 borderline (3/10)
#   LLaVA-1.5:  L25 (4/10), L20 (1/10) — weak
#   Idefics2:   L25 (1/10) — anomaly
#   InternVL3:  untestable (baseline=1)
#
# Focused n=30 sweep on the positive-or-borderline layers:
#   - Qwen L10, L25 (2 canonical positions)
#   - LLaVA-Next L20, L25 (both 10/10 at n=10)
#   - LLaVA-1.5 L25 (strongest n=10 signal)
#   - Idefics2 L25 (anomaly check)
#
# Total ~3-4h GPU. Each per-model layer sweep is ~30 min.
#
# Usage:
#   nohup bash scripts/chain_c3_n30.sh > outputs/chain_c3_n30.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_c3_n30.log
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

#============ Qwen — L10 + L25 × n=30 ============
log "==== Qwen n=30 — L10 + L25 ===="
QWEN_OUT="outputs/sec4_6_qwen_layer_sweep_n30_$(date +%Y%m%d-%H%M%S)"
run "Qwen L10+L25 n=30" \
    uv run python scripts/sec4_6_qwen_layer_sweep_unified.py \
        --layers 10,25 \
        --n-seeds 30 \
        --output-dir "$QWEN_OUT" \
        --device "$DEVICE"

#============ LLaVA-Next — L20 + L25 × n=30 ============
log "==== LLaVA-Next n=30 — L20 + L25 ===="
NEXT_OUT="outputs/sec4_6_llava_next_layer_sweep_n30_$(date +%Y%m%d-%H%M%S)"
run "LLaVA-Next L20+L25 n=30" \
    uv run python scripts/sec4_6_llava_next_layer_sweep_unified.py \
        --layers 20,25 \
        --n-seeds 30 \
        --output-dir "$NEXT_OUT" \
        --device "$DEVICE"

#============ LLaVA-1.5 — L25 × n=30 ============
log "==== LLaVA-1.5 n=30 — L25 ===="
L15_OUT="outputs/sec4_6_llava15_layer_sweep_n30_$(date +%Y%m%d-%H%M%S)"
run "LLaVA-1.5 L25 n=30" \
    uv run python scripts/sec4_6_llava15_layer_sweep_unified.py \
        --layers 25 \
        --n-seeds 30 \
        --output-dir "$L15_OUT" \
        --device "$DEVICE"

#============ Idefics2 — L25 × n=30 (anomaly check) ============
log "==== Idefics2 n=30 — L25 ===="
IDF_OUT="outputs/sec4_6_idefics2_layer_sweep_n30_$(date +%Y%m%d-%H%M%S)"
run "Idefics2 L25 n=30" \
    uv run python scripts/sec4_6_idefics2_layer_sweep_unified.py \
        --layers 25 \
        --n-seeds 30 \
        --output-dir "$IDF_OUT" \
        --device "$DEVICE"

log "==== Chain C3 n=30 DONE ===="
