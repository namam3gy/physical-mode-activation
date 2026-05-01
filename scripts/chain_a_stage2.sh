#!/usr/bin/env bash
# Auto-chain: poll for A Stage 1 (PID 215511) completion, then launch A Stage 2 on cuda:1
# GPU 0 is NOT touched.

STAGE1_PID=215511
STAGE1_DIR="outputs/lmswap_run_a_stage1_20260429-213151"
LOG="outputs/chain_a_stage2.log"

cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation

echo "[chain] $(date) Polling Stage 1 PID $STAGE1_PID every 60s ..." | tee -a "$LOG"
while kill -0 $STAGE1_PID 2>/dev/null; do
    sleep 60
done
echo "[chain] $(date) Stage 1 PID $STAGE1_PID gone. Checking checkpoint ..." | tee -a "$LOG"

# Find the highest step checkpoint
CKPT=$(ls -d "$STAGE1_DIR"/step* 2>/dev/null | sort -t p -k2 -n | tail -1)
if [ -z "$CKPT" ]; then
    echo "[chain] ERROR: no checkpoint found in $STAGE1_DIR" | tee -a "$LOG"
    exit 1
fi
echo "[chain] Using checkpoint: $CKPT" | tee -a "$LOG"

# Expect step17000 — warn if lower (early exit = NaN abort)
STEP=$(basename "$CKPT" | sed 's/step//')
if [ "$STEP" -lt 16000 ]; then
    echo "[chain] WARNING: final ckpt is step $STEP (< 16000). Stage 1 may have failed." | tee -a "$LOG"
fi

STAGE2_DIR="outputs/lmswap_run_a_stage2_$(date +%Y%m%d-%H%M%S)"
echo "[chain] Launching Stage 2 -> $STAGE2_DIR on cuda:1" | tee -a "$LOG"

uv run python scripts/m_lmswap_train.py \
    --variant A \
    --stage 2 \
    --output-dir "$STAGE2_DIR" \
    --stage1-ckpt "$CKPT" \
    --max-steps 21000 \
    --device cuda:1 \
    2>&1 | tee -a "$LOG"

echo "[chain] $(date) Stage 2 done (exit $?)." | tee -a "$LOG"
