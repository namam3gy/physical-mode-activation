#!/usr/bin/env bash
# Chain B5: Qwen 32B FC label-free + 32B post-proj SAE.
#
# B3 was attempted as a precursor but discovered that --n-stim 30 caps at
# 10 because M2 stim has seeds_per_cell=10 per cell. Replicated runs at n=10
# for Qwen + Idefics2 land in `outputs/sae_intervention/*_n30_circle_filled_blank_both/`
# (despite the n30 tag, they are effectively n=10). Real n expansion needs
# either stim regen with seeds_per_cell=30 or multi-cell aggregation.
#
# Usage:
#   nohup bash scripts/chain_b5.sh > outputs/chain_b5.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_b5.log
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

#============ B5.1: Qwen 32B FC label-free ============
log "==== B5.1 — Qwen 32B FC label-free ===="

run "B5.1 generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/fc_label_free_qwen_32b.py
run "B5.1 inference" \
    uv run python scripts/02_run_inference.py --config configs/fc_label_free_qwen_32b.py
run "B5.1 score" \
    bash -c 'set -e; latest=$(ls -td outputs/fc_label_free_qwen_32b_* 2>/dev/null | head -1); [ -n "$latest" ] && uv run python scripts/03_score_and_summarize.py --run-dir "$latest"'

log "==== B5.1 done ===="

#============ B5.2: Qwen 32B post-projection SAE ============
log "==== B5.2 — Qwen 32B post-proj SAE chain ===="

QWEN32B_CAP=outputs/post_projection_qwen_32b
QWEN32B_TAG=qwen_32b_post_proj_5120
QWEN32B_PRED=outputs/m2_qwen_32b_label_free_20260501-022944_46fb52c7/predictions_scored.parquet

run "B5.2a 32B post-proj capture" \
    uv run python scripts/m5b_capture_post_projection.py \
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
        --output-dir "$QWEN32B_CAP" \
        --model-id Qwen/Qwen2.5-VL-32B-Instruct \
        --device "$DEVICE"

if [ $? -eq 0 ]; then
    run "B5.2b 32B SAE train" \
        uv run python scripts/sae_train.py \
            --activations-dir "$QWEN32B_CAP" \
            --predictions "$QWEN32B_PRED" \
            --layer-key post_projection_visual \
            --pmr-abs-threshold 0.5 \
            --n-features 5120 \
            --tag "$QWEN32B_TAG" \
            --device "$DEVICE"

    if [ $? -eq 0 ]; then
        run "B5.2c 32B intervention circle/filled/blank+both" \
            uv run python scripts/sae_intervention.py \
                --sae-dir "outputs/sae/$QWEN32B_TAG" \
                --layer-key post_projection_visual \
                --hook-target merger \
                --top-k-list 20,40,80,160 \
                --rank-by cohens_d \
                --random-controls 3 \
                --n-stim 10 \
                --model-id Qwen/Qwen2.5-VL-32B-Instruct \
                --prompt-mode open \
                --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
                --test-subset filled/blank/both \
                --label circle \
                --device "$DEVICE" \
                --tag "${QWEN32B_TAG}_circle_filled_blank_both"
    fi
fi

log "==== Chain B5 DONE ===="
