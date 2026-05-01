#!/usr/bin/env bash
# Chain B3 + B5: 5-model n=30 intervention + Qwen 32B FC label-free + 32B post-proj SAE.
#
# Order rationale: B3 (n=30 ladder strengthening) is highest paper-impact / shortest.
# B5.1 (32B FC) extends §4.8. B5.2 (32B post-proj) extends encoder-vs-LM dissociation
# to 32B scale.
#
# Usage:
#   nohup bash scripts/chain_b5_b3.sh > outputs/chain_b5_b3.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_b5_b3.log
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

#============ B3: 5-model intervention with n=30 on circle/filled/blank+both ============
log "==== B3 — 5-model n=30 intervention ===="

# Each model: tag, sae-dir, model-id, hook-target, layer-key
declare -a B3_MODELS=(
    "qwen|outputs/sae/qwen_post_proj_14336|Qwen/Qwen2.5-VL-7B-Instruct|merger|post_projection_visual"
    "idefics2|outputs/sae/idefics2_connector_5120|HuggingFaceM4/idefics2-8b|merger|post_projection_visual"
    "llava_next|outputs/sae/llava_next_post_proj_5120|llava-hf/llava-v1.6-mistral-7b-hf|merger|post_projection_visual"
    "llava15|outputs/sae/llava15_post_proj_5120|llava-hf/llava-1.5-7b-hf|merger|post_projection_visual"
    "internvl3|outputs/sae/internvl3_post_proj_5120|OpenGVLab/InternVL3-8B-hf|merger|post_projection_visual"
)

for spec in "${B3_MODELS[@]}"; do
    IFS='|' read -r tag sae_dir model_id hook layer_key <<< "$spec"
    out_tag="${tag}_post_proj_n30_circle_filled_blank_both"
    if [ -d "outputs/sae_intervention/$out_tag" ]; then
        log "SKIP $tag — output already exists"
        continue
    fi
    run "B3.$tag n=30 intervention" \
        uv run python scripts/sae_intervention.py \
            --sae-dir "$sae_dir" \
            --layer-key "$layer_key" \
            --hook-target "$hook" \
            --top-k-list 20,40,80,160 \
            --rank-by cohens_d \
            --random-controls 3 \
            --n-stim 30 \
            --model-id "$model_id" \
            --prompt-mode open \
            --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
            --test-subset filled/blank/both \
            --label circle \
            --device "$DEVICE" \
            --tag "$out_tag"
done

log "==== B3 done ===="

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
        run "B5.2c 32B intervention" \
            uv run python scripts/sae_intervention.py \
                --sae-dir "outputs/sae/$QWEN32B_TAG" \
                --layer-key post_projection_visual \
                --hook-target merger \
                --top-k-list 20,40,80,160 \
                --rank-by cohens_d \
                --random-controls 3 \
                --n-stim 30 \
                --model-id Qwen/Qwen2.5-VL-32B-Instruct \
                --prompt-mode open \
                --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
                --test-subset filled/blank/both \
                --label circle \
                --device "$DEVICE" \
                --tag "${QWEN32B_TAG}_circle_filled_blank_both"
    fi
fi

log "==== Chain B5+B3 DONE ===="
