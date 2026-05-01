#!/usr/bin/env bash
# Phase 2: Post-projection SAE intervention extension to Idefics2 + LLaVA-Next
#
# Each model: capture → train SAE → intervention (4 cells: ball/circle × 2 cells)
# Wait for Phase 1 PID before starting.

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/phase2_post_proj_cross_model.log
PHASE1_PID="${1:-1049587}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { local label="$1"; shift; log "BEGIN $label"; "$@"; local rc=$?; log "END $label rc=$rc"; return $rc; }

# Wait for Phase 1
log "Polling Phase 1 PID $PHASE1_PID every 30s..."
while kill -0 $PHASE1_PID 2>/dev/null; do sleep 30; done
log "Phase 1 done. Starting Phase 2."
sleep 20  # GPU memory release buffer

# Common args for all 4-cell intervention runs
intervene() {
    local model_id="$1" sae_dir="$2" tag_prefix="$3"
    for label_cell in "ball:filled/blank/both" "ball:shaded/blank/none" "circle:filled/blank/both" "circle:shaded/blank/none"; do
        local label=$(echo $label_cell | cut -d: -f1)
        local cell=$(echo $label_cell | cut -d: -f2)
        local cell_tag=$(echo $cell | tr / _)
        local tag="${tag_prefix}_open_${label}_${cell_tag}"
        run "intervention $tag" \
            uv run python scripts/sae_intervention.py \
                --sae-dir "$sae_dir" \
                --layer-key post_projection_visual \
                --hook-target merger \
                --top-k-list 20,40,80,160 \
                --rank-by cohens_d \
                --random-controls 3 \
                --n-stim 10 \
                --model-id "$model_id" \
                --prompt-mode open \
                --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
                --test-subset $cell \
                --label $label \
                --device "$DEVICE" \
                --tag "$tag"
    done
}

#============ Model 1: Idefics2-8B ============
IDEFICS2_CAP_DIR=outputs/post_projection_idefics2
IDEFICS2_TAG=idefics2_connector_5120
IDEFICS2_PRED=outputs/cross_model_idefics2_label_free_20260426-112042_70fe3bfc/predictions_scored.parquet

run "Idefics2 capture" \
    uv run python scripts/m5b_capture_post_projection_llava.py \
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
        --output-dir "$IDEFICS2_CAP_DIR" \
        --model-id HuggingFaceM4/idefics2-8b \
        --projector-path model.connector \
        --device "$DEVICE"

if [ $? -eq 0 ]; then
    run "Idefics2 SAE train" \
        uv run python scripts/sae_train.py \
            --activations-dir "$IDEFICS2_CAP_DIR" \
            --predictions "$IDEFICS2_PRED" \
            --layer-key post_projection_visual \
            --pmr-abs-threshold 0.5 \
            --n-features 5120 \
            --tag "$IDEFICS2_TAG" \
            --device "$DEVICE"

    if [ $? -eq 0 ]; then
        intervene "HuggingFaceM4/idefics2-8b" "outputs/sae/$IDEFICS2_TAG" "idefics2_post_proj"
    fi
fi

#============ Model 2: LLaVA-Next-Mistral-7B ============
NEXT_CAP_DIR=outputs/post_projection_llava_next
NEXT_TAG=llava_next_post_proj_5120
NEXT_PRED=outputs/cross_model_llava_next_label_free_20260426-111126_7e6fc5aa/predictions_scored.parquet

run "LLaVA-Next capture" \
    uv run python scripts/m5b_capture_post_projection_llava.py \
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
        --output-dir "$NEXT_CAP_DIR" \
        --model-id llava-hf/llava-v1.6-mistral-7b-hf \
        --projector-path model.multi_modal_projector \
        --device "$DEVICE"

if [ $? -eq 0 ]; then
    run "LLaVA-Next SAE train" \
        uv run python scripts/sae_train.py \
            --activations-dir "$NEXT_CAP_DIR" \
            --predictions "$NEXT_PRED" \
            --layer-key post_projection_visual \
            --pmr-abs-threshold 0.5 \
            --n-features 5120 \
            --tag "$NEXT_TAG" \
            --device "$DEVICE"

    if [ $? -eq 0 ]; then
        intervene "llava-hf/llava-v1.6-mistral-7b-hf" "outputs/sae/$NEXT_TAG" "llava_next_post_proj"
    fi
fi

#============ Model 3: InternVL3 (stretch — same projector path as LLaVA) ============
INTERN_CAP_DIR=outputs/post_projection_internvl3
INTERN_TAG=internvl3_post_proj_5120
INTERN_PRED=$(ls -td outputs/cross_model_internvl3_label_free_*/predictions_scored.parquet 2>/dev/null | head -1)

if [ -z "$INTERN_PRED" ]; then
    log "SKIP InternVL3 — no predictions parquet found"
else
    run "InternVL3 capture" \
        uv run python scripts/m5b_capture_post_projection_llava.py \
            --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
            --output-dir "$INTERN_CAP_DIR" \
            --model-id OpenGVLab/InternVL3-8B-hf \
            --projector-path model.multi_modal_projector \
            --device "$DEVICE"

    if [ $? -eq 0 ]; then
        run "InternVL3 SAE train" \
            uv run python scripts/sae_train.py \
                --activations-dir "$INTERN_CAP_DIR" \
                --predictions "$INTERN_PRED" \
                --layer-key post_projection_visual \
                --pmr-abs-threshold 0.5 \
                --n-features 5120 \
                --tag "$INTERN_TAG" \
                --device "$DEVICE"

        if [ $? -eq 0 ]; then
            intervene "OpenGVLab/InternVL3-8B-hf" "outputs/sae/$INTERN_TAG" "internvl3_post_proj"
        fi
    fi
fi

log "==== Phase 2 (cross-model post-projection) DONE ===="
