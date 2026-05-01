#!/usr/bin/env bash
# Chain B9 + B13 + C1.
#
# B9  = ball-label mirror of B1: 5 models × {filled/blank/none,cast_shadow,motion_arrow} × label=ball.
# B13 = shaded object_level mirror: 5 models × {shaded/blank/none,cast_shadow,motion_arrow} × label=circle.
# C1  = real n=30 intervention. Regenerate M2 stim with seeds_per_cell=30,
#       then 5 models × 1 cell (filled+blank+both, label=circle) × n=30.
#
# Total ~40 min.
#
# Usage:
#   nohup bash scripts/chain_b9_b13_c1.sh > outputs/chain_b9_b13_c1.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_b9_b13_c1.log
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

# Existing M2 stim
M2_STIM="inputs/mvp_full_20260424-093926_e9d79da3"

declare -a MODELS=(
    "qwen|outputs/sae/qwen_post_proj_14336|Qwen/Qwen2.5-VL-7B-Instruct"
    "idefics2|outputs/sae/idefics2_connector_5120|HuggingFaceM4/idefics2-8b"
    "llava_next|outputs/sae/llava_next_post_proj_5120|llava-hf/llava-v1.6-mistral-7b-hf"
    "llava15|outputs/sae/llava15_post_proj_5120|llava-hf/llava-1.5-7b-hf"
    "internvl3|outputs/sae/internvl3_post_proj_5120|OpenGVLab/InternVL3-8B-hf"
)

#============ B9 — multi-cell × LABEL=BALL ============
log "==== B9 — 5 models × 3 cells × label=ball ===="

declare -a B9_CELLS=(
    "filled/blank/none"
    "filled/blank/cast_shadow"
    "filled/blank/motion_arrow"
)

for spec in "${MODELS[@]}"; do
    IFS='|' read -r tag sae_dir model_id <<< "$spec"
    for cell in "${B9_CELLS[@]}"; do
        cell_tag=$(echo "$cell" | tr '/' '_')
        out_tag="${tag}_post_proj_ball_${cell_tag}"
        if [ -d "outputs/sae_intervention/$out_tag" ]; then
            log "SKIP B9 $tag $cell — exists"
            continue
        fi
        run "B9.$tag.$cell_tag" \
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
                --stimulus-dir "$M2_STIM" \
                --test-subset "$cell" \
                --label ball \
                --device "$DEVICE" \
                --tag "$out_tag"
    done
done

log "==== B9 done ===="

#============ B13 — shaded × multi-cell × LABEL=CIRCLE ============
log "==== B13 — 5 models × shaded/blank/3-cells × label=circle ===="

declare -a B13_CELLS=(
    "shaded/blank/none"
    "shaded/blank/cast_shadow"
    "shaded/blank/motion_arrow"
)

for spec in "${MODELS[@]}"; do
    IFS='|' read -r tag sae_dir model_id <<< "$spec"
    for cell in "${B13_CELLS[@]}"; do
        cell_tag=$(echo "$cell" | tr '/' '_')
        out_tag="${tag}_post_proj_circle_${cell_tag}"
        if [ -d "outputs/sae_intervention/$out_tag" ]; then
            log "SKIP B13 $tag $cell — exists"
            continue
        fi
        run "B13.$tag.$cell_tag" \
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
                --stimulus-dir "$M2_STIM" \
                --test-subset "$cell" \
                --label circle \
                --device "$DEVICE" \
                --tag "$out_tag"
    done
done

log "==== B13 done ===="

#============ C1 — real n=30 intervention ============
log "==== C1 — regenerate M2 with seeds_per_cell=30 + n=30 intervention ===="

run "C1.0 generate stim (seeds=30)" \
    uv run python scripts/01_generate_stimuli.py --config configs/m2_seeds30.py

# Resolve newly created stim dir
NEW_STIM=$(ls -td inputs/m2_seeds30_2026* 2>/dev/null | head -1)
if [ -z "$NEW_STIM" ]; then
    log "ERROR: m2_seeds30 stim dir not found — abort C1"
else
    log "Using new stim dir: $NEW_STIM"
    for spec in "${MODELS[@]}"; do
        IFS='|' read -r tag sae_dir model_id <<< "$spec"
        out_tag="${tag}_post_proj_circle_filled_blank_both_n30real"
        if [ -d "outputs/sae_intervention/$out_tag" ]; then
            log "SKIP C1 $tag — exists"
            continue
        fi
        run "C1.$tag (n=30)" \
            uv run python scripts/sae_intervention.py \
                --sae-dir "$sae_dir" \
                --layer-key post_projection_visual \
                --hook-target merger \
                --top-k-list 20,40,80,160 \
                --rank-by cohens_d \
                --random-controls 3 \
                --n-stim 30 \
                --model-id "$model_id" \
                --prompt-mode open \
                --stimulus-dir "$NEW_STIM" \
                --test-subset filled/blank/both \
                --label circle \
                --device "$DEVICE" \
                --tag "$out_tag"
    done
fi

log "==== Chain B9+B13+C1 DONE ===="
