#!/usr/bin/env bash
# Chain B4 + B1: Pixtral M-Add6 + 5-model multi-cell intervention.
#
# B4 = Pixtral 6th non-Qwen model (chat template fix applied to vlm_runner.py).
# B1 = Multi-cell aggregation: filled+blank+{none, cast_shadow, motion_arrow}
#      across 5 models on label=circle (filled+blank+both already done).
#
# Total ~2-3 h GPU.
#
# Usage:
#   nohup bash scripts/chain_b4_b1.sh > outputs/chain_b4_b1.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_b4_b1.log
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

#============ B4: Pixtral M-Add6 ============
log "==== B4 — Pixtral M-Add6 6th model ===="

run "B4 Pixtral generate stim" \
    uv run python scripts/01_generate_stimuli.py --config configs/m_add6_pixtral_m8a.py
run "B4 Pixtral inference" \
    uv run python scripts/02_run_inference.py --config configs/m_add6_pixtral_m8a.py
run "B4 Pixtral score" \
    bash -c 'set -e; latest=$(ls -td outputs/m_add6_pixtral_m8a_2026* 2>/dev/null | head -1); [ -n "$latest" ] && uv run python scripts/03_score_and_summarize.py --run-dir "$latest"'

log "==== B4 done ===="

#============ B1: Multi-cell intervention (5 models × 3 new cells) ============
log "==== B1 — multi-cell intervention 5 models × 3 cells ===="

declare -a MODELS=(
    "qwen|outputs/sae/qwen_post_proj_14336|Qwen/Qwen2.5-VL-7B-Instruct"
    "idefics2|outputs/sae/idefics2_connector_5120|HuggingFaceM4/idefics2-8b"
    "llava_next|outputs/sae/llava_next_post_proj_5120|llava-hf/llava-v1.6-mistral-7b-hf"
    "llava15|outputs/sae/llava15_post_proj_5120|llava-hf/llava-1.5-7b-hf"
    "internvl3|outputs/sae/internvl3_post_proj_5120|OpenGVLab/InternVL3-8B-hf"
)

declare -a CELLS=(
    "filled/blank/none"
    "filled/blank/cast_shadow"
    "filled/blank/motion_arrow"
)

for spec in "${MODELS[@]}"; do
    IFS='|' read -r tag sae_dir model_id <<< "$spec"
    for cell in "${CELLS[@]}"; do
        cell_tag=$(echo "$cell" | tr '/' '_')
        out_tag="${tag}_post_proj_circle_${cell_tag}"
        if [ -d "outputs/sae_intervention/$out_tag" ]; then
            log "SKIP $tag $cell — output exists"
            continue
        fi
        run "B1.$tag.$cell_tag" \
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
                --test-subset "$cell" \
                --label circle \
                --device "$DEVICE" \
                --tag "$out_tag"
    done
done

log "==== Chain B4+B1 DONE ===="
