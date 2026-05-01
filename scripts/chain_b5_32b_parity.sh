#!/usr/bin/env bash
# Chain — bring Qwen 32B post-projection SAE intervention to parity with 7B.
#
# State going in (2026-05-01):
#   - 32B SAE trained at outputs/sae/qwen_32b_post_proj_5120 ✓
#   - 32B intervention done on circle/filled/blank/both (n=10 only) ✓
#   - Headline n=10 finding: 32B needs k≥40 to break (k=20 retains PMR=1)
#     vs 7B which breaks at k=20. Scaling-axis sub-claim.
#
# This chain adds:
#   1. C1 n=30 — circle/filled/blank/both at n=30 (m2_seeds30 stim).
#      Lands 32B in the regime-cross capacity ladder w/ Wilson CI.
#   2. Round-2 3-cell — circle/filled/blank/{cast_shadow, motion_arrow, none}
#      at n=10. Confirms the k≥40 finding holds across cue conditions.
#   3. B9 — ball label × filled/blank/{none, cast_shadow, motion_arrow}.
#      Tests ball-prior strength at 32B (7B: only ball+motion_arrow breakable).
#   4. B13 — circle × shaded/blank/{none, cast_shadow, motion_arrow}.
#      Cue-vs-abstraction axis at 32B scale.
#
# Total ~20 min wall (single H200, 86 s per cell at n=10, ~3 min per cell at n=30).
#
# Usage:
#   nohup bash scripts/chain_b5_32b_parity.sh > outputs/chain_b5_32b_parity.log 2>&1 &

set -u
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
unset CUDA_VISIBLE_DEVICES

DEVICE=cuda:0
LOG=outputs/chain_b5_32b_parity.log
mkdir -p outputs

MODEL_ID="Qwen/Qwen2.5-VL-32B-Instruct"
SAE_DIR="outputs/sae/qwen_32b_post_proj_5120"
M2_STIM="inputs/mvp_full_20260424-093926_e9d79da3"
M2_SEEDS30="inputs/m2_seeds30_20260501-181649_e8c94bb3"

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

intervene() {
    local out_tag="$1"; local stim_dir="$2"; local n_stim="$3"
    local cell="$4"; local label="$5"
    if [ -d "outputs/sae_intervention/$out_tag" ]; then
        log "SKIP $out_tag — exists"
        return 0
    fi
    run "$out_tag" \
        uv run python scripts/sae_intervention.py \
            --sae-dir "$SAE_DIR" \
            --layer-key post_projection_visual \
            --hook-target merger \
            --top-k-list 20,40,80,160 \
            --rank-by cohens_d \
            --random-controls 3 \
            --n-stim "$n_stim" \
            --model-id "$MODEL_ID" \
            --prompt-mode open \
            --stimulus-dir "$stim_dir" \
            --test-subset "$cell" \
            --label "$label" \
            --device "$DEVICE" \
            --tag "$out_tag"
}

#============ 1. C1 n=30 ============
log "==== 32B C1 — circle/filled/blank/both n=30 (m2_seeds30) ===="
intervene "qwen_32b_post_proj_5120_circle_filled_blank_both_n30real" \
          "$M2_SEEDS30" 30 "filled/blank/both" circle

#============ 2. Round-2 3-cell extension ============
log "==== 32B round-2 — circle/filled/blank × {cast_shadow, motion_arrow, none} n=10 ===="
for cue in cast_shadow motion_arrow none; do
    intervene "qwen_32b_post_proj_5120_circle_filled_blank_${cue}" \
              "$M2_STIM" 10 "filled/blank/${cue}" circle
done

#============ 3. B9 ball mirror ============
log "==== 32B B9 — ball × filled/blank × {none, cast_shadow, motion_arrow} n=10 ===="
for cue in none cast_shadow motion_arrow; do
    intervene "qwen_32b_post_proj_5120_ball_filled_blank_${cue}" \
              "$M2_STIM" 10 "filled/blank/${cue}" ball
done

#============ 4. B13 shaded mirror ============
log "==== 32B B13 — circle × shaded/blank × {none, cast_shadow, motion_arrow} n=10 ===="
for cue in none cast_shadow motion_arrow; do
    intervene "qwen_32b_post_proj_5120_circle_shaded_blank_${cue}" \
              "$M2_STIM" 10 "shaded/blank/${cue}" circle
done

log "==== Chain B5-32B-parity DONE ===="
