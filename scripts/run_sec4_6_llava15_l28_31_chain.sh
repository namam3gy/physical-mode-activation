#!/bin/bash
# §4.6 LLaVA-1.5 deeper-layer disambiguation chain (background).
#
# Mirror of the 2026-04-28 Idefics2 L26-31 disambiguation. Tests whether
# the §4.6 LLaVA-1.5 "L25 only weak shortcut" is depth-coverage artifact
# (deeper layers might still flip) or genuine encoder-saturation null.
#
# Phase 1: M2 capture at L28/30/31 (~50-70 min on H200).
# Phase 2: Extract v_L at L28/30/31 from the new capture (~1 min).
# Phase 3: §4.6 unified layer sweep at L28/30/31, n_seeds=5, eps=0.1
#          → 30 runs (3 layers × 5 baselines × 2 configs) (~30-40 min).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_sec4_6_llava15_l28_31_chain.sh
#
# Predicted: 0/30 v_unit + 0/30 random clean shortcuts at L28/30/31
# (encoder-saturation cluster — LLaVA-1.5 CLIP-ViT-L can't pixel-encode
# regardless of depth). Null result tightens H-shortcut framing by
# closing depth coverage, mirror of Idefics2 9-layer evidence.

set +e
LOG_BASE=outputs/sec4_6_llava15_l28_31_chain
M2_STIM=inputs/mvp_full_20260424-093926_e9d79da3
mkdir -p outputs
echo "===== LLaVA-1.5 §4.6 deeper-layer chain start: $(date) ====="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# ---- Phase 1: M2 capture at L28/30/31 ----
echo
echo "===== Phase 1: LLaVA-1.5 M2 capture L28/30/31 (~50-70 min) ====="
uv run python scripts/02_run_inference.py \
    --config configs/cross_model_llava_l28_31.py \
    --stimulus-dir "$M2_STIM" 2>&1 \
    | tee "${LOG_BASE}_phase1.log"
PHASE1_RC=${PIPESTATUS[0]}
echo "----- Phase 1 done (rc=$PHASE1_RC) at $(date) -----"

# Gate: only proceed to Phase 2 if a capture dir actually exists.
CAPTURE_DIR=$(ls -dt outputs/cross_model_llava_capture_l28_31_*/ 2>/dev/null | head -1)
if [ -z "$CAPTURE_DIR" ] || [ ! -d "${CAPTURE_DIR}activations" ]; then
    echo "ABORT: Phase 1 produced no capture dir with activations; skipping Phase 2+3."
    exit 1
fi
echo "Phase 1 capture: $CAPTURE_DIR"

# ---- Phase 2: extract v_L at L28/30/31 ----
echo
echo "===== Phase 2: v_L extraction L28/30/31 ====="
uv run python scripts/sec4_6_llava15_extract_v_L_l28_31.py 2>&1 \
    | tee "${LOG_BASE}_phase2.log"
PHASE2_RC=${PIPESTATUS[0]}
echo "----- Phase 2 done (rc=$PHASE2_RC) at $(date) -----"

# Gate: only proceed to Phase 3 if steering NPZ exists.
NEW_NPZ=$(ls -t outputs/cross_model_llava_capture_l28_31_*/probing_steering/steering_vectors.npz 2>/dev/null | head -1)
if [ -z "$NEW_NPZ" ] || [ ! -f "$NEW_NPZ" ]; then
    echo "ABORT: Phase 2 produced no steering NPZ; skipping Phase 3."
    exit 1
fi

# ---- Phase 3: layer sweep at L28/30/31 ----
echo
echo "===== Phase 3: §4.6 layer sweep L28/30/31 (~30-40 min) ====="
TS=$(date +%Y%m%d-%H%M%S)
echo "Using steering vectors: $NEW_NPZ"

uv run python scripts/sec4_6_llava15_layer_sweep_unified.py \
    --layers 28,30,31 \
    --n-seeds 5 \
    --eps 0.1 \
    --steering-npz "$NEW_NPZ" \
    --output-dir "outputs/sec4_6_llava15_layer_sweep_l28_31_${TS}" 2>&1 \
    | tee "${LOG_BASE}_phase3.log"
PHASE3_RC=${PIPESTATUS[0]}
echo "----- Phase 3 done (rc=$PHASE3_RC) at $(date) -----"

echo
echo "===== Chain end: $(date) ====="
echo "Logs: ${LOG_BASE}_phase{1,2,3}.log"
echo "Sweep output: outputs/sec4_6_llava15_layer_sweep_l28_31_${TS}"
