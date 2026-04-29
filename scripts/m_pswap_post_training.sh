#!/usr/bin/env bash
# Post-training pipeline for M-PSwap (Pillar B G3 fix).
#
# Sequence:
#   1. Regression-eval gate (POPE F1 ≥ 0.70 + VQA sanity)
#   2. M2 capture on swapped variant (for v_L re-extraction)
#   3. v_L extraction (steering vectors)
#   4. §4.6 layer sweep (pixel-encodability test on Idefics2-MLP)
#   5. M5b SAE intervention (encoder-side, vision-encoder unchanged)
#
# Run: bash scripts/m_pswap_post_training.sh <ckpt_dir>
# Example: bash scripts/m_pswap_post_training.sh outputs/mpswap_run_20260429-033238/step5000

set -e

CKPT="${1:?Usage: $0 <ckpt_dir>}"
LOG_DIR="$CKPT/post_training_logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$CKPT/mlp_pool_resampler.pt" ]; then
    echo "ERROR: $CKPT/mlp_pool_resampler.pt missing — checkpoint incomplete?"
    exit 1
fi

echo "================================================================="
echo "M-PSwap post-training pipeline"
echo "  ckpt: $CKPT"
echo "  log_dir: $LOG_DIR"
echo "================================================================="

# Stage 1: Regression-eval gate
echo
echo "--- Stage 1/5: Regression-eval (POPE + VQA sanity) ---"
uv run python scripts/m_pswap_regression_eval.py \
    --ckpt "$CKPT" \
    --n-pope 9000 \
    --n-vqa 50 \
    --pope-batch-size 8 \
    2>&1 | tee "$LOG_DIR/01_regression_eval.log"

# Check the gate
F1=$(uv run python -c "
import json, sys
with open('$CKPT/regression_eval.json') as f:
    s = json.load(f)
print(s['pope']['f1'])
")
echo "POPE F1: $F1"
if uv run python -c "import sys; sys.exit(0 if float('$F1') >= 0.70 else 1)"; then
    echo "✓ Gate PASS (F1 ≥ 0.70). Continuing pipeline."
else
    echo "✗ Gate FAIL (F1 < 0.70). Halting; review logs and decide on retraining."
    exit 2
fi

# Stage 2: M2 capture on swapped variant
echo
echo "--- Stage 2/5: M2 capture on Idefics2-MLP variant ---"
# Update config to point to this ckpt before running
uv run python -c "
import re
from pathlib import Path
p = Path('configs/cross_model_idefics2_mpswap_capture.py')
text = p.read_text()
text = re.sub(r'DEFAULT_CKPT = Path\(.+?\)', f'DEFAULT_CKPT = Path(\"$CKPT\")', text)
p.write_text(text)
print('updated config DEFAULT_CKPT to $CKPT')
"
uv run python scripts/02_run_inference.py \
    --config configs/cross_model_idefics2_mpswap_capture.py \
    2>&1 | tee "$LOG_DIR/02_m2_capture.log"

# Stage 3: v_L extraction
echo
echo "--- Stage 3/5: v_L extraction ---"
uv run python scripts/m2_extract_per_model_steering.py \
    2>&1 | tee "$LOG_DIR/03_v_L_extract.log"

# Stage 4: §4.6 layer sweep
echo
echo "--- Stage 4/5: §4.6 pixel-encodability layer sweep ---"
SWAPPED_RUN=$(ls -dt outputs/cross_model_idefics2_mpswap_capture_* | head -1)
SWAPPED_NPZ="$SWAPPED_RUN/probing_steering/steering_vectors.npz"
if [ ! -f "$SWAPPED_NPZ" ]; then
    echo "ERROR: $SWAPPED_NPZ missing — v_L extraction failed?"
    exit 3
fi
TS=$(date +%Y%m%d-%H%M%S)
SEC46_OUT="outputs/sec4_6_idefics2_mpswap_layer_sweep_${TS}"
uv run python scripts/sec4_6_idefics2_layer_sweep_unified.py \
    --swapped-ckpt "$CKPT" \
    --steering-npz "$SWAPPED_NPZ" \
    --layers 5,10,15,20,25 \
    --n-seeds 5 \
    --eps 0.1 \
    --output-dir "$SEC46_OUT" \
    2>&1 | tee "$LOG_DIR/04_sec4_6_layer_sweep.log"

# Stage 5: M5b SAE intervention (encoder-side; vision encoder is frozen, so SAE applies as-is)
echo
echo "--- Stage 5/5: M5b SAE intervention on Idefics2-MLP variant ---"
# Use the existing Idefics2 SAE trained on vision-encoder layer 26 (unchanged by swap)
SAE_DIR=$(ls -dt outputs/sae/idefics2_vis26_* 2>/dev/null | head -1)
if [ -z "$SAE_DIR" ]; then
    echo "WARN: no idefics2_vis26 SAE found — skipping Stage 5."
    echo "(Run M5b SAE training on Idefics2-base first if absent.)"
else
    echo "Using SAE: $SAE_DIR"
    uv run python scripts/sae_intervention.py \
        --swapped-ckpt "$CKPT" \
        --sae-dir "$SAE_DIR" \
        --model-id HuggingFaceM4/idefics2-8b \
        --vision-block-idx 26 \
        --prompt-mode open \
        --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
        --test-subset filled/blank/both \
        --top-k-list 20,40,80,160 \
        --rank-by cohens_d \
        --tag "idefics2_mpswap_vis26_open_$(date +%Y%m%d-%H%M%S)" \
        2>&1 | tee "$LOG_DIR/05_m5b_sae.log"
fi

echo
echo "================================================================="
echo "M-PSwap post-training pipeline complete."
echo "Logs at: $LOG_DIR"
echo "================================================================="
