# §4.6 — cross-model layer sweep, 5 models × n=10 (run log, 2026-04-27 → 2026-04-28)

## Combined run history

| Date | Stage | Wall clock |
|---|---|---|
| 2026-04-26 morning | Qwen-only, 5 baseline circle stim × 7 configs × 200 Adam steps | ~30 min |
| 2026-04-26 overnight | LLaVA-1.5 layer sweep (L5/15/20/25) | ~12 min |
| 2026-04-27 afternoon | Qwen + LLaVA-Next layer sweep (3-arch closure) | ~35 min |
| 2026-04-27 night | Idefics2 + InternVL3 added (5-model n=10 chain) | ~50 min |
| 2026-04-28 (Idefics2 disambiguation) | Idefics2 deeper layers L26/28/30/31 (9-layer total) | ~65 min |

## Per-model commands (n=10, ε=0.1, 200 Adam steps)

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_qwen_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 --eps 0.1 --n-steps 200 \
    --output-dir outputs/sec4_6_qwen_layer_sweep_n10_<ts>

CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_llava15_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_llava_next_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_idefics2_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...
CUDA_VISIBLE_DEVICES=0 uv run python scripts/sec4_6_internvl3_layer_sweep_unified.py \
    --layers 5,10,15,20,25 --n-seeds 10 ...

# PMR re-scoring per sweep
for d in outputs/sec4_6_*_layer_sweep_n10_*/; do
    uv run python scripts/sec4_6_summarize.py --run-dir "$d" --model-id <matching-model-id>
done

# 5-model aggregator + figure
uv run python scripts/sec4_6_cross_model_layer_summary.py
```

## Output dirs

- `outputs/sec4_6_{qwen,llava15,llava_next,idefics2,internvl3}_layer_sweep_n10_<ts>/`
- `outputs/sec4_6_idefics2_l26_31_<ts>/` (Idefics2 deeper-layer disambiguation)
- Figure: `docs/figures/sec4_6_cross_model_layer_sweep.png`

## Per-model shortcut layer profile (5-model × 5 LM layers × n=10, aggregate 250 trials)

| Model | Encoder + projector | AUC | Shortcut layers (≥ 80 % v_unit flip) | Random across all layers |
|---|---|---:|---|---|
| Qwen2.5-VL | SigLIP + MLP | 0.99 | L5/10/15/20/25 (all 5; Wilson lower bounds 0.49–0.72) | 1/50 (L10 only) |
| LLaVA-Next | CLIP + AnyRes + MLP | 0.81 | L20 + L25 (10/10 each) | 0/50 |
| LLaVA-1.5 | CLIP + MLP | 0.73 | L25 only (4/10) | 0/50 |
| Idefics2 | SigLIP-SO400M + perceiver | 0.93 | **0** at L5/10/15/20/25/26/28/30/31 (16-97 % depth) | 0/90 |
| InternVL3 | InternViT + MLP | 0.89 | untestable (baseline_pmr=1.0) | n/a |

**Aggregate random rate**: 1/250 trials across the 25 (model × layer) cells tested. **24 of 25 random-control cells = 0/10**; only Qwen L10 hit 1/10 (10 %), well below v_unit's 10/10 at the same layer.

## Headlines

1. **Pixel-encodability is architecture-conditional**: each architecture has its own shortcut LM-layer profile.
2. **Shortcut breadth correlates with encoder probe AUC for MLP-projector models**: Qwen (5 layers, AUC 0.99) > LLaVA-Next (2 layers, AUC 0.81) > LLaVA-1.5 (1 layer, AUC 0.73).
3. **Idefics2 anomaly resolved (2026-04-28)**: 0 clean shortcuts across 9 LM layers (16-97 % depth) despite v_L projection ascending +28 to +163 in every run. Wrong-relative-depth hypothesis falsified. **Perceiver-resampler is the leading remaining mechanism candidate** (Idefics2 differs from MLP-projector models on encoder + projector + AnyRes simultaneously, so the 5-model design doesn't isolate projector — controlled projector-swap test would be needed for a clean causal claim, → M-PSwap).
4. **InternVL3 protocol-saturated**: baseline PMR=1.0 under §4.6 "circle" prompt leaves no abstract baseline cells to flip.

## Deep dive

`docs/insights/sec4_6_cross_model_revised.md`.
