# M5b — SIP + LM activation patching, Qwen2.5-VL (run log, 2026-04-26)

- **Command**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_activation_patching.py --n-pairs 20 --device cuda:0`
- **Manifest**: `outputs/m5b_sip/manifest.csv` (20 SIP pairs from M2 mvp_full predictions)
- **Output dirs**:
  - `outputs/m5b_sip/per_pair_results.csv` — per-pair × per-layer first-letter responses
  - `outputs/m5b_sip/per_layer_ie.csv` — aggregated IE per layer
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (28 LM layers)
- **Wall clock**: ~8.3 min on H200 (40 forward passes × 28 layers)
- **N pairs**: 20 (cue=both clean × cue=none corrupted, paired by seed within each (object_level, bg_level, event_template))
- **Deep dive**: `docs/insights/m5b_sip_activation_patching.md`.

## Per-layer Indirect Effect (visual-token full-state patching)

| Layer | IE | Patched physics rate (n=20) |
|---:|---:|---:|
| L0-L9 | **+1.0** | 20/20 (full recovery) |
| L10 | +0.6 | 12/20 |
| L11 | +0.6 | 12/20 |
| L12 | +0.3 | 6/20 |
| L13 | +0.1 | 2/20 |
| L14-L27 | 0.0 | 0/20 |

Baseline corrupted physics rate: 0/20 (all "D — abstract"). Clean control: 20/20 physics by SIP construction.

## Headline

**Sharp L10 decision-lock-in boundary.** Patching corrupted's visual-token hidden state with clean's full state recovers physics-mode at any L0-L9 (IE=+1.0); partial at L10/L11 (IE=+0.6); zero at L14+. Refines M5a (single-causal-layer steering at L10 α=40) into "L10 = lock-in boundary, with information present at every L0-L9".

## SIP construction

For each (object_level ∈ {line, filled, shaded, textured}) × (bg_level ∈ {blank, ground, scene}) × (event="fall") cell:
- Clean: cue=both seeds where the M2 forced-choice response is physics-mode (PMR=1, abs_rate=0).
- Corrupted: cue=none seeds where the response is abstract reject (PMR=0, abs_rate=1).
- Pair by seed index within the cell. Total: 20 pairs across 12 cells.
