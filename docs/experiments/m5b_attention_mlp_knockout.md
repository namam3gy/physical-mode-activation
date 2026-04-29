# M5b — attention + MLP knockout, Qwen2.5-VL (run log, 2026-04-26)

- **Command**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_attention_mlp_knockout.py --n-pairs 20 --device cuda:0`
- **Output dirs**:
  - `outputs/m5b_knockout/per_pair_results.csv`
  - `outputs/m5b_knockout/per_layer_ie_attention.csv`
  - `outputs/m5b_knockout/per_layer_ie_mlp.csv`
  - `docs/figures/m5b_knockout_per_layer_ie.png`
- **Model**: Qwen2.5-VL-7B (28 LM layers).
- **Wall clock**: ~18 min on H200 (20 stim × 28 layers × 2 ablations = 1120 forward passes).
- **N stim**: 20 clean SIP stim (cue=both, baseline PMR=1) reused from `outputs/m5b_sip/manifest.csv`.
- **Deep dive**: `docs/insights/m5b_attention_mlp_knockout.md`.

## Hypothesis (necessity vs sufficiency)

Prior SIP patching (`m5b_sip_activation_patching.md`) tested *sufficiency*: does clean's hidden state at L suffice to flip corrupted? This experiment tests *necessity*: does ablating L's component break physics commitment in the clean run?

- (a) Attention knockout: zero `self_attn` output at prefill at each L individually.
- (b) MLP knockout: zero `mlp` output at prefill at each L individually.

`IE_necessity = baseline_phys_rate − ablated_phys_rate`.

## Attention knockout result

| Layer | n | ablated phys rate | IE_necessity |
|------:|--:|------------------:|-------------:|
| **All layers L0-L27** | 20 | **1.000** | **0.0** |

→ Single-layer attention is **fully redundant**: knockout never breaks physics commitment at any layer.

## MLP knockout result

| Layer | n | ablated phys rate | IE_necessity | comment |
|------:|--:|------------------:|-------------:|---------|
| L0-L7 | 20 | 1.0 | 0.0 | redundant |
| L8 | 20 | 0.6 | +0.4 | partially necessary |
| **L9** | **20** | **0.0** | **+1.0** | **uniquely fully necessary** |
| L10 | 20 | 0.4 | +0.6 | partially necessary |
| L11 | 20 | 0.6 | +0.4 | partially necessary |
| L12-L13 | 20 | 1.0 | 0.0 | redundant |
| L14 | 20 | 0.6 | +0.4 | partially necessary |
| L15-L27 | 20 | 1.0 | 0.0 | redundant |

## Headline

**L9 MLP is uniquely necessary** (0/20 retain physics when knocked out). Surrounding ring L8 / L10 / L11 / L14 partially necessary (IE=+0.4 to +0.6). Attention has no narrow IE band at single-layer resolution. Triangulating with M5b SIP (sufficiency) + M5a (steering): **L9 MLP constructs physics-mode commitment in the residual stream; L10 reads it via attention** — explains the off-by-one between M5a (steering @ L10) and M5b SIP (sufficient @ L0-L9).
