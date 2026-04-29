# M5b — per-head attention knockout, Qwen2.5-VL (run log, 2026-04-27)

- **Command**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_per_head_attention_knockout.py --n-pairs 20 --layers 8,9,10,11,12,13,14 --device cuda:0`
- **Output dirs**:
  - `outputs/m5b_per_head/per_pair_results.csv`
  - `outputs/m5b_per_head/per_head_ie.csv`
  - `docs/figures/m5b_per_head_attention_ie.png` (all-zero heatmap)
- **Model**: Qwen2.5-VL-7B (28 LM layers, 28 attention heads, head_dim=128, hidden=3584).
- **Wall clock**: ~36 min on H200 (3940 forward passes total = 20 stim × 7 layers × 28 heads + 20 baselines).
- **Layer range**: L8-L14 (the partial-MLP-necessity zone from layer-level knockout).
- **Deep dive**: `docs/insights/m5b_per_head_attention_knockout.md`.

## Method

For each (layer L ∈ {8,9,10,11,12,13,14}, head h ∈ [0..27]):
- `forward_pre_hook` on `layers[L].self_attn.o_proj`
- Zero the slice `x[..., h*head_dim : (h+1)*head_dim]` at prefill (seq_len > 1) — sets head h's contribution to o_proj output to zero
- Clean per-Q-head ablation independent of GQA's K/V sharing (`num_key_value_heads=4`)
- Score letter response → A/B/C physics, D abstract

`IE_necessity_per_(L,h) = baseline_phys_rate − ablated_phys_rate`.

## Result

Baseline phys rate: 20/20 (1.000) on all 20 SIP clean stim.

| (L, h) sweep | n | ablated phys rate | IE_necessity |
|---|---:|---:|---:|
| **All 196 (layer, head) pairs** (L8–L14 × all 28 heads) | 20 each | **1.000** | **0.0** |

**Per-stim summary**: every stim has `broken = 0/196` head-ablations.

## Headline (null, the right kind)

**Every single attention head in the L8-L14 partial-necessity zone is dispensable.** Combined with M5b layer-level attention knockout (also IE=0 at every layer), attention is redundant at *both* layer and head resolution. L9 MLP constructs physics-mode commitment; L10 reads it via attention that is genuinely diffuse — no specific head matters. H10 ("2-3 narrow IE bands") fully resolves: 1 dominant L9 MLP band + 4 partial echoes; attention has *zero* narrow IE bands at any resolution.

## Limitation

Single-head ablation only. Multi-head combination ablation (top-N visual-attention heads in L9 simultaneously) might reveal cumulative necessity.
