# M5b — cross-model SIP + activation patching, LLaVA-1.5 (run log, 2026-04-26)

- **Command**: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m5b_sip_cross_model.py --model-id llava-hf/llava-1.5-7b-hf --capture-pattern "cross_model_llava_capture_*" --label ball --n-pairs 15 --model-tag llava15 --device cuda:0`
- **Output dirs**:
  - `outputs/m5b_sip_cross_model/llava15_manifest.csv` — 15 SIP pairs
  - `outputs/m5b_sip_cross_model/llava15_per_pair_results.csv`
  - `outputs/m5b_sip_cross_model/llava15_per_layer_ie.csv`
  - `docs/figures/m5b_sip_cross_model_llava15_per_layer_ie.png`
- **Models attempted**: Qwen (already done in `m5b_sip_activation_patching.md`), **LLaVA-1.5 ✓**, LLaVA-Next / Idefics2 / InternVL3 skipped (n_pos / n_neg too imbalanced for SIP — open-prompt PMR saturated at ≈1.0 leaves no clean abstract baselines: 2 / 0 / 0 candidates respectively).
- **Wall clock**: ~12 min on H200 (n=15 × 32 LM layers).
- **Class balance source**: LLaVA-1.5 M2 captures (n_pos=375, n_neg=105).
- **Deep dive**: `docs/insights/m5b_sip_cross_model.md`.

## Per-layer Indirect Effect (LLaVA-1.5, n=15 pairs, 32 LM layers)

| Layer range | IE | Notes |
|---|---:|---|
| L0–L19 | **+0.40** | 15/15 PMR=1 after patch (baseline 9/15 PMR=1 + 6/15 flipped) |
| L20 | +0.33 | first layer where patching loses some recovery |
| L21–L23 | +0.27 to +0.33 | gradual decline |
| L24–L28 | +0.13 | late-layer patching nearly useless |
| L29–L30 | −0.07 | slight negative — patching hurts |
| L31 | 0.0 | last layer, no effect |

Decision-lock-in starts at **L20 (62.5% relative depth)** vs Qwen L10 (36%). Per-pair recovery on the 6/15 genuinely-corrupted re-inference pairs: 6/6 at L0-L19 patching, progressively failing at L20+.

## Headline

**Curve shape replicates cross-model; locus shifts.** Both Qwen and LLaVA-1.5 show sharp-then-declining IE × layer profiles. Each model's decision-lock-in layer sits at a model-specific relative depth, not a fixed absolute index. H-locus reframed: cross-model exists with model-specific positions.

## Caveat

LLaVA-1.5 corrupted SIP partly drifted toward PMR=1 at re-inference (9/15 capture-time PMR=0 → re-inference PMR=1). The clean SIP signal lives on the 6/15 genuinely-corrupted pairs; the "+0.40 baseline" reflects the 9/15 already-recovered population. The IE-profile shape is robust either way.
