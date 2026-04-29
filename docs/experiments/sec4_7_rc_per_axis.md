# §4.7 — Decision-consistency per axis (5-model M8a label-free) (run log, 2026-04-26)

- **Command**: `uv run python scripts/sec4_7_rc_per_axis.py`
- **Output**: `outputs/sec4_7_rc_per_axis.csv` — per-(model × axis × level) mean / std RC
- **Figure**: `docs/figures/sec4_7_rc_per_axis.png` (3-panel bar chart, 5 models × 3 axes)
- **Setup**: analysis-only over existing M8a label-free runs (5 models × 400 stim × 5 seeds each).
- **No new GPU run** — reuses M8a outputs.
- **Deep dive**: `docs/insights/sec4_7_rc_per_axis.md`.

## Reframe

Pilot couldn't measure RC under T=0 (degenerate). M2 used T=0.7 with 5 seeds per cell; reinterpret RC as **per-axis decision stability** rather than a single global number.

## Headline (5-model)

- `cue_level=both` is the dominant decision stabilizer for 3 saturated models:
  - Qwen 0.84 → 1.00 (+16 pp RC)
  - Idefics2 0.88 → 0.99 (+11 pp)
  - InternVL3 0.89 → 0.98 (+9 pp)
- The effect inverts/vanishes for LLaVA-1.5 and LLaVA-Next (CLIP encoders).
- `bg_level=ground` is a secondary stabilizer (+3-8 pp).
- `object_level` is the weakest stabilizer.

## Reading

Saturation is not just a behavioral PMR ceiling but **also a decision-stability ceiling**. Non-CLIP models converge to the same PMR call across all 5 seeds when cues fire; CLIP-based models retain seed-level variance even under strong cues. Separate signature of H-encoder-saturation alongside M9 PMR ceiling and §4.6 pixel-encodability.
