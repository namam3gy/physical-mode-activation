# §4.11 — Label-regime distribution (5-model M8d) (run log, 2026-04-26)

- **Command**: `uv run python scripts/sec4_11_regime_distribution.py`
- **Output**: `outputs/sec4_11_regime_distribution.csv` (long-form regime fractions)
- **Figures**: `docs/figures/sec4_11_regime_distribution_5model.png`, `_4model.png`
- **Setup**: analysis-only over existing M8d label-free + labeled runs (5 models × 480 stim, horizontal-event subset).
- **No new GPU run**.
- **Deep dive**: `docs/insights/sec4_11_regime_distribution.md`.

## Method

Apply `classify_regime` (from `src/physical_mode/metrics/pmr.py`, extended in M8d to handle car/person/bird with category-specific keyword sets) to all 4-then-5-model M8d runs. For each (model × category × label_role) cell on the **horizontal-event** subset, compute fraction of responses classified as kinetic / static / abstract / ambiguous.

## Headline

- **LLaVA-1.5** cleanly selects regime by label:
  - `person × no label`: 40 % kinetic + 40 % static
  - `person × physical label`: 62 % kinetic
  - `car × abstract / silhouette`: 28 % kinetic + 70 % ambiguous
- **Saturated models (Qwen / Idefics2 / InternVL3)** kinetic everywhere except `person × exotic` (statue):
  - Qwen: ~30 % static
  - **InternVL3: ~65 % static** (strongest single label-driven static commit in the project — saturated-encoder architectures defer to language when the label uniquely disambiguates)
- **LLaVA-Next**: intermediate, 3-way split on `person × exotic` (30 % kinetic + 25 % static + 25 % abstract).

## Reading

5-model gradient is the granular form of M9's H7 finding. Saturation modulates how strongly the label can override the visual prior — fully-saturated models still defer to a label that uniquely names a static object (statue), but otherwise the kinetic prior wins.

## Caveats

1. `classify_regime` is keyword-based with 5.6 % hand-annotation error per the M8d insight doc.
2. n ≈ 40 per cell — ±5 pp swings are noise.
