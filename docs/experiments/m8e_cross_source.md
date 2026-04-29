# M8e — Cross-source paired analysis (M8a synth × M8c photos × models) (run log, 2026-04-25)

- **Command**: `uv run python scripts/m8e_cross_source.py --out-dir outputs/m8e_summary`
- **Output**:
  - `outputs/m8e_summary/{m8e_synth_pmr_nolabel.csv, m8e_photo_pmr_nolabel.csv, m8e_paired_delta.csv, m8e_h7_cross_source.csv}`
  - Figure: `docs/figures/m8e_cross_source_heatmap.png` (3-panel: synthetic PMR / photo PMR / H7 photo − synthetic delta per (model × category))
- **Setup**: analysis-only — consolidates M8a + M8d + M8c into single (model × category × source_type) view.
- **No new GPU run**.
- **Deep dive**: `docs/insights/m8e_cross_source.md`.

## Headline

Paper Table 1 candidate. Cross-source PMR shifts pattern:

- Qwen photos universally **lower** PMR than synthetic
- LLaVA photos category-asymmetric (some up, some down)
- Highlights synthetic-stim minimality as a co-factor of behavioral PMR saturation

`m8e_cross_source_heatmap.png` is the "external validity" headline figure.
