# M9 — Generalization audit (paper Table 1, 3 models × 3 stim) (run log, 2026-04-25)

- **Command**: `uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit`
- **Output**:
  - `outputs/m9_audit/m9_table1.csv` — per-(stim × model × shape × role) raw PMR
  - `outputs/m9_audit/m9_summary.csv` — bootstrap CI summary
  - Figures: `docs/figures/m9_summary.png`, `docs/figures/m9_table1_heatmap.png`
- **Setup**: analysis-only — consolidates M8a (5 shapes) + M8d (3 categories) + M8c (5 photo categories) × {Qwen, LLaVA-1.5, Idefics2} into a single 9-cell paper Table 1.
- **Statistical method**: 5000 bootstrap iterations on mean PMR(_nolabel) and mean H7 paired-difference per (model × stim) cell.
- **No new GPU run**.
- **Deep dive**: `docs/insights/m9_generalization_audit.md`.

## Headlines (with bootstrap 95% CIs)

1. **Encoder family causally drives synthetic-stim PMR(\_nolabel) ceiling** — robust:
   - SigLIP CIs **[0.800, 0.917]**
   - CLIP CIs **[0.140, 0.371]**
   - Fully separated on M8a + M8d.

2. **Photos compress the encoder gap** — robust:
   - All 3 models converge into [0.183, 0.667] on M8c.
   - 5× synthetic ratio shrinks to ~1.5–2× on photos.

3. **H7 measurability is robust only on LLaVA-on-synthetic**:
   - LLaVA M8a CI [+0.30, +0.42]
   - LLaVA M8d CI [+0.25, +0.36]
   - Both separated from 0.
   - LLaVA M8c CI [−0.03, +0.23] crosses 0 (n=12 underpowered).
   - All Qwen + Idefics2 H7 CIs cross 0 except Idefics2 M8d CI [+0.000, +0.094] which just touches.

4. **LM-modulation of H7 at saturation** — *suggested only*. Idefics2 M8d H7 CI just above 0 vs Qwen M8d H7 CI crosses 0; PASS-rate gap (0.667 vs 0.333) driven by single shape (`car`) crossing the strict threshold. Demoted to flagged-future-work.

## Methodological contribution

Replaces PASS/FAIL binarization (M8a, §4.5) with **bootstrap CIs on the mean H7 delta**. Reveals that the `Qwen 1/5 PASS` pattern was a noise-floor binarization (true mean H7 = 0 with CI crossing 0), not a real "1 out of 5" finding.

## Limitations

1. M8c n=12/category underpowered for H7.
2. M8d 3 shapes thin for cross-shape variance.
3. No encoder probe AUC for Idefics2 yet at audit time (closed in M6 r3).
