# §4.2 — Reverse prompting on real photos (run log, 2026-04-25)

- **Setup**: reuses M8c labeled-arm runs (5 models × 5 photo categories × 3 label roles × 12 seeds = 720 inferences/model = 3600 total).
- **No new GPU run** — analysis-only over existing outputs.
- **Driver**:
  ```bash
  uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit
  uv run python -c "
  import pandas as pd
  df = pd.read_csv('outputs/m9_audit/m9_table1.csv')
  m8c = df[df['stim']=='m8c'].copy()
  m8c['shape_class'] = m8c['shape'].apply(lambda s: 'physical' if s in ['ball','car','person','bird'] else 'abstract')
  ..."
  ```
- **Output**: `outputs/m9_audit/m9_table1.csv` — per-(stim × model × shape × role) PMR.
- **Deep dive**: `docs/insights/sec4_2_reverse_prompting.md`.

## Question

H2 inverse: when an "abstract" label (`circle / drawing / diagram`) is attached to a real photo of a physical object, does PMR drop relative to a "physical" label (`ball / car / person / bird`)? If image-prior dominates, paired delta `phys − abs` should be small on photos.

## Result (5 models × 4 physical categories × paired delta `phys_role − abs_role` PMR)

Image-prior dominates label-prior on real physical photos: paired delta `phys_minus_abs ≤ +0.146` across all 5 models, vs LLaVA-1.5's M8d synthetic delta of **+0.306** (label effect halved on photos).

**Notable cell**: LLaVA-Next `phys − abs = 0.000` on physical photos — calling a real ball `"circle"` does *not* lower PMR vs `"ball"`. The image vs label trade-off is the saturation effect viewed from the input side: rich image → image dominates; impoverished image → label dominates.

## Headline

H2 holds **conditional on visual prior strength**: synthetic stim (impoverished, encoder-saturated) → label dominates; real photos (rich, encoder-grounded) → image dominates. The trade-off curve is the saturation effect viewed from the input axis.
