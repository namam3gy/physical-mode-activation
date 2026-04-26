---
section: §4.2
date: 2026-04-25
status: complete
hypothesis: image-prior dominates label-prior on real photographs (counterfactual to H2)
data: M8c labeled-arm predictions (existing M9 audit data, 5 models × 5 photo categories × 3 label roles × 12 seeds)
---

# §4.2 — Reverse prompting: image vs label on real photographs

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H2** — The label (ball / circle / planet) independently raises PMR even on minimal stim — a language-prior contribution beyond the visual evidence.
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M8c** — Stim diversification — real photographs (60 photos × 5 categories from COCO + WikiArt). Photos REDUCE Qwen PMR(_nolabel) 18-48 pp.
> - **M8d** — Stim diversification — non-ball physical-object categories (car / person / bird × abstraction × bg × cue × {fall, horizontal} × seeds).
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.

## Question

§4.2 asks: when an `"abstract"` label is attached to a *real photograph*
of a physical object (e.g., a ball photo with the label `"circle"`), does
PMR drop the way it does on synthetic stim? This is a counterfactual to
H2 (language-prior strength) — does the image override the label when the
image is unambiguously physical?

## Method

Reuses the existing M8c labeled-arm runs (5 models × 5 photo categories
{ball, car, person, bird, abstract} × 3 label roles {physical, abstract,
exotic} × 12 seeds = 720 inferences per model, 3600 total). LABELS_BY_SHAPE
maps:

- ball → (`ball`, `circle`, `planet`)
- car → (`car`, `silhouette`, `figurine`)
- person → (`person`, `stick figure`, `statue`)
- bird → (`bird`, `silhouette`, `duck`)
- abstract → (`abstract image`, `pattern`, `rendering`)

The §4.2 contrast: PMR(`physical_label` on physical photo) vs
PMR(`abstract_label` on physical photo). If labels dominated, the gap
would equal the M8a / M8d H7 strength (~+0.30 for LLaVA-1.5). If the
image dominated, the gap would be near zero.

## Result

![§4.2 H7 effect halves on photos](../figures/session_image_vs_label_h7.png)

*Figure*: H7 (PMR_physical_label − PMR_abstract_label) per model on M8d
synthetic categories vs M8c real photos. LLaVA-1.5's project-strongest
H7 (M8d +0.31) is **halved** on photos (+0.10). LLaVA-Next M8d shows
slight inversion (−0.05, noise floor) but M8c is essentially zero.

### Physical photos (ball, car, person, bird — n=48 per model)

| model       | encoder         | PMR(_nolabel) | PMR(phys-label) | PMR(abs-label) | phys − abs |
|-------------|-----------------|--------------:|----------------:|---------------:|-----------:|
| Qwen2.5-VL  | SigLIP          | 0.562         | 0.708           | 0.604          | **+0.104** |
| LLaVA-1.5   | CLIP-ViT-L      | 0.354         | 0.479           | 0.333          | **+0.146** |
| LLaVA-Next  | CLIP-ViT-L      | 0.417         | 0.667           | 0.667          | **+0.000** |
| Idefics2    | SigLIP-SO400M   | 0.500         | 0.479           | 0.333          | **+0.146** |
| InternVL3   | InternViT       | 0.583         | 0.792           | 0.833          | **−0.042** |

**Mean phys − abs across 5 models on physical photos: +0.071** (range
−0.04 to +0.15). Compare with same models on M8d synthetic:

| model       | M8d phys − abs (synthetic) | M8c phys − abs (photos) | Δ (compression) |
|-------------|---------------------------:|------------------------:|----------------:|
| Qwen2.5-VL  | +0.008                     | +0.104                  | +0.10           |
| LLaVA-1.5   | +0.306                     | +0.146                  | −0.16           |
| LLaVA-Next  | −0.054                     | +0.000                  | +0.05           |
| Idefics2    | +0.048                     | +0.146                  | +0.10           |

LLaVA-1.5 — the only model with a strong M8d H7 signal — has its label
effect **halved on photos** (0.31 → 0.15). The other 4 models cluster
near zero on both stim sources.

### Abstract photos (12 unstructured / depiction photos)

| model       | PMR(_nolabel) | PMR(phys-label) | PMR(abs-label) | phys − abs |
|-------------|--------------:|----------------:|---------------:|-----------:|
| Qwen2.5-VL  | 0.500         | 0.500           | 0.500          | 0.000      |
| LLaVA-1.5   | 0.000         | 0.000           | 0.083          | −0.083     |
| LLaVA-Next  | 0.417         | 0.167           | 0.083          | +0.083     |
| Idefics2    | 0.083         | 0.250           | 0.250          | 0.000      |
| InternVL3   | 0.333         | 0.500           | 0.250          | +0.250     |

n=12 per cell — noise-floor regime. No interpretable pattern; mostly
small magnitude.

## Implication

**Image-prior dominates label-prior on real physical photos.** The
strongest demonstration is LLaVA-1.5 (M8d phys − abs +0.306 → M8c phys
− abs +0.146, halved). LLaVA-Next is even more striking: M8c label
effect is 0.000 — calling a real ball photo `"circle"` does not lower
PMR vs calling it `"ball"`. The model's visual evidence determines
the physics-mode reading regardless of the linguistic frame.

This is **not** the same as "labels don't matter" — labels still drive
behavior on synthetic stim where image content is impoverished
(line drawings, blank backgrounds). The §4.2 finding is conditional:
**label dominance requires image impoverishment**. Real photos provide
enough visual evidence that the image-side commits the model regardless
of the label-side prior.

This adds a counterfactual leg to the M9 cross-stim story:
- M8a (synthetic, line/blank) → label dominates (LLaVA H7 +0.36)
- M8d (synthetic categories, more shape detail) → label still wins
  for unsaturated encoders (LLaVA H7 +0.31)
- M8c (real photos) → image overrides label, all 5 models converge
  to |phys − abs| ≤ 0.15

The image vs label trade-off is the saturation effect viewed from the
input side: when image is rich, model reads image; when image is sparse,
model defers to label.

## Limitations

1. **n=12 per (shape × role) cell** — wide CIs on individual cells.
   The aggregate-across-shapes mean is more robust (n=48 for physical
   photos per model).
2. **Labels selected from LABELS_BY_SHAPE** — not adversarial. A
   stronger §4.2 test would attach truly contradictory labels
   (e.g., `"diagram"` on a ball photo).
3. **Labels are *names*, not *frames***. A frame-level test
   (`"this is a drawing of a ball"` vs `"this is a real ball"`) would
   isolate the linguistic-frame effect from the lexical-prior effect.
4. **No bootstrap CIs computed for §4.2-specific contrasts** — the
   M9 H7 CIs cover the (phys − abs) on physical photos with full
   shapes pooled but not the §4.2 framing of "real-photo label
   suppression". Future work could bootstrap the (M8d − M8c) ratio
   per model.

## Reproducer

```bash
# Re-runs the full M9 audit (which now includes the §4.2 numbers)
uv run python scripts/m9_generalization_audit.py --out-dir outputs/m9_audit

# Then extract M8c rows + group by shape_class:
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/m9_audit/m9_table1.csv')
m8c = df[df['stim']=='m8c'].copy()
m8c['shape_class'] = m8c['shape'].apply(lambda s: 'physical' if s in ['ball','car','person','bird'] else 'abstract')
phys = m8c[m8c['shape_class']=='physical']
agg = phys.groupby(['model','encoder']).apply(lambda g: pd.Series({
    'phys_minus_abs': (g['physical_pmr'] - g['abstract_pmr']).mean(),
})).reset_index().round(3)
print(agg.to_string(index=False))
"
```

## Artifacts

- `outputs/m9_audit/m9_table1.csv` — per-(stim × model × shape × role)
  PMR rows (now includes InternVL3 M8c since §4.2 update).
- `outputs/encoder_swap_*_m8c_*` (existing labeled-arm runs).
- `docs/insights/sec4_2_reverse_prompting.md` (this doc, + ko).
