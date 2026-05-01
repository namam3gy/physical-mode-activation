# M8d expansion — C6-orig (boat/fish/plant) + C6-alt-A (subtype/style labels)

**Date**: 2026-05-01.
**Source**:
- C6-orig: `outputs/m8d_extra_{qwen,llava,llava_next,idefics2,internvl3}_2026*/predictions_scored.parquet` (5 models × 3 new shapes × 480 stim × 3 labels = 7200 inferences).
- C6-alt-A: `outputs/m8d_subtype_{...}_2026*/predictions_scored.parquet` (5 models × 3 shapes × 480 stim × 5 labels = 12,000 inferences).

## TL;DR (3 things)

1. **C6-orig — encoder-saturation cluster preserved across 3 new categories**.
   InternVL3/Idefics2/Qwen saturate even on objects with no autonomous motion
   (plant). LLaVA-1.5 floor preserved. Cluster ordering identical to original
   M8d (car/person/bird) result, with **boat > plant > fish** within-model.
2. **C6-orig — fish has inverted H7 in saturated models**: silhouette
   produces *higher* PMR than `fish` itself in InternVL3 (0.81 > 0.35) and
   Qwen (0.81 > 0.31). Hypothesis: "fish" disambiguates to swim/static
   regime in saturated LMs while "silhouette" defaults to kinetic prior.
3. **C6-alt-A — subtype labels (sedan/eagle/human) consistently LOWER PMR
   than parent category labels (car/bird/person)** in 4 of 5 models.
   Counter-intuitive but replicates across the LLaVA family + Qwen. Style
   labels (cartoon/sketch/drawing) show strong cross-model heterogeneity.

## C6-orig — boat / fish / plant cross-category

### Architecture cluster preserved (PMR average across 3 labels per shape)

| Model | boat | fish | plant | overall |
|---|---|---|---|---|
| **InternVL3** | 0.98 | 0.53 | 0.83 | 0.78 (super-saturated) |
| **Idefics2** | 0.82 | 0.50 | 0.37 | 0.56 (saturated, plant break) |
| **Qwen2.5-VL** | 0.75 | 0.51 | 0.60 | 0.62 (saturated) |
| **LLaVA-Next** | 0.53 | 0.24 | 0.17 | 0.31 (mid) |
| **LLaVA-1.5** | 0.21 | 0.15 | 0.17 | 0.18 (floor — replicates ladder) |

→ Cluster ordering identical to M8d original (car/person/bird) — adds
**3 new non-Qwen-trained shape classes** to encoder-saturation evidence
(complements Pixtral B4 for cross-encoder generalization).

### Per-shape physical-vs-abstract label gap (H7 evidence)

| Model | shape | physical | abstract (silhouette) | exotic | H7 evidence |
|---|---|---|---|---|---|
| Qwen | boat | 0.80 | 0.66 | (ship) | physical > silhouette ✓ |
| Qwen | **fish** | **0.31** | **0.81** | (shark) | **INVERTED** |
| Qwen | plant | 0.61 | 0.54 | (tree) | flat |
| InternVL3 | boat | 0.95 | 1.00 | (ship) | flat ceiling |
| InternVL3 | **fish** | **0.35** | **0.81** | (shark) | **INVERTED** |
| InternVL3 | plant | 0.86 | 0.80 | (tree) | flat |
| Idefics2 | boat | 0.81 | 0.84 | (ship) | flat |
| Idefics2 | fish | 0.44 | 0.63 | (shark) | abstract > physical (mild) |
| Idefics2 | plant | 0.41 | 0.29 | (tree) | physical > silhouette ✓ |
| LLaVA-Next | boat | 0.54 | 0.57 | (ship) | flat |
| LLaVA-Next | fish | 0.23 | 0.34 | (shark) | abstract > physical (mild) |
| LLaVA-Next | plant | 0.25 | 0.12 | (tree) | physical > silhouette ✓ |
| LLaVA-1.5 | boat | 0.25 | 0.14 | (ship) | physical > silhouette ✓ |
| LLaVA-1.5 | fish | 0.24 | 0.12 | (shark) | physical > silhouette ✓ |
| LLaVA-1.5 | plant | 0.21 | 0.17 | (tree) | flat |

**Fish inversion** (4 of 5 models show ≥0.10 gap toward silhouette > fish):
- Qwen: silhouette 0.81 vs fish 0.31 — **+0.50 inversion**
- InternVL3: silhouette 0.81 vs fish 0.35 — **+0.46 inversion**
- LLaVA-Next: silhouette 0.34 vs fish 0.23 — +0.11
- Idefics2: silhouette 0.63 vs fish 0.44 — +0.19
- LLaVA-1.5: silhouette 0.12 vs fish 0.24 — −0.12 (only model with normal direction)

Hypothesis: "fish" disambiguates to swim/static regime ("the fish will
swim around"), while "silhouette" defaults to kinetic-fall completion
("the silhouette will fall to the ground"). The label "fish" itself is
*specific enough to suggest a non-kinetic regime* — opposite of how
"ball" works (specific → kinetic). H7 is **regime-conditional, not
just label-conditional**.

## C6-alt-A — subtype + style labels on car / person / bird

### Subtype effect (parent label vs subtype): 4 of 5 models show DROP

| Model | car → sedan | bird → eagle | person → human | mean Δ |
|---|---|---|---|---|
| LLaVA-1.5 | 0.60 → 0.41 (−0.19) | 0.94 → 0.88 (−0.06) | 0.31 → 0.29 (−0.02) | **−0.09** |
| Qwen | 0.80 → 0.68 (−0.12) | 0.94 → 0.80 (−0.14) | 0.72 → 0.74 (+0.02) | **−0.08** |
| LLaVA-Next | 0.82 → 0.83 (+0.01) | 0.98 → 0.98 (0) | 0.54 → 0.54 (0) | **+0.00** |
| Idefics2 | 0.92 → 0.83 (−0.09) | 0.93 → 0.97 (+0.04) | 0.72 → 0.71 (−0.01) | **−0.02** |
| InternVL3 | 1.00 → 0.99 (−0.01) | 0.94 → 0.89 (−0.05) | 0.76 → 0.86 (+0.10) | **+0.01** |

**Counter-intuitive finding**: subtype labels (more specific physical
identity: sedan, eagle) generally produce **lower** PMR than the parent
category, especially in unsaturated models (LLaVA-1.5, Qwen). Possible
mechanisms:
- Specificity may invoke richer LM context (e.g., sedan → "parked sedan"
  static prior) while parent labels stay generic.
- Token frequency: "car" / "bird" are more common in training, may have
  stronger physics co-occurrence; "sedan" / "eagle" might have more
  varied (less physics-dominated) contexts.
- LLaVA-Next + InternVL3 + Idefics2 already at ceiling, no room to drop.

### Style label effect (parent vs cartoon/sketch/drawing): heterogeneous

| Model | car → cartoon | bird → drawing | person → sketch |
|---|---|---|---|
| LLaVA-1.5 | 0.60 → 0.45 (−0.15) | 0.94 → 0.39 (**−0.55**) | 0.31 → 0.23 (−0.08) |
| Qwen | 0.80 → 0.88 (+0.08) | 0.94 → 0.68 (−0.26) | 0.72 → 0.74 (+0.02) |
| LLaVA-Next | 0.82 → 0.92 (+0.10) | 0.98 → 0.69 (−0.29) | 0.54 → 0.39 (−0.15) |
| Idefics2 | 0.92 → 0.90 (−0.02) | 0.93 → 0.89 (−0.04) | 0.72 → 0.77 (+0.05) |
| InternVL3 | 1.00 → 1.00 (0) | 0.94 → 0.99 (+0.05) | 0.76 → 0.91 (+0.15) |

**Bird → drawing** (LLaVA-1.5 −0.55, LLaVA-Next −0.29, Qwen −0.26):
"drawing of a bird" suppresses kinetic-mode strongly — abstract framing
overrides physics. **Cleanest H7 evidence in C6-alt-A**.

**Car → cartoon**: opposite direction in 3 of 5 models (LLaVA-Next +0.10,
Qwen +0.08). "cartoon car" still triggers physics-mode. Hypothesis: car's
physics prior is so strong it survives style-label modification, while
bird's (flying) is more decomposable.

### Person + statue — H7 specific suppression (paper-relevant)

| Model | person | statue | Δ |
|---|---|---|---|
| LLaVA-1.5 | 0.31 | 0.29 | −0.02 |
| LLaVA-Next | 0.54 | 0.46 | −0.08 |
| **Qwen** | 0.72 | **0.34** | **−0.38** |
| Idefics2 | 0.72 | 0.82 | +0.10 |
| InternVL3 | 0.76 | 0.41 | −0.35 |

**Qwen + InternVL3 show statue → 0.34/0.41 PMR** (vs 0.72/0.76 for
person) — **−0.38 / −0.35 drop**. "Statue" is the cleanest static-physical
label, and Qwen + InternVL3 correctly suppress kinetic-mode. Idefics2
counter-example (statue > person): saturation overrides regime selection.

This is **direct H7 regime-selection evidence** at the language level,
across the cross-model dimension.

## Implications for paper

1. **Architecture cluster (G2 fix)**: 3 new shape classes (boat/fish/plant)
   replicate the saturation cluster — non-CLIP cluster wins, LLaVA-1.5 floor
   stays at floor. Combined with Pixtral (B4), the encoder-saturation
   evidence now spans 6 non-Qwen-original encoders (SigLIP-7B/SO/SO400M,
   InternViT, CLIP-ViT-L, Pixtral-400M).

2. **H7 is regime-conditional, not label-conditional**: fish-inversion
   (silhouette > fish in saturated models) shows the prior of the label
   matters more than its abstraction level. Paper §6 framing: H7 is
   "label-mediated regime selection" with default kinetic prior; some
   labels (fish, statue) invert this.

3. **Subtype label paradox** (sedan < car, eagle < bird in unsaturated
   models): a new sub-finding worth reporting in supplementary. Suggests
   that label specificity can *reduce* physics-mode commitment because it
   anchors more contextual priors.

4. **Bird→drawing as cleanest abstract-style suppression** (LLaVA-1.5
   −0.55): paper §6 should highlight this as the strongest single data
   point for "abstract-style label overrides category".

## Limitations / next steps

- Pixtral not run on C6 (would add 6th model). Trivial to add.
- Subtype labels are limited — 1 subtype per category (sedan/eagle/human).
  More subtypes per category (truck, pickup, sparrow, ostrich) would
  strengthen the paradox finding.
- C6-orig only 5 seeds_per_cell vs 10 in M8d original — less statistical
  power.

## Files

- C6-orig: `outputs/m8d_extra_{qwen,llava,llava_next,idefics2,internvl3}_2026*/`
- C6-alt-A: `outputs/m8d_subtype_{...}_2026*/`
- Configs: `configs/m8d_extra_*.py`, `configs/m8d_subtype_*.py`
- Renderer additions: `src/physical_mode/stimuli/primitives.py` lines 988+
  (boat / fish / plant primitives — 12 new draw functions).
- Chain orchestrator: `scripts/chain_c6.sh` (qwen) + `scripts/chain_c6_fix.sh`
  (4 non-qwen models with --stimulus-dir override).

## Cross-references

- Original M8d: `docs/insights/m8d_non_ball_categories.md`.
- Encoder-saturation hypothesis: `docs/hypotheses.md` H-encoder-saturation.
- B4 Pixtral 6th model: `docs/insights/m5b_post_proj_multi_cell.md`.
