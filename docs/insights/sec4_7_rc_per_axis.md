---
section: §4.7
date: 2026-04-26
status: complete (5-model M8a)
hypothesis: which input axis (object_level / bg_level / cue_level) stabilizes the model's decision under T=0.7 sampling?
---

# §4.7 — Per-axis decision stability (RC) on M8a

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H1** — PMR rises in an S-shape along the abstraction axis (line → filled → shaded → textured); ground introduction adds the largest single jump.
> - **H-encoder-saturation** — Behavioral PMR(_nolabel) saturation on synthetic stim is determined at the architecture level (joint encoder + LM), not encoder representational capacity alone.
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).

## Reframe

The pilot couldn't measure RC at T=0 (all responses identical, RC=1).
Under T=0.7 (the M8a setting), RC measures within-cell stability across
the 5 seeds. §4.7 asks: **which input axis is the strongest decision
stabilizer?** And does this differ across the 5 architectures?

## Method

For each (model × shape × object_level × bg_level × cue_level) cell with
n_seeds = 5, compute RC = max(count(pmr=v)) / 5 over v ∈ {0, 1}. RC = 1
means all 5 seeds agree on PMR; RC = 0.6 means 3-of-5 majority.

Per axis, average RC across all cells where that axis takes a given
level (e.g., mean RC across all cells with `cue_level=both`). 5 models,
M8a label-free arm only (n=400/model = 80 cells × 5 seeds).

## Result

![§4.7 5-model per-axis RC](../figures/sec4_7_rc_per_axis.png)

*Figure*: per-(model × axis × level) mean RC. Error bars = std across
cells. Higher RC = more decision-stable across seeds.

### Headline reads

1. **`cue_level=both` is the dominant decision stabilizer** (9–16 pp
   gain) for the 3 saturated models:
   - Qwen2.5-VL: cue=none 0.84 → cue=both **1.00** (+0.16) — perfect
     consistency under cue.
   - Idefics2: 0.88 → 0.99 (+0.11)
   - InternVL3: 0.89 → 0.98 (+0.09)

   For the 2 less-saturated models the cue effect inverts or vanishes:
   - LLaVA-1.5: 0.85 → 0.85 (no change)
   - LLaVA-Next: 0.78 → 0.78 (no change)

2. **`bg_level=ground` is a secondary stabilizer** (3–8 pp) for
   saturated models:
   - Qwen: blank 0.88 → ground 0.96 (+0.08)
   - Idefics2: 0.92 → 0.95
   - InternVL3: 0.92 → 0.96

   Inverts for LLaVA-1.5 (0.88 → 0.82, **−0.06**); barely moves for
   LLaVA-Next (0.77 → 0.80).

3. **`object_level` is the weakest stabilizer**, with model-specific
   patterns:
   - Saturated models are roughly flat across line / filled / shaded /
     textured (Qwen 0.90–0.95, Idefics2 0.92–0.96, InternVL3 0.91–1.00
     with textured = perfect).
   - LLaVA-1.5 prefers low-abstraction (line 0.90 / filled 0.88) over
     high (shaded 0.81 / textured 0.80).
   - LLaVA-Next is U-shaped (line 0.69, shaded 0.89, textured 0.78).

### Reading

The saturated models (SigLIP / SigLIP-SO400M / InternViT) commit
confidently when contextual cues are present (motion arrow + ground
plane). Once these visual cues fire, the model converges to the same
PMR call across all 5 seeds. **Saturation is not just a behavioral PMR
ceiling — it is also a decision-stability ceiling**.

LLaVA-1.5 and LLaVA-Next (CLIP encoders) retain decision noise even
under strong cues. The cue_level effect on RC is essentially zero for
these models, suggesting the CLIP+LM pipeline doesn't use the cues to
disambiguate between physics-mode and abstract-mode at the same
confidence as non-CLIP architectures.

This is a separate signature of the architecture-level reframe:
**stim-y AUC = 1.0 cross-architecture (encoder discriminability), but
decision stability under cue varies sharply (joint architecture effect)**.

## Limitations

1. **n_seeds = 5** is the bare minimum for RC; CIs are wide. The ≥10 pp
   differences (cue effect on saturated models) are robust; the 3–8 pp
   bg_level effects are suggestive.
2. **Single arm (label-free)**. Labeled arms might show different RC
   structure since labels themselves stabilize commitment.
3. **PMR is binary**. RC over a binary outcome plateaus at 1.0 quickly;
   regime-RC (consistency in regime classification) would be a more
   sensitive metric for the saturated models.
4. **No bootstrap CIs**. The error bars are within-cell std, not
   model-level resampled CI. Not strictly statistical.

## Implication for hypotheses

- **H-encoder-saturation extension**: saturation manifests as both
  PMR-ceiling and decision-stability-ceiling. The two are related —
  if the encoder fires confidently for "physics-mode" stim, the LM
  commits without seed-level variance.
- **H1 (ramp)**: object_level shows weak RC pattern. The ramp is
  PMR-direction (from low to high mean PMR), not RC-direction (most
  cells are already high-RC for saturated models).

## Reproducer

```bash
uv run python scripts/sec4_7_rc_per_axis.py
```

Outputs:
- `outputs/sec4_7_rc_per_axis.csv` — per-(model × axis × level) mean/std RC
![3-panel bar chart](../figures/sec4_7_rc_per_axis.png)

## Artifacts

- `scripts/sec4_7_rc_per_axis.py` — driver
![5-model × 3-axis figure](../figures/sec4_7_rc_per_axis.png)
- `outputs/sec4_7_rc_per_axis.csv` — underlying numbers
- `docs/insights/sec4_7_rc_per_axis.md` (this doc, + ko)
