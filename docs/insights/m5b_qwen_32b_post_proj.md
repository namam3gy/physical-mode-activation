# M5b post-projection — Qwen 7B vs 32B parity (B5.2 follow-up)

**Date**: 2026-05-01.
**Source**: 10 new intervention runs in `outputs/sae_intervention/qwen_32b_post_proj_5120_*` paired against existing `outputs/sae_intervention/qwen_post_proj_*`.
**Chain**: `scripts/chain_b5_32b_parity.sh` (15 min wall, 1× H200, 32B bf16).
**Companion**: `m5b_post_proj_b9_b13_c1.md` (5-model n=30 + B9 + B13 paper-ready ladder), `m5b_post_projection_cross_model.md` (round-2 4-cell baseline).

## TL;DR (4 findings)

1. **k-threshold +20 shift at scale on circle cells** — 7B breaks at
   k=20, 32B needs k≥40. n=30 Wilson CI separation: 32B k=20 → 30/30
   PMR=1 ([0.886, 1.000]); k=40 → 0/30 PMR=0 ([0.000, 0.114]); randoms
   30/30 retain. Projector representation of physics-mode commitment
   is **more redundantly encoded** at scale.
2. **Cue sensitivity +1 cue at scale** — at 7B, single `cast_shadow`
   alone is insufficient to commit (baseline PMR=0 on both
   filled+blank+cast_shadow AND shaded+blank+cast_shadow). At 32B,
   single `cast_shadow` IS sufficient (baseline PMR=1 on both). Scale
   lowers the cue-strength threshold for commitment ignition.
3. **Ball+cast_shadow becomes projector-localizable at scale** — 7B's
   ball+cast_shadow NULLs across all k≤160; at 32B the same cell
   breaks at k≥80. Reading: 7B routes ball+cue commitment LM-side
   (not in projector); 32B routes more of it through the projector
   representation.
4. **Same-direction k threshold on ball+motion_arrow** — 7B breaks at
   k=20 (the unique 7B ball cell where projector ablation overcomes
   the physics-label prior); 32B breaks at k≥80. Same scale-redundancy
   pattern as #1, just translated to ball-prior cells.

These compose to a coherent reading: **scale moves more of the
world-type-recognition computation into the post-projection
representation, and within that representation encodes it more
redundantly**.

## Methodology

Identical to round-2 + C1: hook `model.merger` (Qwen post-projection
visual stream), train 5120-feature SAE on PMR≥0.5 abs threshold from
M2 captures, intervene with Cohen's-d-ranked top-k ablation + 3
mass-matched random controls. Single change: model = `Qwen/Qwen2.5-VL-32B-Instruct`,
SAE trained on `outputs/post_projection_qwen_32b/` from B5.2a (chain_b5
2026-05-01 14:43 → 14:47, ~4 min capture+train).

## Comparison table

`bl` = baseline PMR (no ablation); `inv` = intervention PMR (mean
across n samples); 7B values from existing round-2/B9/B13/C1 outputs.

### Circle / filled cells

| cell | 7B-bl | 7B k=20 | 7B k=40 | 7B k=80 | 7B k=160 | 32B-bl | 32B k=20 | 32B k=40 | 32B k=80 | 32B k=160 |
|---|---|---|---|---|---|---|---|---|---|---|
| filled+blank+both (n=30) | 1.00 | 0.00 ★ | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 | 0.00 |
| filled+blank+cast_shadow | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 | 0.00 |
| filled+blank+motion_arrow | 1.00 | 0.00 ★ | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 | 0.00 |
| filled+blank+none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

★ = first k at which PMR drops to 0.

### Ball cells

| cell | 7B-bl | 7B k=20 | 7B k=40 | 7B k=80 | 7B k=160 | 32B-bl | 32B k=20 | 32B k=40 | 32B k=80 | 32B k=160 |
|---|---|---|---|---|---|---|---|---|---|---|
| ball / filled+blank+none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ball / filled+blank+cast_shadow | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 ★ | 0.00 |
| ball / filled+blank+motion_arrow | 1.00 | 0.00 ★ | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 |

### Circle / shaded cells (B13)

| cell | 7B-bl | 7B k=20 | 7B k=40 | 7B k=80 | 7B k=160 | 32B-bl | 32B k=20 | 32B k=40 | 32B k=80 | 32B k=160 |
|---|---|---|---|---|---|---|---|---|---|---|
| shaded+blank+none | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| shaded+blank+cast_shadow | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 | 0.00 |
| shaded+blank+motion_arrow | 1.00 | 0.00 ★ | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 ★ | 0.00 | 0.00 |

### Random controls — specificity

All 32B cells with baseline=1 retain PMR=1.00 across `random_0/1/2`
(mass-matched k=160). At baseline=0 cells, randoms also retain PMR=0
(no spurious activation). Specificity preserved at 32B scale.

Exception: 7B `circle/shaded/blank/cast_shadow` shows random_0/1/2
all PMR=1 despite baseline=0 — random ablation flips abstract→physics
on this specific cell. Recorded as a 7B-specific scoring/SAE artifact;
does not affect the 32B scaling claim.

## Scale-axis story (paper §6 sub-claim)

The four findings together suggest:

> **At scale, the post-projection representation absorbs more of
> the physical-vs-abstract commitment, and absorbs it more
> redundantly.**

Two complementary movements:

1. **Coverage broadens**: cells that were not projector-accessible at
   7B (single `cast_shadow` cue, ball+cast_shadow) become
   projector-accessible at 32B. Reading: more of the commitment
   routes through the projector at scale, less routed through pure
   LM-side memorization of label associations.
2. **Threshold rises**: cells that *were* projector-accessible at 7B
   require ~2× more features to ablate at 32B (k=20 → k=40-80).
   Reading: the projector encoding is more polysemantic /
   distributed at scale, requiring broader ablation to suppress.

Net effect: the discriminating axis at the projector level is still
**cue intensity** (B13 finding from 7B replicates: shaded ≠ stronger
than filled), but the cue-strength threshold for activation drops
at scale (B5.2-32B finding: cast_shadow alone enough at 32B).

## Implications for paper

1. **§6.3 Mechanistic level — add scale dimension**: the
   regime-cross capacity ladder of `m5b_post_projection_cross_model.md`
   covered 5 architectures at 7B-class. The 32B addition gives a
   clean **within-architecture scale axis**: same encoder/projector/LM
   family (Qwen 7B → 32B), same SAE methodology, only model size varies.
2. **§6.4 Cross-level triangulation strengthens**: §4.8 PMR scaling
   showed aggregate PMR is invariant 7B → 32B (0.931 → 0.926); this
   M5b extension shows the **mechanism** is *not* invariant (k threshold
   shifts, cue sensitivity shifts, ball-prior accessibility shifts).
   Aggregate PMR invariance hides mechanism reorganization at scale.
3. **§9 Discussion — scale interpretation**: the 32B finding suggests
   that as VLMs scale, world-type-recognition computation migrates
   from LM-side memorization toward post-projection representation.
   This is an **implicit world-model formation** signature — the
   model develops a more localizable "what kind of input is this?"
   read at the visual-language interface as it scales.

## Caveats

- **n=10 on B9 + B13 + round-2 cells** (only C1 has n=30 Wilson CI).
  Single 32B retests at n=30 on cast_shadow-cell candidates would
  tighten the CI on findings #2 + #3 if reviewers push.
- **SAEs are independently trained per model** — features at 7B
  vs 32B are not the "same" features. Cross-scale claims rest on
  *aggregate behavior* of top-k Cohen's-d ranked features per model,
  not feature-identity correspondence.
- **Single architecture**: the scale axis here is Qwen 7B → 32B only.
  Idefics2 / LLaVA-Next don't have publicly released same-architecture
  scale variants, so this scale axis is currently Qwen-only. Mistral
  3B → 7B (LLaVA-Next family) would be the cleanest extension if the
  weights become available.
- **Ball+cast_shadow 7B NULL is a slightly weak baseline** — 7B's
  random-ablation flip on shaded+cast_shadow indicates the SAE
  features can be noisy in some cells. The ball+cast_shadow NULL
  may understate 7B's actual projector accessibility on that cell.

## Files

- New 10 cells: `outputs/sae_intervention/qwen_32b_post_proj_5120_*`.
- 32B SAE (existing): `outputs/sae/qwen_32b_post_proj_5120/`.
- Chain: `scripts/chain_b5_32b_parity.sh`.
- Chain log: `outputs/chain_b5_32b_parity.log`.

## Cross-references

- §4.8 PMR scaling (aggregate invariance): `docs/insights/sec4_8_pmr_scaling.md`.
- Round-2 5-model 4-cell ladder: `docs/insights/m5b_post_projection_cross_model.md`.
- C1 + B9 + B13 paper-ready: `docs/insights/m5b_post_proj_b9_b13_c1.md`.
- Hypothesis row: `docs/hypotheses.md` H-regime-cross (extend with scale-axis sub-claim).
- Paper draft target: `docs/paper/draft_v1.md` §6.3 + §6.4 + §9.
