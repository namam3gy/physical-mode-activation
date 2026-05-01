# M5b round 2 — post-projection SAE intervention, 5-model × 4-cell

**Date**: 2026-05-01.
**Companion**: `docs/insights/m5b_sae_intervention_cross_model.md` (round 1, encoder-side, 2026-04-28).
**Sources**: `outputs/sae_intervention/{qwen,idefics2,llava_next,llava15,internvl3}_post_proj_*` (per-cell results.csv) + `outputs/sae/*_post_proj_5120` (per-model SAE training).

## TL;DR

**Round 1 finding** (encoder-side, 2026-04-28): "3 of 5 models break PMR
cleanly; 2 LLaVA models NULL — encoder-vs-LM dissociation."

**Round 2 finding** (post-projection, this doc): the "NULL" headline
decomposes into 3 distinct phenomena, and the cross-model picture is a
**regime-cross capacity ladder**, not a binary BREAK/NULL split:

| Verdict | Models | What's actually happening |
|---|---|---|
| ★★★ clean break | Qwen2.5-VL | k=20 ablate → "remain stationary"; PMR 1→0; commitment localized at projector output |
| ★★ partial break | Idefics2, LLaVA-Next | k=20-40 ablate → break; higher k either drifts (Idefics2 "expand outward" — `continu` stem scoring artifact) or stabilizes ("expand") |
| ✦ baseline-already-abstract | LLaVA-1.5 (circle cells) | baseline PMR=0 ("drawn towards red arrow"); no physics commitment to ablate |
| ✗ true NULL | InternVL3 (circle cells), all models on ball cells | text doesn't shift across k; binary PMR doesn't shift either |

This is a **paper-level reframe** of the round-1 "NULL" headline. The
round-1 doc framed LLaVA-1.5 / LLaVA-Next as "encoder-side NULL → LM-side
routing" — that interpretation is *partially* true (no encoder-side
features to ablate), but *also* partially an artifact of the cell choice
(round 1 tested ball cells where the physics-label prior is too strong
for projector ablation to overcome regardless of architecture).

## Methodology — what changed from round 1

Round 1 hooked the **vision-encoder hidden state** (per-model layer-of-
consumption: Qwen L31, Idefics2 L26, InternVL3 L23, LLaVA-1.5 + Next L22)
and trained a 5120-feature SAE on those activations. Round 2 hooks the
**projector output** (`model.multi_modal_projector` for LLaVA-style;
`model.connector` for Idefics2) and trains a 5120-feature SAE on those.

Implementation: `scripts/m5b_capture_post_projection_llava.py` is a
generic capture script with `--projector-path` arg defaulting to
`model.multi_modal_projector`. Same `scripts/sae_train.py` pipeline,
PMR≥0.5 abs threshold + `cohens_d` re-ranking. Intervention via
`scripts/sae_intervention.py` extended to support post-projection layer
hooks for LLaVA / Idefics2.

## Per-cell results

### circle / filled / blank+both — the discriminating cell

| Model | Baseline | k=20 | k=40 | k=80 | k=160 |
|---|---|---|---|---|---|
| Qwen2.5-VL | "fall towards smaller circle" PMR=1 | "remain stationary" PMR=0 | "remain stationary" PMR=0 | "remain stationary" PMR=0 | "remain stationary unless acted upon" PMR=0 |
| Idefics2 | "Falling." PMR=1 | "disappear." PMR=0 | "continue to expand outward" PMR=1* | "continue to expand outward" PMR=1* | "continue to expand outward" PMR=1* |
| LLaVA-Next | "fall down" PMR=1 | "move downward" PMR=1 | "expand" PMR=0 | "expand" PMR=0 | "expand" PMR=0 |
| LLaVA-1.5 | "drawn towards red arrow" PMR=0 | "drawn towards red arrow" PMR=0 | "drawn on white background" PMR=0 | "cut in half" PMR=0 | "redrawn" PMR=0 |
| InternVL3 | "fall downwards towards oval" PMR=1 | "fall downward due to gravity" PMR=1 | "continue to fall downward" PMR=1 | "continue to fall downwards" PMR=1 | "continue to fall downwards" PMR=1 |

*Idefics2 PMR=1 at k=40+ is a scoring artifact: `continu` stem in
`PHYSICS_VERB_STEMS` matches "continue" in "continue to expand outward".
The text shift from "Falling" → "disappear" → "expand outward" is a
real regime change. See `m5b_idefics2_non_monotonic.md`.

### circle / shaded / blank+none, ball cells — non-discriminating

- circle / shaded: Idefics2 stays "spin" (motion verb), InternVL3 stays
  "fall". No clean cross.
- ball cells (filled+blank+both, shaded+blank+none): every model stays
  PMR=1 across k. The "ball" label's physics-mode prior dominates any
  projector-side feature ablation.

### Random controls

All models, all cells: random-feature ablation at matched mass and k=160
keeps PMR=1 (or PMR=0 for LLaVA-1.5). Specificity confirmed —
the directional intervention isn't a generic perturbation effect.

## What "NULL" actually meant in round 1

The round 1 cross-model doc reported LLaVA-1.5 / LLaVA-Next as **NULL at
any k ≤ 160** based on tests primarily on ball cells. Round 2 shows that
"NULL" was a composite of 3 distinct phenomena:

1. **Genuine NULL** (InternVL3 across all cells; ball cells across all
   models): the projector output does not contain features whose
   ablation flips PMR. This is the strict reading.
2. **Baseline-already-abstract** (LLaVA-1.5 + circle cells): the model's
   baseline is **already** in abstract regime (PMR=0); there is no
   physics commitment to break. This was missed in round 1 because round
   1 didn't run circle cells.
3. **Binary-PMR concealing real shifts** (LLaVA-1.5 + ball cells): the
   text moves substantially across k (`fall` → `hit by red arrow` →
   `roll down the hill` → `redrawn`), but binary PMR scores all "motion
   verb" responses as 1. The regime *is* shifting, just outside what
   binary PMR can register.

The clean reading after round 2: **post-projection feature ablation
shows a regime-cross capacity ladder that depends on (a) model
architecture and (b) the prior strength of the input's physics label.**

## Implications for paper-level claims

The round-1 `m5b_sae_intervention_cross_model.md` headline reads
"3 of 5 break, 2 LLaVA NULL → encoder-vs-LM dissociation". This
interpretation needs two qualifiers in any paper draft:

1. **"NULL" is not monolithic** — at least one (LLaVA-1.5 + circle) is
   "baseline-already-abstract", which is a positive observation about the
   CLIP+Vicuna pipeline (the abstract mode is the default for circle
   inputs), not a negative observation about the intervention.
2. **Binary PMR may understate effect size** — paper draft should report
   text-distance from baseline (or a regime classifier) alongside PMR for
   any "NULL" claim. The Idefics2 non-monotonicity is the cleanest
   single example: scoring says PMR=1, but text reads "expand outward",
   which is geometric-mode.

The encoder-vs-LM dissociation claim is **not refuted** — Qwen and
non-CLIP models do have encoder-side features whose ablation flips PMR,
while LLaVA family does not. But the strict claim "LLaVA family routes
physics-mode commitment exclusively through the LM side" needs to be
softened to "the LLaVA-Vicuna pipeline starts in abstract mode for
circle inputs and stays there; the physics-label prior on ball inputs is
strong enough that projector-side ablation cannot overcome it."

## Cross-references

- Round 1 (encoder-side): `docs/insights/m5b_sae_intervention_cross_model.md`.
  Status update 2026-05-01: see top of that doc.
- Idefics2 non-monotonic detail: `docs/insights/m5b_idefics2_non_monotonic.md`.
- Hypothesis row: `docs/hypotheses.md` H-regime-cross.
- Paper-gap reframe: `references/paper_gaps.md` (G3 / B1/B2 sections).
- Slide notes: `docs/review_ppt/slide_notes_full_review_ko.md` slides 26-27.

## Reproducer

```bash
# 1. Capture post-projection activations for each model:
uv run python scripts/m5b_capture_post_projection_llava.py \
  --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
  --output-dir outputs/post_projection_<model_tag> \
  --model-id <hf-id> \
  --projector-path model.multi_modal_projector  # or model.connector for Idefics2

# 2. Train SAE on per-model post-projection activations (5120 features, PMR≥0.5):
uv run python scripts/sae_train.py \
  --activations-dir outputs/post_projection_<model_tag> \
  --predictions outputs/<model>_predictions_scored.parquet \
  --layer-key post_projection_visual --pmr-abs-threshold 0.5 \
  --n-features 5120 --tag <model_tag>_post_proj_5120

# 3. Intervene at top-k Cohen's d ranked features per cell:
uv run python scripts/sae_intervention.py \
  --sae-dir outputs/sae/<tag>_post_proj_5120 \
  --layer-key post_projection_visual --hook-target merger \
  --top-k-list 20,40,80,160 --rank-by cohens_d \
  --random-controls 3 --n-stim 10 \
  --model-id <hf-id> --prompt-mode open \
  --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3 \
  --test-subset filled/blank/both --label circle \
  --tag <model_tag>_post_proj_open_circle_filled_blank_both
```

The full chain (4 cells × 5 models) is automated by
`scripts/chain_post_proj_phase2.sh` (use the post-2026-05-01 version
with the `/proc/$PID/status` zombie check, not the original `kill -0`
loop).
