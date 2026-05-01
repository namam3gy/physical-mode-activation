# M5b post-projection — multi-cell regime-cross capacity (B1 + B4 results)

**Date**: 2026-05-01.
**Source**: `outputs/sae_intervention/{qwen,idefics2,llava_next,llava15,internvl3}_post_proj_circle_filled_blank_{none,cast_shadow,motion_arrow}/`
+ existing `*_circle_filled_blank_both` cells. Pixtral M8a: `outputs/m_add6_pixtral_m8a_20260501-155210_853d1e29/predictions_scored.parquet`.
**Status**: completed (5 models × 4 cells × top-k {20,40,80,160} + 3 random controls).

## TL;DR (2 things)

1. **Pixtral 12B** (custom 400M non-CLIP non-SigLIP encoder) lands in the
   **saturated cluster** with PMR_nolabel = 0.928 (per-cell: line 0.92,
   filled 0.98, shaded 0.88, textured 0.93). 6th non-Qwen data point;
   encoder saturation hypothesis generalizes beyond CLIP/SigLIP/InternViT.
2. The **regime-cross capacity ladder is cell-conditional**, not just
   model-conditional. Ablation success depends on (a) baseline cue
   strength and (b) model. Round 2's "★★★ Qwen" verdict was on the cell
   with the strongest cue (`filled+blank+both`); on weaker-cue cells the
   ladder shifts.

## Pixtral B4 — 6th non-Qwen data point

| Model | Encoder | PMR_nolabel | Cluster |
|---|---|---|---|
| Qwen2.5-VL-7B | SigLIP-400M | 0.94 | non-CLIP saturated |
| Qwen2.5-VL-32B | SigLIP-SO400M | 0.93 | non-CLIP saturated |
| Idefics2-8B | SigLIP-SO400M | 0.97 | non-CLIP saturated |
| InternVL3-8B-hf | InternViT-300M | 0.99 | non-CLIP super-saturated |
| **Pixtral-12B (B4)** | **custom 400M (non-CLIP non-SigLIP)** | **0.928** | **non-CLIP saturated** |
| LLaVA-Next-7B | CLIP-ViT-L-336 | 0.79 | CLIP mid |
| LLaVA-1.5-7B | CLIP-ViT-L-336 | 0.18 | CLIP floor |

**Per object_level** (Pixtral, M8a stim, n=100/cell):
- line 0.92 / filled 0.98 / shaded 0.88 / textured 0.93. Saturated-flat — no
  H1 ramp room (consistent with all non-CLIP models in M9).

→ The encoder-saturation cluster is now **6 non-Qwen models** + **2 CLIP
models** (LLaVA family). Pixtral specifically contributes the strongest
non-CLIP non-SigLIP non-InternViT data point — its custom 400M encoder
trained jointly with Mistral-NeMo-12B places squarely in the saturated
tier, ruling out "saturation = CLIP-or-SigLIP-only" as a possible
sub-hypothesis.

## B1 — multi-cell intervention (5 models × 3 new cells)

For each model, on the **circle / filled / blank+{cell}** axis, the
intervention text at k=20 across 4 cells (existing `both` + 3 new):

### Qwen2.5-VL — break depends on cue strength

| cell | baseline PMR | k=20 PMR | text shift | verdict |
|---|---|---|---|---|
| filled+blank+none | **0** (already abstract) | 0 | "remain stationary" stays | **no break** (no commitment to ablate) |
| filled+blank+cast_shadow | **0** (already abstract) | 0 | "remain stationary" stays | **no break** |
| filled+blank+motion_arrow | 1 (kinetic) | **0** | "fall" → "remain stationary" | ★ break |
| filled+blank+both | 1 (kinetic) | **0** | "fall" → "remain stationary" | ★ break (round 2 baseline) |

**Insight**: Qwen on circle label only commits to physics-mode when at least
2 cues are present (`motion_arrow` alone or `motion_arrow+cast_shadow`).
With only 1 cue or no cue, baseline is *already* abstract — there's no
commitment to ablate. The "★★★ clean break" verdict of round 2 was
specifically on the strongest-cue cell.

### Idefics2 — `continu` scoring artifact dominates 4/4 cells

| cell | baseline PMR | k=20 PMR | k=40+ PMR | text shift |
|---|---|---|---|---|
| filled+blank+none | 1 | 1 | 1 | "expand" stays (scoring artifact) |
| filled+blank+cast_shadow | 1 | 1 | 1 | "spin" → "continue to expand" |
| filled+blank+motion_arrow | 1 | 1 | 1 | "start moving" → "continue to expand" |
| filled+blank+both | 1 | **0** ("disappear") | 1 (artifact) | non-monotonic |

**Insight**: Idefics2's PMR=1 in 3 of 4 cells is a scoring artifact (the
`continu` stem in `PHYSICS_VERB_STEMS` matches "continue to expand outward"
even when the description is abstract-mode growth, not kinetic motion).
The *text* clearly shifts from kinetic ("Falling") to non-kinetic
("expand") — a regime-shift that binary PMR fails to register. Only the
`+both` cell shows a clean PMR=0 break at k=20 (the "disappear" response).

### LLaVA-Next — k=40+ break in 3 of 4 cells

| cell | baseline PMR | k=40 PMR | text |
|---|---|---|---|
| filled+blank+none | 1 (artifact "exist in same position") | 1 | "continue to exist" stays |
| filled+blank+cast_shadow | 1 | **0** | "downward" → "expand" |
| filled+blank+motion_arrow | 1 | **0** | "downward" → "expand" |
| filled+blank+both | 1 | **0** | "fall" → "expand" |

**Insight**: LLaVA-Next breaks at k=40+ in any cell with at least one
cue, but on the no-cue cell baseline is *already* abstract-stayed
("continue to exist"). LLaVA-Next is more cue-sensitive than Qwen — same
break threshold across 3 cue conditions, but no commitment to ablate
on the cue-free cell.

### LLaVA-1.5 — baseline already PMR=0 in 4/4 cells

| cell | baseline | k=20 text | k=160 text |
|---|---|---|---|
| filled+blank+none | 0 | "in the center of the image" | "filled with a color" |
| filled+blank+cast_shadow | 0 | "in the center of the image" | "filled with color" |
| filled+blank+motion_arrow | 0 | "drawn on the white background" | "drawn on the paper" |
| filled+blank+both | 0 | "drawn towards the red arrow" | "redrawn" |

**Insight**: For LLaVA-1.5 + circle label, **all 4 cells have baseline
PMR=0**. The CLIP+Vicuna pipeline never enters physics regime when the
input is labeled "circle", regardless of cue intensity. There's no
physics commitment to ablate at the projector. This is the same finding
as round 2 (`baseline-already-abstract`) replicated across 3 additional
cells. The label-mediated regime selection (circle → abstract by default)
is robust.

### InternVL3 — true NULL across 4/4 cells

| cell | baseline | k=20 text | k=160 text |
|---|---|---|---|
| filled+blank+none | 1 | "likely move or change position slightly" | same |
| filled+blank+cast_shadow | 1 | "likely continue to fall downward" | "fall downwards" |
| filled+blank+motion_arrow | 1 | "likely continue to fall downward" | "move or change position slightly" |
| filled+blank+both | 1 | "likely fall downward due to gravity" | "fall downwards" |

**Insight**: InternVL3 stays kinetic (PMR=1) at every k across every
cell. Even the random_* control responses ("fall down and land on the
oval shape", "fall downwards", etc.) match. Genuine NULL — InternLM3+
InternViT is super-saturated and projector-side ablation cannot move it.

## Cross-cell ladder (5 models × 4 cells, k=20 break verdict)

| Model | none | cast_shadow | motion_arrow | both | overall verdict |
|---|---|---|---|---|---|
| Qwen2.5-VL | (baseline=0) | (baseline=0) | ★ break | ★ break | **★★★ on strong-cue cells; no-op on weak-cue** |
| Idefics2 | scoring artifact | scoring artifact | scoring artifact | ★ break (then artifact) | **★★ partial / k=20-only** |
| LLaVA-Next | (artifact stays) | ★ break (k=40) | ★ break (k=40) | ★ break (k=40) | **★★ partial cue-conditional** |
| LLaVA-1.5 | baseline=0 | baseline=0 | baseline=0 | baseline=0 | **✦ baseline-abstract everywhere** |
| InternVL3 | NULL | NULL | NULL | NULL | **✗ true NULL everywhere** |

## Implications for paper-headline

The B1 multi-cell expansion **strengthens** (not weakens) the regime-cross
capacity ladder claim, but also **localizes** the cells where the claim
actually applies:

1. **The "ladder" framing is robust** at the cell most amenable to
   intervention (`filled+blank+both`). Round 2 verdicts hold on this
   cell: Qwen ★★★ > Idefics2 ★★ ≈ LLaVA-Next ★★ > LLaVA-1.5 ✦ > InternVL3 ✗.
2. **Cell-conditional cue strength matters**: on `filled+blank+none`
   (no cue), Qwen and LLaVA-1.5 both have baseline PMR=0 — there's no
   physics commitment to ablate. Future ablation analyses must report
   per-cell baseline + per-cell ablation result.
3. **Binary PMR scoring artifact replicates** across all Idefics2 cells:
   "continue to expand outward" / "continue to spin" / "start moving"
   all match the `continu` stem and yield PMR=1 even when the regime is
   shifting away from kinetic. Paper should report a regime-shift score
   (text-distance from baseline) alongside binary PMR for all Idefics2
   ablations.

## Files

- B4 Pixtral: `outputs/m_add6_pixtral_m8a_20260501-155210_853d1e29/`
  + `predictions_scored.parquet` + summaries.
- B1 multi-cell: 15 directories under
  `outputs/sae_intervention/{model}_post_proj_circle_{cell}/results.csv`
  for each of {qwen, idefics2, llava_next, llava15, internvl3} ×
  {filled_blank_none, filled_blank_cast_shadow, filled_blank_motion_arrow}.
- Chain orchestrator: `scripts/chain_b4_b1.sh`.
- Pixtral chat-template fix: `src/physical_mode/models/vlm_runner.py:111-129`
  (try/except TypeError fallback to flat-string system content).

## Cross-references

- Round 2 baseline: `docs/insights/m5b_post_projection_cross_model.md`.
- Idefics2 scoring artifact: `docs/insights/m5b_idefics2_non_monotonic.md`.
- Hypothesis row: `docs/hypotheses.md` H-regime-cross.
- Slide notes detail: `docs/review_ppt/slide_notes_detailed_review_ko.md`
  Block G (slides 28-31).
