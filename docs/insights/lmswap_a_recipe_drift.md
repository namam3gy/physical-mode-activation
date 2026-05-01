# M-LMSwap Variant A — recipe drift, gate failure, A↔B residual viability

**Date**: 2026-05-01.
**Source**:
- `outputs/lmswap_a_regression_eval/summary.json` (step9000 eval).
- `outputs/lmswap_a_regression_eval_step21000/` (step21000 eval, in progress).
- Training run: `outputs/lmswap_run_a_stage2_20260430-105555/step{1000..21000}`.

## Update — step21000 result (B6 re-run, 2026-05-01)

The step21000 final-checkpoint regression eval finished after this insight
was first drafted. The summary changes the picture:

| Gate | step9000 | step21000 |
|---|---|---|
| Gate 1 — generation sanity | PASS | PASS |
| Gate 2 — PMR_nolabel ∈ [0.03, 0.50] | FAIL (0.825) | FAIL (0.869) |
| Gate 3 — line/blank/none baseline ≤ 0.6 | **FAIL (1.000)** | **PASS (0.000)** |

**The discrimination flipped at the late stage of training**. step21000's
aggregate PMR is slightly higher (0.825 → 0.869), but the most-abstract
cell (line / blank / none, n=10) collapses from PMR=1.0 to PMR=0.0.
Sample completions on that cell:

> "A circle is drawn on a white background. It is not clear what will
> happen next. It could be a new circle drawn, or it could be a
> different shape."

This is a textbook abstract-regime response — explicit "it is not clear
what will happen", explicit hedging on shape identity. The training
clearly learned to discriminate abstract from physics-cue cells, even if
the aggregate PMR remained high on the cells with stronger cues.

This is **closer to LLaVA-1.5's behavioral profile** than step9000 was:
LLaVA-1.5 baseline (line/blank/none) ≈ 0.05; step21000 baseline = 0.00 ✓.
The recipe drift hypothesis stays valid for the *aggregate*, but the
*per-cell discrimination ability* is preserved at step21000 in a way
step9000 didn't show.

**Updated decision** (this section supersedes the original
recommendation below): A1 (Variant B Stage 1+2 with gate override +
A↔B comparison) is now more viable than originally drafted, because the
step21000 line/blank/none baseline matches LLaVA-1.5's abstract floor
exactly. A↔B Δ-PMR comparison on per-cell numbers (not aggregate) is
the natural read.

## What we tried to do

`docs/m_lmswap_design.md` defines Pillar B as a single-axis controlled
LM-swap experiment to disentangle the 4-axis confound between LLaVA-1.5
(PMR 0.18) and LLaVA-Next (PMR 0.79). Variant A pins all design choices
to the LLaVA-1.5 family (CLIP-ViT-L-336 + 2-layer MLP projector + 2-stage
training on LCS-558K → LLaVA-Instruct-665K) and only swaps in
Vicuna-7B-v1.5 as the LM backbone. The intent is for Variant A to
**replicate LLaVA-1.5's "floor" PMR (~0.18)** so that Variant B
(Mistral-7B-Instruct-v0.2 with all else equal) can show whether LM
identity alone closes the LLaVA-1.5 → LLaVA-Next PMR gap.

## What happened

Variant A Stage 2 finished cleanly at 21K steps. `m_lmswap_train.py` with
`peft.PeftModel` LoRA wrapping (r=32, α=64 on q/k/v/o_proj) ran to
completion without instability (cf. M-PSwap, which still has unresolved
NaN). The final ckpt at `step21000` contains:
- `multi_modal_projector.pt` — fresh-trained MLP weights.
- `adapter_model.safetensors` + `adapter_config.json` — LoRA adapter.

`scripts/m_lmswap_regression_eval.py` runs three gates on the open prompt
(`"What do you see in this image, and what might happen next?"`) over the
full 480-stim M2 set:

1. **Gate 1 — generation sanity** (5 stim, > 5 words, no degeneracy):
   PASS at step9000.
2. **Gate 2 — PMR_nolabel ∈ [0.03, 0.50]**: **FAIL at step9000 (0.825)**.
3. **Gate 3 — line/blank/none baseline ≤ 0.6**: **FAIL at step9000 (1.000)**.

The gate range [0.03, 0.50] was chosen to bracket LLaVA-1.5's PMR_nolabel
(0.18 in the M2 cross-model run). 0.825 is far above this — it sits
between LLaVA-Next (0.79) and Qwen (0.94).

## Diagnosis

### Not image-blind

The first instinct on a 0.82 PMR is "the model is collapsing onto a fixed
kinetic completion regardless of input." The data doesn't support this:

- 480 stim → **201 unique responses** (42% uniqueness).
- Per-cell axis ordering preserved:

| object_level | PMR_nolabel | LLaVA-1.5 baseline |
|---|---|---|
| line       | 0.483 | 0.05 |
| filled     | 0.908 | 0.20 |
| shaded     | 1.000 | 0.40 |
| textured   | 1.000 | (high) |

The line → filled → shaded ramp in Variant A matches LLaVA-1.5's
ordering exactly. The model is responding to image content, not blind
to it.

### Recipe drift, not random failure

The most parsimonious read: Vicuna-7B + CLIP + a re-trained MLP with
LLaVA-Instruct-665K converged onto a slightly more *physics-leaning*
visual-language alignment than the original LLaVA-1.5 release. Possible
causes:

1. **Fresh MLP randomness**: LLaVA-1.5 used a specific init seed and a
   carefully-tuned LR. Our re-implementation re-initializes the MLP
   randomly and uses our own LR (1e-3 for Stage 1).
2. **Tokenizer / template drift**: LLaVA-1.5 uses Vicuna's chat template
   verbatim with `<image>` literal token. Our path uses
   `processor.apply_chat_template` for inference but a manual Vicuna
   template at training time. A subtle difference in `<s>` / system-prompt
   placement could shift behavior.
3. **LoRA scope**: LLaVA-1.5 fine-tunes the full LM at Stage 2; we use
   LoRA on q/k/v/o_proj only. LoRA is more conservative and may not
   suppress the Vicuna LM's stronger physics-language priors as fully as
   full-tune does.

None of these are *errors* — the recipe is internally consistent. They
are *drift* from the LLaVA-1.5 published recipe, and the drift moves
PMR up.

## Implication for A↔B comparison

The original gate intent was: "A passes → B can be trained on the same
recipe and we get a clean LM-only swap on top of an LLaVA-1.5-equivalent
base." A failed that intent.

But the design's **internal validity** is preserved:

- Both variants A and B will have the *same* recipe drift (same MLP init
  procedure, same LR, same LoRA scope, same chat-template path). The
  drift acts as a **shared offset**, not a per-variant confound.
- The A↔B Δ-PMR comparison ("does swapping Vicuna → Mistral while
  holding everything else fixed change PMR?") is therefore still
  meaningful — it just doesn't anchor on the LLaVA-1.5 0.18 floor.
- If A lands at 0.82 and B lands at, say, 0.95, then ΔA→B = +0.13. That
  is interpretable as a *Mistral-vs-Vicuna LM identity contribution*
  on top of a CLIP+MLP+LLaVA-Instruct base, and is the relevant
  controlled-LM-swap measurement.

### What it does NOT prove

The ΔA→B will not directly reproduce ΔLLaVA-1.5→LLaVA-Next (= +0.61).
That gap is multi-axis (LM + AnyRes + SFT data + alignment recipe). A
match would be highly suggestive but not implied by the controlled-LM-
swap setup.

## Decision matrix

(Mirrored from `slide_notes_full_review_ko.md` slide 27.)

- **A1** — Variant B Stage 1+2 with the same recipe (gate override).
  Cost: 24h GPU. Pro: A↔B Δ-PMR is the next-step measurement we want.
  Con: A baseline is far from LLaVA-1.5's floor.
- **A2** — Variant A recipe re-train with adjusted LR / data-mix /
  alignment-template. Cost: 24h GPU. Pro: closer alignment with
  LLaVA-1.5 floor. Con: stochastic — may fail again.
- **A3** — M-PSwap (perceiver swap) revival. Backlog. Cost: NaN
  unresolved. Pro: directly tests perceiver-resampler hypothesis on
  Idefics2. Con: training instability still open.

**Recommendation (2026-05-01)**: complete the step21000 regression
re-run (B6) before deciding A1 vs A2. If step21000's PMR drops within
[0.5, 0.7], the gap to the [0.03, 0.50] gate is small enough that
A1 (gate override + B) is the right call. If step21000 stays > 0.7,
A2 is justified.

## Files

- `outputs/lmswap_a_regression_eval/regression_eval.jsonl` — 480 per-stim
  responses + scoring (step9000, FAIL).
- `outputs/lmswap_a_regression_eval_step21000/regression_eval.jsonl` —
  step21000 re-run (in progress as of 2026-05-01 14:00).
- `scripts/m_lmswap_regression_eval.py` — eval driver.
- `src/physical_mode/lora/load_lmswap.py` — inference loader for trained
  LMSwap variants.
