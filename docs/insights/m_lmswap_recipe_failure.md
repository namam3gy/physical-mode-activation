# M-LMSwap recipe failure — deprioritized 2026-05-03

## TL;DR

Both Variant A (CLIP + Vicuna-7B) and Variant B (CLIP + Mistral-7B) failed
their regression gates from opposite failure modes. Neither reproduces
LLaVA-1.5's saturation regime (PMR_nolabel ≈ 0.18), so the planned A↔B
controlled comparison can't isolate "LM family identity" — both baselines
are unusable. **M-LMSwap is dropped from Pillar B paper scope.** The LM-vs-
encoder mechanism dissociation claim is carried instead by §4.6 + M5b
cross-model results (LLaVA family M5a positive / M5b NULL → LM-side flip,
encoder-side null).

## Final regression eval results

| Gate | A (Vicuna, step21000) | B (Mistral, step21000) | LLaVA-1.5 expected |
|---|---|---|---|
| gate_1 (sanity ≥5 words, no degenerate repetition) | PASS | **FAIL** (5/5 = "The ball will bounce", 4 words) | — |
| gate_2 (PMR_nolabel ∈ [0.03, 0.50]) | **FAIL 0.869** | **FAIL 0.996** | ~0.18 |
| gate_3 (baseline cell line/blank/none ≤ 0.6) | PASS 0.000 | **FAIL 1.000** | ≤ 0.6 |
| overall | FAIL | FAIL | — |

## A's failure mode — verbose runaway (4th bug: missing-EOS-label)

**Symptom**: 480/480 responses run to max_new_tokens=80 mid-sentence;
sanity test with max_new_tokens=512 shows **0/5 emit `</s>`**.

**Root cause**: `format_chat[A]` (m_lmswap_train.py L115) omits trailing `</s>`:
```python
full = f"{_VICUNA_SYSTEM} USER: {user_text} ASSISTANT: {assistant_text}"
```
Vicuna's LlamaTokenizer does NOT auto-append eos (verified empirically:
`tok("...world")['input_ids'][-1] == 3186`, not 2). A's training labels never
contained `</s>` → LM never learned to emit EOS.

This is the 4th distinct bug discovered on this pipeline, and the only one
on A's path. The B branch correctly appends `</s>`; the A branch was missed.

**Truncation rescue analysis** (`scripts/m_lmswap_a_rescore_truncated.py`):
re-scoring A's responses by truncating at the first sentence-ending mark
gives PMR_nolabel = 0.504 — within 1σ of the gate ceiling (0.50, σ ≈ 0.023
on n=480), but methodologically reviewer-vulnerable: 175/480 of the original
PMR=1 cases had their physics commitment in the *second* sentence (e.g.,
"A circle and a dot are in the center. The dot is moving towards the
center.") — truncation discards real signal.

A is "almost defensible" with truncation, but defending the truncation rule
is harder than just retraining. Either way, A retrain only matters if there's
a B to compare against — see below.

## B's failure mode — mode collapse (recipe-level, not bug-level)

**Symptom**: 480 stim → only **8 unique responses across the entire eval**.
Top 3 cover 449/480 (94%):
- "Ball will bounce" × 226
- "The ball will bounce" × 158
- "Ball will fall" × 65

Eval ran in 50s (vs. A's 580s) because B emits ~4 tokens then `</s>` for
every input — EOS *was* learned, but the response distribution collapsed
to a small set of physics-prior phrases regardless of image content.
Baseline cell (line/blank/none — abstract circle on white) produces "The ball
will bounce" × 10/10. **The model isn't conditioning on the image.**

**Suspected mechanism**: LoRA r=32 / α=64 / LR=2e-4 on Mistral-7B over
LLaVA-Instruct-665K over-adapted the LM's attention so cross-modal grounding
broke. The model learned the M2 prompt distribution ("What will happen
next?") but stopped reading the visual tokens.

This is recipe-level — not fixable by a one-line bug fix. A redesign would
need: smaller LoRA rank/α, possibly frozen LM layers, different data
mixture, lower LR, or warmup tuning. No guarantee a redesigned recipe lands
in the LLaVA-1.5 saturation band on the first try (cost: another 12h+ per
attempt, with the same risk profile).

## Why the controlled-reproduction premise didn't hold

The M-LMSwap design (`docs/m_lmswap_design.md`) assumed the canonical
2-stage LLaVA-1.5 recipe (MLP-only stage 1 → MLP+LoRA stage 2) would
reproduce LLaVA-1.5's PMR profile when applied symmetrically to Vicuna and
Mistral. It didn't, in either direction:
- A under-fit on EOS (and possibly other tokens) → over-verbose, PMR inflated
- B over-fit the prompt distribution → mode collapse, PMR saturated to 1.0

This isn't an "A is the right number, B is the wrong number" problem. It's
"the recipe doesn't reproduce LLaVA-1.5 regardless of LM choice." The
A↔B comparison can't isolate LM family identity if neither variant lands
in LLaVA-1.5's working regime.

## Bug cascade (full record)

Four bugs were discovered during M-LMSwap training, three on B's path and
one on A's:

| # | Bug | Affects | Root cause | Fix | Commit |
|---|---|---|---|---|---|
| 1 | Double-BOS | B only | Literal `<s>` in `format_chat[B]` + tokenizer auto-prepend | Drop literal `<s>` | `7c6db1e` |
| 2 | Left-padding | B only | Mistral tokenizer default left-pad vs cut formula assuming right-pad → label leak | Force `padding_side="right"` | `5978cc2` |
| 3 | EOS-as-PAD | B only | `pad_token=None` defaulted to `eos_token=2` → masking masked legitimate `</s>` | Prefer UNK over EOS for pad | `a542f2e` |
| 4 | Missing-EOS-label | A only | `format_chat[A]` omits trailing `</s>`, Vicuna tokenizer doesn't auto-append | (Not applied — recipe deprioritized) | — |

Bugs 1–3 are Mistral-tokenizer-pipeline bugs. Bug 4 is a format-chat-string-
construction bug that A escaped on the tokenizer side but tripped on the
template side. The B branch comment in `format_chat` actually documents the
intent (line 121: "Trailing `</s>` is kept so the assistant span ends with
an explicit EOS for the LM head to learn") — but the A branch was missed.

The cascade itself is a meta-signal: a paper-grade recipe should not require
this much debugging, and even after three rounds of bug fixes neither variant
trains to the target regime. That argues against further investment.

## What replaces M-LMSwap in Pillar B

The LM-vs-encoder mechanism dissociation claim — originally targeted by
M-LMSwap's controlled A↔B comparison — is now carried by the cross-model
M5a/M5b results (`docs/insights/m5b_sae_intervention_cross_model.md`):

- **LLaVA family** (LLaVA-1.5, LLaVA-Next): M5a positive (LM-side hook
  flips PMR 10/10) + M5b NULL at any k ≤ 160 (encoder-side SAE features
  absent or too distributed) → physics-mode commitment routes through LM,
  not encoder.
- **Idefics2 + InternVL3**: M5b positive at k=160 → encoder-side commitment
  detectable, perceiver-resampler / cross-attn projection.
- **Qwen2.5-VL**: M5b positive at k=20 (0.4% of features) — both sides
  contribute.

This dissociation is *cross-architecture* and *measured via causal intervention*,
which is stronger evidence than a controlled-LM-swap re-training would have
provided.

## Concrete artifacts preserved

- All 4 chain scripts: `scripts/chain_lmswap_b{,_tail,_s2_retry,_s2_resume}.sh`
- Training script with bug fixes 1–3: `scripts/m_lmswap_train.py`
- Diagnostic scripts: `scripts/m_lmswap_a_eos_sanity.py`,
  `scripts/m_lmswap_a_rescore_truncated.py`
- A stage 2 ckpt: `outputs/lmswap_run_a_stage2_20260430-105555/step21000/`
- B stage 2 ckpt: `outputs/lmswap_run_b_stage2_20260502-152743/step21000/`
- A regression eval: `outputs/lmswap_a_regression_eval_step21000/`
- B regression eval: `outputs/lmswap_b_regression_eval/`

These are kept on disk for opportunistic future revival (recipe redesign).
Not gitignored cleanups — just paused.

## Decision authority

Drop decision authorized by user 2026-05-02 23:30: "B 끝난 다음에 advisor의
조언에 따라 A 재학습 여부를 정하자. 내 허락맡지말고 advisor가 재학습 하라고하면
그대로 진행하면 돼." Advisor recommended drop after B's mode-collapse
diagnostic confirmed (8 unique responses across 480 stim, 94% covered by 3
phrases). Auto-executed per pre-authorization.
