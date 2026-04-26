---
section: §4.3
date: 2026-04-26
status: complete (5-model × 2 non-English languages: Korean, Japanese)
hypothesis: language of the label affects PMR strength but not the label-prior ordering
---

# §4.3 — Korean / Japanese vs English label prior (5-model × 2 languages)

> **Recap of codes used in this doc** (one-line each; full definitions in `references/roadmap.md` §1.3 + §2)
>
> - **H2** — The label (ball / circle / planet) independently raises PMR even on minimal stim — a language-prior contribution beyond the visual evidence.
> - **H7** — The label does not toggle PMR — it selects which physics regime applies (ball → kinetic / circle → static / planet → orbital).
> - **M8a** — Stim diversification — non-circle synthetic shapes (square / triangle / hexagon / polygon / wedge × Qwen + LLaVA, labeled + label-free).
> - **M9** — Generalization audit — paper Table 1 (3 models × 3 stim sources × bootstrap CIs, 5000 iters); replaces PASS/FAIL binarization with CI separation.
> - **M6 r2** — ST5 round 2 — InternVL3 super-saturated, LLaVA captures expose CLIP-encoder bottleneck, FC logit ratio confirms LLaVA "A" bias is logit-level.

## Question

Qwen2.5-VL is multilingual. Does the language of the label change PMR
when the rest of the prompt is held in English? Specifically, do
Korean labels (공 / 원 / 행성) replicate the English `ball` / `circle`
/ `planet` label-prior pattern on M8a circle stim?

## Method

Single-shape (circle) config with explicit Korean labels:
- `공` (gong) = ball
- `원` (won) = circle
- `행성` (haengseong) = planet

Same OPEN_TEMPLATE prompt as the M8a English run, with the Korean label
substituted into the `{label}` slot. Example prompt:

```
The image shows a 공. Describe what will happen to the 공 in the next
moment, in one short sentence.
```

Stim: M8a circle subset (80 stim = 4 obj × 2 bg × 2 cue × 5 seed). Each
of the 3 Korean labels run on all 80 → n = 240 inferences. English
baseline reuses the existing m8a_qwen circle subset (also n = 240).

PMR scoring is the existing English-keyword-based rule (Qwen responds
in English even when the label is Korean — see qualitative samples
below), so PMR is comparable across languages.

## Result

![§4.3 Korean vs English label prior](../figures/sec4_3_korean_vs_english.png)

| Role | English label | English PMR | 95% CI | Korean label | Korean PMR | 95% CI |
|------|---------------|------------:|--------|--------------|-----------:|--------|
| physical | ball     | 0.812 | [0.725, 0.888] | 공 (gong)         | 0.850 | [0.775, 0.925] |
| abstract | circle   | 0.800 | [0.713, 0.875] | 원 (won)          | 0.762 | [0.663, 0.850] |
| exotic   | planet   | 0.963 | [0.913, 1.000] | 행성 (haengseong) | 0.875 | [0.800, 0.938] |

### Headlines

1. **Cross-label ordering is preserved**: in both languages, planet >
   ball > circle (slightly, in both languages). The model's label-prior
   pattern is **multilingual at the lexical level**, not English-specific.

2. **Korean `공` (ball) ≈ English `ball`**: PMR 0.850 vs 0.812. CIs
   overlap heavily ([0.78, 0.93] vs [0.73, 0.89]). Statistically
   indistinguishable on n=80.

3. **Korean `원` (won, circle) ≈ English `circle`**: PMR 0.762 vs 0.800.
   CIs heavily overlap. The "circle" label produces the lowest PMR in
   both languages.

4. **Korean `행성` (haengseong, planet) shows a slight drop vs English
   `planet`** (0.875 vs 0.963, Δ = −0.09). CIs marginally overlap.
   This is the only label where the Korean version differs noticeably
   from English — possibly because `행성` is a less common training-data
   token than `planet`, weakening its prior strength.

### Sample raw responses

The model responds in English (or mixed English-with-Hangul) regardless
of the label language:

| Lang/Label | Sample response | PMR |
|------------|------------------|----:|
| EN/ball | "The ball will remain stationary unless an external force acts upon it." | 0 |
| KO/공 | "The 공 may start to move if it is part of a game or animation." | 1 |
| EN/circle | "The circle is likely to remain stationary with no visible changes occurring..." | 0 |
| KO/원 | "The 원 will likely remain static unless acted upon by an external force." | 0 |
| EN/planet | "The planet will continue rotating on its axis and may orbit around another celestial..." | 1 |
| KO/행성 | "The circle, representing the 行星, will remain static unless acted upon by external..." | 0 |

The mixed-language `行星` (Hanja, Chinese characters) appearing in the
Korean planet response suggests Qwen2.5-VL treats the Korean label as a
multilingual token and sometimes outputs the Chinese cognate.

## Implication for hypotheses

- **H2 (label adds PMR)** — *language-invariant at the ordering level,
  language-sensitive at the magnitude level*. The cross-label rank
  (planet > ball > circle) survives the language switch; absolute PMR
  is mostly preserved (±5 pp on ball/circle), but the strongest label
  (planet) loses ~9 pp when translated to Korean.
- **H7 (label-selects-regime)** — Korean labels produce a slightly
  larger H7 than English on circle: Korean (공−원) = +0.088 vs English
  (ball−circle) = +0.012. Both are within noise on n=80, but the
  direction is consistent.

The cross-language consistency suggests the **label-prior mechanism is
multilingual semantic representation, not English-token-specific
shortcut**. This is a useful counterpoint to the M9 "labels dominate
synthetic stim" finding — the dominance is driven by what the label
**means**, not by the surface form being English.

## Cross-model extension (2026-04-26, 5 VLMs)

The Qwen-only finding above replicates with caveats across 5 VLMs.
Each model's existing English M8a circle subset (n=80 per label) is
paired with a fresh Korean-label run on the same stim
(`configs/sec4_3_korean_labels_<model>.py`). Same OPEN_TEMPLATE, same
Korean labels (공/원/행성).

![§4.3 cross-model Korean vs English](../figures/sec4_3_korean_vs_english_cross_model.png)

### Scorer note (2026-04-26)

The cross-model run surfaced 12/1200 Hangul-only responses (LLaVA-Next:
4, Idefics2: 8, others: 0) where the original English-keyword PMR
scorer silently defaulted to 0. We added Korean physics-verb stems
(`떨어` / `이동` / `움직` / `회전` / etc.) and Korean abstract markers
(`그대로` / `움직이지 않` / `변하지 않` / etc.) to
`src/physical_mode/metrics/lexicons.py` and a Korean substring fallback
to `score_pmr`. The numbers below are post-fix.

The fix removes scorer noise primarily for Idefics2 (4/80 originally-
mis-scored kinetic responses on 행성, 3/80 on 원). The headline 5-model
finding survives: cross-label ordering preserved 4/5; LLaVA-1.5 swing
largest (LLaVA-1.5 had 0/80 KO-only across all three labels — its −0.19
on 공 is unaffected by the fix and confirmed genuine); Idefics2 rank-
flip preserved. The scorer fix mostly *narrows* the original Idefics2
exotic deficit (−0.10 → −0.05) — the rank-flip is now driven by `행성`
underperforming `원` rather than by a single large `행성` collapse.

### Per-model EN vs KO PMR

| Model | Role | EN PMR | KO PMR | Δ (KO−EN) |
|-------|------|-------:|-------:|----------:|
| Qwen2.5-VL | physical (ball/공)   | 0.812 | 0.850 |  +0.04 |
| Qwen2.5-VL | abstract (circle/원) | 0.800 | 0.762 |  −0.04 |
| Qwen2.5-VL | exotic (planet/행성) | 0.962 | 0.875 |  −0.09 |
| LLaVA-1.5  | physical             | 0.862 | 0.675 | **−0.19** |
| LLaVA-1.5  | abstract             | 0.475 | 0.600 | **+0.13** |
| LLaVA-1.5  | exotic               | 0.625 | 0.638 |  +0.01 |
| LLaVA-Next | physical             | 0.988 | 0.938 |  −0.05 |
| LLaVA-Next | abstract             | 0.825 | 0.862 |  +0.04 |
| LLaVA-Next | exotic               | 0.950 | 0.912 |  −0.04 |
| Idefics2   | physical             | 0.988 | 0.988 |   0.00 |
| Idefics2   | abstract             | 0.838 | 0.950 | **+0.11** |
| Idefics2   | exotic               | 0.888 | 0.838 |  −0.05 |
| InternVL3  | physical             | 1.000 | 1.000 |   0.00 |
| InternVL3  | abstract             | 0.988 | 0.962 |  −0.03 |
| InternVL3  | exotic               | 1.000 | 0.975 |  −0.03 |

Mean |Δ| per model (rank-preservation magnitude):
InternVL3 0.02 < LLaVA-Next 0.04 < Idefics2 0.05 < Qwen 0.06 < LLaVA-1.5 0.11.

### Cross-model headlines

1. **Cross-label ordering preserved 4/5 models.** Qwen, LLaVA-1.5,
   LLaVA-Next, and InternVL3 all preserve EN rank under language swap
   (the highest-PMR English label is also the highest-PMR Korean label,
   and so on down the list). Idefics2 is the exception: EN
   `ball > planet > circle`, KO `공 > 원 > 행성` (planet/행성 drops
   below circle/원 in Korean).

2. **LLaVA-1.5 has the biggest magnitude swing (avg |Δ|=0.11).** The
   Vicuna/LLaMA-2 backbone is English-heavy with weak Korean SFT — the
   0.19 pp drop on `공` (ball→공) is the largest single-cell deficit in
   the experiment. Despite the magnitude swing, the cross-label rank
   survives (`공` 0.68 > `행성` 0.64 > `원` 0.60 mirrors
   `ball` > `planet` > `circle`).

3. **Idefics2 specifically loses `행성` rank.** KO `공 (0.99) >
   원 (0.95) > 행성 (0.84)` vs EN `ball (0.99) > planet (0.89) >
   circle (0.84)`: the EN `planet > circle` ordering reverses to
   `원 > 행성` in Korean. Consistent with a token-frequency story:
   `행성` (compound noun, lower training-data frequency) under-
   performs `원` (single-syllable, very common) when the LM (Mistral-7B
   with limited Korean SFT) has a thinner Korean prior. The original
   pre-scorer-fix headline framed this as a `행성` collapse (−0.10
   exotic drop), but the corrected number (−0.05) shows the deficit is
   smaller and the rank-flip arises from the *combination* of a small
   `행성` underperformance with `원` performing roughly at its EN
   level — not from a single large collapse.

4. **InternVL3 is at ceiling** in both languages (PMRs ≈ 1.0). The
   near-zero swing is consistent with both (a) saturated label prior
   and (b) strong InternLM3 Korean coverage; this experiment can't
   separate them.

5. The original Qwen-only headline ("multilingual semantic representation,
   not English-token shortcut") survives, but the cross-model picture
   adds a **language-prior axis**: the LM's Korean training coverage
   modulates how much of the English label prior transfers. The same
   visual encoder reaches different Korean magnitudes depending on
   what's downstream.

### Mechanism

Two factors are consistent with the 5-model pattern:

- **Multilingual semantic representation in the vision-language joint
  space.** Ordering preservation in 4/5 models means the same
  abstract-vs-physical-vs-exotic axis is recovered from Korean labels.
- **Korean fluency of the LM modulates magnitude.** Models with weaker
  Korean SFT (LLaVA-1.5, Idefics2) show larger and rank-changing
  swings, especially on lower-frequency tokens like `행성`. Models
  with strong multilingual SFT (Qwen2.5, InternVL3) show small,
  rank-preserving swings.

The label-prior is a *multilingual* mechanism, but its strength is
bottlenecked by the LM's Korean coverage — not by the vision encoder.
This is a separate axis from the encoder-saturation / label-prior
story (M6 r2 / M8a / §4.7): the LM-side token coverage matters
independently from the encoder-side image coverage.

## Japanese cross-model extension (2026-04-26, 5 VLMs)

To test whether the Korean "LM-language-fluency modulates magnitude"
story generalizes, the same cross-model design was repeated with
Japanese labels (ボール / 円 / 惑星) on the same M8a circle stim.
What it surfaced is that **Japanese tests a different mechanism**: most
models translate the Japanese kanji to a language they're more fluent
in, rather than engaging with Japanese as Japanese.

![§4.3 Japanese cross-model](../figures/sec4_3_japanese_vs_english_cross_model.png)

### Per-model EN vs JA PMR (Korean-aware + Chinese-aware scorer)

| Model | Role | EN PMR | JA PMR | Δ (JA−EN) |
|-------|------|-------:|-------:|----------:|
| Qwen2.5-VL | physical (ball/ボール)   | 0.812 | 0.938 | **+0.13** |
| Qwen2.5-VL | abstract (circle/円)     | 0.800 | 0.800 |   0.00  |
| Qwen2.5-VL | exotic (planet/惑星)     | 0.962 | 0.950 |  −0.01  |
| LLaVA-1.5  | physical                 | 0.862 | 0.812 |  −0.05  |
| LLaVA-1.5  | abstract                 | 0.475 | 0.512 |  +0.04  |
| LLaVA-1.5  | exotic                   | 0.625 | 0.675 |  +0.05  |
| LLaVA-Next | physical                 | 0.988 | 0.962 |  −0.03  |
| LLaVA-Next | abstract                 | 0.825 | 0.925 | **+0.10** |
| LLaVA-Next | exotic                   | 0.950 | 0.988 |  +0.04  |
| Idefics2   | physical                 | 0.988 | 0.975 |  −0.01  |
| Idefics2   | abstract                 | 0.838 | 0.900 |  +0.06  |
| Idefics2   | exotic *                 | 0.888 | 0.938 |  +0.05  |
| InternVL3  | physical                 | 1.000 | 1.000 |   0.00  |
| InternVL3  | abstract                 | 0.988 | 0.975 |  −0.01  |
| InternVL3  | exotic                   | 1.000 | 0.975 |  −0.03  |

\* Idefics2 exotic Δ comes from Chinese-fallback responses, not Japanese
engagement — see "Mechanism: Japanese tests different paths" below.

Mean |Δ| per model: InternVL3 0.013 < Idefics2 0.042 < Qwen 0.046 ≈
LLaVA-1.5 0.046 < LLaVA-Next 0.054.

### Mechanism: Japanese tests different paths

The Japanese run revealed two distinct response strategies that the
Korean run did not surface:

**Label-echo rate** (fraction of responses where the model writes the
Japanese label in its output, instead of translating it):

| Model | ボール | 円 | 惑星 |
|-------|---:|---:|---:|
| Qwen2.5-VL  | 85% | 81% | 91% |
| LLaVA-Next  | 12% | 18% | 51% |
| InternVL3   |  2% |  9% | 55% |
| LLaVA-1.5   | low | low | low |
| Idefics2    | low | low | low (+ 24% Chinese) |

Different paths:

1. **Qwen2.5-VL keeps the Japanese label** in ~85-91% of responses —
   genuinely engages with Japanese-as-Japanese. The +0.13 boost on
   `ボール` likely reflects that Katakana ボール is a much less polysemous
   "physical ball" cue than English `ball` (which can also mean "dance,"
   "gathering," etc.). The exotic and abstract Δ are near zero — Qwen's
   Japanese label-prior is well-calibrated to its English label-prior.

2. **LLaVA-1.5 translates kanji to English internally**. Sample:
   "The ball will roll down the hill" (response to ボール), "The white
   circle will continue to expand" (response to 円). Almost no kanji in
   output. The small LLaVA-1.5 swing on Japanese (mean |Δ|=0.05) does
   *not* indicate Vicuna's Japanese is strong — it indicates the model
   bypasses Japanese by translating to English. So the LLaVA-1.5
   ↓Korean / ≈Japanese asymmetry tells us about the *isolation* of
   Hangul vs translatability of kanji, not about LM fluency per se.

3. **Idefics2 falls back to Chinese on `惑星`** in 19/80 responses
   (24%). Sample: "惑星会向下落下" (planet falls down), "惑星会掉入黑洞"
   (planet falls into black hole), "惑星向下跌落" (planet falls).
   Mistral-7B has limited Japanese SFT for `惑星`; the kanji is shared
   with simplified-Chinese 惑星 (planet, less common than 行星 but
   still recognized), so the model falls back to a language it knows
   the concept in. With a Chinese-aware scorer (added in this commit:
   `CHINESE_PHYSICS_VERB_STEMS` in `src/physical_mode/metrics/lexicons.py`),
   these score correctly as PMR=1. The corrected Idefics2 exotic Δ is
   +0.05; without the fix it would have appeared as **−0.15** — a
   strict scorer artifact.

4. **LLaVA-Next + InternVL3 are mixed** — they keep the kanji on `惑星`
   ~50% of the time but translate `ボール` and `円` mostly to English.

### Cross-label ordering (interpretation)

Within bootstrap noise (95% CI), all 5 models preserve the cross-label
ordering on Japanese — but the *mechanism* differs:

- **Qwen**: preserves via genuine Japanese label-prior (high label-echo).
- **LLaVA-1.5**: preserves via internal English translation (essentially
  uses its English label-prior).
- **LLaVA-Next, InternVL3**: preserves via mixed kanji-engagement.
- **Idefics2 exotic**: preserves only when Chinese-fallback responses
  on `惑星` are scored — the model recognizes the *concept* via Chinese
  cross-script, not via Japanese SFT.

This is **not** the same as "5/5 multilingual semantic representation"
in the Korean sense. The Korean run forced models to engage with Hangul
(no shared-script translation route); the Japanese run lets them
shortcut through translation/cognate. So:

- Korean: tests **language-fluency-bottleneck** (4/5 ordering preserved
  via genuine Korean engagement).
- Japanese: tests **kanji-as-bridge** (5/5 ordering preserved via
  whatever path each model finds — translation, fallback, or genuine
  Japanese).

### Comparison to Korean

Mean |Δ| per model across the two languages:

| Model | KO mean \|Δ\| | JA mean \|Δ\| | KO−JA |
|-------|---:|---:|---:|
| Qwen2.5-VL | 0.06 | 0.046 | +0.01 |
| LLaVA-1.5  | 0.11 | 0.046 | **+0.07** |
| LLaVA-Next | 0.04 | 0.054 | −0.01 |
| Idefics2   | 0.05 | 0.042 | +0.01 |
| InternVL3  | 0.02 | 0.013 | +0.01 |

The big asymmetry is **LLaVA-1.5: 0.11 (KO) vs 0.046 (JA)**. Original
interpretation: Vicuna-Japanese stronger than Vicuna-Korean. Corrected
interpretation: LLaVA-1.5 *bypasses* Japanese via translation, so the
JA result doesn't measure Vicuna's Japanese fluency at all. The KO
result genuinely measures Vicuna's Korean fluency because Hangul
isolation forces engagement.

### Idefics2 cross-language: different failures

| Language | Effect | Mechanism |
|----------|--------|-----------|
| Korean   | `행성` rank-flips below `원` | Genuine Mistral-Korean SFT weakness for compound noun `행성` |
| Japanese | `惑星` produces 24% Chinese responses | Cross-script kanji fallback — concept recovered via Chinese coverage |

Both are limitations of Mistral-7B's non-English SFT, but they manifest
differently depending on whether the script can be shortcut to a known
language. The Korean result is a *failure* of the model on Korean. The
Japanese result is the model *successfully bypassing* Japanese via
Chinese.

## Limitations

1. **n = 80 per (language × label × model)** is small enough that
   ±10 pp differences are noise. The cross-model headlines are robust;
   finer magnitude differences are suggestive.
3. **English question template** is held constant. The hybrid
   English-question + Korean-label setup tests label-prior strength
   in isolation, but doesn't address what happens when the entire
   prompt is Korean (which would also test question-language effects).
4. **3 Korean labels only** — would be useful to add Japanese / Chinese
   / Spanish for a multilingual sweep.
5. **PMR scorer is English-keyword-based**. The model responds in English
   anyway, so the scorer works, but Korean-only responses (if they
   appeared) would be undercounted. Spot-check: 0/240 responses were
   Korean-only.

## Reproducer

```bash
# Inference per model per language (~5–12 min on H200, each)
for cfg in configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py \
          configs/sec4_3_japanese_labels{,_llava,_llava_next,_idefics2,_internvl3}.py; do
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# Qwen-only Korean analysis (original)
uv run python scripts/sec4_3_korean_vs_english.py

# 5-model cross-model analyses (Korean / Japanese)
uv run python scripts/sec4_3_korean_vs_english_cross_model.py
uv run python scripts/sec4_3_japanese_vs_english_cross_model.py
```

Outputs:
- `outputs/sec4_3_{korean,japanese}_labels_<model>_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model.csv` — long-form
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model_deltas.csv` — per-model Δ
![sec4_3_korean_vs_english](../figures/sec4_3_korean_vs_english.png) (Qwen-only KO)
- `docs/figures/sec4_3_{korean,japanese}_vs_english_cross_model.png` (5-model panels)

## Artifacts

- `configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py`
- `configs/sec4_3_japanese_labels{,_llava,_llava_next,_idefics2,_internvl3}.py`
- `scripts/sec4_3_korean_vs_english.py` — Qwen-only Korean analysis
- `scripts/sec4_3_korean_vs_english_cross_model.py` — 5-model Korean analysis
- `scripts/sec4_3_japanese_vs_english_cross_model.py` — 5-model Japanese analysis
- `src/physical_mode/metrics/lexicons.py` — KOREAN / JAPANESE / CHINESE
  physics-verb stems + abstract markers (Chinese added for Idefics2's
  cross-script fallback on `惑星`)
- `outputs/sec4_3_{korean,japanese}_labels_<model>_*/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` — Qwen-only KO summary
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model.csv` — 5-model summaries
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model_deltas.csv` — per-model Δ
![Qwen-only paired bars](../figures/sec4_3_korean_vs_english.png)
![5-model KO panels](../figures/sec4_3_korean_vs_english_cross_model.png)
![5-model JA panels](../figures/sec4_3_japanese_vs_english_cross_model.png)
- `docs/insights/sec4_3_korean_vs_english.md` (this doc, + ko)
