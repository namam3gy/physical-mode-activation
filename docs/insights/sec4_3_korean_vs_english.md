---
section: §4.3
date: 2026-04-26
status: complete (5-model: Qwen2.5-VL, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3)
hypothesis: language of the label affects PMR strength but not the label-prior ordering
---

# §4.3 — Korean vs English label prior (5-model)

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
to `score_pmr`. The numbers below are post-fix; the original 5-model
finding stands but with cleaner Idefics2 magnitudes (the −0.10 exotic
drop shrinks to −0.05; abstract +0.08 grows to +0.11; rank-flip
preserved).

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

3. **Idefics2 specifically loses `행성` rank.** The −0.05 pp drop on
   the exotic role combined with a +0.11 pp rise on the abstract role
   flips the rank against `원`: KO `공 > 원 > 행성` vs EN
   `ball > planet > circle`. Consistent with a token-frequency story:
   `행성` (compound noun, lower training-data frequency) under-
   performs `원` (single-syllable, very common) when the LM (Mistral-7B
   with limited Korean SFT) has a thinner Korean prior. Note that the
   Idefics2 exotic drop is smaller after the Korean-scorer fix
   (−0.10 → −0.05) — the original measurement over-stated the
   single-cell deficit but the rank-flip story is unchanged.

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

## Limitations

1. **n = 80 per (language × label × model)** is small enough that
   ±10 pp differences are noise. The cross-model headline (ordering
   preserved 4/5; LLaVA-1.5 swing largest; Idefics2 exotic flip)
   is robust; finer magnitude differences are suggestive.
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
# Inference per model (~4–8 min on H200, each)
for cfg in configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py; do
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# Qwen-only analysis (original)
uv run python scripts/sec4_3_korean_vs_english.py

# 5-model cross-model analysis
uv run python scripts/sec4_3_korean_vs_english_cross_model.py
```

Outputs:
- `outputs/sec4_3_korean_labels_<model>_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` (Qwen-only)
- `outputs/sec4_3_korean_vs_english_cross_model.csv` (5-model long-form)
- `outputs/sec4_3_korean_vs_english_cross_model_deltas.csv` (per-model Δ)
- `docs/figures/sec4_3_korean_vs_english.png` (Qwen-only)
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` (5-model panels)

## Artifacts

- `configs/sec4_3_korean_labels.py` — Qwen Korean labels config
- `configs/sec4_3_korean_labels_{llava,llava_next,idefics2,internvl3}.py` — cross-model configs
- `scripts/sec4_3_korean_vs_english.py` — Qwen-only analysis driver
- `scripts/sec4_3_korean_vs_english_cross_model.py` — 5-model analysis driver
- `outputs/sec4_3_korean_labels_<model>_*/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` — Qwen-only summary
- `outputs/sec4_3_korean_vs_english_cross_model.csv` — 5-model summary
- `outputs/sec4_3_korean_vs_english_cross_model_deltas.csv` — per-model Δ
- `docs/figures/sec4_3_korean_vs_english.png` — Qwen-only paired bars
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` — 5-model panel grid
- `docs/insights/sec4_3_korean_vs_english.md` (this doc, + ko)
