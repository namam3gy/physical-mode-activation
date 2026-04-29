# §4.3 — Korean / Japanese vs English label prior (run log, 2026-04-26)

## Setup

- **Stim subset**: M8a circle subset (n=80 per label per language per model = 80 stim × 3 labels × 5 models × 3 languages).
- **Languages**: English (existing M8a circle) + Korean (공 / 원 / 행성) + Japanese (ボール / 円 / 惑星).
- **Configs**: `configs/sec4_3_korean_labels_<model>.py` and `configs/sec4_3_japanese_labels_<model>.py` per model.
- **Sampling**: T=0.7, top_p=0.95, max_new_tokens=96 (matches M8a / OPEN_TEMPLATE).
- **Models**: Qwen2.5-VL-7B, LLaVA-1.5-7B, LLaVA-Next-7B, Idefics2-8B, InternVL3-8B.
- **Scorer**: PMR scorer extended with Korean physics-verb stems + Chinese-fallback stems for Idefics2 Japanese.
- **Deep dive**: `docs/insights/sec4_3_korean_vs_english.md`.

## Korean cross-model EN→KO Δ (KO − EN)

| Model | physical | abstract | exotic | mean \|Δ\| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | +0.04 | −0.04 | −0.09 | 0.06 |
| LLaVA-1.5  | **−0.19** | **+0.13** | +0.01 | 0.11 |
| LLaVA-Next | −0.05 | +0.04 | −0.04 | 0.04 |
| Idefics2   |  0.00 | **+0.11** | −0.05 | 0.05 |
| InternVL3  |  0.00 | −0.03 | −0.03 | 0.02 |

## Japanese cross-model EN→JA Δ (JA − EN)

| Model | physical | abstract | exotic | mean \|Δ\| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | **+0.13** | 0.00 | −0.01 | 0.05 |
| LLaVA-1.5  | −0.05 | +0.04 | +0.05 | 0.05 |
| LLaVA-Next | −0.03 | +0.10 | +0.04 | 0.05 |
| Idefics2   | −0.01 | +0.06 | +0.05 \* | 0.04 |
| InternVL3  |  0.00 | −0.01 | −0.03 | 0.01 |

\* Idefics2 exotic +0.05 on Japanese comes from Chinese-fallback responses scored correctly by the new `CHINESE_PHYSICS_VERB_STEMS` lexicon. Without the fix the apparent Δ would have been **−0.15** — pure scorer artifact (19/80 responses fell back to Chinese reading of `惑星`).

## Headlines

1. **Cross-label ordering preserved 4/5 models** in Korean (Qwen / LLaVA-1.5 / LLaVA-Next / InternVL3); Idefics2 Korean order shifts (`공 > 원 > 행성` vs EN `ball > planet > circle`).
2. **Japanese 5/5 ordering preserved** within bootstrap noise.
3. **LLaVA-1.5 Korean swing largest** (avg \|Δ\|=0.11; Vicuna LM has weak Korean SFT). InternVL3 Korean smallest (0.02; ceiling + strong InternLM3 Korean coverage).
4. **Korean tests language-fluency-bottleneck**; **Japanese tests kanji-as-bridge** (mixed paths: genuine engagement Qwen, internal translation LLaVA-1.5, cross-script fallback Idefics2). LLaVA-1.5 ↓Korean / ≈Japanese asymmetry reflects **script translatability**, not LM SFT depth.

## Output dirs

- `outputs/sec4_3_korean_<model>_<ts>/`
- `outputs/sec4_3_japanese_<model>_<ts>/`
- Figures: `docs/figures/sec4_3_korean_vs_english_cross_model.png`
