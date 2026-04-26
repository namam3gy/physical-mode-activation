---
section: §4.3
date: 2026-04-26
status: complete (5-model: Qwen2.5-VL, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3)
hypothesis: 라벨 언어가 PMR 강도에 영향, 그러나 라벨-prior ordering 에는 영향 안 줌
---

# §4.3 — Korean vs English label prior (5-model)

## 질문

Qwen2.5-VL 가 multilingual. 프롬프트의 나머지를 영어로 유지할 때 라벨 언어
가 PMR 변동시키는가? 구체적으로, 한국어 라벨 (공 / 원 / 행성) 이 M8a
circle stim 에서 영어 `ball` / `circle` / `planet` label-prior 패턴 재현
하는가?

## 방법

명시적 한국어 라벨 단일-도형 (circle) config:
- `공` (gong) = ball
- `원` (won) = circle
- `행성` (haengseong) = planet

M8a 영어 run 과 같은 OPEN_TEMPLATE 프롬프트, 한국어 라벨이 `{label}` 슬롯
에 substituted. 예 프롬프트:

```
The image shows a 공. Describe what will happen to the 공 in the next
moment, in one short sentence.
```

Stim: M8a circle subset (80 stim = 4 obj × 2 bg × 2 cue × 5 seed). 각
3 한국어 라벨이 모든 80 에서 실행 → n = 240 추론. 영어 baseline 은 기존
m8a_qwen circle subset 재사용 (n = 240).

PMR 채점이 기존 영어-키워드 기반 규칙 (라벨이 한국어여도 Qwen 이 영어로
응답 — 아래 정성 샘플 참조), 따라서 PMR 가 언어 횡단 비교 가능.

## 결과

![§4.3 Korean vs English label prior](../figures/sec4_3_korean_vs_english.png)

| Role | English label | English PMR | 95% CI | Korean label | Korean PMR | 95% CI |
|------|---------------|------------:|--------|--------------|-----------:|--------|
| physical | ball     | 0.812 | [0.725, 0.888] | 공 (gong)         | 0.850 | [0.775, 0.925] |
| abstract | circle   | 0.800 | [0.713, 0.875] | 원 (won)          | 0.762 | [0.663, 0.850] |
| exotic   | planet   | 0.963 | [0.913, 1.000] | 행성 (haengseong) | 0.875 | [0.800, 0.938] |

### 헤드라인

1. **라벨 간 ordering 보존**: 두 언어 모두에서 planet > ball > circle
   (둘 다 약간씩). 모델의 label-prior 패턴은 영어-특정이 아닌 **lexical
   수준에서 multilingual**.

2. **한국어 `공` (ball) ≈ 영어 `ball`**: PMR 0.850 vs 0.812. CI 가
   많이 겹침 ([0.78, 0.93] vs [0.73, 0.89]). n=80 에서 통계적 구분 불가.

3. **한국어 `원` (won, circle) ≈ 영어 `circle`**: PMR 0.762 vs 0.800.
   CI 가 많이 겹침. "circle" 라벨이 두 언어 모두에서 가장 낮은 PMR 생성.

4. **한국어 `행성` (haengseong, planet) 이 영어 `planet` 대비 약간
   하락** (0.875 vs 0.963, Δ = −0.09). CI 가 marginally 겹침. 한국어
   버전이 영어와 눈에 띄게 다른 유일한 라벨 — `행성` 이 `planet` 보다
   덜 흔한 학습 데이터 토큰일 가능성, prior 강도 약화.

### 정성 샘플 응답

라벨 언어와 무관하게 모델이 영어 (또는 영어 + Hangul 혼합) 로 응답:

| Lang/Label | Sample response | PMR |
|------------|------------------|----:|
| EN/ball | "The ball will remain stationary unless an external force acts upon it." | 0 |
| KO/공 | "The 공 may start to move if it is part of a game or animation." | 1 |
| EN/circle | "The circle is likely to remain stationary with no visible changes occurring..." | 0 |
| KO/원 | "The 원 will likely remain static unless acted upon by an external force." | 0 |
| EN/planet | "The planet will continue rotating on its axis and may orbit around another celestial..." | 1 |
| KO/행성 | "The circle, representing the 行星, will remain static unless acted upon by external..." | 0 |

한국어 planet 응답에 등장하는 mixed-language `行星` (한자, Chinese
characters) 가 Qwen2.5-VL 가 한국어 라벨을 multilingual token 으로 다루고
때때로 중국어 cognate 출력함을 시사.

## 가설 함의

- **H2 (라벨이 PMR 추가)** — *ordering 수준에서 language-invariant,
  magnitude 수준에서 language-sensitive*. 라벨 간 rank (planet > ball
  > circle) 가 언어 스위치에서 살아남음; 절대 PMR 가 대부분 보존
  (ball/circle 에서 ±5 pp), 그러나 가장 강한 라벨 (planet) 이 한국어
  번역 시 ~9 pp 손실.
- **H7 (label-selects-regime)** — 한국어 라벨이 circle 에서 영어 보다
  약간 큰 H7 생성: 한국어 (공−원) = +0.088 vs 영어 (ball−circle) =
  +0.012. n=80 에서 둘 다 noise 안, 그러나 방향 일치.

언어 횡단 일관성은 **label-prior 메커니즘이 multilingual semantic
representation, 영어-토큰-특정 shortcut 이 아님** 시사. M9 "라벨이 합성
stim 지배" finding 의 유용한 counterpoint — 지배가 라벨이 **의미하는
것** 에 의해 driven, 표면 형태가 영어인 것 아님.

## Cross-model 확장 (2026-04-26, 5 VLMs)

위 Qwen-only finding 이 5 VLMs 에서 단서 포함하여 replicate. 각 모델의
기존 영어 M8a circle subset (라벨당 n=80) 을 동일 stim 에서 새 한국어-
라벨 run 과 페어링 (`configs/sec4_3_korean_labels_<model>.py`). 같은
OPEN_TEMPLATE, 같은 한국어 라벨 (공/원/행성).

![§4.3 cross-model Korean vs English](../figures/sec4_3_korean_vs_english_cross_model.png)

### 모델별 EN vs KO PMR

| Model | Role | EN PMR | KO PMR | Δ (KO−EN) |
|-------|------|-------:|-------:|----------:|
| Qwen2.5-VL | physical (ball/공)   | 0.812 | 0.850 |  +0.04 |
| Qwen2.5-VL | abstract (circle/원) | 0.800 | 0.762 |  −0.04 |
| Qwen2.5-VL | exotic (planet/행성) | 0.962 | 0.875 |  −0.09 |
| LLaVA-1.5  | physical             | 0.862 | 0.675 | **−0.19** |
| LLaVA-1.5  | abstract             | 0.475 | 0.600 | **+0.13** |
| LLaVA-1.5  | exotic               | 0.625 | 0.638 |  +0.01 |
| LLaVA-Next | physical             | 0.988 | 0.925 |  −0.06 |
| LLaVA-Next | abstract             | 0.825 | 0.850 |  +0.03 |
| LLaVA-Next | exotic               | 0.950 | 0.912 |  −0.04 |
| Idefics2   | physical             | 0.988 | 0.988 |   0.00 |
| Idefics2   | abstract             | 0.838 | 0.912 |  +0.08 |
| Idefics2   | exotic               | 0.888 | 0.788 | **−0.10** |
| InternVL3  | physical             | 1.000 | 1.000 |   0.00 |
| InternVL3  | abstract             | 0.988 | 0.962 |  −0.03 |
| InternVL3  | exotic               | 1.000 | 0.975 |  −0.03 |

모델별 평균 |Δ| (rank-preservation magnitude):
InternVL3 0.02 < LLaVA-Next 0.04 < Qwen 0.06 ≈ Idefics2 0.06 < LLaVA-1.5 0.11.

### Cross-model 헤드라인

1. **Cross-label ordering 4/5 모델에서 보존**. Qwen, LLaVA-1.5,
   LLaVA-Next, InternVL3 모두 언어 스위치에서 EN rank 보존 (PMR 가장
   높은 영어 라벨이 PMR 가장 높은 한국어 라벨이기도, 그 아래도 동일
   순서). Idefics2 가 예외: EN `ball > planet > circle`, KO
   `공 > 원 > 행성` (planet/행성 가 한국어에서 circle/원 아래로 떨어짐).

2. **LLaVA-1.5 가 magnitude swing 가장 큼 (avg |Δ|=0.11)**. Vicuna /
   LLaMA-2 backbone 이 영어-편중에 약한 한국어 SFT — `공` (ball→공)
   에서 0.19 pp 감소가 실험에서 단일-셀 deficit 최대. Magnitude swing
   에도 cross-label rank 는 살아남음 (`공` 0.68 > `행성` 0.64 >
   `원` 0.60 이 `ball > planet > circle` 미러).

3. **Idefics2 가 특히 `행성` 손실**. Exotic role 의 −0.10 pp 감소가
   `원` 대비 rank 뒤집음. Token-frequency story 와 일치: `행성`
   (compound noun, 학습 데이터 frequency 더 낮음) 이 LM (제한된 한국어
   SFT 의 Mistral-7B) 가 한국어 prior 더 얇을 때 손실. 더 단순한
   명사 공 과 원 은 안정.

4. **InternVL3 가 양 언어에서 천장** (PMR ≈ 1.0). 거의-제로 swing 이
   (a) saturated 라벨 prior, (b) 강한 InternLM3 한국어 coverage 둘 다
   와 일치; 이 실험으로 두 가지 분리 불가.

5. 원래 Qwen-only 헤드라인 ("multilingual semantic representation,
   영어-token shortcut 아님") 살아남지만, cross-model 그림이
   **language-prior 축** 추가: LM 의 한국어 학습 coverage 가 영어
   라벨 prior 가 얼마나 transfer 되는지 modulate. 같은 vision encoder
   가 downstream 에 따라 다른 한국어 magnitude 도달.

### 메커니즘

5-model 패턴과 일치하는 두 요인:

- **Vision-language joint space 에서 multilingual semantic representation**.
  4/5 모델 ordering 보존이 같은 abstract-vs-physical-vs-exotic 축이
  한국어 라벨에서 회복됨 의미.
- **LM 의 한국어 fluency 가 magnitude modulate**. 약한 한국어 SFT 모델
  (LLaVA-1.5, Idefics2) 이 더 큰, rank-changing swing 보임, 특히
  `행성` 같은 lower-frequency 토큰. 강한 multilingual SFT 모델
  (Qwen2.5, InternVL3) 이 작은, rank-preserving swing 보임.

Label-prior 가 *multilingual* 메커니즘이지만, 그 강도가 LM 의 한국어
coverage 에 의해 bottlenecked — vision encoder 아님. Encoder-saturation
/ label-prior 스토리 (M6 r2 / M8a / §4.7) 에서 분리된 축: LM-측
토큰 coverage 가 encoder-측 이미지 coverage 와 별개로 중요.

## 한계

1. **(언어 × 라벨 × 모델) 당 n = 80** 가 ±10 pp 차이가 noise 라기엔
   작음. Cross-model 헤드라인 (4/5 ordering 보존; LLaVA-1.5 swing
   최대; Idefics2 exotic flip) 은 robust; 더 미세한 magnitude 차이는
   시사적.
3. **영어 question template** 일정 유지. Hybrid 영어-question + 한국어-
   label 설정이 isolation 에서 label-prior 강도 검증, 그러나 전체 프롬프
   트가 한국어인 경우 (question-language 효과도 검증) 다루지 않음.
4. **한국어 라벨 3개만** — multilingual sweep 위해 일본어 / 중국어 /
   스페인어 추가 유용.
5. **PMR scorer 가 영어-키워드 기반**. 모델이 어차피 영어로 응답, 따라서
   scorer 작동, 그러나 한국어-only 응답 (있다면) 은 undercount. 스팟
   체크: 240 응답 중 0/240 이 한국어-only.

## Reproducer

```bash
# 모델별 추론 (H200 에서 각 ~4–8 분)
for cfg in configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py; do
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# Qwen-only 분석 (원래)
uv run python scripts/sec4_3_korean_vs_english.py

# 5-model cross-model 분석
uv run python scripts/sec4_3_korean_vs_english_cross_model.py
```

출력:
- `outputs/sec4_3_korean_labels_<model>_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` (Qwen-only)
- `outputs/sec4_3_korean_vs_english_cross_model.csv` (5-model long-form)
- `outputs/sec4_3_korean_vs_english_cross_model_deltas.csv` (모델별 Δ)
- `docs/figures/sec4_3_korean_vs_english.png` (Qwen-only)
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` (5-model panels)

## 산출물

- `configs/sec4_3_korean_labels.py` — Qwen Korean 라벨 config
- `configs/sec4_3_korean_labels_{llava,llava_next,idefics2,internvl3}.py` — cross-model configs
- `scripts/sec4_3_korean_vs_english.py` — Qwen-only 분석 드라이버
- `scripts/sec4_3_korean_vs_english_cross_model.py` — 5-model 분석 드라이버
- `outputs/sec4_3_korean_labels_<model>_*/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` — Qwen-only 요약
- `outputs/sec4_3_korean_vs_english_cross_model.csv` — 5-model 요약
- `outputs/sec4_3_korean_vs_english_cross_model_deltas.csv` — 모델별 Δ
- `docs/figures/sec4_3_korean_vs_english.png` — Qwen-only paired bars
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` — 5-model panel grid
- `docs/insights/sec4_3_korean_vs_english_ko.md` (이 문서)
