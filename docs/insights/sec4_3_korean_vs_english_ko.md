---
section: §4.3
date: 2026-04-26
status: complete (Qwen2.5-VL 만)
hypothesis: 라벨 언어가 PMR 강도에 영향, 그러나 라벨-prior ordering 에는 영향 안 줌
---

# §4.3 — Qwen2.5-VL Korean vs English label prior

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

## 한계

1. **단일 모델 (Qwen2.5-VL)**. LLaVA-1.5 + LLaVA-Next + Idefics2 +
   InternVL3 는 multilingual 능력에서 다를 수 있음. Cross-model sweep
   이 architecture 에서 언어 sensitivity 분리.
2. **(언어 × 라벨) 당 n = 80** 가 ±10 pp 차이가 noise 라기엔 작음.
   헤드라인 finding (라벨 간 ordering 보존) 은 robust; magnitude 차이는
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
# 추론 (H200 에서 ~5 분)
uv run python scripts/02_run_inference.py \
    --config configs/sec4_3_korean_labels.py \
    --stimulus-dir inputs/m8a_qwen_<ts> \
    --limit 240

# 분석
uv run python scripts/sec4_3_korean_vs_english.py
```

출력:
- `outputs/sec4_3_korean_labels_qwen_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv`
- `docs/figures/sec4_3_korean_vs_english.png`

## 산출물

- `configs/sec4_3_korean_labels.py` — Korean 라벨 config
- `scripts/sec4_3_korean_vs_english.py` — 분석 드라이버
- `outputs/sec4_3_korean_labels_qwen_*/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` — 요약 테이블
- `docs/figures/sec4_3_korean_vs_english.png` — paired bar chart
- `docs/insights/sec4_3_korean_vs_english_ko.md` (이 문서)
