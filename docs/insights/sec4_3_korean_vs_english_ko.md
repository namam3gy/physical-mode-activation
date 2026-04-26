---
section: §4.3
date: 2026-04-26
status: complete (5-model × 2 non-English languages: Korean, Japanese)
hypothesis: 라벨 언어가 PMR 강도에 영향, 그러나 라벨-prior ordering 에는 영향 안 줌
---

# §4.3 — Korean / Japanese vs English label prior (5-model × 2 languages)

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

### Scorer 수정 (2026-04-26)

Cross-model run 에서 1200 응답 중 12개 한국어-only 응답 발견 (LLaVA-Next 4,
Idefics2 8, 나머지 0) — 원래 영어-키워드 PMR scorer 가 조용히 0 으로
default. Korean physics-verb stems (`떨어` / `이동` / `움직` / `회전` 등)
와 Korean abstract markers (`그대로` / `움직이지 않` / `변하지 않` 등) 를
`src/physical_mode/metrics/lexicons.py` 에 추가 + `score_pmr` 에 한국어
substring fallback 추가. 아래 수치는 fix 후.

Fix 가 주로 Idefics2 의 scorer noise 제거 (행성에서 원래 mis-score 된 4/80
kinetic 응답, 원에서 3/80). 헤드라인 5-model finding 은 유지: cross-label
ordering 4/5 보존; LLaVA-1.5 swing 최대 (LLaVA-1.5 는 세 라벨 모두 0/80
KO-only — 공의 −0.19 는 fix 와 무관하게 유지, Vicuna-LM-bias 스토리가
scorer artifact 가 아닌 진짜임을 확인); Idefics2 rank-flip 보존. Scorer
fix 가 주로 원래 Idefics2 exotic deficit 을 *줄임* (−0.10 → −0.05) —
rank-flip 이 이제 단일 `행성` collapse 가 아닌 `행성` 이 `원` 보다
underperform 함으로 driven.

### 모델별 EN vs KO PMR

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

모델별 평균 |Δ| (rank-preservation magnitude):
InternVL3 0.02 < LLaVA-Next 0.04 < Idefics2 0.05 < Qwen 0.06 < LLaVA-1.5 0.11.

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

3. **Idefics2 가 특히 `행성` rank 손실**. KO `공 (0.99) >
   원 (0.95) > 행성 (0.84)` vs EN `ball (0.99) > planet (0.89) >
   circle (0.84)`: EN 의 `planet > circle` ordering 이 한국어에서
   `원 > 행성` 로 뒤집음. Token-frequency story 와 일치: `행성`
   (compound noun, 학습 데이터 frequency 더 낮음) 이 `원` (단일 음절,
   매우 흔함) 대비 LM (제한된 한국어 SFT 의 Mistral-7B) 가 한국어
   prior 더 얇을 때 underperform. 원래 pre-scorer-fix 헤드라인이 이를
   `행성` collapse (−0.10 exotic drop) 로 framing 했으나, 수정된 수치
   (−0.05) 가 deficit 이 더 작고 rank-flip 이 *단일 큰 collapse 가 아닌*
   `행성` 의 작은 underperformance + `원` 이 EN level 에서 거의 그대로
   유지하는 *조합* 에서 발생함을 보임.

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

## Japanese cross-model 확장 (2026-04-26, 5 VLMs)

Korean 의 "LM-언어-fluency 가 magnitude 변조" 스토리가 일반화되는지
검증하기 위해 같은 cross-model 디자인을 Japanese 라벨 (ボール / 円 /
惑星) 로 동일 M8a circle stim 에 반복. 결과적으로 **Japanese 가 다른
mechanism 을 테스트** 함이 드러남: 대부분의 모델이 Japanese 를
Japanese 로 engage 하지 않고 더 fluent 한 언어로 번역.

![§4.3 Japanese cross-model](../figures/sec4_3_japanese_vs_english_cross_model.png)

### 모델별 EN vs JA PMR (Korean-aware + Chinese-aware scorer)

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

\* Idefics2 exotic Δ 가 Japanese engagement 가 아닌 Chinese-fallback
응답에서 옴 — 아래 "메커니즘: Japanese 가 다른 path 테스트" 참조.

모델별 평균 |Δ|: InternVL3 0.013 < Idefics2 0.042 < Qwen 0.046 ≈
LLaVA-1.5 0.046 < LLaVA-Next 0.054.

### 메커니즘: Japanese 가 다른 path 테스트

Japanese run 이 Korean run 에서 노출되지 않은 두 가지 응답 전략을
드러냄:

**Label-echo rate** (모델이 Japanese 라벨을 번역하지 않고 출력에 그대로
쓴 응답 비율):

| Model | ボール | 円 | 惑星 |
|-------|---:|---:|---:|
| Qwen2.5-VL  | 85% | 81% | 91% |
| LLaVA-Next  | 12% | 18% | 51% |
| InternVL3   |  2% |  9% | 55% |
| LLaVA-1.5   | low | low | low |
| Idefics2    | low | low | low (+ 24% Chinese) |

다른 path:

1. **Qwen2.5-VL 이 Japanese 라벨 유지** ~85-91% — 진짜로 Japanese-as-
   Japanese 로 engage. `ボール` 의 +0.13 boost 는 Katakana ボール 가
   영어 `ball` (춤, 모임 등 polysemous) 보다 훨씬 덜 polysemous 한
   "physical ball" cue 임을 반영. Exotic + abstract Δ 가 거의 0 —
   Qwen 의 Japanese label-prior 가 영어 label-prior 와 잘 calibrated.

2. **LLaVA-1.5 가 kanji 를 영어로 내부 번역**. 샘플:
   "The ball will roll down the hill" (ボール 응답), "The white circle
   will continue to expand" (円 응답). 출력에 kanji 거의 없음. Japanese
   에서 LLaVA-1.5 의 작은 swing (mean |Δ|=0.05) 가 *Vicuna 의 Japanese
   가 강함을 의미하지 않음* — 모델이 영어로 번역하여 Japanese 를
   bypass 함을 의미. 즉 LLaVA-1.5 ↓Korean / ≈Japanese 비대칭이 LM
   fluency 자체가 아닌 Hangul 의 *고립* vs kanji 의 *번역가능성* 에
   대해 알려줌.

3. **Idefics2 가 `惑星` 에서 Chinese 로 fallback** 19/80 응답 (24%).
   샘플: "惑星会向下落下" (planet falls down), "惑星会掉入黑洞" (planet
   falls into black hole), "惑星向下跌落" (planet falls). Mistral-7B 가
   `惑星` 에 제한된 Japanese SFT; kanji 가 simplified-Chinese 惑星
   (planet, 行星 보다는 덜 흔하지만 인식됨) 와 공유, 모델이 concept 를
   아는 언어로 fallback. Chinese-aware scorer (이 commit 에 추가됨:
   `src/physical_mode/metrics/lexicons.py` 의 `CHINESE_PHYSICS_VERB_STEMS`)
   적용 시 PMR=1 로 정확하게 점수. 수정된 Idefics2 exotic Δ 는 +0.05;
   fix 없으면 **−0.15** 로 보였을 것 — 순수 scorer artifact.

4. **LLaVA-Next + InternVL3 가 mixed** — `惑星` 에서 ~50% kanji 유지,
   `ボール`/`円` 에서는 대부분 영어 번역.

### Cross-label ordering (해석)

Bootstrap noise (95% CI) 안에서 5 모델 모두 Japanese 의 cross-label
ordering 보존 — 그러나 *메커니즘* 이 다름:

- **Qwen**: 진짜 Japanese label-prior 로 보존 (high label-echo).
- **LLaVA-1.5**: 내부 영어 번역으로 보존 (essentially 영어 label-prior
  사용).
- **LLaVA-Next, InternVL3**: mixed kanji-engagement 로 보존.
- **Idefics2 exotic**: `惑星` 의 Chinese-fallback 응답이 점수될 때만
  보존 — 모델이 *concept* 를 Japanese SFT 가 아닌 Chinese cross-script
  로 인식.

이는 Korean 의미의 "5/5 multilingual semantic representation" **과
다름**. Korean run 이 모델들에게 Hangul engage 강제 (shared-script
번역 route 없음); Japanese run 은 번역/cognate 으로 shortcut 허용.
따라서:

- Korean: **language-fluency-bottleneck** 테스트 (4/5 ordering 진짜
  Korean engagement 로 보존).
- Japanese: **kanji-as-bridge** 테스트 (5/5 ordering 각 모델이 찾은
  path — 번역, fallback, 또는 진짜 Japanese — 로 보존).

### Korean 과 비교

모델별 평균 |Δ| 두 언어 비교:

| Model | KO mean \|Δ\| | JA mean \|Δ\| | KO−JA |
|-------|---:|---:|---:|
| Qwen2.5-VL | 0.06 | 0.046 | +0.01 |
| LLaVA-1.5  | 0.11 | 0.046 | **+0.07** |
| LLaVA-Next | 0.04 | 0.054 | −0.01 |
| Idefics2   | 0.05 | 0.042 | +0.01 |
| InternVL3  | 0.02 | 0.013 | +0.01 |

큰 비대칭은 **LLaVA-1.5: 0.11 (KO) vs 0.046 (JA)**. 원래 해석:
Vicuna-Japanese 가 Vicuna-Korean 보다 강함. 수정된 해석: LLaVA-1.5 가
Japanese 를 번역으로 *bypass*, 따라서 JA 결과가 Vicuna 의 Japanese
fluency 를 측정하지 않음. KO 결과는 Hangul 고립이 engagement 강제하므로
Vicuna 의 Korean fluency 진짜 측정.

### Idefics2 cross-language: 다른 실패

| Language | Effect | Mechanism |
|----------|--------|-----------|
| Korean   | `행성` 이 `원` 아래로 rank-flip | 진짜 Mistral-Korean SFT 의 compound noun `행성` 약점 |
| Japanese | `惑星` 가 24% Chinese 응답 생성 | Cross-script kanji fallback — Chinese coverage 로 concept 회복 |

둘 다 Mistral-7B 의 non-English SFT 한계지만, script 가 알려진 언어로
shortcut 가능한지에 따라 다르게 발현. Korean 결과는 모델의 Korean *실패*.
Japanese 결과는 모델이 Chinese 로 Japanese 를 *성공적으로 우회*.

## 한계

1. **(언어 × 라벨 × 모델) 당 n = 80** 가 ±10 pp 차이가 noise 라기엔
   작음. Cross-model 헤드라인은 robust; 더 미세한 magnitude 차이는
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
# 모델별 + 언어별 추론 (H200 에서 각 ~5–12 분)
for cfg in configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py \
          configs/sec4_3_japanese_labels{,_llava,_llava_next,_idefics2,_internvl3}.py; do
    uv run python scripts/02_run_inference.py \
        --config "$cfg" \
        --stimulus-dir inputs/m8a_qwen_<ts>
done

# Qwen-only Korean 분석 (원래)
uv run python scripts/sec4_3_korean_vs_english.py

# 5-model cross-model 분석 (Korean / Japanese)
uv run python scripts/sec4_3_korean_vs_english_cross_model.py
uv run python scripts/sec4_3_japanese_vs_english_cross_model.py
```

출력:
- `outputs/sec4_3_{korean,japanese}_labels_<model>_<ts>/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model.csv` — long-form
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model_deltas.csv` — 모델별 Δ
- `docs/figures/sec4_3_korean_vs_english.png` (Qwen-only KO)
- `docs/figures/sec4_3_{korean,japanese}_vs_english_cross_model.png` (5-model panels)

## 산출물

- `configs/sec4_3_korean_labels{,_llava,_llava_next,_idefics2,_internvl3}.py`
- `configs/sec4_3_japanese_labels{,_llava,_llava_next,_idefics2,_internvl3}.py`
- `scripts/sec4_3_korean_vs_english.py` — Qwen-only Korean 분석
- `scripts/sec4_3_korean_vs_english_cross_model.py` — 5-model Korean 분석
- `scripts/sec4_3_japanese_vs_english_cross_model.py` — 5-model Japanese 분석
- `src/physical_mode/metrics/lexicons.py` — KOREAN / JAPANESE / CHINESE
  physics-verb stems + abstract markers (Idefics2 의 cross-script
  fallback `惑星` 위해 Chinese 추가)
- `outputs/sec4_3_{korean,japanese}_labels_<model>_*/predictions.{jsonl,parquet,csv}`
- `outputs/sec4_3_korean_vs_english.csv` — Qwen-only KO 요약
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model.csv` — 5-model 요약
- `outputs/sec4_3_{korean,japanese}_vs_english_cross_model_deltas.csv` — 모델별 Δ
- `docs/figures/sec4_3_korean_vs_english.png` — Qwen-only paired bars
- `docs/figures/sec4_3_korean_vs_english_cross_model.png` — 5-model KO panels
- `docs/figures/sec4_3_japanese_vs_english_cross_model.png` — 5-model JA panels
- `docs/insights/sec4_3_korean_vs_english_ko.md` (이 문서)
