---
companion_to: physical_mode_paper_ko.pptx
date: 2026-04-26
audience: 도메인 처음인 사람도 이해할 수 있게
---

# 본 연구 발표 자료 — 슬라이드별 상세 설명

> 슬라이드 PPT (`physical_mode_paper_ko.pptx`) 와 함께 보세요.
> 각 슬라이드에 대해 "슬라이드 내용 + 왜 이게 중요한지 + 도메인 처음인 사람도 이해할 수 있는 배경 설명" 을 제공합니다.

## 목차

1. [Title](#slide-1-title)
2. [Section 1 — Introduction (서론)](#section-1--introduction)
   - [Slide 2: 동기 — VLM 의 shortcut 문제](#slide-2-동기--vlm-의-shortcut-문제)
   - [Slide 3: 본 연구의 핵심 질문](#slide-3-본-연구의-핵심-질문)
   - [Slide 4: 주요 기여 3 가지](#slide-4-주요-기여-3-가지)
3. [Section 2 — Related Work (관련 연구)](#section-2--related-work)
   - [Slide 5: VLM grounding-failure / shortcut](#slide-5-vlm-grounding-failure--shortcut)
   - [Slide 6: Probing / Causal interpretability](#slide-6-probing--causal-interpretability)
4. [Section 3 — Problem & Definitions (문제 정의)](#section-3--problem--definitions)
   - [Slide 7: 문제 정의 — 두 차원 측정](#slide-7-문제-정의--두-차원-측정)
   - [Slide 8: 자극 설계 — M2 5축 factorial](#slide-8-자극-설계--m2-5축-factorial)
   - [Slide 9: 메트릭 정의](#slide-9-메트릭-정의)
   - [Slide 10: 테스트 모델 — 5 개 오픈소스 VLM](#slide-10-테스트-모델--5-개-오픈소스-vlm)
5. [Section 4 — Implementation (구현)](#section-4--implementation)
   - [Slide 11: 파이프라인](#slide-11-파이프라인)
   - [Slide 12: 활성화 캡처 + 선형 프로빙](#slide-12-활성화-캡처--선형-프로빙)
   - [Slide 13: 인과 개입 — VTI steering + §4.6 픽셀 ascent](#slide-13-인과-개입--vti-steering--46-픽셀-ascent)
6. [Section 5 — Experiments / Results (실험 / 결과)](#section-5--experiments--results)
   - [Slide 14: 5-model M2-stim PMR 사다리](#slide-14-5-model-m2-stim-pmr-사다리)
   - [Slide 15: H1 abstraction ramp](#slide-15-h1-abstraction-ramp)
   - [Slide 16: H2 paired-delta — 3 가지 패턴](#slide-16-h2-paired-delta--3-가지-패턴)
   - [Slide 17: Vision encoder probe](#slide-17-vision-encoder-probe)
   - [Slide 18: M5a VTI causal steering](#slide-18-m5a-vti-causal-steering)
   - [Slide 19: §4.6 픽셀-공간 gradient ascent (Qwen)](#slide-19-46-픽셀-공간-gradient-ascent-qwen)
   - [Slide 20: §4.6 cross-model NULL](#slide-20-46-cross-model-null)
   - [Slide 21: 외부 타당성](#slide-21-외부-타당성)
   - [Slide 22: Multilingual labels](#slide-22-multilingual-labels)
7. [Section 6 — Discussion (논의)](#section-6--discussion)
   - [Slide 23: Architecture-level reframe](#slide-23-architecture-level-reframe)
   - [Slide 24: 한계](#slide-24-한계)
8. [Section 7 — Conclusion (결론)](#section-7--conclusion)
   - [Slide 25: 결론](#slide-25-결론)
   - [Slide 26: Q&A](#slide-26-qa)

---

## Slide 1: Title

**슬라이드 내용**: "오픈소스 VLM 은 언제 원을 공으로 보는가?" + 부제 "5 개 오픈소스 VLM 에서 추상→물리 shortcut 의 행동 / 인과 / 픽셀 수준 분석" + 모델 목록.

**핵심 메시지**: 발표 한 줄 — "VLM 이 검은 원을 공으로 보는 현상의 원인을 5 개 모델에 걸쳐, 행동 / 메커니즘 / 픽셀 세 차원에서 분리한다."

**도메인 배경**:
- **VLM (Vision-Language Model)**: 이미지 + 텍스트를 함께 입력 받아 자연어 응답을 생성하는 모델. 본 연구에서는 Qwen2.5-VL, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3 5 종을 분석.
- **Shortcut**: 시각 증거가 여러 해석을 허용하지만 모델이 한 쪽으로 collapse 하는 현상. 본 연구의 핵심 관심사는 *추상 도형 (검은 원) → 물리 객체 (공)* 의 shortcut.

---

## Section 1 — Introduction

### Slide 2: 동기 — VLM 의 shortcut 문제

**슬라이드 내용**: 좌측에 5 개 bullet 으로 shortcut 문제 설명, 우측에 baseline 자극 (`line/blank/none` 흰 배경 위 검은 원).

**핵심 메시지**: 모델은 "흰 배경 위 검은 원" 만 봤는데 "공이 떨어진다" 라고 응답한다. 이게 shortcut.

**상세 설명**:
- 이미지에는 *물리적 단서가 전혀 없다*: 지면도, 그림자도, 텍스처도, 운동을 시사하는 화살표도 없음.
- 인간이 같은 자극을 보면 "흰 배경 위 검은 원" 즉 추상 도형으로 묘사한다.
- 그러나 Qwen2.5-VL 같은 모델은 동일 프롬프트 ("이 다음에 무슨 일이 일어날까?") 에 대해 "공이 중력으로 떨어진다", "공이 굴러간다" 같이 *물리* 응답을 한다.
- 시각 증거는 추상과 물리 둘 다와 호환되는데, 모델은 한쪽으로 collapse — 이게 shortcut.
- 기존 문헌 (Eyes Wide Shut, Tong et al. 2024) 가 anecdotal 하게 보고했지만, *모델 내부의 어디에서* 이 collapse 가 일어나는지는 미해결.

**우측 자극 그림**:
이 검은 원 자극이 본 연구 전체의 baseline. 슬라이드 18 (M5a steering) 에서 이 자극이 LM L10 α=40 개입으로 "추상 → 물리 객체" 로 뒤집힘. 슬라이드 19 (§4.6) 에서 이 자극에 작은 픽셀 perturbation 만 가해도 모델이 물리로 응답함.

---

### Slide 3: 본 연구의 핵심 질문

**슬라이드 내용**: 3 가지 차원 — 행동 / 메커니즘 / 픽셀 — 의 질문.

**핵심 메시지**: 본 연구는 shortcut 을 단순히 *발견* 하지 않고 *어디서 / 어떻게* 일어나는지 localize 한다.

**상세 설명**:

- **[행동 차원]**: 어떤 자극에서 / 어떤 모델에서 shortcut 이 강하게 나타나는가? 모델 간 차이가 *visual encoder 때문* 인가, *LM 때문* 인가, 아니면 *combination* 인가?

- **[메커니즘 차원]**: 모델 내부에서 "추상 vs 물리" 결정이 *어느 LM 레이어* 에서 일어나는가? 단일 선형 방향이 그 결정을 *causally* 만드는가?
  - "Causally" 는 단순 상관 관계가 아닌 *개입 (intervention)* 으로 검증한다는 뜻. 즉 "이 방향을 강제로 더하면 행동이 변하는가?"

- **[픽셀 차원]**: shortcut 이 이미지 자체에 *encode 가능* 한가? 픽셀에 작은 perturbation 만 가해서 행동을 뒤집을 수 있는가? 가능하다면, 어떤 architecture 에서 가능한가?
  - 이 질문이 제일 새롭고 중요. 픽셀 수준에서 인코드 가능하다면, shortcut 은 "input 의 속성" 이지 단순 "model 의 quirk" 가 아니다.

**연결**: 이 세 질문이 슬라이드 4 (3 가지 기여) 와 직접 매칭된다.

---

### Slide 4: 주요 기여 3 가지

**슬라이드 내용**: 본 연구의 paper-grade contribution 3 개.

**핵심 메시지**: 각 차원에서 명확한 finding 을 얻었음.

**상세 설명**:

**기여 1 — Architecture-level reframe (행동 차원)**:
- 5 모델 × 3 stim source × bootstrap CI 로, 행동 PMR 의 ceiling 이 *encoder representational capacity 만으로* 결정되지 않음을 disconfirm.
- 결정자는 **joint encoder + LM (architecture 수준)**.
- 가장 깨끗한 disconfirmer: LLaVA-1.5 와 LLaVA-Next — 같은 CLIP-ViT-L 인코더 계열을 쓰는데도 PMR 이 0.18 vs 0.70. 인코더가 같은데 행동이 0.52 만큼 차이남 → 인코더만으론 설명 불가.

**기여 2 — Causal localization (메커니즘 차원)**:
- Qwen2.5-VL 의 LM layer 10 에 단일 선형 방향 `v_L10` 이 존재.
- forward-hook 으로 `+α=40 · v_L10` 더하면 line/blank/none 자극의 10/10 응답이 "D: abstract" → "B: stays still" 로 flip.
- 다른 layer (15, 20, 25) 는 같은 α 로 안 움직임 — *L10 에 특정* 한 causal locus.
- M5a-ext: −α 도 D → B flip. v_L10 은 단방향 activator 가 아니라 *regime axis* (sign 이 +면 동적 / −면 정적).

**기여 3 — Pixel encodability — architecture-conditional (픽셀 차원)** (압축 high-level summary; Slide 19-20 에서 상세):
- 픽셀-공간 gradient ascent 가 **5 모델 중 3 모델 testable** 에서 PMR flip 가능 (Qwen broad / LLaVA-Next L20+L25 / LLaVA-1.5 L25 weak).
- **Idefics2 9 LM layers (L5-L31, 16-97% depth) 모두 0 shortcuts** → wrong-relative-depth falsified, **perceiver-resampler 가 leading remaining candidate**.
- **InternVL3** protocol-saturated.
- M9 PMR-ceiling / §4.7 결정-안정성 ceiling 과 평행한 *세 번째 architecture-level signature*, **projector design (MLP vs perceiver)** 를 disambiguating axis 로 노출.
- 자세한 데이터 + figure + perceiver isolation 한계는 Slide 19-20 참조.

---

## Section 2 — Related Work

### Slide 5: VLM grounding-failure / shortcut

**슬라이드 내용**: VLM shortcut 문헌 리뷰.

**핵심 메시지**: 기존 연구는 shortcut 의 *행동적* 관찰 위주이고 *메커니즘* 분석은 부재. 본 연구가 그 갭을 채움.

**상세 설명**:

- **Eyes Wide Shut (Tong et al., 2024)**: VLM 이 visual primitives 를 어떻게 놓치는지를 행동 수준에서 분류. MoF (Mixture-of-Features) 라는 처방을 제안. 본 연구의 출발점이지만 *왜* 그곳에서 *어떻게* 실패하는지 안 다룸.

- **Vision-language grounding failure**: VLM 이 "공간 추론", "객체 카운팅", "인과 관계" 등에서 어떻게 실패하는지 행동 통계 보고. 본 연구의 shortcut 은 이 우산 안에 있는 specific case (추상 → 물리).

- **Language-prior dominance**: VLM 이 시각 증거보다 텍스트/라벨에 더 의존하는 경향. 본 연구의 H7 (label-selects-regime) 와 직접 연결 — 같은 이미지에 circle vs ball 라벨을 붙이면 다른 응답을 유도.

**기존 한계**:
- 행동 수준에서 정지. *model 내부의 어디서 / 어떻게* 결정이 일어나는지를 mechanism-level 로 localize 한 작업이 VLM 에 거의 없음.
- 본 연구가 Qwen2.5-VL 의 L10 에서 v_L10 방향을 발견하고 그것이 인과적임을 보임으로써 이 갭을 채움.

---

### Slide 6: Probing / Causal interpretability

**슬라이드 내용**: Probing / 인과 분석 도구 4-5 개 review.

**핵심 메시지**: 본 연구가 사용한 메커니즘 분석 도구의 출처. 대부분 LM 전용 도구를 VLM 에 *처음으로* 적용.

**상세 설명**:

- **Linear probing (Alain & Bengio, Belrose et al.)**: 모델 hidden state 위에 logistic regression 을 올려서 "이 hidden state 가 어떤 정보를 인코드하는가?" 를 측정. 본 연구의 vision encoder probe (M3) 와 LM logit lens (M4) 가 이 기법.

- **Logit lens (nostalgebraist 외)**: LM 의 각 레이어 출력을 vocab projection 으로 변환해 "이 레이어에서 모델이 어떤 단어를 생각하고 있는가?" 추적. Tuned lens (Belrose et al.) 가 개선 버전.

- **Activation patching / SIP (Wang et al., Conmy et al.)**: clean-run 의 hidden state 를 corrupted-run 의 동일 위치에 patch 해서 그 위치의 *indirect effect* 측정. 본 연구의 후속 (M5b) 에서 사용 예정 — 현재는 보류.

- **VTI steering vectors / class-mean directions (Burns et al. 외)**: physics-mode = 1 vs = 0 인 stim 의 hidden state 평균 차이로 "physics-mode 방향" 을 추출. M5a 의 v_L10 이 이 기법.

- **Adversarial / feature visualization (Goodfellow, Madry, Olah 외)**: 픽셀 공간에서 모델 출력을 조작. §4.6 은 Olah 의 *feature visualization* 에 가까움 — class-mean 방향을 target 으로 한 픽셀-공간 gradient ascent.

---

## Section 3 — Problem & Definitions

### Slide 7: 문제 정의 — 두 차원 측정

**슬라이드 내용**: 좌측 행동 측정 (PMR/GAR/RC), 우측 메커니즘 측정 (probe / steering / pixel-encoding).

**핵심 메시지**: 본 연구의 측정은 행동과 메커니즘 두 축으로 나뉜다.

**상세 설명**:

행동 (Behavioral) 측정은 모델의 *output text* 에서 추출한 정량 값. 메커니즘 (Mechanistic) 측정은 모델의 *내부 hidden state* 에서 추출.

- **PMR**: 응답에 물리 동사가 포함된 비율. 예: "공이 떨어진다" → PMR=1; "원은 그대로 있다" → PMR=0. Rule-based scorer (다국어 stem 매칭). 자세한 정의는 슬라이드 9 참조.
- **GAR**: PMR 의 부분 — 응답이 *하방* 운동을 묘사한 비율. "떨어진다"는 GAR=1, "굴러간다"는 GAR=0.
- **RC**: 같은 자극을 N번 sampling 했을 때 응답이 일치하는 비율 (T=0.7).

- **Vision encoder probe**: 인코더의 hidden state 가 추상 vs 물리 셀을 *선형 분리* 할 수 있는가?
- **LM logit lens**: 각 LM 레이어의 hidden state 가 PMR 을 *예측* 하는가?
- **Causal intervention**: forward hook 으로 hidden state 에 vector 를 더하면 *output 이 변하는가*?
- **Pixel encoding**: 픽셀 공간 gradient ascent 가 *output 을 변하게 하는가*?

이 두 축이 결합되어 본 연구의 핵심 finding (architecture-level reframe + causal locus + pixel-encodability) 을 만든다.

---

### Slide 8: 자극 설계 — M2 5축 factorial

**슬라이드 내용**: 6 개 sample stim (line/blank/none ~ textured/blank/none) + 우측에 5축 설명.

**핵심 메시지**: 본 연구의 핵심 자극 set 은 5축 factorial — 추상화 / 배경 / 물리 cue / 이벤트 / seed 를 모두 조합한 2880 stim.

**상세 설명**:

5 축 정의:

1. **object_level** (추상화 축): line / filled / shaded / textured.
   - line: 윤곽만; filled: 회색 채움; shaded: 3D 음영 (공처럼 보임); textured: 재료 cue (나무/돌).
   - "얼마나 물리적으로 보이는가?" 의 4 단계.

2. **bg_level** (배경 축): blank / ground / scene.
   - blank: 흰 배경; ground: 한 줄 지면; scene: 풍경 (지평선 + 하늘).

3. **cue_level** (물리 cue 축): none / cast_shadow / motion_arrow / both.
   - cast_shadow: 객체 아래 그림자; motion_arrow: 움직임 화살표.

4. **event** (방향 축): fall / rise / horizontal.
   - 프롬프트 framing. "What will happen next?" 같은 open-ended 사용.

5. **seed**: 10 개 — 각 셀에서 stim 위치 / 모양을 약간 randomize.

총 4 × 3 × 4 × 1 × 10 = 480 stim (single event), × 6 prompt-variants (3 라벨 + 3 prompt mode) = 2880 추론.

**왜 이렇게?**: shortcut 의 강도가 어떤 axis 에 의존하는지 찾기 위해. 예를 들어 "ground line 추가만으로 PMR 이 +21pp 증가" 같은 효과를 axis 분리로 측정.

이 자극 set 이 M2 (Qwen 만), 그 후 M6 r1 (LLaVA-1.5), M6 r7 (5 모델 모두) 로 확장.

---

### Slide 9: 메트릭 정의

**슬라이드 내용**: 7 개 메트릭 (PMR variants, GAR, RC, paired-delta, v_L, projection) 정의 표.

**핵심 메시지**: 본 연구의 모든 정량 결과는 이 7 개 메트릭으로 표현됨.

**상세 설명**:

각 메트릭의 *왜 이게 필요한가*:

- **PMR(_label)**: 라벨 ("the ball...", "the circle...") 이 물리 응답에 미치는 영향 측정.
- **PMR(_nolabel)**: 라벨 없는 baseline. 모델의 *image-only* 추론을 측정 — H2 null test 의 anchor.
- **GAR**: 응답이 단순 "물리적이다" 가 아니라 *gravity-aligned* 인지. ball 응답은 GAR 높고, planet 응답은 GAR 낮다 (orbital).
- **RC**: 모델의 결정이 안정적인가? T=0.7 sampling 에서 5/5 같은 응답이면 RC=1. 천장에 가까운 모델은 RC=1 로 saturate.
- **Paired-delta H2**: PMR(label) − PMR(_nolabel). 라벨이 baseline 위에 *얼마만큼* 추가 PMR 을 만드는지. **per-stim 매칭** — 같은 이미지의 두 응답을 비교하여 noise 줄임.
- **Class-mean v_L**: M5a 의 핵심 — physics-mode 와 abstract 의 layer-L hidden state 평균 차이. 이게 "physics-mode 방향" 의 추정치.
- **v_L projection**: 임의 hidden state h 를 v_unit_L 에 사영. \"physics-mode 강도\" 의 단일 스칼라.

Bootstrap 5000-iter CI 가 각 메트릭에 적용됨 (prediction-level resampling).

---

### Slide 10: 테스트 모델 — 5 개 오픈소스 VLM

**슬라이드 내용**: 5 모델 비교 표 (Vision encoder, LM, 이미지 처리).

**핵심 메시지**: 본 연구는 5 개 모델을 동일 protocol 로 비교 — encoder × LM 의 조합 효과를 분리 가능.

**상세 설명**:

| Model | Vision encoder | LM | 이미지 처리 |
|---|---|---|---|
| Qwen2.5-VL-7B | SigLIP | Qwen2-7B | 동적 504×504 |
| LLaVA-1.5-7B | CLIP-ViT-L/14 | Vicuna-7B (LLaMA-2) | 고정 336×336 |
| LLaVA-Next-7B | CLIP-ViT-L/14 | Mistral-7B | AnyRes 5-tile |
| Idefics2-8B | SigLIP-SO400M | Mistral-7B | 384×384 |
| InternVL3-8B | InternViT-300M | InternLM3-8B | 동적 448×448 |

**Encoder 계열**:
- SigLIP × 2 (Qwen, Idefics2 SO400M-variant)
- CLIP × 2 (LLaVA-1.5, LLaVA-Next)
- InternViT × 1 (InternVL3)

**LM 계열**:
- Qwen2 × 1
- Vicuna (LLaMA-2 기반) × 1
- Mistral × 2 (LLaVA-Next, Idefics2)
- InternLM3 × 1

**왜 이 다양성이 중요한가**:
- Encoder 효과를 분리: SigLIP-Mistral (Idefics2) vs CLIP-Mistral (LLaVA-Next) — 같은 LM, 다른 인코더.
- LM 효과를 분리: CLIP-Vicuna (LLaVA-1.5) vs CLIP-Mistral (LLaVA-Next) — 같은 인코더, 다른 LM + 다른 이미지 처리.
- Cross-encoder swap: §4.5 에서 Idefics2 가 SigLIP-계열 saturation 을 *LM-independent* 임을 입증.

이 5 모델 set 이 cross-architectural 분리의 도구 역할을 한다.

---

## Section 4 — Implementation

### Slide 11: 파이프라인

**슬라이드 내용**: 6 단계 파이프라인 — stim 생성 → inference → capture → score → probe → causal intervention.

**핵심 메시지**: 본 연구의 모든 결과는 이 6 단계의 조합으로 만들어진다.

**상세 설명**:

1. **Stimulus generation**: PIL/matplotlib 으로 결정론적 stim 생성. EvalConfig + FactorialSpec 으로 axes 정의 → 480 stim 자동 생성. seed 가 randomize 영역만 흔들고 cell label 은 결정론적.

2. **Inference**: `AutoModelForImageTextToText` 로 모든 모델 동일 코드 path. predictions.{jsonl, parquet, csv} 가 *streaming* 으로 저장 (crash-safe).

3. **Activation capture (옵션)**: `capture_lm_layers=(5,10,15,20,25)` 로 forward hook 등록 → 매 stim 의 LM hidden state 를 safetensors 로 저장. 비전 인코더도 동일 (`capture_vision_layers`).
   - 디스크: M2 (480 stim, 5 LM layer + 6 vision layer) → ~5-15 GB.

4. **Scoring**: `score_pmr()` rule-based scorer. ABSTRACT_MARKERS 로 "no movement" / "remain stationary" 같은 추상 응답을 *gate* 해서 false-positive 방지. PHYSICS_VERB_STEMS 로 물리 동사 (fall, roll, drop 등) 매칭. 다국어 (영/한/일/중) stem 모두 지원.
   - Hand-validation 5-6% disagreement vs 사람 라벨링.

5. **Probing**: scikit-learn logistic regression. 5-fold CV AUC. behavioral_y (per-stim PMR) 또는 stim_y (factorial cell label) 두 가지 target.

6. **Causal intervention**: `scripts/06_vti_steering.py` (M5a runtime steering) 와 `scripts/sec4_6_counterfactual_stim.py` (§4.6 픽셀-공간 gradient ascent). 다음 슬라이드 (12-13) 에서 자세히.

---

### Slide 12: 활성화 캡처 + 선형 프로빙

**슬라이드 내용**: 캡처/프로빙 방법 + 5-model encoder chain figure.

**핵심 메시지**: "인코더가 추상 vs 물리를 선형 분리하는가?" 와 "LM 의 어느 레이어가?" 라는 질문을 logistic regression 으로 정량화.

**상세 설명**:

**Vision encoder probe**:
- 인코더의 selected layers (3, 7, 11, 15, 19, 23) 에 forward hook → 각 stim 의 *시각 토큰 위치* 의 hidden state 를 mean-pool → (n_stim, hidden_dim) tensor.
- 이 tensor 위에 logistic regression (sklearn) 5-fold CV → AUC.

**LM probe**:
- LM 의 selected layers (5, 10, 15, 20, 25) 에 동일 처리.
- 단, *시각 토큰 위치만* 마스킹 (image_token_id 매칭).
- 결과: 5 모델 × 5-6 layer × 2 y-target = 50-60 AUC 셀.

**Y target 두 가지**:
- **behavioral_y**: 각 stim 의 per-model PMR call (0/1). "이 모델 응답이 물리였는가?"
- **stim_y**: factorial cell 라벨 (예: "이 stim 이 line/blank/none 인가 textured/ground/both 인가?"). 인코더 식별 능력 측정.

**Headline (figure)**:
5 모델 모두 stim_y AUC = 1.0 (모든 인코더가 자극을 trivially 분리). behavioral_y AUC 만 0.73 (CLIP) ↔ 0.99 (SigLIP) 로 분리.

→ **인코더 차이는 단순 "정보를 인코드하는가" 가 아니라 "PMR 을 어떻게 매핑하는가" — 이 차이는 LM-side 처리가 결정**. 슬라이드 23 의 architecture-level reframe 의 기초.

---

### Slide 13: 인과 개입 — VTI steering + §4.6 픽셀 ascent + M5b SAE 개입

**슬라이드 내용**: 세 가지 인과 개입 방법.

**핵심 메시지**: 단순 상관 관계가 아닌 *causal* 분석을 위해 *세 가지* 보완적 개입을 수행. (1) LM-side runtime steering, (2) pixel-space gradient ascent (역방향), (3) encoder-side SAE feature ablation.

**상세 설명**:

**M5a — runtime VTI steering (forward hook)**:

```
v_L = mean(h_L | PMR=1) − mean(h_L | PMR=0)
v_unit_L = v_L / ||v_L||
```

- v_L 을 M2 captures 의 480 stim 위에서 계산.
- *개입*: model.model.language_model.layers[L] 의 forward 출력 hidden_states 에 `α · v_unit_L` 을 더한다. 모든 token position 균일.
- L ∈ {10, 15, 20, 25}, α ∈ {0, 5, 10, 20, 40, −α}. T=0 (deterministic).
- **이게 인과 개입인 이유**: hidden state 를 *직접 조작* 해서 *그 결과 출력이 어떻게 변하는가* 측정. 단순 correlation 이 아니라 do-operator.

**§4.6 — pixel-space gradient ascent**:

- *최적화 변수*: post-processor pixel_values tensor.
  - Qwen2.5-VL: T_patches × 1176 (patch-flattened normalized 표현).
  - LLaVA-1.5: 1×3×336×336 (표준 CLIP 형식).
- *목적함수*: `<mean(h_L10[visual]), v_L10>` 최대화.
  - 즉 "L10 의 시각 토큰 hidden state 를 v_L10 방향으로 가능한 한 멀리 밀어라."
- *최적화*: Adam, lr=1e-2, n_steps=200. float32 leaf → bf16 cast 로 vision tower → projector → LM 0..10 미분 가능 (forward 만 differentiable, generate 는 안 함).
- *제약*: L_∞-bounded `clamp_(δ, -ε, +ε)` ε ∈ {0.05, 0.1, 0.2} 또는 unconstrained.
- **Random control**: 매칭 ε=0.1 에서 random unit direction 에 대해 동일 최적화. magnitude 매칭 controls falsify "any perturbation flips" alternative.

**M5b — encoder-side SAE feature ablation (2026-04-27 → 2026-04-28 cross-model)**:

- **개입 위치**: vision encoder 의 마지막 (모델별로 LM 이 *실제로 소비* 하는) layer 의 forward output 에 hook 으로 *features 빼기*.
- **모델별 hook layer** (LLaVA `vision_feature_layer=-2` 같은 model-specific routing 반영):
  - Qwen2.5-VL: `model.visual.blocks[31]` (마지막 layer, 32 layers).
  - LLaVA-1.5 / LLaVA-Next: `vision_tower.encoder.layers[22]` (penultimate, `vision_feature_layer=-2` convention; layers[23] 은 *버려짐*).
  - Idefics2: `vision_model.encoder.layers[26]` (마지막 layer of 27, `last_hidden_state` 로 들어감).
  - InternVL3-hf: `vision_tower.encoder.layer[23]` (마지막 layer of 24, `vision_feature_layer=-1`).
- **SAE 학습**: 4× overcomplete (n_features = 4 × d_in), λ=1.0 L1 sparsity, 5K steps Adam lr=1e-3 batch=4096, input z-score, tied weights.
- **Feature ranking**: `delta = mean(z|physics) − mean(z|abstract)`; **Cohen's d = delta / pooled_std** (high-baseline outlier 필터링). Phys / Abs 라벨링은 per-stim mean PMR ≥ 0.667 (phys) 또는 ≤ 0.333 (abs) — saturated 모델 (LLaVA-Next / Idefics2 / InternVL3) 은 ≤ 0.5 로 완화.
- **개입 방법**: top-k physics-cue features (Cohen's d 순) 의 *raw-scale contribution* 을 vision encoder hidden state 에서 빼기 (Bricken et al. 2023 trick: SAE 의 z-score 정규화 round-trip 을 보존하면서 다른 features 와 reconstruction residual 은 bit-identical 유지).
- **Random control (3 sets)**: high-mass non-top-k pool 에서 랜덤 추출, [0.7×, 2×] top-k mass window 에서 magnitude-matched. 3 random sets × top-k features 의 ablation 으로 비교.
- **k sweep**: {5, 10, 20, 40, 80, 160}; LLaVA-1.5 는 high-k extension {200, 300, 500, 800} 까지 확장.

**세 개입의 상보성**:
- M5a (LM-side): "이 LM direction 이 행동을 인과적으로 결정하는가?" 를 묻는 *forward* 개입.
- §4.6 (pixel-side): "이 LM direction 이 *역방향* 으로 픽셀에서 인코딩 가능한가?" 를 묻는 *inverse* 개입.
- M5b (encoder-side): "encoder 가 이 LM direction 의 *국소화된 feature* 표현을 가지는가?" 를 묻는 *encoder-side localization* 개입.

이 세 개입이 슬라이드 18 (M5a Qwen), 18b (M5a cross-model), 18c (M5b cross-model), 19-20 (§4.6) 에서 결과로 보여짐.

---

## Section 5 — Experiments / Results

### Slide 14: 5-model M2-stim PMR 사다리

**슬라이드 내용**: m2_cross_model_pmr_ladder.png 그림 + 우측에 표 + 핵심 인용.

**핵심 메시지**: 5 모델이 M2 stim 에서 보이는 PMR 을 동일 스케일로 비교 — 3 cluster 로 분리.

**상세 설명**:

**5-model M2-stim PMR(_nolabel) 사다리** (480-stim 평균 ± 95% bootstrap CI):

| Model | PMR(_nolabel) | 95% CI | Cluster |
|---|---:|---|---|
| LLaVA-1.5 | **0.383** | [0.34, 0.43] | Floor (CLIP+Vicuna) |
| LLaVA-Next | 0.790 | [0.75, 0.83] | Mid (CLIP+Mistral+AnyRes) |
| Qwen2.5-VL | 0.938 | [0.92, 0.96] | Saturated |
| Idefics2 | 0.967 | [0.95, 0.98] | Saturated |
| InternVL3 | 0.988 | [0.98, 1.00] | Saturated |

**3 cluster 의 의미**:

- **Floor (LLaVA-1.5)**: PMR 0.18 — 모델이 abstract 응답을 자주 함. CLIP-ViT-L 인코더 + Vicuna LM. *Unsaturated*.
- **Mid (LLaVA-Next)**: PMR 0.70 — LLaVA-1.5 와 같은 CLIP 인코더 family 를 쓰는데도 PMR 0.52 만큼 높음.
- **Saturated**: Qwen / Idefics2 / InternVL3 모두 0.94 이상. 이미지 보고 거의 항상 물리 응답.

**Critical point**: LLaVA-1.5 vs LLaVA-Next 는 *같은 CLIP-ViT-L 인코더 family*. PMR 0.52 차이는 인코더로 설명 안 됨. LM 변경 (Vicuna → Mistral) + AnyRes tiling + projector + training 이 4축 confound 라 어느 게 결정적인지 isolating 안 되지만, *인코더만으론 부족하다* 는 disconfirmation 은 깨끗함.

→ **architecture-level reframe** 의 가장 강한 증거. 슬라이드 23 으로 연결.

---

### Slide 15: H1 abstraction ramp

**슬라이드 내용**: m2_cross_model_h1_ramp.png + 우측 해석.

**핵심 메시지**: 추상화 ramp (line→filled→shaded→textured) 가 *unsaturated 모델 (LLaVA-1.5)* 에서만 깨끗하게 보이고, 다른 모델은 천장.

**상세 설명**:

H1 (abstraction ramp) 가설:
- "더 추상적인 stim (line) 에서 더 사실적인 stim (textured) 으로 갈수록 PMR 이 monotonic 증가한다."

**모델별 ramp range** (line PMR → textured PMR):

| Model | line | filled | shaded | textured | range |
|---|---:|---:|---:|---:|---:|
| LLaVA-1.5 | 0.51 | 0.66 | 0.75 | 0.81 | **+0.30** |
| LLaVA-Next | 0.84 | 0.94 | 0.98 | 0.98 | +0.14 |
| Idefics2 | 0.87 | 0.96 | 0.96 | 0.96 | +0.09 |
| Qwen | 0.88 | 0.89 | 0.91 | 0.93 | +0.05 |
| InternVL3 | 0.98 | 0.99 | 1.00 | 1.00 | +0.02 |

**해석**:
- LLaVA-1.5 에서 H1 의 깨끗한 monotonic S-curve. 추상화 axis 의 효과를 axis 별로 분리 측정 가능.
- 다른 모델은 천장에 너무 가까워서 axis 효과가 *압축*. line 이미 ~0.87-0.98 PMR, textured 도 비슷한 수준.

**시사점**: H1 ramp 의 측정은 *unsaturated 인코더* 가 필요한 *operationally constrained* 가설이다. M8a 에서 5 도형 cross-shape strict scoring 결과 — Qwen 1/4, LLaVA 4/4 PASS — 이 비대칭이 saturation 가설의 cross-shape validation.

---

### Slide 16: H2 paired-delta — 3 가지 패턴

**슬라이드 내용**: m2_cross_model_h2_paired_delta.png + 우측에 3 패턴 설명.

**핵심 메시지**: H2 는 단순 "라벨이 항상 PMR 을 더한다" 가 아님. **encoder saturation 이 어떤 패턴이 적용되는지를 결정**.

**상세 설명**:

H2 paired-delta = `PMR(label) − PMR(_nolabel)` (per-stim 매칭).

3 가지 architecture-conditional 패턴:

**패턴 1 — LLaVA-1.5 / LLaVA-Next** (unsaturated CLIP 인코더):
- ball Δ = +0.475 (LLaVA-1.5) / +0.190 (LLaVA-Next). 모두 양수.
- circle Δ = +0.173 / +0.081
- planet Δ = +0.244 / +0.160
- "Classical H2" — 모든 라벨이 baseline 위로 PMR 을 끌어올림.

**패턴 2 — Qwen / Idefics2** (포화 SigLIP-계열):
- Qwen ball ≈ 0, circle −0.088, planet ≈ 0. circle 라벨이 *baseline 아래* 로 억제.
- Idefics2 ball ≈ 0, circle −0.040, planet −0.085. 비-물리 라벨이 모두 *억제*.
- "**Circle override**" 패턴 — abstract-leaning 라벨이 saturated image-prior 를 less commitment 방향으로 override.

**패턴 3 — InternVL3** (완전 포화):
- 모든 Δ ≈ 0 (ball +0.012, circle +0.010, planet −0.004).
- 라벨이 작용할 *headroom* 자체가 없음.

**의의**:
- H2 의 원래 framing (Hypothesis 2: 라벨이 PMR 을 끌어올림) 은 *unsaturated 인코더 한정* 으로 좁혀짐.
- 포화된 모델은 *반대* 패턴 (label override) 보임 — 이게 M4b ("circle override on Qwen") 의 cross-model 일반화.
- → **H2 가 architecture-level 효과** (인코더 포화도와 LM 의 라벨-prior 결합) 임이 5 모델로 입증.

---

### Slide 17: Vision encoder probe

**슬라이드 내용**: encoder_chain_5model.png + 우측에 모델별 AUC 리스트.

**핵심 메시지**: 인코더가 추상 vs 물리를 *얼마나 분리하는가* 는 5 모델에 걸쳐 0.73 ↔ 0.99 로 분포. 그러나 이 차이는 *직접* PMR 을 설명하지 못함.

**상세 설명**:

5-model behavioral-y AUC chain (M8a stim, apples-to-apples):

| Model | Encoder | behavioral-y AUC | PMR(_nolabel) |
|---|---|---:|---:|
| Qwen2.5-VL | SigLIP | 0.99 | 0.84 |
| Idefics2 | SigLIP-SO400M | 0.93 | 0.88 |
| InternVL3 | InternViT | 0.89 | 0.92 |
| LLaVA-Next | CLIP-ViT-L | 0.77 | 0.70 |
| LLaVA-1.5 | CLIP-ViT-L | 0.73 | 0.18 |

**Stim-defined y 로 모든 인코더 AUC = 1.0**:
- y target 을 "이 stim 이 어떤 factorial cell 인가?" 로 바꾸면 모든 인코더가 trivially 1.0 분리.
- → 인코더의 *표현 능력* 자체는 균일.

**Behavioral-y AUC 의 의미**:
- "인코더 hidden state 가 *모델 자신의 PMR 출력* 을 얼마나 잘 예측하는가?"
- 0.99 (Qwen) vs 0.73 (LLaVA-1.5) 의 차이는 인코더가 "이 stim 에서 모델이 어떤 PMR 응답을 할지" 정보를 얼마나 인코드하는가의 차이.
- LLaVA-1.5 의 인코더가 "정보 부족" 한 게 아니라 "PMR 매핑이 noisy" 한 것 — encoder 의 weak link.

**시사점**:
- 인코더 AUC 의 차이가 PMR 의 *직접 원인은 아님*. 0.77 (LLaVA-Next) → 0.73 (LLaVA-1.5) 으로 AUC 가 비슷해도 PMR 은 0.70 → 0.18 로 0.52 차이.
- Architecture-level 효과 — encoder × LM 의 결합이 결정자.

---

### Slide 17b: M4 LM logit-lens cross-model — 5-model × 5-layer probe AUC

**슬라이드 내용**: m4_lm_probing_cross_model.png 그림 + AUC ladder 표.

**핵심 메시지**: Encoder probe AUC ladder (Slide 17) 와 *동일한 architecture-level 분류* 를 LM probe 에서도 발견. **추가**: Idefics2 의 LM AUC (0.995) 가 vision AUC (0.93) *보다 높음* — perceiver-resampler 가 정보를 *제거하지 않고 오히려 집중*. encoder-saturation 의 *두 번째 downstream 시그니처*.

**상세 설명**:

**Setup**:
- 기존 M2 cross-model captures 재사용 (no new inference). 5 models × 5 LM layers (5/10/15/20/25) 의 visual-token mean hidden state 를 feature 로.
- Y label: per-stim mean PMR 의 binary threshold (≥ 0.667 phys, ≤ 0.333 abs).
- Probe: scikit-learn `LogisticRegression` with `class_weight="balanced"`, 5-fold StratifiedKFold (saturated 모델은 imbalanced — n_pos / n_neg 비율 매우 큼).
- 메트릭: 5-fold mean AUROC.

**Result — 5-model × 5-layer LM probe AUC**:

| Model | L5 | L10 | L15 | L20 | L25 | M3 vision AUC | n_phys / n_abs |
|---|--:|--:|--:|--:|--:|--:|---:|
| **Idefics2-8B** | **0.995** | **0.995** | **0.995** | **0.995** | **0.995** | 0.93 | 390 / 5 |
| Qwen2.5-VL-7B | 0.965 | 0.965 | 0.962 | 0.959 | 0.957 | 0.99 | 310 / 19 |
| LLaVA-Next-Mistral-7B | 0.732 | 0.762 | 0.751 | 0.786 | 0.791 | 0.81 | 393 / 9 |
| LLaVA-1.5-7B | 0.758 | 0.760 | 0.762 | 0.763 | 0.768 | 0.73 | 327 / 13 |
| InternVL3-8B | NaN | NaN | NaN | NaN | NaN | 0.89 | 479 / 1 |

**3 가지 발견**:

1. **LM AUC ladder 가 vision AUC ladder 와 정렬**:
   - Non-CLIP 인코더 (Qwen / Idefics2 / InternVL3-경향) → LM AUC 0.96 ~ 0.99.
   - CLIP 인코더 (LLaVA-1.5 / LLaVA-Next) → LM AUC 0.73 ~ 0.79.
   - **H-encoder-saturation 이 downstream LM 으로 propagate** — 인코더 표현능력이 LM 의 visual-token 위치에서의 PMR-relevant 정보 인코딩 정도를 결정.

2. **Idefics2 LM AUC (0.995) > Idefics2 vision AUC (0.93)**:
   - Vision encoder 출력 (1296-token SigLIP-SO400M) 이 perceiver-resampler 로 320-token 으로 *압축* 됨.
   - 압축된 budget 에서 LM probe AUC 가 *더 높음* — perceiver 가 정보를 stripping 하지 않고, 오히려 *physics-mode 신호를 시각 토큰 위치로 집중* 시킴.
   - **§4.6 Idefics2 anomaly 와의 dissociation**: §4.6 에서 Idefics2 의 픽셀-공간 gradient ascent 는 9 layers 모두 0 flip; 그러나 M4 LM probe 는 0.995 — 즉 *정보는 LM 에 있는데 픽셀-공간에서 방향이 routable 하지 않다*.

3. **Architecture-level 의미**:
   - "정보 존재" (M4 LM probe) ≠ "픽셀-공간 shortcut routability" (§4.6 픽셀 ascent).
   - Perceiver-resampler 의 bottleneck 효과는 *정보 stripping* 이 아니라 *픽셀-공간 gradient routability 의 차단*.
   - H-shortcut 가설의 정밀화: pixel-encodability 는 *encoder 가 정보를 가지는가* 가 아니라 *projector 디자인이 픽셀-공간에서 그 방향을 routable 하게 두는가* 의 함수.

**Limitations**:
- InternVL3 untestable (n_neg=1, probe degenerate).
- Per-stim mean across visual tokens 으로 positional structure 손실 (perceiver-compressed 320 vs CLIP 576 vs AnyRes 2928 — 토큰 수 다름).
- Idefics2 LM dim 4096 (Mistral) vs vision dim 1152 (SigLIP-SO400M) — LM AUC 의 일부는 *higher-dim 인코딩 공간* 효과 (4096-d probe 가 1152-d 보다 capacity 더 큼). 단 dissociation 논거는 AUC *크기* 가 아니라 ≫ chance 에 의존하므로 robust.

**시사점**:
- 본 결과가 §4.6 cross-model + M5a cross-model 과 triangulation 됨.
- Idefics2 의 case: LM 에 신호 있음 (M4 0.995) + forward-hook 작동 (M5a 10/10) + 픽셀-공간 inverse 차단 (§4.6 0/9) → **perceiver 가 inverse pathway 만 차단**.

---

### Slide 18: M5a VTI causal steering (Qwen)

**슬라이드 내용**: First-letter 응답 분포 표 + line/blank/none 자극 그림 + 응답 비교.

**핵심 메시지**: Qwen2.5-VL 의 LM L10 에 forward-hook 으로 +α=40·v_L10 더하면 line/blank/none 자극의 10/10 응답이 *D (abstract) → B (stays still)* 로 flip. 다른 layer 는 안 움직임.

**상세 설명**:

**Setup**:
- Test stim: line/blank/none × 10 seeds.
- Prompt: forced-choice with options A (rolls), B (stays still), C (moves up), D (abstract). Label = "circle" (baseline PMR ≈ 0).
- α ∈ {0, 5, 10, 20, 40}, layer L ∈ {10, 15, 20, 25}.
- T=0 (deterministic). 200 inferences total.

**Result table** (first-letter 응답 분포):

| layer | α=0 | α=5 | α=10 | α=20 | α=40 |
|---|---|---|---|---|---|
| 10 | 10 D | 10 D | 10 D | 10 D | **10 B** |
| 15 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 20 | 10 D | 10 D | 10 D | 10 D | 10 D |
| 25 | 10 D | 10 D | 10 D | 10 D | 10 D |

L10 α=40 만 flip. 다른 layer 는 같은 magnitude 로 절대 안 움직임.

**응답 텍스트 비교**:

baseline (α=0):
> "D — This is an abstract shape and as such, it does not have physical properties that would allow it to fall, move, or change in any way..."

L10 α=40 (intervention):
> "B) It stays still — Justification: The circle in the image appears to be floating or suspended in space without any external force acting upon it..."

→ "abstract shape" → "physical object (stationary)" 의 *categorical shift*. 인터벤션이 단순 텍스트 변경이 아니라 *해석* 의 변경.

**M5a-ext follow-up**:
- Exp 3: −α=40 도 D → B flip (textured/blank/none baseline 에서). 즉 v_L10 은 binary toggle 이 아니라 **regime axis**.
- +α: kinetic (rolls / falls); −α: static (stays still). Sign 이 regime 선택, 둘 다 physics-mode 활성화.

**시사점**:
- shortcut 의 *causal locus* 가 LM L10 에 단일 선형 방향으로 존재.
- 이는 **H-locus** 가설 (mid-LM 결정 layer) 의 직접 증거.
- 후속 슬라이드에서 (a) cross-model M5a (18b), (b) cross-model M5b SAE encoder-side (18c), (c) §4.6 픽셀 reverse (19) 에서 *세 가지 보완적 시그니처* 로 확인.

---

### Slide 18b: M5a runtime steering — cross-model (3 of 4 testable models)

**슬라이드 내용**: 4 models × 응답 비교 표 + 모델별 sweet-spot α + 응답 텍스트.

**핵심 메시지**: Qwen 만의 결과로 보였던 **forward-hook v_L injection** 이 **3 of 4 testable models** 로 확장됨. **Idefics2 결과 (10/10 flip at L25 α=20)** 가 §4.6 cross-model 의 perceiver-resampler 가설을 정밀화 — *forward 방향은 작동, inverse 픽셀-공간만 차단*.

**상세 설명**:

**Setup (per model)**:
- 모델별 v_L 추출: M2 cross-model captures 재사용 (M6 r7 결과). v_L = mean(h_L | PMR=1) − mean(h_L | PMR=0) at 모델별 LM hidden state.
- 모델별 hook layer (각 모델의 baseline-PMR=0 가능한 layer 선택):
  - Qwen2.5-VL: L10 (M5a 원본), L15/20/25 sweep.
  - LLaVA-Next-Mistral-7B: L20, L25 (cross-model §4.6 에서 가장 강한 shortcut layer).
  - Idefics2-8B (Mistral-7B LM): L20, L25.
  - LLaVA-1.5-7B (Vicuna-7B LM): L25 (cross-model §4.6 에서 4/10 weak shortcut layer).
  - InternVL3-8B: 모든 cell baseline=1 → 측정 불가능 (saturated, no abstract baseline).
- α sweep: 모델별 dynamic range 다름 — Qwen 0~60, LLaVA-Next 0~20, Idefics2 0~60, LLaVA-1.5 0~60.
- Stim cell: per-model baseline-PMR=0 cell 선택.
  - Qwen: line/blank/none × circle (PMR=0).
  - LLaVA-Next: line_blank_both × circle (PMR=0.3, 가장 abstract-mode 가 많은 cell).
  - Idefics2: line_blank_none × circle (PMR=0, perfect abstract baseline).
  - LLaVA-1.5: line_blank_none × circle (PMR=0).
- Prompt: open-ended ("circle 다음에 무슨 일이 일어날지 한 문장으로"). PMR scorer 로 점수.

**Result table — 모델별 sweet-spot α 에서의 PMR flip rate**:

| Model | Layer | α sweet spot | PMR flip | Stim cell (baseline) | 대표 응답 |
|---|---:|---:|---:|---|---|
| Qwen2.5-VL-7B | L10 | 40 | **10/10** | line/blank/none × circle (0) | "The ball is falling down due to gravity." |
| **LLaVA-Next-Mistral-7B** | **L20** | **10** | **10/10** | line_blank_both × circle (0) | "The ball will roll down the ramp." |
| **LLaVA-Next-Mistral-7B** | **L25** | **15-20** | **10/10** | line_blank_both × circle (0) | "The ball will bounce up." |
| **Idefics2-8B** | **L25** | **20** | **10/10** | line_blank_none × circle (0) | "The tip of the arrow will hit the center of the circle." |
| **Idefics2-8B** | **L20** | **20** | **10/10** | line_blank_none × circle (0) | (paper-quality physics commit) |
| LLaVA-1.5-7B | L25 | (any 0-60) | **0/10** | line_blank_none × circle (0) | (NULL — 응답 변하지만 motion stem 안 잡힘) |
| InternVL3-8B | — | — | (untestable) | n/a (baseline=1) | n/a |

**LLaVA-1.5 NULL 의 의미**:
- 응답은 의미 변화 — α=0 "The circle will be filled with a color" → α=20 "in the center" → α=60 "on the floor" — 그러나 PMR scorer 가 motion stem (falls/rolls/bounces) 매칭 못함.
- α=60 "on the floor" 는 location 함의 (gravity) 만 — motion verb 부재.
- 즉 *runtime steering 은 부분적으로 작동* 하지만 PMR 으로 측정 가능한 break 임계점에 못 도달.
- 이는 §4.6 LLaVA-1.5 weak shortcut (L25 4/10) 와 일관된 *encoder bottleneck* 시그니처.

**Idefics2 의 paper-changing 결과**:
- Idefics2 L25 α=20: 10 stim 모두 동일 응답 — "The tip of the arrow will hit the center of the circle." → **regime-attractor**: sweet-spot α 에서 LM 이 *deterministic physics-mode attractor* 로 들어감.
- 이는 M5a-ext 의 regime-axis finding 과 일관 — sufficient α 에서 regime 이 *forcibly selected*, visual content tracking 은 moderate α 에서.

**Triangulation with M4 + §4.6 (Idefics2 case)**:

| Test on Idefics2 | Result | Implication |
|---|---|---|
| M4 LM probe AUC (Slide 17b) | 0.995 across L5-L25 | 정보가 *LM 에 도달* |
| **M5a runtime steering** | **L25 α=20 → 10/10 flip** | Forward-direction `v_L` 이 LM 에서 *작동* |
| §4.6 픽셀-공간 gradient ascent (L5-L31) | 0/90 v_unit + 0/90 random | Inverse pixel→LM **routability 차단** |

**Refined H-shortcut + perceiver-resampler 가설**:
- Perceiver-resampler 는 physics-mode 신호를 *strip 하지 않음* — LM 이 정보 보유 (M4 0.995) + forward-hook 이 그 정보 활용 (M5a 10/10).
- Perceiver 가 제거하는 것은 *픽셀-공간 gradient routability* — 즉 *역방향* (pixels → v_L direction) pathway 의 차단.
- Bottleneck 은 *inverse* (픽셀-사이드) pathway 이지 *forward* (v_L direction → LM commitment) pathway 가 아님.
- 이 refinement 가 §4.6 cross-model 결과를 정밀화 — paper-grade.

**α dynamic range 가 모델별 다름**:
- Qwen 40, LLaVA-Next 5-15, Idefics2 20.
- α > sweet spot 시 모델별 *token degeneracy* 발생 — Qwen 은 "rock rock rock", LLaVA-Next 도 비슷, Idefics2 는 "tip tip tip tip...".
- PMR scorer 가 repetition 의 motion stem 을 false-positive 로 잡을 수 있음 → α=40+ rates 는 pure flip 아닌 mixed 측정.

**시사점**:
- Causal localization 이 Qwen-only 에서 **3-model cross-model** 로 확장 (paper Contribution 2 강화).
- LLaVA-1.5 의 NULL 은 encoder-bottleneck 시그니처 — *어떤 layer 에서도 LM-side direction 이 PMR break 임계점 도달 못함*.
- Idefics2 forward-positive + inverse-negative dissociation 으로 perceiver-resampler 가설 정밀화.

---

### Slide 18c: M5b SAE intervention — encoder-side cross-model (round 2, 2026-04-28)

**슬라이드 내용**: m5b_sae_intervention_cross_model.png drop curves + 5-model k-break 표.

**핵심 메시지**: Qwen 의 M5b SAE intervention (top-20 ablation 으로 PMR break) 을 **5-model 로 확장**. **3 of 5 모델 (Qwen / Idefics2 / InternVL3) clean break, 2 of 5 LLaVA NULL**. Encoder-side feature localization 이 *architecture-conditional* — non-CLIP 클러스터에만 존재. **LLaVA-Next 의 M5a positive + M5b NULL dissociation** 로 *physics-mode commitment 가 LM-side direction 으로 라우팅* (encoder localization 없이도 LM 이 신호 보유) 임을 입증.

**상세 설명**:

**개요**:
- M5b 의 핵심 질문: **encoder 가 LM 의 v_L direction 에 *국소화된 feature 표현* 을 가지는가?**
- 측정 방법: encoder 의 마지막 (실제 사용) layer 의 hidden state 에서 *top-k physics-cue features 를 zero-out* (Bricken et al. 2023 trick); PMR drop 측정.
- Random control: magnitude-matched 랜덤 features (3 sets) 로 비교 — direction-specificity 검증.

**Round-1 → Round-2 methodological correction (2026-04-28)**:
- Round 1 (오전): 5 모델 모두 `vision_hidden_23` 에서 SAE 학습. **그러나 LLaVA family 는 `vision_feature_layer=-2` convention 으로 layer 22 만 LM 에 전달 — layer 23 의 perturbation 은 downstream 에 도달하지 않음**. Idefics2 도 SigLIP-SO400M (27 layers) 의 last_hidden_state 가 layer 26 from `post_layernorm` — layer 23 은 mid-encoder.
- Round 2 (저녁): **모델별 actually-consumed layer** 에서 fresh capture + SAE retrain + intervention.
  - Qwen: L31 (last) — round 1 = round 2 (이미 정확).
  - LLaVA-1.5 / LLaVA-Next: L22 (penultimate, `vision_feature_layer=-2`).
  - Idefics2: L26 (last block of 27, `last_hidden_state` 직전).
  - InternVL3-hf: L23 (last, `vision_feature_layer=-1`).

**Setup (round 2)**:
- 모델별 vision-only forward 로 layer-N capture (480 stim, ~5 min on H200, no LM forward — vision encoder activations 만).
- 모델별 SAE retrain: 4× overcomplete (n_features = 4 × d_in), λ=1.0, 5K Adam steps. Saturated 모델 (LLaVA-Next / Idefics2 / InternVL3) 은 `--pmr-abs-threshold 0.5` 로 완화 (strict 0.333 에서 abs sample 부족).
- Stim cell: per-model OPEN+circle baseline-PMR=1 cell 선택.
  - Qwen: filled/blank/both (1.0).
  - LLaVA-1.5: shaded/ground/cast_shadow (1.0).
  - LLaVA-Next: shaded/blank/both (1.0).
  - Idefics2: filled/blank/both (1.0).
  - InternVL3: filled/blank/both (1.0).
- k sweep: {5, 10, 20, 40, 80, 160}; LLaVA-1.5 추가로 {200, 300, 500, 800} 까지 high-k extension (NULL robustness check).
- 3 mass-matched random feature sets per model.

**Result — 5-model OPEN-uniform k-break table**:

| Model | k=5 | k=10 | k=20 | k=40 | k=80 | k=160 | random | layer | n_features |
|---|--:|--:|--:|--:|--:|--:|--:|---:|---:|
| **Qwen2.5-VL-7B** | 1.00 | 1.00 | 1.00 | **0.00** | **0.00** | **0.00** | 1.00 | 31 | 5120 |
| LLaVA-1.5-7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 22 | 4096 |
| LLaVA-Next-7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 22 | 4096 |
| **Idefics2-8B** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 | 26 | 4608 |
| **InternVL3-8B** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 | 23 | 4096 |

**LLaVA-1.5 high-k extension**: k=200/300/500/800 모두 1.0 (NULL). 4096 features 의 19.5% 까지 ablate 해도 break 없음 — **NULL 은 sample-size 또는 threshold artifact 가 아님**.

**Qwen FC vs OPEN 비교**:
- 원본 Qwen FC (forced-choice + label=circle): k=20 break.
- Cross-model uniform OPEN: k=40 break.
- 차이: FC 의 binary letter answer space 가 OPEN 의 free-text PMR scoring 보다 약간 빠른 break threshold. 정성적으로 동일 (random NULL, top-k clean break).

**3 가지 발견**:

1. **Encoder-side SAE physics-cue features 가 3 of 5 architectures 에서 causally bound**:
   - Top-k ablation 이 PMR 을 1.0 → 0.0 로 cleanly drive, mass-matched random 은 1.0 retain.
   - Feature selectivity 확인 (non-specific perturbation 이 아님).

2. **Effect concentration 이 architecture 별로 다름**:
   - Qwen: k=40 break (0.78% of 5120) — 가장 *concentrated*.
   - Idefics2 / InternVL3: k=160 break (3.5-3.9% of features) — 더 *distributed*.
   - 이 ladder 가 **M3 vision-encoder probe AUC ladder** 와 정렬: Qwen 0.99 > Idefics2 0.93 > InternVL3 0.89 > LLaVA 0.7-0.8.
   - → 인코더 *discriminability 가 높을수록 SAE features 가 더 concentrated*.

3. **CLIP-기반 LLaVA 모델 NULL — paper-grade dissociation**:
   - LLaVA-Next 는 M5a 에서 LM-side L20+L25 10/10 flip (positive); 그러나 M5b encoder-side 는 NULL.
   - 이 dissociation 은 mechanistic claim 을 정밀화: **CLIP family 의 physics-mode commitment 는 encoder-side localized features 가 아닌 LM-side residual-stream direction 으로 라우팅**.
   - Non-CLIP cluster (Qwen / Idefics2 / InternVL3) 는 *둘 다* 라우팅 — encoder-side features + LM-side direction 모두 causal.

**Triangulation with prior signatures**:

| Architecture | M3 vision AUC | M5a LM-side flip | M5b encoder-side break | Encoder cluster |
|---|--:|---|---|---|
| Qwen2.5-VL | 0.99 | 10/10 (L10) | k=40 break | High-saturation, non-CLIP |
| Idefics2 | 0.93 | 10/10 (L25) | k=160 break | Mid-saturation, non-CLIP, perceiver |
| InternVL3 | 0.89 | n/a (saturated) | k=160 break | Mid-saturation, non-CLIP, MLP |
| LLaVA-Next | 0.81 | 10/10 (L20+L25) | NULL | Low-saturation, CLIP+AnyRes |
| LLaVA-1.5 | 0.73 | 0/10 (NULL) | NULL | Low-saturation, CLIP |

**시사점**:
- **두 번째 downstream 시그니처** for H-encoder-saturation (after M3 probe AUC + §4.6 pixel-encodability + M4 LM probe).
- **Architecture-level reframe 강화**: shortcut localization 의 *위치* (encoder vs LM) 가 architecture-conditional. CLIP cluster 는 LM-only, non-CLIP 은 dual.
- **Paper Contribution 2 확장**: Qwen-only causal localization 에서 5-model architecture-conditional decomposition 으로.

---

### Slide 19: §4.6 픽셀-공간 gradient ascent (Qwen)

**슬라이드 내용**: 4-panel canonical seed (baseline / ε=0.05 / ε=0.1 / unconstrained) + 응답 비교.

**핵심 메시지**: Qwen 의 픽셀에 gradient ascent (ε=0.05) 로 *5/5* PMR flip. 매칭 magnitude random direction 은 *0/15*. 픽셀-인코드 가능성 입증.

**상세 설명**:

**Question**: M5a 가 runtime 에 hidden state 를 더해서 행동을 뒤집을 수 있다 — 그러면 *반대 방향* — 픽셀에 perturbation 을 가해서 모델이 *스스로* h_L10 을 v_L10 방향으로 밀게 만들 수 있는가?

**Method**:
- 최적화 변수: Qwen2.5-VL 의 post-processor `pixel_values` (T_patches × 1176, patch-flattened normalized 표현).
- 목적함수: `<mean(h_L10[visual]), v_L10>` 최대화 (M5a 와 *동일* target).
- Adam, lr=1e-2, n_steps=200. ε ∈ {0.05, 0.1, 0.2} L_∞-bounded + unconstrained + 3 random unit dir at ε=0.1.
- 5 baseline circle stim × 7 configs × 200 step = 35 runs.

**Result**:

| Config | n flipped (PMR 0→1) | 평균 final projection |
|---|--:|--:|
| `bounded_eps0.05` | **5 / 5** | 43.7 |
| `bounded_eps0.10` | **5 / 5** | 100.6 |
| `bounded_eps0.20` | **5 / 5** | 125.9 |
| `unconstrained` | **5 / 5** | 181.1 |
| `control_v_random_*` | **0 / 15** | 73-85 |

**핵심 관찰**:
- 5/5 v_L10 flip at ε=0.05 — 가장 작은 ε 에서도 baseline 의 모든 5 seed 가 flip.
- 0/15 random control flip — magnitude 매칭 (random projection 73-85, bounded ε=0.1 v_L10 ~100) 인데도 flip 안 함. *방향 특이성* 이 magnitude 가 아닌 결정 요인.

**4-panel figure 의 의미**:
- baseline → ε=0.05 → ε=0.1 → unconstrained 으로 갈수록 perturbation 의 시각적 amplitude 가 커짐.
- ε=0.05 는 가까이서 보면 보이는 *옅은 dotted texture*. abstract 원의 형태는 보존됨.
- 인간이 보면 "흰 배경 위 검은 원" 으로 묘사. 그러나 모델은 "공이 떨어진다" 라고 응답.

**Sample 응답**:
- Baseline: "The circle will remain unchanged as it is a static image."
- v_L10 ε=0.05: "The circle will continue to fall downward due to gravity."
- Random ε=0.1: "The circle will remain stationary as there is no indication of movement..."

→ **Pixel-encodability 확립**. shortcut 은 *이미지에 인코드 가능*.

다음 슬라이드 (20) 에서 이게 Qwen 한정 임을 보임.

---

### Slide 20: §4.6 cross-model layer sweep (n=10) + Idefics2 9-layer disambiguation

**슬라이드 내용**: 5-panel figure (모델별 LM 레이어 vs PMR flip rate, Wilson 95% CI) + 우측 모델별 결과 + 하단 해석.

**핵심 메시지**: 픽셀-인코드 가능성은 architecture-conditional. **5 모델 중 3 모델 testable** 가 지원 (Qwen / LLaVA-Next / LLaVA-1.5); **Idefics2 는 L5-L31 9 레이어 모두 falsifies** — wrong-relative-depth 가설은 falsified, **perceiver-resampler 가 leading remaining candidate** (5-model design 은 isolate 못함).

**상세 설명**:

**Layer sweep 핵심 결과 (n=10 stim, ε=0.1)**:

| Model | Vision encoder | Projector | LM | Clean shortcut layers (Wilson lower > random upper) |
|---|---|---|---|---|
| Qwen2.5-VL | SigLIP | MLP | Qwen2-7B | L5/10/15/20/25 (5 layers, all ≥ 80%) |
| LLaVA-Next | CLIP+AnyRes | MLP | Mistral-7B | L20+L25 (모두 10/10) |
| LLaVA-1.5 | CLIP-ViT-L | MLP | Vicuna-7B | L25 only (4/10, weaker than n=5 morning suggested) |
| **Idefics2** | SigLIP-SO400M | **perceiver** | Mistral-7B | **0 across L5-L31 (9 layers, 16-97% depth)** |
| InternVL3 | InternViT | MLP | InternLM3-8B | (untestable — baseline_pmr=1.0) |

**Idefics2 9-layer disambiguation** (T1b from data audit, 2026-04-28):

기존 5-model n=10 sweep (2026-04-27 night) 에서 Idefics2 L5/10/15/20/25 모두 0 clean shortcuts despite v_L projection rising +38. 두 가지 후보 가설:

(a) **Wrong-relative-depth**: Mistral-7B 32 layers, 우리는 L25 (78% depth) 까지만 테스트. Idefics2 의 shortcut 이 L26-31 (81-97% depth) 에 있을 수 있음 — 이전 morning §4.6 의 LLaVA-1.5 L10 null 이 L25 에서 fix 된 것과 같은 layer-choice artifact?

(b) **Perceiver-resampler bottleneck**: Idefics2 만 perceiver projector (다른 모델은 MLP). Perceiver 가 64 tile-tokens 를 fixed visual-token budget 으로 압축 — v_L-aligned information 이 LM 에 도달하기 전에 stripping 됐을 수 있음.

**T1b 실험 (2026-04-28)**: 새 M2 LM activation capture at L26/28/30/31 → v_L 추출 (n_pos=470 / n_neg=10) → 80-run counterfactual sweep.

**Result**:
- L26: 0/10 v_unit, 0/10 random (proj -10.7 → +27.1)
- L28: **1/10 v_unit** (line_blank_none_fall_006: "Appear." → "Move." PMR=1, isolated noise on different stim than L25 hit), 0/10 random (proj -10.6 → +28.1)
- L30: 0/10 v_unit, 0/10 random (proj -10.9 → +30.3)
- L31: 0/10 v_unit, 0/10 random (proj -72.0 → +163, larger v_L magnitude)

**Aggregate Idefics2 9 layers**: 1/90 v_unit hit (Wilson [0.0025, 0.07]) + 0/90 random. v_L projection 은 정상 ascending at every depth — gradient ascent 은 mechanically 작동함, PMR 만 flip 안 함.

**해석**:
- (a) **Wrong-relative-depth 가설 falsified**: 9 layers 모두 0 → layer-choice artifact 아님.
- (b) **Perceiver-resampler 는 leading remaining candidate**: encoder→LM 채널이 v_L-aligned info 를 strip 하는 게 9-layer 무시 패턴과 일관.
- **5-model design isolation 한계**: Idefics2 는 MLP-projector 모델들과 encoder (SigLIP-SO400M vs CLIP/SigLIP/InternViT) + projector (perceiver vs MLP) + AnyRes (없음 vs 있음 in LLaVA-Next) **동시에** 다름. Projector axis 만 isolate 하려면 controlled projector-swap test (동일 encoder/LM 에서 perceiver↔MLP 교체 + 재학습) 필요 — out of scope for v1 paper.

**시사점**:
- **H-shortcut framing 정정**: 픽셀-인코드 가능성은 architecture-conditional. **3 of 5 testable** (Qwen / LLaVA-Next / LLaVA-1.5) 가 지원, Idefics2 falsifies 9-layer evidence, InternVL3 untestable. **Encoder saturation 만으로는 부족** (Idefics2 AUC 0.93 > LLaVA-Next 0.81 인데 Idefics2 는 0 shortcuts) — projector design 이 disambiguating axis.
- M9 PMR-천장 (행동), §4.7 결정-안정성 천장, §4.6 pixel-encodability — 3 architecture-level signature 가 동일한 architectural property 의 다른 표현이며, 이번 update 로 **projector design (MLP vs perceiver)** 이 추가 disambiguating axis 로 노출됨.

**Random control**: 5 모델 합쳐 1/250 trials (49 of 50 random-control cells = 0/10; Qwen L10 random 1/10 만 non-zero, v_unit 10/10 보다 훨씬 낮음) — **방향 특이성 보존**.

---

### Slide 21: 외부 타당성

**슬라이드 내용**: M8a / M8d / M8c 3 stim source figures + 핵심 결과.

**핵심 메시지**: 본 연구의 결과 (architecture-level reframe, H1 unsaturated-only, H7 cross-category) 는 비-원 도형 / 비-공 카테고리 / 실사진 으로 일반화.

**상세 설명**:

**M8a (5 도형 cross-shape)**:
- circle / square / triangle / hexagon / polygon × 4 추상화 × 2 bg × 2 cue × 5 seed.
- 사전 등록된 strict scoring: **Qwen 1/4 PASS, LLaVA 4/4 PASS**.
- 비대칭 그 자체가 cross-shape validation of saturation hypothesis. Qwen 천장이라 H1/H7 측정 불가; LLaVA 측정 가능.

**M8d (3 카테고리 cross-category)**:
- car / person / bird × 4 추상화 × 2 bg × 2 cue × 2 event (fall + horizontal) × 5 seed.
- LLaVA **3/3 H7 PASS**: car +0.525, person +0.138, bird +0.550 on PMR_regime (physical − abstract).
- → Label-selects-regime 가 circle-only 가 아니라 *category-general*. H7 의 paper-grade evidence.

**M8c (실사진)**:
- 60 photos × 5 categories from COCO 2017 + WikiArt.
- **Photos REDUCE Qwen PMR(_nolabel) by 18-48 pp** across categories. Synthetic-stim minimality 가 saturation 의 co-factor.
- 모든 3 모델이 사진에서 PMR [0.18, 0.67] 로 *수렴* — encoder 격차가 사진에서 압축.

**시사점**:
- Architecture-level reframe 이 stim-source-general (M9 paper Table 1 의 핵심).
- H1 / H7 가설이 axis-축 / category-general 로 확장.
- 사진은 saturation 을 자연스럽게 풀어주는 *input-context* 요소 — synthetic-stim 의 minimality 가 saturation 의 co-driver.

---

### Slide 21b: §4.8 — Qwen 7B vs 32B PMR scaling

**슬라이드 내용**: Qwen 7B (M2 baseline) 와 Qwen 32B (M2 동일 stim) 의 PMR 비교 표 + cue-cell breakdown.

**핵심 메시지**: **5× scale 증가가 PMR aggregate 를 거의 안 움직임** (0.931 → 0.926). 단 *cue=none* (가장 약한 cue) cell 에서 PMR 8.6 pp 하락 + abstract_reject 35× 증가 — *scaling 이 약-cue 에서만 제한적으로 도움*. H2 label gap 도 절반으로 약화 (+0.071 → +0.010), 그러나 dissolved 는 아님. **MechBench-style "scale doesn't fix grounding" 지지**.

**상세 설명**:

**Setup**:
- 동일 M2 stim (480 stim × 3 labels = 1440 inferences), 동일 OPEN prompt.
- Qwen2.5-VL-7B (M2 baseline) vs Qwen2.5-VL-32B (5× more parameters).
- 32B inference: 1440 inferences in 16 min wall on H200 (single-GPU bf16, ~67 GB weight + 30 GB activation).
- 동일 PMR scorer 사용.

**Result — aggregate PMR**:

| Model | Aggregate PMR | Δ vs 7B |
|---|--:|--:|
| Qwen2.5-VL-7B (baseline) | 0.931 | — |
| Qwen2.5-VL-32B | **0.926** | **−0.005** (effectively zero) |

**Per-cell breakdown — cue_level 별**:

| cue_level | 7B PMR | 32B PMR | Δ |
|---|--:|--:|--:|
| both | 0.978 | 0.972 | −0.006 |
| cast_shadow | 0.957 | 0.945 | −0.012 |
| motion_arrow | 0.992 | 0.987 | −0.005 |
| **none** | **0.797** | **0.711** | **−0.086** |

→ **cue=none (가장 약한 cue) 에서만 의미있는 drop**. 즉 32B 는 *cue 부재 시* 약간 더 신중함 (덜 자동으로 physics-mode 안 함).

**abstract_reject 변화** (response 가 explicit "이는 abstract shape 입니다" 패턴):

| Cell | 7B abstract_reject | 32B abstract_reject | factor |
|---|--:|--:|--:|
| Aggregate | 0.002 | 0.065 | **35×** |
| cue=none | 0.014 | 0.157 | 11× |

→ **Scaling 이 abstract-reject 행동을 35× 증폭** — 32B 는 *명시적으로 abstract shape 으로 인지* 하는 frequency 가 매우 큼.

**H2 label gap (`ball − circle` PMR delta)**:

| Model | gap |
|---|--:|
| 7B | +0.071 |
| 32B | **+0.010** |

→ Label gap 이 절반 이하로 *축소*. H2 (label-prior dominance) 가 32B 에서 약화 — 그러나 *완전히 dissolved* 는 아님.

**시사점**:

1. **MechBench-style "scale doesn't fix grounding" 가 PMR 에서 재확인**:
   - Aggregate PMR 변화 −0.005 는 noise 수준 — 5× parameters 가 overall PMR 천장을 못 깨뜨림.

2. **약-cue 에서 작은 개선**:
   - cue=none 에서 −8.6 pp drop + abstract_reject 11× 증가 → 32B 는 *cue 가 약할 때 더 abstract-mode 로 인식*.
   - 즉 scaling 이 약 cue 에서 *visual prior 의 underweighting* 을 완화함 (visual-prior under-weighting 가설의 작은 confirmatory evidence).

3. **H2 label gap 약화 — 절반**:
   - 7B 의 +0.071 → 32B 의 +0.010 으로 약화.
   - 그러나 0 으로 사라진 건 아님 — H2 dissolved 가 아님.

4. **paper 함의**:
   - "Scale doesn't help PMR aggregate" — non-trivial claim, MechBench analog.
   - "약-cue cell 에서만 도움" — visual-prior under-weighting 가 saturation 의 *부분적* 메커니즘.
   - 32B 는 7B 의 같은 architecture cluster (SigLIP + Qwen2-LM) 에 속함 — encoder-saturation 의 architecture 결정성 *유지*. Scaling 이 architecture cluster 를 안 바꿈.

**Limitations**:
- 단일 model family (Qwen). Cross-family (e.g., Llama4 Vision 8B vs 80B) 는 미실시.
- 72B (Qwen 의 더 큰 버전) 도 후속 — 144 GB bf16 → dual-GPU 또는 quantization 필요. Predicted to land near 32B based on 32B/7B null pattern.

---

### Slide 22: Multilingual labels

**슬라이드 내용**: 5-model Korean vs English figure + 일본어 mechanism 차이 설명.

**핵심 메시지**: H2 (label-prior dominance) 가 cross-language 일반화하지만, 모델 별로 *engaging mechanism* 이 다름.

**상세 설명**:

**한국어 (공/원/행성) on M8a circle stim** (5 models):
- Cross-label ordering 보존 4/5 모델 (planet > ball > circle 순).
- LLaVA-1.5 가 가장 큰 swing (avg |Δ|=0.11) — Vicuna 의 한국어 coverage 가 가장 약함.

**일본어 (ボール / 円 / 惑星)** — 다른 mechanism 노출:
- **Qwen genuinely engages JA**: label-echo 85-91%. ボール 응답 한 개에서 ball Δ +0.13.
- **LLaVA-1.5 internally translates kanji to English**: 응답에 kanji 거의 없음. "The ball will roll" 같은 영어 응답.
- **Idefics2 falls back to Chinese on 惑星**: 24% 의 응답이 Chinese (惑星会向下落下). Mistral 의 일본어 SFT 부족 → 한자 공유로 Chinese fallback. **Cross-script bypass mechanism**.

**Scorer 다국어 확장**:
- KO physics-verb stems (떨어, 굴러, 움직 등), JP (落ち, 転が, 動く 등), CN (下落, 跌落 등) 추가.
- 51 → 54 PMR 테스트 케이스 확장.
- 비대칭 추가: abstract markers (한국어 "그대로", 일본어 "そのまま", 중국어 "保持") 가 physics-verb 매칭을 *gate*.

**시사점**:
- Label-prior 는 multi-lingual semantic 수준에서 작동 (token-frequency 가 아닌).
- 그러나 모델별로 engagement 형태가 다름 — paper 의 inter-model 다양성 강화.

---

## Section 6 — Discussion

### Slide 23: Architecture-level reframe

**슬라이드 내용**: 5 가지 saturation 시그니처 — PMR 천장 / 결정-안정성 천장 / 픽셀-인코드 가능성 / LM probe AUC / encoder-side SAE intervention.

**핵심 메시지**: 본 연구는 saturation 의 **5 가지 distinct downstream 시그니처** 를 발견 (2026-04-28 기준). 모두 같은 architectural property 의 다른 측면이며, 모두 *동일한 architecture 클러스터링* 을 만든다 — 비-CLIP 클러스터 (Qwen / Idefics2 / InternVL3) 와 CLIP 클러스터 (LLaVA-1.5 / LLaVA-Next) 분리.

**상세 설명**:

**Signature 1 — PMR ceiling** (M9 paper Table 1):
- 행동 PMR(_nolabel) 이 non-CLIP cluster [0.80, 0.92] 와 CLIP cluster [0.14, 0.37] 로 fully separated.
- 사진에서 모든 3 모델이 [0.18, 0.67] 로 수렴.
- → Encoder + LM 결합이 PMR 천장을 결정.

**Signature 2 — Decision-stability ceiling** (§4.7):
- Non-CLIP 모델 (Qwen / Idefics2 / InternVL3) 이 cue 발화 시 5 seed 모두 같은 PMR call 로 수렴 (RC ≈ 1.0).
- CLIP-기반 모델 (LLaVA-1.5 / LLaVA-Next) 은 강한 cue 에서도 seed-level variance 보유 (RC < 1).
- → Saturation 이 단지 행동 천장만이 아니라 *결정-안정성 천장* — 같은 자극에 같은 답을 반복.

**Signature 3 — Pixel encodability** (§4.6 5-model n=10 layer sweep + Idefics2 9-layer disambiguation, 2026-04-28):
- **3 of 5 testable architectures** 가 픽셀-공간 gradient ascent 로 PMR flip 가능: Qwen broad (5 layers ≥ 80%), LLaVA-Next L20+L25 (10/10), LLaVA-1.5 L25 weak (4/10).
- **Idefics2 falsifies universal claim**: 9 LM layers (L5-L31, 16-97% relative depth) 모두 0 clean shortcuts despite v_L projection ascending +28~+163 cleanly. **Wrong-relative-depth 가설 falsified** by 9-layer evidence.
- **Perceiver-resampler 는 leading remaining mechanism candidate**: Idefics2 만 perceiver projector (다른 모델은 MLP). 5-model design 은 isolate 못함 — controlled projector-swap test (동일 encoder/LM 에서 perceiver↔MLP 교체 + 재학습) 필요.
- → Saturation 의 *세 번째* signature: pixel-encodability 가 **encoder + projector design** 결합 속성. Encoder saturation 만으론 부족 (Idefics2 는 AUC 0.93 으로 LLaVA-Next 0.81 보다 saturated 인데도 0 shortcuts).

**Signature 4 — LM logit-lens probe AUC** (M4 cross-model, 2026-04-28 — Slide 17b):
- 5-model × 5-layer LM hidden-state probe AUC ladder: Idefics2 0.995, Qwen 0.96, LLaVA-Next 0.79, LLaVA-1.5 0.76, InternVL3 untestable.
- Vision encoder probe AUC ladder (Slide 17) 와 *동일한 architecture clustering*.
- **Idefics2 LM AUC > vision AUC (0.995 > 0.93)** — perceiver-resampler 가 *정보를 stripping 하지 않고 압축으로 집중*.
- → "정보 LM 도달" 시그니처 — encoder downstream 에서도 saturation cluster 가 유지됨.

**Signature 5 — Encoder-side SAE intervention** (M5b cross-model round 2, 2026-04-28 — Slide 18c):
- 모델별 actually-consumed encoder layer 에서 SAE retrain + top-k feature ablation.
- **3 of 5 모델 break PMR**: Qwen k=40 (0.78%), Idefics2 k=160 (3.5%), InternVL3 k=160 (3.9%).
- **2 LLaVA 모델 NULL** at k ≤ 800 (LLaVA-1.5 high-k extension).
- → encoder 에 *국소화된 physics-cue feature 표현* 이 비-CLIP 클러스터에만 존재. Random control 모두 1.0 (specificity 확인).
- LLaVA-Next 의 M5a positive (LM-side 10/10 flip) + M5b NULL dissociation → CLIP 클러스터의 commitment 가 *LM-side direction 으로만 라우팅*.

**5 signature 의 통합 의미**:
- 모두 *동일한 architecture-level saturation 속성* 의 다른 표현.
- "encoder representational capacity 만으로 결정" 이라는 단순 인코더 가설은 disconfirm.
- joint encoder + LM + **projector design** (architecture 수준) 이 결정자.
- **2026-04-28 update**: projector design (MLP vs perceiver) 이 disambiguating axis 로 추가 노출. encoder + LM 만으로는 Idefics2 의 0-shortcut 패턴을 설명 못함.
- **2026-04-28 evening update**: encoder-side localization (M5b) + LM probe AUC (M4) 가 추가 시그니처로 합류 → architecture clustering 의 *5-fold redundancy* — 단일 architecture-level property 가 5 가지 distinct downstream signature 로 표현됨.
- 본 연구의 paper 의 가장 큰 single contribution — 단순 행동 통계가 아닌 *5 차원 mechanism-level 정렬* 이 architecture-level reframe 을 강화.

**3-cluster decomposition (5 signature 모두에서 일관)**:

| Cluster | Models | M3 vision AUC | M9 PMR ceiling | §4.6 pixel-flip | M4 LM AUC | M5b SAE break |
|---|---|--:|--:|---|--:|---|
| **High-saturation** (non-CLIP, broad shortcut) | Qwen2.5-VL | 0.99 | 0.84 | 5 layers ≥ 80% | 0.96 | k=40 |
| **Mid-saturation** (non-CLIP, distributed) | Idefics2, InternVL3 | 0.93 / 0.89 | 0.88 / 0.92 | 0/9 (Idefics2), n/a | 0.995 / n/a | k=160 / k=160 |
| **Low-saturation** (CLIP) | LLaVA-1.5, LLaVA-Next | 0.73 / 0.81 | 0.18 / 0.70 | weak / 2 layers | 0.76 / 0.79 | NULL / NULL |

---

### Slide 24: 한계

**슬라이드 내용**: 7 가지 한계 (2026-04-28 evening update — M5b cross-model intervention 완료, Qwen 72B 미실시, projector isolation 미검증).

**핵심 메시지**: 본 연구의 결과를 *반증* 할 수 있는 후속 실험과 paper scope 결정을 솔직히 listing.

**상세 설명**:

1. **Projector isolation 미검증**: §4.6 5-model design 은 Idefics2 가 MLP-projector 모델들과 encoder + projector + AnyRes 동시에 다름. **Perceiver-resampler 가 leading remaining candidate** 이지만 controlled projector-swap test (동일 encoder/LM 에서 perceiver↔MLP 교체 + 재학습) 가 isolation 의 정통한 방법 — 재학습 비용으로 v1 paper out of scope. 9 LM layers (L5-L31) 에서 0 shortcuts 라는 negative result 가 가장 strong 한 currently available evidence. **2026-04-28 evening 추가**: M4 cross-model + M5a cross-model + M5b cross-model 의 triangulation 으로 "perceiver 가 inverse pixel-routability 만 차단, forward 는 작동" 가 더 정밀한 가설로 정리됨.

2. **LM-only counterfactual 부재**: LLaVA-1.5 → LLaVA-Next 의 0.52 PMR jump 가 4축 confound (AnyRes / projector / training / LM family). 동일 인코더 (CLIP-ViT-L) 에서 LM 만 swap 한 controlled experiment 가 paper-grade LM-modulation evidence 를 줌. 미실시. **부분적 evidence**: LLaVA-Next 의 M5a positive + M5b NULL dissociation 이 "CLIP cluster 의 commitment 가 LM-side direction 으로 라우팅" 을 시사하지만, controlled LM swap 으로만 fully isolated.

3. **InternVL3 §4.6 protocol untestable**: `line_blank_none_fall_*` baseline_pmr=1.0 이라 abstract-baseline 으로 못 쓰임. mvp_full label-free 에서 InternVL3 가 abstract-mode 로 응답하는 cell 이 없음. Alternative-baseline stim 탐색 필요 (다른 prompt 또는 다른 stim category). **단 M5b cross-model 에서는 InternVL3 의 baseline=1 stim 에서 top-160 ablation break 확인됨** — H-shortcut 의 alternative 시그니처로는 testable.

4. ~~**v_L10 은 1-d class-mean 축**~~: ✅ **M5b cross-model intervention 완료 (2026-04-28 evening)**. Per-model SAE retrain at actually-consumed layer (LLaVA L22 / Idefics2 L26 / InternVL3 L23 / Qwen L31) → 3 of 5 모델 break (Qwen k=40, Idefics2 k=160, InternVL3 k=160), 2 LLaVA NULL even at k=800 (19.5% of features). Encoder-side feature localization 이 architecture-conditional 임을 입증. **Multi-axis SAE / non-linear feature decomposition** 은 v2 paper scope.

5. **Qwen 72B PMR scaling 미실시**: §4.8 에서 Qwen 7B (PMR 0.931) vs 32B (PMR 0.926) 비교 완료. 72B (~144 GB bf16, dual-GPU 또는 quantization 필요) 는 confirmatory 역할 — 32B/7B null pattern 으로부터 72B 도 ~0.93 가까이 land 예측. 시간 + 자원 비용 trade-off 로 v1 paper out of scope.

6. **Single-task evaluation**: "next-state-prediction" 만 검증. 다른 shortcut 행동 (counting / spatial / causality) 미검증.

7. **Human baseline 미수집**: M7 Prolific (20 raters × 50 stim) paper-blocking. 다음 단계.

**ST5 prompt-steering retire**: 원래 project scope (Gavrikov 2024 식 explicit "treat as abstract / treat as physical" prompt steering) 은 paper 에서 retire. **Reframe**: prompt-variation axis 는 §4.3 KO/JA + label-free + open vs FC 로 cover 됨 — Gavrikov 의 명시적 instantiation 은 reviewer 에게 직접 referrable.

**솔직성의 의의**:
- 본 연구의 강점은 5-model × 3-stim × bootstrap CI 의 *robustness* + **9-layer Idefics2 disambiguation** + **5-model M5b cross-model SAE intervention** 의 *5-fold downstream signature alignment*.
- 약점은 **single-task**, **LM-only counterfactual 부재**, **projector axis isolation 미검증** (forward/inverse dissociation 으로 가설 정밀화는 완료).
- Future work clearly identified — controlled projector-swap, controlled LM-swap, multi-axis SAE decomposition, M7 Prolific human baseline.

---

## Section 7 — Conclusion

### Slide 25: 결론

**슬라이드 내용**: 3 가지 paper-grade contribution 재진술.

**핵심 메시지**: 본 연구는 VLM 의 추상→물리 shortcut 을 행동 / 인과 / 픽셀 3차원에서 localize. 단일 architectural 속성 (encoder-saturation) 이 3 차원 모두의 결정자임을 보임.

**상세 설명**:

**기여 1 — Architecture-level reframe**:
- 5 model × 3 stim source × bootstrap CI 로, 행동 PMR ceiling 이 encoder representational capacity 만으로 결정되지 않음을 disconfirm.
- 2-CLIP-point insight (LLaVA-1.5 vs LLaVA-Next) 가 가장 깨끗한 disconfirmer.
- **paper-grade defensible**.

**기여 2 — 인과 localization (LM-side + encoder-side cross-model, 2026-04-28 update)**:
- **M5a runtime steering**: Qwen L10 α=40 + LLaVA-Next L20+L25 + Idefics2 L25 의 *3 of 4 testable models* 가 10/10 PMR flip — *Qwen-only 에서 cross-model 로 확장*. LLaVA-1.5 NULL (encoder-bottleneck), InternVL3 untestable (saturated baseline=1).
- **M5b SAE encoder-side intervention (round 2)**: 모델별 actually-consumed layer (LLaVA L22 / Idefics2 L26 / InternVL3 L23 / Qwen L31) 에서 SAE retrain + top-k feature ablation. *3 of 5 break* (Qwen k=40, Idefics2 k=160, InternVL3 k=160), *2 LLaVA NULL* at k≤800.
- **LLaVA-Next 의 M5a positive + M5b NULL dissociation**: physics-mode commitment 가 *CLIP cluster 에서는 LM-side direction 으로만 라우팅*, *non-CLIP cluster 에서는 둘 다 (encoder feature + LM direction) 사용*. 단순 "Qwen-only" 결과보다 mechanistic claim 정밀화.
- M5a-ext: v_L 은 regime axis (+ kinetic / − static), binary toggle 아님 — Qwen-only 결과로 유지.
- **paper-grade defensible cross-model**.

**기여 3 — Pixel encodability 는 architecture-conditional**:
- Qwen ε=0.05 픽셀 perturbation 으로 5/5 PMR flip + 매칭 random 0/15.
- **5-model n=10 layer sweep + Idefics2 9-layer disambiguation (2026-04-28)**: **3 of 5 testable architectures** 지원 (Qwen broad, LLaVA-Next L20+L25, LLaVA-1.5 L25 weak). Idefics2 9 layers (L5-L31) 모두 0 → wrong-relative-depth falsified, **perceiver-resampler 가 leading remaining candidate** (5-model design isolation 한계 명시). InternVL3 untestable.
- Random controls 1/250 in aggregate — direction specificity 보존.
- M9 PMR-ceiling / §4.7 결정-안정성 ceiling 과 평행한 *세 번째 architecture-level signature*, **projector design (MLP vs perceiver) 을 추가 disambiguating axis** 로 노출.
- **paper-grade defensible** + **cross-model cross-signature 정렬** (3 signatures 의 alignment).

**Big picture**:
- VLM shortcut 은 단순 "model quirk" 가 아니라 *architecture-level saturation 의 다차원 표현*.
- 행동 PMR ceiling, 결정 stability ceiling, pixel encodability — 이 3 가지가 같은 architectural property 의 *signature*.
- **2026-04-28 update**: projector design (MLP vs perceiver) 이 architecture-level disambiguating axis 로 추가 발견 — encoder + LM 만으로는 Idefics2 의 9-layer 0-shortcut 패턴을 설명 못함.
- Future work: SAE / multi-axis / LM-only counterfactual 로 saturation 의 mechanism-level decomposition.

---

### Slide 26: Q&A

**슬라이드 내용**: 감사 인사 + repo 링크 + 동반 자료 안내.

**핵심 메시지**: Q&A 환영. 후속 토론 + 다음 단계 우선순위.

**저장소 + 동반 자료**:
- Repo: github.com/namam3gy/physical-mode-activation
- 본 동반 MD: `docs/review_ppt/physical_mode_paper_ko.md` — 슬라이드별 상세 설명 (이 파일).
- Roadmap: `references/roadmap.md` — 전체 milestone, hypothesis scorecard, change log.
- Insight docs: `docs/insights/m{1,3,4,5,6,8a,8c,8d,9}_*.md`, `docs/insights/sec4_{2,3,5,6,7,10,11}_*.md`.

**Open backlog**:
- M5b — SIP / activation patching / SAE (mechanism gap).
- §4.4 — Michotte 2-frame causality.
- §4.8 — Qwen 32B / 72B scaling.
- M7 — Prolific human baseline + paper draft.
- §4.6 cross-model 후속 — 더 어려운 stim 으로 saturated 모델 v_L10 추출.

---

## 부록 — 메트릭 한 줄 요약

| 메트릭 | 한 줄 |
|---|---|
| **PMR** | 응답 텍스트에 물리 동사가 포함된 비율. Rule-based scorer. |
| **GAR** | PMR 의 부분 — 응답이 *하방* 운동을 묘사. |
| **RC** | 같은 자극을 N seed 로 sampling 했을 때 응답 일치 비율. T=0.7. |
| **AUC** | Logistic regression probe 의 5-fold CV AUC. behavioral_y 또는 stim_y. |
| **Bootstrap CI** | 5000-iter prediction-level resampling 으로 mean 의 95% CI. |
| **v_L** | mean(h_L \| PMR=1) − mean(h_L \| PMR=0). \"physics-mode 방향\". |
| **Paired-delta H2** | PMR(label) − PMR(_nolabel), per-stim 매칭. |

## 부록 — 가설 한 줄 요약

| ID | 한 줄 |
|---|---|
| **H1** | 추상화 axis (line→textured) 에서 PMR 이 monotonic 증가. **unsaturated 인코더 한정**. |
| **H2** | 라벨이 PMR 을 끌어올림. **3 가지 architecture-conditional 패턴** (saturation 에 따라). |
| **H7** | 라벨이 PMR 을 toggle 하지 않고 *어느 physics regime* 인지 선택 (ball → kinetic, planet → orbital). |
| **H-encoder-saturation** | 행동 PMR ceiling 은 architecture 수준 (encoder + LM 결합) 에서 결정. |
| **H-locus** | 인과 결정 layer 는 LM mid (L10). |
| **H-direction-bidirectional** | v_L10 은 regime axis (+ kinetic / − static), binary toggle 아님. |
| **H-shortcut** | shortcut 이 이미지에 픽셀-수준에서 인코드 가능. **Qwen-scoped (saturation 특이적)**. |
