# 원이 공으로 보이는 순간 — 슬라이드별 상세 해설

> 본 문서는 `docs/review_ppt/physical_mode_storyline_ko.pptx` (26 슬라이드)의 동반 자료입니다.
> 발표 청중이 도메인 전문가가 아닐 때, 슬라이드의 각 시각 요소가 무엇을 의미하는지·왜 그렇게 구성했는지·이어지는 슬라이드와 어떻게 연결되는지를 풀어 적었습니다.
> 청중에게 발표할 때 그대로 읽기 좋도록 자연스러운 한국어 호흡으로 정리했고, 도메인 용어가 처음 등장할 때마다 직관적인 비유를 함께 덧붙였습니다.

---

## 슬라이드 1 — 표지

**한 문장**: VLM(Vision-Language Model)이 추상 도형을 마치 물리 객체처럼 \"착각\"하는 현상을 5개 모델 위에서 행동·메커니즘·픽셀 3차원으로 분석한 연구입니다.

- 왼쪽의 "원 → 공" 도식이 이 발표의 핵심 메시지를 상징합니다. 사람이 보면 그냥 \"원\"인 그림인데, 비전-언어 모델은 그걸 \"공\"으로 처리하고 \"떨어진다\"고 답해 버립니다.
- 오른쪽 부제 "VLM이 추상 도형을 물리 객체로 \"착각\"하는 현상을 행동·메커니즘·픽셀 수준에서 분석한다"는 본 발표의 3축 (When / Where / How) 을 압축한 표현입니다.
- 5개 모델은 Qwen2.5-VL, LLaVA-1.5, LLaVA-Next, Idefics2, InternVL3입니다. 인코더 계열(SigLIP, CLIP, InternViT)과 LM 계열(Qwen2, Vicuna, Mistral, InternLM3)을 골고루 섞은 것이 핵심.

---

## 슬라이드 2 — 놀라운 관찰 (the hook)

**한 문장**: 가장 단순한 자극(흰 배경 위 검은 원) 하나에 사람과 AI의 응답이 정반대로 나뉩니다.

- 왼쪽의 자극 이미지(`docs/figures/01_line_blank_none.png`)는 이 연구가 사용하는 가장 추상적인 "최소 자극"입니다. 중력 단서·지면·텍스처·움직임 단서가 모두 없습니다.
- 가운데 카드(파랑) — 사람의 응답: "흰 배경 위 검은 원이 그려져 있다." 즉 추상적 도형이라고 묘사하고, 운동을 언급하지 않습니다.
- 오른쪽 카드(주황) — Qwen2.5-VL의 실제 응답: "The ball will fall down due to gravity." 갑자기 \"공\"·\"중력\"·\"낙하\"라는 단어들이 등장합니다.
- 아래 \"관찰 — 무엇이 이상한가?\" 콜아웃 — 핵심을 3 줄로 정리:
  1. 이미지에는 중력 단서가 **전혀 없다**.
  2. 그런데 모델은 추상↔물리 두 해석 중 **한쪽으로만 무너진다(collapse)**.
  3. 이런 shortcut은 알려져 있지만 \"어디서, 왜\" 일어나는지가 연구의 미해결 문제.
- 발표 팁: 청중에게 \"여러분은 이 그림을 보면 뭐라고 답하시겠어요?\"라고 물어보고 1초 정도 휴지를 두면 자연스럽게 \"AI가 다르게 본다\"는 메시지가 강조됩니다.

---

## 슬라이드 3 — 우리의 핵심 질문 (3축)

**한 문장**: 현상을 보는 데서 끝나지 않고, 행동·메커니즘·픽셀 세 차원으로 \"어디서·어떻게\" 일어나는지를 좁혀 들어갑니다.

- 카드 1 (파랑, WHEN — 행동): "어떤 자극에서, 어떤 모델에서, shortcut이 얼마나 강한가? 행동 시그니처는?"
  - 측정 도구: PMR / GAR / RC. (정의는 슬라이드 10).
- 카드 2 (청록, WHERE — 메커니즘): "모델 내부의 어떤 레이어 / 어떤 방향이 \"물리 모드\" 결정을 인과적으로 일으키는가?"
  - 도구: linear probe AUC, logit lens, VTI steering, SAE feature ablation, SIP activation patching.
- 카드 3 (주황, HOW — 픽셀): "shortcut이 픽셀에 인코드 가능한가? 이미지에 작은 노이즈만으로 응답을 뒤집을 수 있나?"
  - 도구: pixel-space gradient ascent (counterfactual stimulus generation, §4.6).
- 아래 종합 띠: \"세 질문에 동시에 답하기 위해 동일 자극·동일 5개 모델로 일관되게 실험\". 본 연구의 큰 차별점입니다.

---

## 슬라이드 4 — 왜 중요한가

**한 문장**: VLM의 shortcut은 단순한 학술 호기심이 아니라 신뢰성·world-model·해석가능성 세 갈래로 응용가치가 있습니다.

1. 신뢰성 / AI 안전성. 의료·자율주행·로봇 등 안전 영역에서 \"이미지 내용\" 대신 \"학습 priors\"를 답으로 내놓으면 위험. 환각의 한 형태.
2. World-model 가설. V-JEPA, RT-2, OpenVLA 등 최근 흐름은 \"VLM이 내부에 세상의 물리 모델을 갖는다\"고 본다. 본 연구는 그 \"물리 모드\"가 언제 켜지는지를 직접 측정.
3. 해석 가능성. 행동만 측정하면 \"어디가 문제인지\" 모른다. 우리는 인코더·LM 레이어·픽셀 공간까지 들어가 \"shortcut의 물리적 위치\"를 좁혀 들어간다.

---

## 슬라이드 5 — VLM 작동 한 장 도식

**한 문장**: VLM은 그림 → 비전 인코더 → projector → 언어 모델 → 텍스트 의 5단계 파이프라인이며, 본 연구는 각 단계를 분리해서 본다.

- 5개 박스 (이미지 → Vision Encoder → Projector → Language Model → 응답)는 거의 모든 오픈소스 VLM의 표준 구조입니다.
- Vision Encoder 옵션: CLIP / SigLIP / InternViT. \"패치 단위 표현\"으로 변환.
- Projector 옵션: MLP / perceiver-resampler. LM의 어휘 공간으로 \"번역\"하는 단계.
- LM 옵션: Qwen / Vicuna / Mistral / InternLM. attention + 추론.
- 응답: 물리 동사를 쓸지, 추상 표현을 쓸지가 본 연구의 출력 변수.
- 하단 띠 - \"본 연구의 측정 위치\":
  - **Vision Encoder** → probe(AUC), SAE feature 학습
  - **LM 레이어** → logit lens, VTI steering(±α·v_L), SIP patching
  - **픽셀** → gradient ascent counterfactual
  - **응답** → PMR / GAR / RC
- 발표 팁: 청중이 VLM을 처음 본다면 이 도식 한 장에 1분 정도 시간을 쓰는 게 좋다. 이후 모든 결과 슬라이드의 \"무엇이 어디서 일어나는지\"가 이 도식을 가리키게 된다.

---

## 슬라이드 6 — 선행 연구 한눈에

**한 문장**: 기존 문헌은 VLM의 shortcut을 \"보고\"는 했지만 \"어디서·왜\"의 메커니즘 분석은 빈 공간으로 남아 있었다.

| 연구 | 무엇을 했나 | 한계 |
|---|---|---|
| Eyes Wide Shut (Tong et al., 2024) | VLM이 놓치는 시각 primitive 모음 | 행동만 측정 |
| Pixels-to-Principles (Ballout et al., 2025) | encoder는 보지만 LM이 사용 안 함 보고 | 사진만, 추상화 axis 없음 |
| VLMs are Blind (Rahmanzadehgervi et al., 2024) | 추상 도형 7 task에서 VLM 58% | 물리 task 없음 |
| Shape vs Texture bias (Gavrikov et al., 2024) | 프롬프트로 모양/질감 편향 조정 | 사람 96%까지 도달 못함 |
| MechBench (Zhang et al., 2024) | 기계적 추론 벤치마크 | scale 키워도 안 풀림 — architectural limit |
| VTI / Activation Patching (Liu+ '25 / Wang+ '23) | LM 인과 개입 도구 | VLM에는 거의 적용 안 됨 |

- 마지막 띠: 본 연구의 차별점 — 동일 자극·5개 모델·세 차원(행동·메커니즘·픽셀)을 한꺼번에 측정한 사례는 없음.

---

## 슬라이드 7 — 추상화 사다리

**한 문장**: \"같은 동그라미를 다른 옷으로 입혀서 모델이 어느 옷에서 무너지는지\"를 보는 것이 본 연구의 핵심 자극 디자인.

- 4단계 사다리 (왼쪽일수록 추상, 오른쪽일수록 물리):
  - **line** — 선만으로 그린 원. 가장 추상.
  - **filled** — 회색으로 채운 원. 약간 \"물체\"스러움.
  - **shaded** — 3D 셰이딩이 들어간 구. 빛-위에서 prior가 자동 발동.
  - **textured** — 가죽 / 점박이 텍스처가 들어간 공. 가장 \"공\"에 가깝다.
- 이 4단계가 H1 가설(PMR이 추상화 따라 S-curve로 오르는가)의 핵심 축입니다.
- 발표 팁: \"왼쪽으로 가면 도형, 오른쪽으로 가면 객체\"라는 식으로 한 손으로 가리키며 설명하면 가시적입니다.

---

## 슬라이드 8 — 5축 factorial

**한 문장**: 추상화 외에도 배경·부가 단서·이벤트·라벨 4가지 축을 더 추가해, 총 2,880개의 자극으로 구성된 통제 디자인.

- 표 — 5개 factor 정의:
  - object_level (4 값): line / filled / shaded / textured. \"도형 → 3D 공\" 사다리.
  - bg_level (3 값): blank / ground / scene. 지면이 있나? 풍경이 있나?
  - cue_level (4 값): none / cast_shadow / motion_arrow / both. 그림자·화살표 등 motion 단서.
  - event (3 값): fall / horizontal / rise. 프레임 위치(낙하/수평/상승).
  - label (4 값): circle / ball / planet / _nolabel. 언어 라벨이 응답을 흔드는지.
- 우측 4컷의 예시 이미지로 \"같은 사건(낙하)이 다른 옷으로 어떻게 보이는지\" 직관 제공.
- 총합: 4 × 3 × 4 × 3 × 10 seed = 1,440 × 3 라벨 + label-free = 2,880 추론 / 모델.
- 외부 타당성 확장: M8a (5 도형, 원 외 추가), M8d (3 카테고리: 차/사람/새), M8c (60 실사진).

---

## 슬라이드 9 — 5개 테스트 모델

**한 문장**: 인코더 계열 × LM 계열을 골고루 섞어 \"어떤 부품이 결정자인지\"를 비교 가능하게 만든 model zoo.

| 모델 | 인코더 | Projector | LM | 역할 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | SigLIP | MLP | Qwen2-7B | 메인 — 인과 실험 anchor |
| LLaVA-1.5-7B | CLIP-ViT-L | MLP | Vicuna-7B | Floor — S-curve 가장 깨끗 |
| LLaVA-Next-7B | CLIP-ViT-L | MLP | Mistral-7B | Mid — 결정자 분리 핵심 |
| Idefics2-8B | SigLIP-SO400M | **perceiver-resampler** | Mistral-7B | 유일한 perceiver |
| InternVL3-8B | InternViT-300M | MLP (pixel-shuffle) | InternLM3-8B | 비-CLIP 비교점 |

- 핵심 비교쌍 — LLaVA-1.5 vs LLaVA-Next: 동일 CLIP-ViT-L 인코더지만 LM과 처리 방식이 다름. PMR이 0.18 ↔ 0.79로 갈리므로 \"인코더 단독 결정자\" 가설을 disconfirm 하는 가장 깨끗한 사례.
- Idefics2의 perceiver-resampler가 shortcut의 \"마지막 미스터리\"의 핵심 단서가 됩니다 (슬라이드 20).

---

## 슬라이드 10 — 측정 지표 정의

**한 문장**: 응답 텍스트 위에서 자동 채점한 6개 지표 — 사람 검수 ~5% 불일치.

- **PMR** (Physics-Mode Reading rate): 응답에 falls / rolls / bounces 같은 물리 동사가 들어간 비율. 본 연구의 메인 지표.
- **PMR(_nolabel)**: 라벨 없는 open-ended 프롬프트의 PMR. 모델 자체 편향 측정.
- **GAR** (Gravity-Align Rate): 물리 응답 중 \"하방으로 떨어진다\"고 답한 비율 — 중력 정합도.
- **RC** (Response Consistency): T=0.7 샘플링에서 N seed의 PMR call 일관도. 결정 안정성.
- **H2 paired-Δ**: PMR(label) − PMR(_nolabel). 라벨 prior가 PMR을 얼마나 끌고 가는지.
- **v_L 방향**: L 레이어 hidden state에서 mean(physics=1) − mean(physics=0). \"물리 모드 방향\" 추정.
- 신뢰구간: Wilson 95% / 5000-iter bootstrap CI.
- 발표 팁: PMR 한 단어만 외우면 됩니다. 나머지는 \"보조 지표\"로 알리는 정도.

---

## 슬라이드 11 — 결과 흐름 한눈에

**한 문장**: 6단계로 좁혀 들어가며 \"행동 → 메커니즘 → 픽셀\" 순으로 답을 좁힌다.

1. **행동 사다리** — 5 모델 PMR이 0.18 ↔ 0.99. 같은 자극인데 모델별 격차.
2. **Encoder boomerang** — 인코더 모두 보고 있다(AUC≈1.0)지만 행동은 갈림.
3. **원인은 architecture** — 같은 CLIP인 LLaVA-1.5(0.18) vs LLaVA-Next(0.70). 인코더 단독 결정자 X.
4. **LM의 한 레이어가 결정** — Qwen L10에 ±α·v_L 더하면 응답이 뒤집힌다.
5. **Encoder의 ~30개 feature** — SAE 학습 후 top-k feature만 ablate → 3 of 5 모델 PMR 무너짐.
6. **픽셀에 인코드 가능** — 픽셀 공간 gradient ascent로 추상 원 → 물리 응답. Idefics2만 차단.

각 단계가 다음 단계의 질문을 낳고, 모든 단계가 슬라이드 23의 5겹 시그니처로 수렴합니다.

---

## 슬라이드 12 — 결과 1: PMR 사다리

**한 문장**: 같은 480개 자극 위에서 5개 모델의 PMR(_nolabel)이 0.38 → 0.99로 무려 0.6 차이가 난다 — 인코더만으로는 설명 안 되는 격차.

- 그래프 (`docs/figures/m2_cross_model_pmr_ladder.png`): x축은 라벨 (ball / circle / planet / _nolabel), y축은 PMR. 각 라벨에서 5개 모델 막대 + 95% CI.
- _nolabel 막대(가장 오른쪽 그룹)에 주목하면 \"라벨 없이 본 모델 자체 편향\"이 보임.
- 표:
  - LLaVA-1.5: 0.38 [0.34, 0.43] — Floor
  - LLaVA-Next: 0.79 [0.75, 0.83] — Mid
  - Qwen2.5-VL: 0.94 [0.92, 0.96] — Saturated
  - Idefics2: 0.97 [0.95, 0.98] — Saturated
  - InternVL3: 0.99 [0.98, 1.00] — Saturated
- 핵심 — 같은 인코더 계열(CLIP) 안에서도 PMR이 0.38 ↔ 0.79로 갈린다 → 인코더 단독 결정자 가설 disconfirm.

---

## 슬라이드 13 — 결과 1b: H1 ramp & H2 라벨 효과

**한 문장**: 추상화가 올라가면 PMR도 올라가지만, 천장이 있는 모델(Qwen / Idefics2 / InternVL3)에서는 측정 헤드룸이 없다. 라벨 효과도 모델별 부호 패턴이 3가지로 갈린다.

- 좌측 그래프 (`m2_cross_model_h1_ramp.png`) — H1 ramp: line → filled → shaded → textured 따라 PMR 증가. LLaVA-1.5만 깨끗한 +0.30 ramp. Qwen / Idefics2 / InternVL3는 첫 단계부터 천장에 붙어 있음.
- 우측 그래프 (`m2_cross_model_h2_paired_delta.png`) — H2 paired-Δ: PMR(label) − PMR(_nolabel) 부호가 모델별로 다름.
  - LLaVA-1.5 / LLaVA-Next (unsaturated CLIP): 모든 라벨 양수. ball Δ=+0.475 (LLaVA-1.5).
  - Qwen / Idefics2 (포화 SigLIP-계열): 비-물리 라벨(circle, planet)이 baseline 아래로 억제 — \"circle override\".
  - InternVL3 (완전 포화): Δ ≈ 0.
- 해석: \"라벨이 항상 PMR을 더한다\"가 아님. 라벨 효과의 부호 자체가 인코더 saturation 상태에 따라 결정.

---

## 슬라이드 14 — 결과 2: Encoder boomerang

**한 문장**: 5개 인코더가 모두 추상↔물리를 0.99 AUC로 \"이미 보고\" 있는데, 행동은 0.18 ↔ 0.99로 갈린다.

- 그래프 (`encoder_chain_5model.png`): 좌 — 3개 모델의 layer별 vision encoder probe AUC. 우 — encoder probe AUC vs behavioral PMR 산점도.
- 결과:
  - Qwen SigLIP: AUC 0.99
  - Idefics2 SigLIP-SO400M: 0.93
  - InternVL3 InternViT: 0.89
  - LLaVA-Next CLIP-ViT-L: 0.81
  - LLaVA-1.5 CLIP-ViT-L: 0.73
- 더 중요한 사실 — stim-defined Y target으로 모든 인코더 AUC = 1.0. 인코더 표현능력은 균일.
- 결론: \"Encoder knows, decoder gates\". 인코더는 정보를 가지고 있으나 LM이 그 정보를 다르게 \"gating\" 한다.

---

## 슬라이드 15 — 결과 3: M9 generalization audit

**한 문장**: 합성 도형이든 카테고리든 실사진이든, 같은 클러스터링이 보존됨 — paper Table 1 재료.

- 그래프 (`m9_summary.png`): 3 모델 × 3 자극 source × 95% bootstrap CI.
- 좌측 패널 (PMR_nolabel):
  - 합성 자극 (M8a): non-CLIP cluster [0.84, 0.92] vs CLIP-1.5 [0.14, 0.21] 완전 분리.
  - 사진 (M8c): 모든 모델 [0.28, 0.55]로 수렴 — 사진은 인코더 격차를 압축한다.
- 우측 패널 (H7 delta — 라벨이 regime을 선택하는 효과):
  - Synthetic stim: LLaVA-1.5 +0.36 (M8a), +0.31 (M8d) — 가장 강한 H7 evidence.
  - Photos (M8c): 모든 모델 H7 ≈ 0 — 사진은 라벨 효과를 무력화.
- 청중 친화적 해석: \"같은 결과가 자극을 바꿔도 보존된다 → robustness\".

---

## 슬라이드 16 — 결과 4: VTI Steering at LM L10

**한 문장**: LM의 10번째 레이어에서 hidden state에 \"+α·v_L10\"이라는 한 줄짜리 개입을 가하면 \"원\"이 \"공\"이 된다 — 인과 관계의 직접 증거.

- v_L10 정의: \"physics=1 응답들의 hidden state 평균\" − \"physics=0 응답들의 hidden state 평균\". \"물리 모드 방향\".
- 자극: line/blank/none (가장 추상한 원).
- 응답 표:
  - α=0 (baseline): \"This is just a circle on a white background.\" — 추상 (D)
  - α=+10 / +20: 여전히 추상.
  - α=+40: \"It stays still — the circle appears to be floating in space without external force.\" — 물리·정지 (B). 10/10 응답 모두 뒤집힘.
  - α=−40: \"The circle remains stationary, suspended.\" — 정적 물리. (M5a-ext에서 발견)
- 핵심: L10에서만 일어나고(L15·L20·L25는 안 됨), |α| > 임계값에서 \"regime axis\"로 작동. +α는 동적 물리, −α는 정적 물리. 어느 방향이든 추상 → 물리로 끌고 간다.
- 발표 팁: 이 슬라이드가 청중에게 가장 \"마법 같은\" 결과로 보임. \"이미지 한 픽셀도 안 건드렸는데 응답이 뒤집힌다\"가 메시지.

---

## 슬라이드 17 — 결과 5: Encoder의 SAE feature ablation

**한 문장**: 인코더 표현에서 SAE를 학습한 뒤 top-k feature 약 30개만 0으로 끄면 PMR이 무너진다 — 인과 좁히기의 다른 방향.

- 그래프 (`m5b_sae_intervention_cross_model.png`): x축 ablate한 SAE feature 수, y축 PMR.
  - Qwen은 k=40에서 PMR이 0으로 떨어짐.
  - LLaVA-1.5 / LLaVA-Next는 어떤 k에서도 1.00 유지.
  - Idefics2 / InternVL3는 k=160 부근에서 떨어짐.
- 표:
  - Qwen2.5-VL: L31, BREAK at k=40
  - LLaVA-1.5: L22, NULL
  - LLaVA-Next: L22, NULL
  - Idefics2: L26, BREAK at k=160
  - InternVL3: L23, BREAK at k=160
- Random k 컨트롤은 모두 1.00 — 방향 특이성 검증.
- 결론:
  - 비-CLIP: 인코더 안에 \"물리 모드\" 표현이 국소화 (~30개 feature).
  - CLIP family: 인코더에 없음 → LM-side direction 으로만 라우팅.
  - 이게 \"5겹 시그니처\" 중 다섯 번째 (encoder vs LM 분기).

---

## 슬라이드 18 — 결과 5b: 부품별 필요성 (knockout)

**한 문장**: Qwen 안에서 어떤 부품이 \"물리 모드 commitment\"를 만드는지 좁혀 들어가면 — L9의 MLP 하나로 수렴한다.

- 좌측 그래프 (`m5b_knockout_per_layer_ie.png`):
  - Attention knockout (necessity): 28 레이어 모두 IE = 0 — attention은 redundant.
  - MLP knockout (necessity): L9에서만 IE = +1.0. L8(0.4), L10(0.6), L11(0.4), L14(0.4)는 부분.
- 우측 그래프 (`m5b_sip_per_layer_ie.png`): SIP activation patching per-layer IE — L9까지 1.0, L14+ 0. 수직 절벽.
- 우측 박스 (Qwen Triangulation):
  - Encoder ~30개 SAE feature가 운반 →
  - L0~L9 visual token이 정보 운반 →
  - **L9 MLP에서 commitment 생성** →
  - L10에서 attention이 redundantly read-out →
  - 출력 letter B/D 결정.
- 발표 팁: \"단 하나의 MLP가 모델의 마음을 결정한다\"가 메시지.

---

## 슬라이드 19 — 결과 6: 픽셀에 인코드 가능 (§4.6)

**한 문장**: 이미지에 \"눈에 거의 안 보이는 작은 노이즈\"만 더해도 응답이 \"공이 떨어진다\"로 뒤집힌다.

- 좌측 그래프 (`sec4_6_counterfactual_stim_trajectory.png`): 픽셀 공간에서 v_L10 방향으로 gradient ascent. 200 step Adam, ε=0.1. 6 가지 설정의 v_L10 projection 추적.
- 응답 비교 (우측):
  - **Baseline (ε=0)**: \"The circle will remain stationary as there is no indication of movement.\" — 추상 응답.
  - **v_L10 ascent (ε=0.05)**: \"The circle will continue to fall downward due to gravity.\" — 물리 응답으로 collapse.
- 통계: 5/5 v_L10 flip vs 매칭 magnitude random 0/15. 즉 \"방향 특이성\" 보장 — 단순한 \"아무 노이즈\"가 응답을 뒤집는 게 아님.
- 결론 (콜아웃): shortcut은 \"runtime hidden injection\"만이 아니라 픽셀 자체에 인코드 가능. 매칭 magnitude random 0/15가 \"any perturbation\" 가설을 falsify.

---

## 슬라이드 20 — 결과 6b: 픽셀 routability cross-model

**한 문장**: 픽셀에 인코드되는 정도가 모델별로 갈린다 — 특히 Idefics2만 9 레이어 모두 0/10. perceiver-resampler가 마지막 미스터리.

- 그래프 (`sec4_6_cross_model_layer_sweep.png`): 5 모델 × LM layer × n=10. 각 패널은 v_unit 방향(파랑)과 random 방향(회색)의 PMR flip rate를 95% Wilson CI로 표시.
  - Qwen: L5/10/15/20/25 모두 ≥ 80%. Broad shortcut.
  - LLaVA-Next: L20+L25에서 10/10 flip.
  - LLaVA-1.5: L25 only, 4/10 — n=10에서 약화.
  - **Idefics2: 9 레이어 (L5-L31, 16-97% 깊이) 모두 0/10**.
  - InternVL3: protocol-saturated (baseline=1).
  - Random control: aggregate 1/250.
- Idefics2 단독 패턴 박스 (우측):
  - 9개 layer 모두 0/10 flip
  - **그러나** v_L projection은 정상 ascending (-11→+28 at L26-30, -72→+163 at L31)
  - **M4 LM probe AUC 0.995** (정보는 LM 도달!)
  - **M5a forward steering 10/10** (forward는 작동)
  - → perceiver-resampler 가설: perceiver는 \"픽셀 → v_L\" **역방향 routability만** 차단. forward는 통과시킴.
- 이 dissociation은 \"정보 LM 도달 ≠ 픽셀-공간 routability\"라는 미묘한 구분의 인과 evidence.

---

## 슬라이드 21 — 외부 타당성 (M8a / M8d / M8c)

**한 문장**: 원에서만 보이는 현상이 아니라, 다른 도형 / 다른 카테고리 / 실사진까지 같은 클러스터링이 보존된다.

- M8a — 5 도형 × 4 추상화 (`m8a_shape_grid.png`). circle / square / triangle / hexagon / polygon × line / filled / shaded / textured.
  - Qwen 1/4 PASS, LLaVA 4/4. 비대칭 자체가 H-encoder-saturation 가설을 cross-shape로 검증.
- M8d — 차 / 사람 / 새 (`m8d_full_scene_samples.png`). 비-공 카테고리.
  - LLaVA 3/3 H7 PASS — 라벨이 regime을 선택. 본 연구의 가장 강한 H7 evidence.
- M8c — 60 실사진 (`m8c_photo_grid.png`). COCO + WikiArt에서 추출.
  - 사진은 인코더 격차를 압축, label 효과 절반으로. 사진의 \"풍부한 cue\" 가 saturation 천장을 깨뜨린다.
- 메시지: 본 연구의 클러스터링이 자극 source에 강건하다 (robustness).

---

## 슬라이드 22 — 추가 강건성: scaling & multilingual

**한 문장**: 모델을 5배 키워도(7B → 32B) PMR 천장은 안 깨지고, 라벨을 한국어/일본어로 바꿔도 모델별 ordering은 보존된다.

- 좌측 표 (§4.8 Qwen 7B vs 32B):
  - Aggregate PMR: 0.931 → 0.926. 거의 변화 없음.
  - abstract_reject: 0.002 → 0.065. 35× 증가 — 32B가 cue 약할 때 abstract-mode를 더 본다.
  - H2 ball−circle: +0.071 → +0.010. 절반으로 줄지만 dissolved는 아님.
  - cue=none PMR: 0.797 → 0.711. -8.6 pp drop.
  - 결론: \"scale doesn't fix grounding\" — MechBench-style finding.
- 우측 그래프 (`sec4_3_korean_vs_english_cross_model.png`): 한국어 라벨 (공/원/행성) vs 영어 (ball/circle/planet) × 5 모델.
  - 라벨 ordering 4/5 모델 보존.
  - LLaVA-1.5가 가장 큰 swing — Vicuna 한국어 SFT 약함의 시그널.
  - Idefics2 일본어 \"惑星\"에서 24% Chinese fallback (Mistral의 일본어 SFT가 약해서 한자를 중국어로 인식).

---

## 슬라이드 23 — 종합: 5겹 다운스트림 시그니처

**한 문장**: 5개 별도의 측정 방식이 모두 같은 architecture clustering을 가리킨다 — 단일 architectural property가 5겹 redundant하게 표현된다는 뜻.

1. **PMR ceiling** — 비-CLIP [0.84, 0.92] vs CLIP [0.14, 0.37] 분리. (행동 1)
2. **Decision-stability ceiling (RC)** — 비-CLIP은 cue 발화 시 5 seed 모두 동일 call. (행동 2)
3. **픽셀 encodability** — Qwen broad shortcut, Idefics2 0/9 (perceiver bottleneck). (메커니즘)
4. **LM logit-lens probe AUC** — encoder probe ladder와 동일 클러스터링. (메커니즘)
5. **Encoder SAE feature ablation** — 3 of 5 break, 2 LLaVA NULL (encoder vs LM 분기). (메커니즘)

→ 5개 시그니처가 같은 3-cluster decomposition을 만든다 (High / Mid / Low saturation). 단일 architectural property의 redundant manifestation.

이게 본 연구의 \"strongest claim\". 단순 \"인코더 capacity\" 가설로는 설명 불가.

---

## 슬라이드 24 — 한계 & Future

**한 문장**: Architecture-level finding은 lock 되었지만 isolation은 아직 미해결 — 다음 단계는 controlled swap 실험들.

1. **Projector isolation 미검증** (Pillar B). Idefics2의 perceiver-resampler가 leading candidate. encoder/LM 동일 + perceiver↔MLP swap으로 검증 필요. M-PSwap LoRA 진행 중 (NaN 미해결).
2. **LM-only counterfactual 부재** (Pillar B). LLaVA-1.5(0.18) → LLaVA-Next(0.70) PMR jump는 4축 confound. CLIP+Vicuna vs CLIP+Mistral controlled swap (M-LMSwap) 진행 중.
3. **Single-task evaluation**. next-state-prediction만 검증. Counting / spatial / causality 등 다른 shortcut 미검증. M-MP Pillar A가 4-prompt로 확장 (cross-method split @ Qwen×MCQ 발견).
4. **Human baseline 미수집**. M7 Prolific 20 raters × 50 stim 계획 — paper-blocking 단계. 인간이 \"같은 자극\"에서 같은 PMR 패턴을 보이는지 직접 검증.
5. **External model 한정**. Pixtral / Phi-3.5-V / GPT-4V / Gemini-VL 등 닫힌 모델은 미테스트. 5 model의 클러스터링이 더 큰 모델에서도 보존되는지 확인 필요.

---

## 슬라이드 25 — 결론

**한 문장**: 행동 → 메커니즘 → 픽셀의 3차원으로 \"shortcut의 위치\"를 좁혀 들어갔다.

1. **Architecture-level reframe**. 행동 PMR ceiling은 인코더 표현력 단독으로 결정 안 됨. 5-fold downstream signature가 동일 architectural property의 redundant manifestation. CLIP 2-point 비교(LLaVA-1.5 vs Next)가 가장 깨끗한 disconfirmer.
2. **Causal localization (LM + Encoder)**. M5a runtime steering: 3 of 4 모델 10/10 PMR flip. M5b SAE encoder ablation: 3 of 5 break, 2 LLaVA NULL. \"CLIP cluster commitment는 LM-side direction으로만 라우팅, 비-CLIP은 encoder + LM 둘 다.\"
3. **Pixel encodability — architecture-conditional**. 픽셀 공간 gradient ascent로 3 of 5 모델 flip. Idefics2 9-layer 0/10 → perceiver-resampler가 forward 정보 통과는 시키되 inverse 픽셀 routability만 차단. M4 + M5a + §4.6 dissociation으로 가설 정밀화.

**Big picture**: VLM의 \"원→공\" shortcut은 단순 model quirk가 아니라 architecture-level saturation의 다차원적 표현이다.

---

## 슬라이드 26 — 감사합니다 / 참고

- 동반 자료
  - 슬라이드별 상세 한국어 해설: `docs/review_ppt/physical_mode_storyline_ko.md` (이 문서).
  - 본 논문 자료 인덱스: `references/roadmap.md` (single source of truth).
  - 가설 evidence chain: `docs/hypotheses.md`. 마일스톤별 인사이트: `docs/insights/*.md`.
- Repo 구성: `src/physical_mode/` (package), `scripts/0{1..6}_*.py`, `scripts/sec4_6_*.py`, `scripts/sae_*.py`, `configs/`, `tests/`.
- 발표 후 토론 우선순위:
  1. M-LMSwap (Vicuna vs Mistral on CLIP+ViT-L) 결과가 LLaVA family NULL을 어떻게 설명할 것인가?
  2. 인간 baseline (M7 Prolific) 우선순위는 어디에 둘 것인가?
  3. Pixtral / GPT-4V 등으로 모델 풀 확장이 paper에 critical 한가?

---

## 부록: 발표 시간 가이드

| 단락 | 슬라이드 | 권장 시간 |
|---|---|---|
| 도입 (hook + question + why) | 1–4 | 4 분 |
| 배경 (VLM 한 장 도식 + 선행 연구) | 5–6 | 3 분 |
| 자극 + 모델 + 지표 | 7–10 | 4 분 |
| 결과 흐름 + 행동 결과 | 11–13 | 5 분 |
| 메커니즘 결과 (encoder boomerang → SAE → MLP) | 14–18 | 7 분 |
| 픽셀 결과 + Idefics2 미스터리 | 19–20 | 4 분 |
| 외부 타당성 + scaling/multilingual | 21–22 | 3 분 |
| 종합 + 한계 + 결론 + Q&A | 23–26 | 5 분 |
| **합계** |  | **~35 분** |

질의응답을 포함한 45 분 슬롯을 가정한 시간 배분입니다. 시간이 짧다면 슬라이드 18 (knockout triangulation) 과 22 (scaling/multilingual) 를 가장 먼저 떨어뜨리고, 슬라이드 16 (VTI steering) 과 19 (픽셀 인코드) 는 어떤 시간 제약에서도 살려두는 것을 추천합니다.
