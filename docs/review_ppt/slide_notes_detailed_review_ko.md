# 원이 공으로 보이는 순간 — 종합 검토 슬라이드 노트 (입문자용 자세한 버전)

> 본 문서는 `docs/review_ppt/physical_mode_detailed_review_ko.pptx` (45 슬라이드)의
> 동반 자료입니다. **본 도메인을 처음 보시는 분도 따라오실 수 있도록** 모든 약어를
> 풀어서 적었고, 각 milestone (M-…) 이 어떤 실험이었는지 자세히 설명했습니다.

---

## 0. 본 deck 의 의도 + 용어집 (먼저 읽어주세요)

이 프로젝트는 \"VLM (비전-언어 모델, Vision-Language Model — 그림과 글을 동시에
이해하는 AI) 이 추상적인 도형 (예: 그냥 \"동그라미\" 하나) 을 보여줬을 때 왜
\"공이 떨어진다\" 같은 물리적 응답을 하는가?\" 를 분석합니다.

### 핵심 용어 (이것만 알면 됩니다)

| 용어 | 풀어 쓰면 | 본 문서에서 의미 |
|---|---|---|
| **VLM** | Vision-Language Model | 이미지 + 텍스트 → 텍스트 응답을 생성하는 AI 모델 (예: ChatGPT 4V, Qwen2.5-VL, LLaVA 등) |
| **Encoder (vision encoder)** | 이미지를 \"AI 가 이해할 수 있는 숫자 (벡터)\" 로 바꿔주는 모듈 | CLIP, SigLIP 등이 대표적 |
| **Projector** | Encoder 가 만든 비전 벡터를 LM 의 어휘 공간으로 \"번역\" 하는 작은 신경망 | 보통 2-layer MLP 또는 cross-attention 기반 perceiver |
| **LM (Language Model)** | 텍스트를 입력받아 다음 토큰을 예측하는 모델 | Vicuna, Mistral, Qwen2 등 |
| **PMR** | Physics-Mode Reading rate — 물리 모드 응답률 | 응답 안에 \"falls / rolls / bounces\" 같은 물리 동사가 있으면 1, 없으면 0. 이 비율이 높을수록 모델이 \"물체가 움직인다\" 라고 답하는 경향 |
| **GAR** | Gravity-Align Rate | 물리 응답 중에서 \"아래로 떨어진다\" 라고 답한 비율 (중력 정합도) |
| **RC** | Response Consistency | 같은 프롬프트를 5 번 돌렸을 때 PMR 결정이 얼마나 일관되는가 |
| **Probe** | linear probe — \"hidden state 에 \"physics-mode\" 정보가 얼마나 들어 있나\" 를 측정하는 간단한 분류기 | AUC 1.0 = 완벽 분리, 0.5 = 무작위 |
| **AUC** | Area Under ROC Curve | probe 의 분리 성능. 1.0 가 max, 0.5 가 chance level |
| **SAE** | Sparse Autoencoder — 모델 hidden state 를 \"해석 가능한 feature\" 로 분해하는 도구 | 5120 개 feature 로 분해 후 어떤 feature 가 physics 와 가장 관련 있는지 ranking |
| **Ablation / knockout** | 특정 component (feature, head, layer) 의 출력을 0 으로 강제하는 개입 | 이 ablation 후에도 응답이 같으면 \"그 component 는 dispensable\", 응답이 바뀌면 \"필수\" |
| **Steering / VTI (Visual Token Intervention)** | LM 의 hidden state 에 특정 방향 vector 를 더하는 개입 | hidden_L10 ← hidden_L10 + α·v_L10 |
| **SIP (Subspace Intervention via Patching)** | 두 개의 다른 입력 (\"clean\" + \"corrupted\") 의 hidden state 를 layer 별로 patching 해서 \"어디서 정보가 lock 되는가\" 를 측정 | 본 연구에서는 \"L0-L9 patching 으로 회복되면 commitment 가 그 이전 layer 에서 발생\" |
| **logit lens** | LM 의 중간 layer hidden 을 unembedding 에 직접 통과시켜 \"이 layer 가 어떤 단어를 \"생각\"하는가\" 를 보는 기법 | layer-wise probe 의 한 종류 |
| **factorial stim** | 여러 axis (추상화 / 배경 / 단서 / event) 를 cross 한 stimulus 디자인 | 본 연구의 M2 = 2,880 자극 |

### Milestone 코드명 풀이 (M0 ~ M9)

| 코드 | 풀어 쓰면 | 무엇을 했나 |
|---|---|---|
| **M0** | Milestone 0 — Infrastructure | 코드 패키지 구조, 자극 생성 / 추론 / 채점 파이프라인 구축. 모든 실험의 토대 |
| **M1** | Milestone 1 — Pilot | 첫 번째 작은 실험 (240 stim × 2 prompt). \"PMR 가 정말 abstraction axis 따라 오르나?\" 의 첫 측정 |
| **M2** | Milestone 2 — MVP-full | 본격 factorial 실험 (2,880 stim × Qwen2.5-VL). 이후 모든 분석의 기준 자극 |
| **M3** | Milestone 3 — Vision encoder probing | 인코더의 hidden state 위에 linear probe 학습. \"인코더 안에 physics-mode 정보가 있는가?\" 측정 |
| **M4** | Milestone 4 — LM logit lens / per-layer probe | LM 의 layer-by-layer hidden state 위에 probe 학습. \"어느 layer 부터 physics 정보가 등장하나?\" |
| **M5a** | Milestone 5a — VTI steering | LM hidden 에 +α·v_L 더하는 인과 개입. \"한 layer 에 노이즈 주면 응답이 뒤집히나?\" |
| **M5b** | Milestone 5b — SIP + MLP/attention knockout + SAE intervention | M5a 보다 더 정밀한 인과 분석. \"어느 component (MLP / attention head / SAE feature) 가 정확히 책임지나?\" |
| **M6** | Milestone 6 — Cross-model rounds (r1-r7) | 5 개 다른 VLM 위에서 M2-M5 를 다 다시 돌림. 결과의 일반화 검증 |
| **M7** | Milestone 7 — Human baseline | (예정) 사람 20명 × 50 stim 으로 \"인간이 같은 자극에서 같은 PMR 패턴을 보이나?\" 검증 |
| **M8a/c/d/e** | Milestone 8 a/c/d/e — External validity | 다른 도형 (M8a), 실사진 (M8c), 비-공 카테고리 (M8d), source consolidation (M8e) |
| **M9** | Milestone 9 — Bootstrap CI generalization | 5000-iter bootstrap 으로 cluster 분리의 통계적 유의성 검증 |

### \"§4.x\" 가 뭔가요?

`references/project.md` 의 \"§4. Additional ideas\" 섹션 항목들. 본 프로젝트의
ROADMAP 에서 \"4.6\" / \"4.8\" 같이 부르는 follow-up 실험들:

| 코드 | 뭐였나 |
|---|---|
| **§4.3** | Korean / Japanese 라벨 swap — 라벨 언어가 PMR 에 영향 주는가? |
| **§4.5** | Cross-encoder swap — SigLIP-SO400M + Mistral 조합도 Qwen 패턴 재현하나? |
| **§4.6** | Pixel-space counterfactual stim — 픽셀에 작은 노이즈만 주면 PMR 뒤집히나? |
| **§4.7** | Decision-consistency boundary — RC 가 axis 별로 어디서 깨지나? |
| **§4.8** | PMR scaling — 7B 와 32B 가 다른가? |
| **§4.10** | Attention visualization UI |
| **§4.11** | H7 follow-up — label-regime category annotation |

### \"Pillar A / B / C\" 는 뭔가요?

본 프로젝트가 paper 까지 가기 위한 3 갈래 보강 작업:

- **Pillar A** = M-MP (Multi-Prompt). 하나의 prompt 만 검증된 한계를 극복 — 3 개
  prompt × 5 모델 × 480 stim 으로 cross-prompt validation.
- **Pillar B** = M-LMSwap + M-PSwap. controlled swap 실험 — \"LM 만 다르게\" 또는
  \"projector 만 다르게\" 학습한 변종 모델로 인과 isolation.
- **Pillar C** = M-Marr. paper §6 을 Marr 3-level (Computational / Representational /
  Mechanistic) 로 재구성.

---

## Block A — Setup (1-7)

### 슬라이드 1 — 표지

**한 문장**: VLM 의 추상→물리 shortcut 을 행동·메커니즘·픽셀 3 차원으로 5 개
모델 (+ 32B 추가 + M-LMSwap 통제 실험) 위에서 분석한 8 일간 의 종합 검토.

본 deck 은 기존 storyline 과 달리 **모든 실험 (성공 + 실패), 디테일, 현재 진행
중인 작업** 을 다룹니다. \"이 프로젝트의 모든 것을 한 번에 알고 싶다\" 일 때
사용하시면 됩니다.

---

### 슬라이드 2 — Hook (놀라운 관찰)

**한 문장**: 가장 단순한 자극 (line/blank/none — 흰 배경 위에 검은 원 하나, 다른
단서 0) 위에서 사람과 AI 의 응답이 정반대로 갈라집니다.

- 사람의 응답: \"흰 배경 위 검은 원이 그려져 있다.\" — 추상적 도형으로 묘사.
- Qwen2.5-VL 의 응답: \"The ball will fall down due to gravity.\" — 갑자기 \"공\" /
  \"중력\" / \"낙하\" 등장.
- 무엇이 이상한가:
  1. 이미지에 중력 단서·지면·텍스처가 **전혀 없다**.
  2. 모델은 추상↔물리 두 해석 중 **한쪽으로만 collapse** 한다.
  3. 이런 shortcut 은 알려져 있지만 **어디서·왜** 일어나는지 미해결.
  4. 모델별 강도가 다르다 — PMR_nolabel: LLaVA-1.5 0.18 → InternVL3 0.99 (5.5×).

PMR_nolabel = 라벨 (\"ball\" / \"circle\") 없이 \"What might happen next?\" 만 물어봤을
때 물리 동사가 응답에 들어간 비율. 즉 **모델 자체의 편향**.

---

### 슬라이드 3 — Project journey timeline (8 일간 여정)

8 일간의 일별 기록. 색상 의미:
- 🟢 초록 = 성공 milestone
- 🔵 파랑 = 메커니즘 발견
- 🟠 주황 = warning / audit
- 🔴 빨강 = 실패 / 학습
- 🟡 액센트 = 현재 진행

각 일별 사건:

- **04-24 (M0–M5a 한 번에 통과)**: 코드 인프라부터 시작해 M1 pilot → M2 본실험
  → M3 인코더 probe → M4 LM probe → M5a 인과 개입 까지 한 번에 통과. 이 날
  하루에 \"L10 α=40 으로 응답 10/10 뒤집힌다\" 발견.
- **04-25 (M5a-ext + M6 r1 + M8 외부 타당성)**: 인과 개입의 음수 부호 발견 →
  regime axis 재해석. LLaVA-1.5 가 \"floor\" (PMR 0.18) 임을 발견. 5 도형 / 3
  카테고리 / 60 실사진 으로 외부 타당성 확장.
- **04-26 (M5b SIP + 5-model M2 + §4.6 픽셀)**: L9 MLP 가 단독으로 책임진다는
  발견 (knockout IE = +1.0). 5 모델 PMR 사다리 lock. 픽셀 공간 v_L10 ascent 5/5.
- **04-27 (M5b per-head + §4.6 5-model n=10)**: 196 (layer, head) cells 모두
  IE = 0 — attention 은 fully redundant. 5 모델 픽셀 sweep.
- **04-28 (M5a/M5b cross-model + §4.8 7B vs 32B + M4 LM AUC)**: cross-model
  steering 3/4 성공. M5b SAE encoder ablation (3 break, 2 NULL). 32B PMR 변화 없음.
- **04-29 (M-PSwap NaN backlogged + M-LMSwap 채택)**: M-PSwap 학습 NaN 미해결로
  대기. M-LMSwap (LM-only swap) 으로 우선순위 이전. 50-step smoke PASS.
- **04-30 (Variant A Stage 1+2 ~24h)**: LCS-558K → LLaVA-Instruct-665K 학습
  21K step 도달. 학습 NaN 없음.
- **05-01 오늘 (Regression split + post-proj round 2 + B5 Qwen 32B)**: step9000
  gate FAIL, step21000 baseline=0.000 PASS (cell-discrimination 학습!). M5b
  round 2 (5×4 cells) regime-cross ladder 발견. Qwen 32B post-proj k=40 break.

---

### 슬라이드 4 — 3축 질문 (WHEN / WHERE / HOW)

**한 문장**: 같은 자극 + 같은 5 모델 위에서 \"행동 → 메커니즘 → 픽셀\" 3 차원으로
좁혀 들어간다.

- **WHEN (행동 — 언제)**: 어떤 자극에서 / 어떤 모델에서 shortcut 이 얼마나 강한가?
  - 도구: PMR / GAR / RC / paired-Δ / open vs forced-choice / _nolabel
  - 결과: 5-model PMR 사다리 0.18 → 0.99, H1 ramp, H2 3 패턴, M9 robust
- **WHERE (메커니즘 — 어디서)**: 모델 내부 어떤 layer / 방향이 결정자인가?
  - 도구: linear probe, logit lens, VTI steering, SAE feature ablation, SIP
    activation patching, MLP/attention knockout
  - 결과: L10 α=40 flip 10/10, L9 MLP IE=+1.0, ~30 SAE feature break, 196
    (L,h) = 0
- **HOW (픽셀 — 어떻게)**: shortcut 이 픽셀에 인코드 가능한가?
  - 도구: v_L gradient ascent on pixels, matched-mass random control
  - 결과: Qwen broad shortcut, LLaVA-Next L20+L25, Idefics2 0/9 (perceiver candidate)

본 연구의 차별점은 \"동일 자극·5 개 모델·3 차원\" 이 **한꺼번에** 측정된다는 것.

---

### 슬라이드 5 — VLM 파이프라인 + 5 model zoo

**한 문장**: VLM 은 \"이미지 → 비전 인코더 → projector → LM → 응답\" 5 단계
파이프라인. 본 연구는 각 단계를 분리해서 측정.

- 이미지 → **Vision Encoder** (CLIP / SigLIP / InternViT). \"이미지 → 비전 패치
  벡터\" 변환. 측정: M3 probe AUC, M5b encoder SAE.
- → **Projector** (MLP 또는 perceiver-resampler). \"비전 벡터 → LM 어휘 공간
  벡터\" 번역. 측정: M5b post-projection SAE round 2.
- → **Language Model** (Qwen / Vicuna / Mistral / InternLM). 추론. 측정: M4 logit
  lens, M5a VTI steering, M5b SIP, MLP/head knockout.
- → **응답 (text)**. 측정: PMR / GAR / RC, open vs FC, _nolabel.

5 model zoo 표 (architecture 사양 + PMR_nolabel + 역할):

| 모델 | Encoder | Projector | LM | PMR_nolabel |
|---|---|---|---|---|
| Qwen2.5-VL 7B/32B | SigLIP-400M / SO400M | merger MLP | Qwen2 | 0.94 |
| LLaVA-1.5 7B | CLIP-ViT-L-336 | 2-layer MLP | Vicuna-7B | 0.18 (Floor) |
| LLaVA-Next 7B | CLIP-ViT-L-336 AnyRes | 2-layer MLP | Mistral-7B | 0.79 |
| Idefics2 8B | SigLIP-SO400M | **perceiver-resampler** | Mistral-7B | 0.97 |
| InternVL3 8B-hf | InternViT-300M | MLP (pixel-shuffle) | InternLM3-8B | 0.99 |
| **M-LMSwap A** (학습완료) | CLIP-ViT-L-336 | 2-layer MLP (fresh) | **Vicuna-7B** | 0.87 (드리프트) |
| **M-LMSwap B** (대기중) | CLIP-ViT-L-336 | 2-layer MLP (fresh) | **Mistral-7B-Instruct** | 미학습 |

핵심 비교쌍 — LLaVA-1.5 vs LLaVA-Next: 같은 CLIP-ViT-L-336 을 쓰는데도 PMR
0.18 ↔ 0.79 의 격차. 이게 \"인코더만으로는 결정 안 된다\" 의 가장 깨끗한 증거.
이게 Pillar B (M-LMSwap) 의 출발점.

---

### 슬라이드 6 — Stim 디자인 (자극 디자인)

**한 문장**: \"같은 동그라미를 다른 옷으로 입혀서 모델이 어디서 무너지는지\"
보는 것이 자극 디자인의 핵심.

- **axis A — 추상화 사다리 (왼쪽 추상 ↔ 오른쪽 물리)**:
  - line: 선만으로 그린 원 (가장 추상)
  - filled: 회색으로 채운 원 (약간 물체스러움)
  - shaded: 3D 셰이딩이 들어간 구 (빛-위에서 prior 가 자동 발동)
  - textured: 가죽/점박이 텍스처가 들어간 공 (가장 \"공\")
- **5-axis factorial** (M2): 4 (object_level) × 3 (bg_level: blank/ground/scene)
  × 4 (cue_level: none/cast_shadow/motion_arrow/both) × 3 (event_template:
  fall/horizontal/rise) × 10 (seeds) × 4 (label: circle/ball/planet/_nolabel)
  = **2,880 추론 / 모델**.
- **외부 타당성 확장** (M8 + §4):
  - M8a: 5 도형 (circle/square/triangle/hexagon/polygon) × 4 추상화
  - M8d: 3 카테고리 (car/person/bird) — 비-공 H7 검증
  - M8c: 60 실사진 (COCO + WikiArt)
  - M8e: cross-source consolidation
  - §4.3: 한국어/일본어/중국어 라벨 swap
  - §4.6: 픽셀 공간 counterfactual stim 생성
  - §4.8: Qwen 7B vs 32B scaling

---

### 슬라이드 7 — 측정 지표 (Metrics)

**한 문장**: 응답 텍스트 위에서 자동 채점한 6 개 지표 (사람 검수 ~5% 불일치).
**최근 발견**: binary PMR 의 한계.

| 지표 | 정의 | 어디 쓰는가 | 주의사항 |
|---|---|---|---|
| **PMR** | 응답에 falls/rolls/bounces 등 물리 동사가 들어간 비율 | 메인 지표 | lexicon stem 매칭 (\"continu\" 등 false positive 가능) |
| **PMR_nolabel** | 라벨 없는 open-ended 의 PMR | 모델 자체 편향 | M9 bootstrap CI 의 핵심 입력 |
| **GAR** | 물리 응답 중 \"하방으로\" 답한 비율 | 중력 정합도 / H7 라벨-regime | ball/circle/planet 패턴이 H7 evidence |
| **RC** | T=0.7 N seed PMR call 일관도 | 결정 안정성 | n=5 seed × 5 모델 |
| **H2 paired-Δ** | PMR(label) − PMR(_nolabel) | 라벨 prior 효과 | 3 모델별 부호 패턴 |
| **v_L direction** | hidden_L[physics=1].mean() − hidden_L[physics=0].mean() | VTI steering / SAE / 픽셀 ascent | saturated 모델은 n_neg<5 → 안 잡힘 |

**최근 발견 (M5b round 2, 2026-05-01)** — binary PMR 의 한계:
Idefics2 post-proj k=20 → \"disappear\" (PMR=0), k=40+ → \"continue to expand
outward\" (PMR=1, BUT \"continu\" stem matching 으로 인한 scoring artifact).
Text 는 fall→hit→roll 로 명확히 이동하지만 binary PMR 은 모든 motion verb 를
동등 점수. → paper draft 에 regime-shift score (text-distance) 보조 지표 권장.

---

## Block B — Behavioral results (8-14)

### 슬라이드 8 — M2 PMR 사다리 (5-model)

**한 문장**: 같은 480 자극 위에서 5 모델 PMR_nolabel 이 0.18 → 0.99 로 5.5×
격차. 인코더만으로 설명 안 되는 격차.

각 모델 막대 + 95% Wilson CI:
- LLaVA-1.5: 0.18 [0.14, 0.21] — Floor (가장 낮음)
- LLaVA-Next: 0.79 [0.75, 0.83] — Mid
- Qwen2.5-VL: 0.94 [0.92, 0.96] — Saturated
- Idefics2: 0.97 [0.95, 0.98] — Saturated
- InternVL3: 0.99 [0.98, 1.00] — Super-saturated

왜 인코더 단독 결정자가 아닌가:
1. 같은 CLIP-ViT-L-336 인코더에서 PMR 0.18 ↔ 0.79
2. M3 stim-defined Y 측정 시 5 인코더 모두 AUC = 1.0
3. 즉 인코더 표현 능력은 균일, 차이는 LM 의 \"읽는 방식\"
4. → \"Encoder knows, decoder gates\" (boomerang)

(boomerang 가설: 인코더는 정보를 다 가지고 있는데 LM 이 다르게 \"gate\" 한다 —
\"부메랑처럼 정보가 인코더를 한 번 거쳤다가 LM 단계에서 갈라진다\" 의 비유.)

---

### 슬라이드 9 — H1 ramp (5-model)

**한 문장**: 추상화 (line→textured) 따라 PMR 이 오르는 정도. LLaVA-1.5 만 깨끗한
+0.30 ramp, 나머지는 첫 단계부터 천장에 붙어 측정 헤드룸 없음.

| 모델 | line PMR | textured PMR | Δ (ramp) |
|---|---|---|---|
| LLaVA-1.5 | 0.51 | 0.81 | **+0.30** (가장 깨끗) |
| LLaVA-Next | 0.65 | 0.79 | +0.14 |
| Idefics2 | 0.88 | 0.97 | +0.09 |
| Qwen | 0.89 | 0.94 | +0.05 |
| InternVL3 | 0.97 | 0.99 | +0.02 |

관측:
- Ramp 측정성은 encoder saturation 과 반비례 — saturated 모델은 천장 효과로 H1
  검증 불가.
- M8a (5 도형) 에서도 동일 패턴 — Qwen 3/5 fail (square/triangle 천장), LLaVA 4/5.
- → H1 은 \"unsaturated-only AND shape-axis-only\" 로 정밀화 (가설 카드 수정).

---

### 슬라이드 10 — H2 paired-Δ — 3 architectural patterns

**한 문장**: 라벨 효과의 부호 자체가 architecture-conditional. 3 가지 다른 패턴이
나타난다.

H2 paired-Δ = PMR(label) − PMR(_nolabel). 즉 \"라벨이 있을 때\" 와 \"없을 때\" 의
차이. 양수면 라벨이 PMR 끌어올림, 음수면 깎음.

3 가지 패턴:

- **① Unsaturated CLIP — LLaVA-1.5 / LLaVA-Next**: 모든 라벨 양수 (classical H2).
  - LLaVA-1.5: ball Δ = +0.475, planet +0.244, circle +0.173.
  - 즉 \"ball 라벨 추가하면 PMR 이 47.5 pp 올라간다\".

- **② Saturated SigLIP — Qwen / Idefics2**: ball ≈ 0, planet/circle 음수.
  - Qwen: ball 0.000, planet/circle 음수 = \"circle override\".
  - 즉 \"비-물리 라벨 (circle) 이 PMR 을 baseline 아래로 깎는다\".

- **③ Super-saturated — InternVL3**: 모든 Δ ≈ 0 (천장 효과).
  - 라벨이 더 끌어올릴 헤드룸이 없음.

교훈: H2 는 \"라벨이 항상 PMR 더한다\" 가 아님. 라벨 효과의 **부호 자체** 가
encoder saturation 상태에 따라 결정.

---

### 슬라이드 11 — M9 generalization audit

**한 문장**: 같은 클러스터링이 자극을 바꿔도 보존된다 (paper Table 1 재료).

3 모델 × 3 자극 source × 5000-iter bootstrap CI (Wilson 95%):

| 자극 source | non-CLIP cluster | CLIP-LLaVA-1.5 | 분리? |
|---|---|---|---|
| M8a 5-도형 합성 | [0.80, 0.92] | [0.14, 0.21] | **완전 분리** ✅ |
| M8d 3-카테고리 | [0.84, 0.92] | [0.18, 0.36] | **완전 분리** ✅ |
| M8c 60 실사진 | [0.28, 0.55] | [0.18, 0.42] | **수렴 (overlap)** |

발견:
1. 합성 자극 (M8a/M8d): non-CLIP vs CLIP cluster 가 95% CI 로 완전 분리.
2. 사진 (M8c): 모든 모델 [0.18, 0.67] 로 수렴 — \"풍부한 cue\" 가 saturation 천장
   을 깨뜨림.
3. 사진은 라벨 효과 (H7) 도 절반으로 — synthetic stim 의 minimality 가 라벨-
   regime selection 의 co-factor.
4. **M8d H7 — LLaVA car +0.525 / person +0.138 / bird +0.550** = 본 프로젝트의
   가장 강한 cross-category H7 evidence.

H7 = \"라벨이 PMR 을 더하는 게 아니라 어떤 physics regime 을 선택한다\" 의 가설.
ball → 떨어진다, planet → 궤도, car → 굴러간다, bird → 난다, 등.

---

### 슬라이드 12 — §4.8 Qwen 7B vs 32B + B5.1 today

**한 문장**: 5× 모델 키워도 aggregate PMR 안 움직임 (MechBench-style \"scale doesn't
fix grounding\"). cue=none cell 에서만 scale 이 도움. **오늘 B5.1**: 32B FC
label-free PMR 0.873.

| 지표 | Qwen 7B | Qwen 32B | Δ | 해석 |
|---|---|---|---|---|
| aggregate PMR (open, M2) | 0.931 | 0.926 | −0.005 | 5× scaling 으로 변화 없음 |
| cue=none PMR | 0.797 | 0.711 | **−8.6 pp** | 32B 가 weak-cue 에서 더 잘 abstain |
| abstract_reject rate | 0.002 | 0.065 | **35×** | 32B 가 \"이건 그림이다\" 로 응답 |
| H2 (ball − circle) | +0.071 | +0.010 | halved | 라벨 prior 효과 약화 |
| **FC PMR_nolabel (B5.1 오늘)** | (별도) | **0.873** | — | FC prompt 에서도 32B 천장 유지 |

scale 이 도움이 되는 곳:
- cue=none cell (5% of M2) 에서만 PMR 이 떨어진다.
- 그곳은 **visual prior 가 가장 약한 곳** — scale 이 grounding 을 강화.
- 32B 는 \"abstract_reject\" 응답이 35× 증가 → \"이미지에 정보가 부족하다\" 라고
  명시적으로 언급.
- → scaling 이 visual-prior 약한 곳에서만 grounding 향상.

scale 이 안 도움 되는 곳:
- Aggregate PMR 0.926 ≈ 7B 0.931 — saturation 그대로.
- shaded/textured cue 가 강한 cell 에서는 7B/32B 모두 천장.
- → 단순 scaling 으로는 architectural saturation 미해결.
- MechBench (Zhang+ 2024) 의 \"scale 으로 안 풀림\" 과 일치.

---

### 슬라이드 13 — §4.3 한국어/일본어 라벨 swap

**한 문장**: 5-model × {EN/KO/JA} 라벨로 텍스트 응답 패턴 비교. 4/5 모델 ordering
보존, LLaVA-1.5 가 가장 큰 swing.

| 모델 | EN ordering | KO swing | JA 특이사항 |
|---|---|---|---|
| Qwen2.5-VL | ball > planet > circle | minor (±0.05) | 정상 |
| LLaVA-1.5 | ball >> circle ~ planet | **ball KO 0.16 → 공 0.42 (+26pp!)** | 한국어 SFT 약함의 시그널 |
| LLaVA-Next | ball > circle > planet | minor | 정상 (Mistral) |
| Idefics2 | ball > circle > planet | minor | **JA 惑星 → Chinese fallback 24%** |
| InternVL3 | saturated all | ≈ 0 | InternLM3 강함 |

발견:
1. 라벨 ordering 은 4/5 모델에서 **언어 invariant** — H7 이 multilingual generalize.
2. LLaVA-1.5 KO \"공\" 응답이 EN \"ball\" 보다 26 pp 높은 PMR — Vicuna 의 한국어
   SFT 약점이 visual grounding 약점을 노출.
3. Idefics2 일본어 \"惑星\" → 중국어 응답 24% — Mistral 의 일본어 SFT 가 약해서
   한자를 중국어로 인식. (관측만, 결과 해석은 보수적.)

---

### 슬라이드 14 — M4b / M4c label-free findings

**한 문장**: 라벨 confound 제거하고 다시 측정해서 \"라벨이 정말 PMR 을 끌어올리나?\"
를 검증. 결과: Qwen 에서는 \"ball ≈ no-label / circle suppress\" 패턴 발견.

**M4b — open_no_label (Qwen, M2 stim)**:
- 프롬프트 변경: \"What will the [ball/circle/planet] do?\" → \"What will happen
  next?\" (라벨 0).
- 결과 (Qwen):
  - ball: baseline +0.000 (no-label 과 동일)
  - circle: baseline −0.065 (suppressed)
  - planet: baseline +0.012 (no-label 과 동일)
- 재해석: 원래 H2 (\"ball 라벨이 PMR 끌어올림\") 는 Qwen 에서 정확히 반대 — ball
  은 baseline 과 같고 circle 만 baseline 아래로. 언어 prior 의 asymmetric 영향:
  양수가 아니라 음수 노이즈.

**M4c — forced_choice_no_label (Qwen + LLaVA)**:
- 프롬프트 (FC = forced choice MCQ):
  - Q: 다음 중 무엇이 일어날까요?
  - A) the depicted object will fall down
  - B) ... will stay still
  - C) ... will rise
  - D) abstract / not enough info
- 결과 (Qwen): M4b 패턴 재현 + planet-suppress 추가 (option set bias: 행성 regime
  은 D 로 collapse).
- 결과 (LLaVA-1.5): **\"A\" 만 477/480 반환**. first-token logit-ratio 도 \"A\"-bias
  확인. **Vicuna model-level pathology**, greedy 차원 아님.
  → FC 분석에서 LLaVA family 제외 (M6 r2c).

---

## Block C — Mechanistic Pillar 1: Encoder (15-17)

### 슬라이드 15 — M3 vision encoder probe AUC ladder

**한 문장**: 인코더 hidden state 위에 linear probe 학습해서 \"인코더 안에 physics
정보가 있는가?\" 측정. Stim-Y 모두 AUC 1.0, behavior-Y 가 PMR 사다리 일치.

probe 학습 = 인코더의 hidden state (예: 512 차원 벡터) 위에 단순 선형 분류기를
얹어서 \"이 hidden 이 physics 자극인지 abstract 자극인지 구분 가능한가?\" 측정.
AUC 1.0 = 완벽 분리. 0.5 = 무작위.

| 모델 | behavior-Y AUC | stim-Y AUC | PMR |
|---|---|---|---|
| Qwen SigLIP | 0.99 | 1.0 | 0.94 |
| Idefics2 SigLIP-SO | 0.93 | 1.0 | 0.97 |
| InternVL3 InternViT | 0.89 | 1.0 | 0.99 |
| LLaVA-Next CLIP | 0.81 | 1.0 | 0.79 |
| LLaVA-1.5 CLIP | 0.73 | 1.0 | 0.18 |

두 가지 Y 의 차이:
- **stim-Y**: 자극의 ground-truth label (\"이 stim 은 우리 디자인상 physics-cell
  이다\"). 모든 인코더 AUC = 1.0 → 표현 능력은 균일.
- **behavior-Y**: 모델의 실제 응답에 따른 label (\"이 stim 에 모델이 PMR=1 응답
  했다\"). 이게 PMR 사다리와 일치.

결론 — boomerang:
- Stim-Y AUC 모두 1.0 → 인코더 **표현 능력은 균일**.
- Behavior-Y AUC 가 PMR 사다리와 일치 → 차이는 **LM-side gating**.
- \"인코더가 본 것\" ↔ \"LM 이 읽은 것\" 의 분기 — *encoder knows, decoder gates*.
- 단, **LLaVA-1.5 에서 boomerang 부재** — 그곳은 인코더가 bottleneck (AUC 0.73).

---

### 슬라이드 16 — §4.5 cross-encoder swap (Idefics2 가 Qwen 패턴 재현)

**한 문장**: SigLIP-SO400M + Mistral-7B (Idefics2 vibe) 가 Qwen 패턴을 재현 — 즉
\"encoder family (SigLIP) 가 결정자\" 의 causal evidence.

**실험 설계**:
- 통제: encoder family 만 swap, 나머지 동일.
- 비교 1: Qwen2.5-VL-7B (SigLIP)
- 비교 2: Idefics2-8B (SigLIP-SO400M + perceiver-resampler + Mistral-7B)
- 측정 항목: vision encoder probe AUC, behavioral PMR + H7 패턴, M5a steering, M5b SAE.
- 예상: encoder-saturation hypothesis 가 맞다면 Idefics2 도 PMR ≥ 0.9 + circle override.

**결과 — H-encoder-saturation 통과**:
1. Idefics2 vision probe AUC: **0.93** (Qwen 0.99 와 같은 saturation tier).
2. Idefics2 PMR_nolabel: **0.97** (Qwen 0.94 와 같은 tier).
3. H2 paired-Δ: **circle override 재현** (planet/circle < 0).
4. M5a L25 steering: **10/10 flip** (Qwen L10 의 Idefics2 layer-equivalent).

→ Encoder family (SigLIP) 가 결정자. CLIP 와 SigLIP 사이의 격차가 PMR 사다리의
주요 원인. Causal evidence at the encoder-family level.

---

### 슬라이드 17 — M5b SAE encoder ablation (round 1, 2026-04-28)

**한 문장**: 5120-feature SAE 학습 후 top-k Cohen's d 랭크된 feature 약 30 개
ablate → 3 of 5 모델에서 PMR 1.0 → 0.0 깨짐.

SAE = Sparse Autoencoder. 4096 차원 hidden 을 5120 개의 sparse 한 \"interpretable
feature\" 로 분해. 각 feature 는 hidden 의 작은 \"방향\" 을 의미. 학습 후 \"어떤
feature 가 physics-mode 와 관련 있는지\" Cohen's d 로 랭크.

ablate 의 의미: SAE feature 의 출력을 0 으로 강제 → 그 feature 가 transport 하던
정보가 LM 까지 도달 못함.

| 모델 | Layer | Break at k = | Verdict |
|---|---|---|---|
| Qwen2.5-VL | L31 (last) | **20 (0.4%)** | ★★★ clean |
| Idefics2 | L26 | 160 (3.5%) | ★ break |
| InternVL3 | L23 (-1) | 160 (3.9%) | ★ break |
| LLaVA-1.5 | L22 (-2) | NULL ≤ 800 | ✗ encoder NULL |
| LLaVA-Next | L22 (-2) | NULL ≤ 160 | ✗ encoder NULL |

Round 1 reading (당시):
- 비-CLIP: ~30 SAE feature 로 encoder 안에 \"physics-mode\" 표현 국소화.
- CLIP family: encoder-side 에 없음 → LM-side direction 만 라우팅?

\* 2026-05-01 round 2 가 이 reading 을 reframe (슬라이드 30).

---

## Block D — Mechanistic Pillar 2: LM (18-21)

### 슬라이드 18 — M4 LM logit lens cross-model

**한 문장**: 5-model × 5-layer LM probe AUC 를 측정. **Idefics2 LM AUC 0.995 >
vision AUC 0.93** — perceiver-resampler 가 정보 strip 안 함.

logit lens / per-layer probe = LM 의 각 hidden state layer 에 linear probe 를
얹어서 \"이 layer 가 physics-mode 정보를 얼마나 가지고 있나\" 측정.

| 모델 | Vision AUC | LM AUC | 방향성 |
|---|---|---|---|
| **Idefics2** | **0.93** | **0.995** | **LM > Vision** |
| Qwen2.5-VL | 0.99 | 0.96 | ≈ |
| LLaVA-Next | 0.81 | 0.79 | ≈ |
| LLaVA-1.5 | 0.73 | 0.76 | ≈ |
| InternVL3 | untestable | untestable | (n_neg=1) |

Idefics2 의 의미:
- Vision AUC 0.93 ≤ LM AUC 0.995.
- Perceiver-resampler 가 정보를 오히려 더 잘 보존 (strip ✗).
- 그런데 §4.6 픽셀 ascent 0/9 → **정보 LM 도달 ≠ 픽셀-공간 routability**.
- Forward 통과 OK, inverse pixel→v_L 차단.
- → 슬라이드 27 에서 perceiver hypothesis 정밀화.
- M5a 도 10/10 flip 으로 forward 작동 확인 (슬라이드 21).

---

### 슬라이드 19 — M5a VTI L10 magic (Qwen)

**한 문장**: LM 의 L10 hidden state 에 +α·v_L10 더하면 응답이 뒤집힌다 (이미지 한
픽셀 안 건드리고).

**v_L10 정의**:
- v_L10 = hidden_L10[physics=1].mean() − hidden_L10[physics=0].mean()
- 즉 \"physics 응답들의 hidden 평균\" 과 \"abstract 응답들의 hidden 평균\" 의 차이.
- ≈ \"physics-mode 방향\" 의 추정.

**개입**:
- hidden_L10 ← hidden_L10 + α·v_L10
- α 스케일 0, 5, 10, 20, 40

**자극**: line/blank/none circle (가장 추상한 원, baseline PMR ≈ 0.05).

**결과**:
- α = 0 (baseline): \"This is just a circle on a white background.\" — 추상 (D),
  10/10.
- α = 10 ~ 20: 여전히 추상 (D).
- α = 40: \"It stays still — the circle appears to be floating in space without
  external force.\" → **물리·정지 (B), 10/10 flip!**

다른 layer (L15·L20·L25): 같은 α 에서 변화 없음. → L10 만의 narrow band intervention.

---

### 슬라이드 20 — M5a-ext bidirectional regime axis

**한 문장**: v_L10 은 \"object-ness\" axis 가 아니라 **regime axis**. +α 는 dynamic
physics, −α 는 static physics. baseline D 는 임계값 아래.

| α 부호 / 크기 | 응답 예시 | regime | 텍스트 카테고리 |
|---|---|---|---|
| α = 0 | \"This is just a circle on white background.\" | abstract / undecided | D |
| α = +40 | \"The circle will continue falling downward due to gravity.\" | kinetic physics | A — falls |
| α = +40 (label=ball) | \"The ball will roll along the surface.\" | kinetic physics | A — falls |
| α = −40 | \"The circle remains stationary, suspended in midair.\" | static physics | B — stays |
| α = −40 (label=ball) | \"The ball is floating without external force.\" | static physics | B — stays |

재해석 (H-direction-bidirectional → H-regime-axis):
- 초기 가설: v_L10 이 \"object-ness\" axis (+ → 물체 / − → 추상).
- 수정: v_L10 은 regime axis (+ → kinetic / − → static), baseline D 는 |α| 임계값
  아래의 \"undecided\" 상태.
- → 양쪽 부호 모두 physics-mode 활성화, 부호가 어떤 physics regime 인지 결정.

---

### 슬라이드 21 — M5a cross-model — 3 of 4 testable models flip 10/10

**한 문장**: 각 모델 자체 v_L_per_model + 자체 α dynamic range 로 cross-model 검증.
LLaVA-1.5 만 0/10 (encoder bottleneck).

| 모델 | Layer | α | Baseline cell | 결과 |
|---|---|---|---|---|
| Qwen2.5-VL | L10 | 40 | line/blank/none circle | **10/10 flip** ✅ |
| LLaVA-Next | L20 / L25 | 10 / 15-20 | line/blank/both | **10/10 flip** ✅ |
| Idefics2 | L25 | 20 | line/blank/none | **10/10 flip** ✅ |
| LLaVA-1.5 | L25 (sweep α=0~60) | — | line/blank/none | **0/10** ✗ |
| InternVL3 | — | — | baseline=1.0 | untestable (ceiling) |

Idefics2 결과 텍스트:
- α = 0 (baseline): \"It is unclear what will happen — the image is just an arrow
  and a circle.\" (10/10 abstract)
- α = 20 @ L25: **\"The tip of the arrow will hit the center of the circle.\"**
  (10/10 physics)

LLaVA-1.5 — encoder bottleneck:
- α 0~60 sweep, layer L20/L25 모두 0/10 flip.
- v_L 방향 자체는 잡히지만 (ratio 정확), L 위에서 흔들어도 응답이 안 바뀜.
- → encoder-side 에 인코딩이 너무 약해서 LM 이 흔들리지 않는 cell.
- → §4.6 weak-shortcut 결과와 일치 (L25 only, n=10 에서 4/10).

---

## Block E — Mechanistic Pillar 3: Triangulation (22-24)

### 슬라이드 22 — M5b SIP (Subspace Intervention via Patching) + activation patching

**한 문장**: \"clean\" + \"corrupted\" 두 입력의 hidden state 를 layer 별로 patching
해서 \"어디까지 patching 하면 physics regime 회복되는가\" 측정.

SIP 작동 방식:
1. clean stim (예: shaded ball) → forward → hidden_clean[L0..L31].
2. corrupted stim (예: line circle) → forward → hidden_corr[L0..L31].
3. corrupted forward 중에 hidden_clean 의 L0..LX 까지 \"덮어쓰기\" → 응답 측정.
4. X 가 어디까지 갔을 때 응답이 clean 응답과 일치하는가? = lock-in layer.

| Layer 범위 | Qwen IE (n=20) | LLaVA-1.5 IE (n=15) |
|---|---|---|
| L0 - L9 | **+1.0** (20/20 회복) | (separate run) |
| L10 - L11 | +0.6 (12/20) | — |
| L14+ | 0 (회복 안 됨) | 0 |
| L20 (LLaVA lock) | — | **+1.0 (62.5% depth)** |

IE = Intervention Effect (= 회복률).

Cross-model 비교:
- Qwen: L10 (relative depth 36 %)
- LLaVA-1.5: L20 (relative depth 62.5 %)
- Curve shape 는 동일 (수직 절벽)
- Locus 는 model-specific
- → \"transition layer\" 가 model-specific 이지만 sharp 하다.
- → paper 기여 2 (causal localization) 의 Qwen-only 였던 것이 LLaVA-1.5 까지
  cross-model 확장.

---

### 슬라이드 23 — M5b MLP knockout (L9 가 단독으로 책임)

**한 문장**: MLP knockout 으로 측정 시 L9 만 IE = +1.0, attention knockout 모든
28 layer IE = 0. \"Construction-and-broadcast\" 메커니즘.

knockout = 특정 layer 의 component (MLP 또는 attention) 의 출력을 0 으로 강제.

| 측정 | 결과 |
|---|---|
| Attention knockout (necessity) | 28 layer 모두 IE = 0 — attention is redundant |
| MLP knockout (necessity) | **L9 만 IE = +1.0** (uniquely necessary) |
| L8 / L10 / L11 / L14 MLP knockout | 부분 (0.4 / 0.6 / 0.4 / 0.4) — partial echoes |

**Triangulation Qwen chain**:
1. **Encoder**: ~30 SAE feature 가 physics signal 운반 (M5b round 1).
2. **Visual tokens**: L0-L9 가 그 signal 을 transport (M5b SIP).
3. **L9 MLP**: commitment 를 *construct* (MLP knockout, +1.0).
4. **L10 attention**: *redundantly read out* via attention (per-head 196 cells = 0).
5. → letter B/D 결정.

M5a/M5b off-by-one 조화:
- M5a: L10 α=40 flip (\"L10 이 결정자\").
- M5b: L9 MLP IE=+1.0 (\"L9 가 결정자\").
- 같은 결정 boundary 의 두 측면 — L9 가 construct, L10 이 read out.

---

### 슬라이드 24 — M5b per-head attention knockout — 196 cells = 0

**한 문장**: 20 stim × 7 layer (L8-L14) × 28 head = 196 (L,h) ablation cells 모두
IE = 0. \"narrow IE band\" 가설 (H10) refuted.

기존 가설 (H10): \"2-3 narrow attention IE bands\" (Wang+ 2023 같은 attention head
분석 따라). 즉 \"몇몇 특정 head 가 결정자\".

관측: **196 cells 모두 IE = 0**.
- 어떤 single head 도 load-bearing 아님.
- attention 은 fully redundant at both layer and head 수준.

재해석:
- 기계 작동은 \"pull through specific head\" 가 아님.
- **\"L9 MLP 가 commitment 를 만들고 L10 attention 이 broadcast\"** — 이 메커니즘은
  단일 head 에 의존하지 않음.
- → H10 은 refuted, 새 framing 채택.

---

## Block F — Pixel-space (25-27)

### 슬라이드 25 — §4.6 Qwen pixel counterfactual

**한 문장**: 픽셀 공간에서 v_L10 방향으로 gradient ascent 200 step → 5/5 flip vs
random 0/15. shortcut 의 픽셀 인코드성.

**Setup**:
- 자극 baseline = line/blank/none circle (가장 추상).
- Adam 200 step, ε = 0.1, v_L10 방향 (LM L10 의 physics 방향).

**응답 비교**:
- Baseline (ε = 0): \"The circle will remain stationary as there is no indication
  of movement.\" → 추상 (D).
- v_L10 ascent (ε = 0.05): **\"The circle will continue to fall downward due to
  gravity.\"** → 물리 응답으로 collapse.

통계: **5/5 v_L10 flip vs 0/15 random**. → 단순한 \"아무 노이즈\" 는 안 되고
**방향 특이성** 보장.

shortcut 의 픽셀 인코드성 (paper 기여 3):
- shortcut 은 \"runtime hidden injection\" 만이 아니라 **픽셀 자체에 인코드 가능**.
- 왜 중요한가: 적대적 공격 (adversarial) 의 한 형태 / 신뢰성 / 안전성: 작은 픽셀
  변화가 응답 뒤집음 / Falsification: 매칭 magnitude random 0/15 가 \"any
  perturbation\" 가설을 falsify.

---

### 슬라이드 26 — §4.6 cross-model layer sweep (5-model × 5-layer × n=10)

**한 문장**: 5-model × 5-layer × n=10 = 250 trials. Aggregate random 1/250 = 0.4%
hit. 픽셀 routability 가 architecture-conditional.

| 모델 | Shortcut layers (10/10 flip) | Random control | 특징 |
|---|---|---|---|
| Qwen2.5-VL | **L5/10/15/20/25 모두 ≥ 80%** | 1/10 (L10), rest 0 | broad shortcut |
| LLaVA-Next | **L20 (10/10), L25 (10/10)** | 0/10 모두 | Mid — clean 2 layer |
| LLaVA-1.5 | L25 only (4/10 at n=10) | 0/10 모두 | weak shortcut |
| Idefics2 | **0/9 layers (L5-L31)** | 0/10 모두 | anomaly — 슬라이드 27 |
| InternVL3 | untestable (baseline=1.0) | — | protocol-saturated |

랜덤 컨트롤 — 방향 특이성 검증:
- Aggregate: 5 models × 5 layers × 10 trials = 250 trials, **1/250 hit** (Qwen
  L10 only).
- → 24/25 random-control cells = 0/10. 방향 특이성 보장.
- → v_L direction-specific shortcut 이 architecture-conditional 한 형태로 5
  모델에 분포.

---

### 슬라이드 27 — Idefics2 9-layer disambiguation + perceiver-resampler hypothesis

**한 문장**: Idefics2 만 9 레이어 (L5 ~ L31) 모두 0/10 flip. 그러나 v_L projection /
LM probe AUC / M5a steering 은 모두 정상 — 무엇이 routing 을 차단하는가?

Idefics2 의 5 가지 측정:

| 측정 | 값 | 해석 |
|---|---|---|
| §4.6 픽셀 ascent (9 layers L5-L31) | 0/10 모두 | 픽셀-공간 routability 차단 |
| v_L projection (L26-L30 ascent) | −11 → +28 정상 상승 | 방향성 자체는 잡힘 |
| v_L projection (L31) | −72 → +163 정상 | 절대값 더 큰 ascent 도 작동 |
| M4 LM probe AUC | 0.995 | 정보가 LM 까지 도달 |
| M5a runtime steering (L25 α=20) | 10/10 flip ✅ | Forward-side hidden injection 작동 |

Perceiver-resampler hypothesis:
- Forward (LM 이 hidden 받기): **통과 OK**.
- Inverse (픽셀 → v_L gradient): **차단**.
- → Perceiver-resampler 가 정보를 forward 로는 잘 보내지만, inverse pixel-space
  gradient routability 만 끊는다.

주의:
- Idefics2 는 encoder + projector + AnyRes 가 동시에 다름.
- Perceiver 단독 isolation 미검증 (controlled projector swap = M-PSwap, NaN 미해결).

---

## Block G — M5b round 2 + B5 today (28-31)

### 슬라이드 28 — M5b round 2 methodology

**한 문장**: Round 1 은 vision-encoder hidden 위, round 2 는 projector output 위.
Round 1 \"NULL\" 은 ball cell 만 테스트한 결과 — circle cell 도 테스트하면 다른
양상이 나올 수 있다는 가설.

Round 1 결과: LLaVA family 가 encoder-side NULL → \"LM-side direction 으로만
라우팅\" 으로 해석.

Round 1 의 한계: ball cell (filled+blank+both, shaded+blank+none) 만 테스트.
Round 2 의 가설: projector output 에 commitment 가 다른 형태로 인코드되어 있을
수 있다. → circle cell 도 테스트 — 라벨이 abstract 인 경우 다른 양상.

| 모델 | Round 1 hook (encoder) | Round 2 hook (post-proj) | shape |
|---|---|---|---|
| Qwen2.5-VL | vision_hidden_31 | model.model.visual.merger | (n_groups, 3584) |
| LLaVA-1.5/Next/LMSwap | vision L22 (-2) | model.multi_modal_projector | (576, 4096) |
| Idefics2 | vision L26 | model.connector (perceiver+MLP) | (64, 4096) |
| InternVL3 | vision L23 (-1) | model.multi_modal_projector | (256, 3072) |

---

### 슬라이드 29 — Round 2 결과 — regime-cross capacity ladder (5 × 4)

**한 문장**: discriminating cell 은 **circle / filled / blank+both**. ball cell 은
모두 PMR=1 유지 (label prior 강함). 모델별 ladder 가 명확히 갈린다.

| 모델 | circle/filled/blank+both | circle/shaded/blank+none | ball/filled/blank+both | verdict |
|---|---|---|---|---|
| **Qwen2.5-VL** | k=20 → \"remain stationary\" PMR 1→0 ✅ | stays kinetic | ball stays PMR=1 | **★★★ 깨끗** |
| **Idefics2** | k=20 → \"disappear\" PMR=0 / k=40+ → \"continue to expand\" PMR=1* | stays \"spin\" | ball stays PMR=1 | **★★ 부분** |
| **LLaVA-Next** | k=40+ → \"expand\" PMR=0 | no break tested | ball stays PMR=1 | **★★ 부분** |
| **LLaVA-1.5** | baseline PMR=0 (\"drawn towards red arrow\") | no break tested | ball stays PMR=1 | **✦ baseline-abstract** |
| **InternVL3** | stays \"fall downwards\" NULL all k ≤ 160 | drift only | ball stays PMR=1 | **✗ true NULL** |
| **Qwen 32B (B5 오늘)** | k=40 → \"remain stationary\" PMR=0 | (no test) | (no test) | **★★★ 깨끗** |

\* k=40+ Idefics2 PMR=1 은 \"continu\" stem matching 의 scoring artifact (슬라이드
30 + `m5b_idefics2_non_monotonic.md` 참고).

핵심: ladder 는 (1) circle vs ball 라벨 의존, (2) Qwen > Idefics2 ~ Next > LLaVA-1.5
> InternVL3 — 5-fold 시그니처와 일치.

---

### 슬라이드 30 — \"NULL\" 의 3 가지 분해

**한 문장**: Round 1 \"NULL\" 헤드라인이 3 가지 distinct phenomena 로 분해.

**(a) Genuine NULL** — InternVL3 (모든 cell + 모든 k ≤ 160):
- Text 자체가 안 움직이고 binary PMR 도 PMR=1 유지.
- encoder + projector + LM 어디에도 ablate 가능한 commitment 없음.
- Super-saturated 의 진짜 NULL.

**(b) Baseline-already-abstract** — LLaVA-1.5 + circle cell:
- Baseline 응답이 \"drawn towards red arrow\" 로 PMR=0.
- 즉 모델이 이미 abstract regime 에 있어서 ablate 할 physics commitment 가
  존재하지 않음.
- \"NULL\" 가 아니라 \"이미 깨진 상태\". CLIP+Vicuna 의 default mode for circle.

**(c) Binary-PMR conceals real shifts** — LLaVA-1.5 + ball cell:
- Text 는 fall → hit by arrow → roll → redrawn 으로 명확히 이동.
- 모든 응답이 motion verb 라 binary PMR 은 1 로 고정.
- 실제 regime 은 흔들리지만 측정 도구가 catch 못함.

→ paper draft: \"NULL → regime-cross capacity ladder\" reframe + regime-shift
score (text-distance) 보조 지표 권장.

---

### 슬라이드 31 — B5 today: Qwen 32B post-proj SAE (오늘 결과)

**한 문장**: Qwen 32B 도 post-projection SAE intervention 에서 같은 ★★★ tier.
k 임계값만 약간 상승 (7B k=20 → 32B k=40). \"scaling 이 mechanism 을 변경하지
않는다\" 의 두 번째 증거.

| k_zeroed | intervention text | PMR | 해석 |
|---|---|---|---|
| baseline | \"The circle will move downward and land on the smaller gray shape below it.\" | 1 | kinetic |
| top_k = 20 | \"The circle will continue moving downward along the path indicated by the arrow.\" | **1** | still kinetic — 7B 와 다름 |
| top_k = 40 | \"The circle will remain stationary as there is no indication of movement or change.\" | **0** | abstract regime ✅ |
| top_k = 80 | \"The circle will remain stationary...\" | 0 | stable abstract |
| top_k = 160 | \"The circle will remain stationary...\" | 0 | stable abstract |
| random_0/1/2 | \"... move downward and collide with surface below.\" | 1 | specificity ✓ |

5-fold 시그니처에 32B 추가:
- 7B 와 32B 가 같은 ★★★ tier — \"scale 이 mechanism 을 바꾸지 않는다\" 의 두 번째
  증거 (§4.8 행동 지표에 더해 mechanism 도 보존).
- 단, 32B 가 k=40 까지 buffer 가 더 두꺼움 — feature 가 더 많이 필요. 5120-feature
  SAE 위에서 0.4% (k=20) 가 아니라 0.8% (k=40).

---

## Block H — Pillar B (32-36)

### 슬라이드 32 — Pillar B motivation (왜 controlled LM swap 이 필요한가)

**한 문장**: LLaVA-1.5 (0.18) → LLaVA-Next (0.79) PMR 점프는 \"LM 이 결정자\" 라고
결론 내릴 만큼 깨끗한 비교가 아니다 — 4 개 변인이 동시에 다르다.

LLaVA-1.5 ↔ LLaVA-Next 의 4 축 confound:
1. **LM 백본**: Vicuna-7B → Mistral-7B-Instruct.
2. **Vision token 처리**: 단일-tile 576 → AnyRes (~4×4 grid).
3. **SFT 데이터셋**: LLaVA-Instruct-150K → LLaVA-Next 760K.
4. **Vision-language 정렬**: 단일 stage → 2-stage refresh.
- → \"인코더만 같다\" 한 가지로는 LM 이 결정자라 결론 불가.

**M-LMSwap — 단일축 controlled swap**:
- 공통 component:
  - CLIP-ViT-L-336 (LLaVA-1.5 와 동일)
  - 2-layer MLP projector (랜덤 초기화)
  - LoRA on q/k/v/o_proj (r=32, α=64)
  - LCS-558K + LLaVA-Instruct-665K
- Variant A: + Vicuna-7B-v1.5
- Variant B: + Mistral-7B-Instruct-v0.2
- → LM identity 만 swap, 나머지 모두 통제.
- → A↔B Δ-PMR 이 LM family 효과의 직접 측정.

LoRA = Low-Rank Adaptation. 전체 LM 파라미터를 미세조정 안 하고 작은 rank-32
matrix 만 학습 (q/k/v/o_proj 에 추가). 학습 비용 이 1% 수준이지만 효과 비슷.

---

### 슬라이드 33 — M-LMSwap training pipeline (2-stage canonical recipe)

**한 문장**: Stage 1 (~17K, ~12h) projector pretrain on LCS-558K. Stage 2 (~21K,
~12h) LoRA tune on LLaVA-Instruct-665K.

학습 타임라인 4 단계:
1. Stage 1 — projector pretrain (LCS-558K, 17K step, MLP only, ~12h H200).
2. Stage 2 — instruction tune (LLaVA-Instruct-665K, 21K step, LoRA + MLP unfrozen, ~12h).
3. Final ckpt (step21000) — MLP weight + LoRA adapter (merged on load).
4. Regression eval (3-gate) — PMR_nolabel + baseline cell + generation sanity.

Recipe details (LLaVA-1.5 모방):
- Stage 1: LR 1e-3, batch 32 (effective), Train MLP only (LM frozen), LCS-558K
  caption 데이터, 17K step ≈ 1 epoch.
- Stage 2: LR 2e-4 (LoRA), batch 32, LoRA on q/k/v/o_proj (r=32, α=64), MLP
  unfrozen after PEFT wrap, LLaVA-Instruct-665K (Mix665k), 21K step ≈ 1 epoch.

학습 결과 (Variant A):
- Stage 1 50-step smoke: loss 8.46 → 3.29 ✅
- Stage 1 17K step: MLP grad-norm 안정 (NaN 없음)
- Stage 2 dry-run: 256 LoRA + 4 MLP grads, no leakage ✅
- Stage 2 21K step: 최종 ckpt step21000 저장 ✅

Bug fix mid-run:
- get_mlp_ref 가 hasattr(\"base_model\") 사용 — HF PreTrainedModel 에서는 항상
  True 라 잘못된 path.
- → isinstance(model, PeftModel) 로 수정.

다음 단계: regression eval (슬라이드 34).

---

### 슬라이드 34 — Variant A regression: gate split

**한 문장**: step21000 의 regression eval 결과 — Aggregate PMR FAIL (recipe drift),
그러나 line/blank/none baseline = 0.000 PASS — cell-discrimination 학습됨.

| Gate | step9000 | step21000 | 변화 |
|---|---|---|---|
| Gate 1 — generation sanity (>5 단어, no degeneracy) | PASS | PASS | — |
| Gate 2 — PMR_nolabel ∈ [0.03, 0.50] | FAIL (0.825) | FAIL (0.869) | 약간 더 높아짐 |
| Gate 3 — line/blank/none baseline ≤ 0.6 | FAIL (1.000) | **PASS (0.000)** ✅ | **−1.000 pp! discrimination 학습** |

step21000 line/blank/none 응답 예시 — 명백한 abstract regime:

> \"A circle is drawn on a white background. It is not clear what will happen next.
> It could be a new circle drawn, or it could be a different shape.\"

해석 — A↔B 비교 의미가 회복:
- step21000 은 cell-discrimination 을 학습 — 가장 추상적 cell 에서는 abstract
  response, cue 강한 cell 에서는 kinetic.
- Per-cell ordering 이 LLaVA-1.5 와 구조적으로 일치 (line ≈ 0, filled high, shaded
  high). Aggregate 만 +0.4 shifted.
- → A↔B Δ-PMR 비교가 per-cell 수준에서 의미 있다. **A1 결정 (Variant B 진행) 권장**.

---

### 슬라이드 35 — Recipe drift hypothesis (왜 0.4 shifted up 했는가)

**한 문장**: LLaVA-1.5 의 published recipe 를 모방했지만 PMR 천장이 0.4 더 높이.
3 가지 후보 가설.

후보 1 — Fresh MLP 랜덤 init (severity high):
- LLaVA-1.5 는 특정 init seed + 정밀 LR 사용.
- 우리 implementation 은 fresh random init + 자체 LR (1e-3 stage 1).
- 같은 LCS-558K 위에서도 다른 minima 로 수렴 가능.

후보 2 — LoRA-only Stage 2 vs full LM tune (severity high):
- LLaVA-1.5 Stage 2 는 전체 LM 을 tune.
- 우리는 LoRA on q/k/v/o_proj 만.
- LoRA 는 보수적 — Vicuna 의 강한 physics-language prior 를 충분히 억제 못함
  → PMR 천장 high.

후보 3 — Chat template / image token 처리 차이 (severity medium):
- LLaVA-1.5 는 Vicuna chat template 그대로 + literal `<image>`.
- 우리는 `processor.apply_chat_template` (inference) + manual Vicuna template
  (training).
- 이 미묘한 차이가 vision-language 정렬에 작은 shift 를 만들 수 있음.

A1/A2 결정에 활용 — 두 옵션 모두 한 번에 테스트 못함. step21000 의 cell-
discrimination 학습 결과로 A1 (recipe 그대로 + Variant B) 진행이 더 합리적.
A2 (recipe 재조정 후 재학습) 는 gate fail 이 paper-blocker 가 될 때만 trigger.

---

### 슬라이드 36 — A1/A2/A3 decision matrix

**한 문장**: 3 가지 옵션 — A1 (B 진행 + Δ 비교) / A2 (A 재학습) / A3 (M-PSwap 부활).
**현재 권장: A1 ★**.

**A1 ★ 추천** — Variant B 진행 (gate override). GPU 24h.
- PRO: A↔B per-cell Δ-PMR 비교 가능 (axis ordering 일치).
- PRO: step21000 baseline=0 으로 A1 risk 감소.
- CON: A baseline aggregate +0.4 shifted (Δ 비교만 valid).

**A2** — Variant A recipe 재학습. GPU 24h, 확률적.
- PRO: LLaVA-1.5 floor 와 정렬 가능 → aggregate 비교도.
- CON: 또 실패할 수 있음 (slide 35 1번/2번 hypothesis 가 맞으면 LR/data 조정만
  으로 부족).
- CON: 시간 손실.

**A3** — M-PSwap (perceiver swap) 부활. GPU 24h+, NaN 미해결.
- PRO: Idefics2 perceiver-resampler 가설 직접 검증 (G3 fix).
- CON: 학습 안정화가 우선 미해결 (NaN at step 1000).
- CON: backlogged 상태.

---

## Block I — Failures (37-40)

### 슬라이드 37 — M-PSwap NaN failure (Pillar B 의 원래 우선순위 → backlog)

**한 문장**: Idefics2 perceiver-resampler ↔ MLP swap 으로 §4.6 Idefics2 anomaly
직접 검증 목적이었으나 학습 NaN 미해결.

| 단계 | 결과 | 특이사항 |
|---|---|---|
| Feasibility spike (bypass-only) | **FAIL** | perceiver 가 forward pass 에 integral — bypass 로는 안 됨 |
| Infra 구축 (LoRA + MLPPoolResampler fp32) | ✅ | src/physical_mode/lora/{idefics2_mlp_resampler.py, load_swapped.py} |
| 50-step smoke | PASS | loss 정상 하강, NaN 없음 |
| Full training step 0 → 1000 | PASS | 안정적 |
| Full training step 1000 NaN | **FAIL** | NaN-abort logic 작동, run aborted at outputs/mpswap_run_20260429-033238/step1000 |
| Diagnostic suite (NaN reproducer) | WIP | long-text / streaming / bf16 / mask 패턴 stress test — 미재현 |
| D0a bf16 forward-only repro (M-LMSwap day-0) | FAIL to reproduce | step 1465 까지 clean — single-bad-batch 가설 약화 |

현재 status — backlogged:
- 결정 (2026-04-29): submission_plan.md §6 Pillar-B drop rule 을 조기 적용
  (week 8 → week 4). 인프라 보존, NaN 진단 일시 중단, M-LMSwap 으로 우선순위 이전.
- 교훈: perceiver-resampler 가 forward pass 에 integral 한 architecture 는
  swap-style controlled experiment 가 더 어렵다. 다음 시도 시 **discriminator 기반
  NaN repro batch** 가 필요.

---

### 슬라이드 38 — M-MP Qwen × MCQ split (cross-method dissociation 발견)

**한 문장**: M5a (steering) 와 M5b (SAE) 가 MCQ prompt 에서 갈린다 — \"Generative-
vs-Categorical\" framing 분해.

audit follow-up 으로 발견한 cross-method dissociation:

| 프롬프트 | M5a (steering α=40) | M5b (SAE k=20) | 패턴 |
|---|---|---|---|
| open (existing baseline) | 10/10 flip ✅ | 20/20 break ✅ | positive/positive |
| describe_scene (generative) | 10/10 flip ✅ | 20/20 break ✅ | positive/positive |
| meta_phys_yesno (categorical) | 0/10 ✗ | 0/20 ✗ | null/null |
| **meta_phys_mcq (audit follow-up)** | 0/10 ✗ | **10/10 break** ✅ | **split! null/positive** |

Reframe — \"Generative-vs-Categorical\" framing 이 부족:
- 초기 framing: \"M5a/M5b 는 generative prompt 에서만 작동, categorical 에서는
  NULL\". MCQ 가 framing 분해.
- 관측: MCQ 는 categorical task 인데 M5b 가 break. M5a 는 NULL. → \"task type\" 이
  아니라 **intervention method × prompt format** 의 interaction.
- 현재 입장 (audit-tightened): \"yes/no 의 M5b immunity 는 yes/no-prompt-specific
  (n=1 categorical-binary)\". 더 많은 categorical-binary prompt 로 axis 확정 필요.

---

### 슬라이드 39 — M-MP Idefics2 single-cell + audit caveat

**한 문장**: 단일 cell 에서 30/30 framing-shift (kinetic→suspended) — 하지만 audit
결과 \"1 cell × 10 stim × 3 k\" 로 reframe.

원래 발견 (2026-04-28 morning):
- Cell: shaded/ground/both (ball)
- Top-k SAE ablation:
  - baseline: \"The ball is falling.\"
  - k=160: \"The ball is in the air.\"
  - k=320: \"The ball is in the air.\"
  - k=500: \"The ball is in the air.\"
- 관측: framing shift kinetic → suspended.
  - 10 stim 모두 동일 intervention text
  - Random 10/10 retains kinetic
  - → SAE features encode kinetic-verb production specifically (specificity ✓)
- 초기 framing: \"30/30 stim across 3 k values\" (advisor 지적 후 reframe).

Audit caveat (2026-04-28 review):
- Reframe: \"30 independent stim\" → \"1 cell × 10 stim × 3 k replicates\".
- Cell 1 audit caveats:
  - 1 unique intervention text 만 (\"in the air\") → diversity 부족
  - verbosity-preference confound 가능
  - 2x2 (task × format) 에서 1 cell 만 채워짐
- Cell 2 test (audit follow-up): textured/ground/cast_shadow ball cell.
  - baseline 자체가 \"ball is in the air\" (suspended).
  - top-k ablation no-op → specificity 재확인.
  - 하지만 cell 1 의 architecture-level 격상은 보류.
- → 3rd cell with kinetic baseline 필요.

---

### 슬라이드 40 — Infra learnings — zombie polling + recipe drift

**한 문장**: 두 인프라 사건이 paper 수준 deliverable 에 직접 영향. 두 사건 모두
정확한 root cause + fix 적용으로 recover.

**Event 1: Phase 2 zombie polling 5h hang (2026-04-30)**:
- Setup: chain_post_proj_phase2.sh 가 `kill -0 $PID` 로 prior chain 의 종료를 대기.
- 버그: `kill -0` 가 Z-state defunct (zombie) 프로세스에서도 success. prior chain
  이 zombie 로 남으면 polling 무한 대기.
- 결과: **5 시간 GPU 0 idle** — Phase 2 작업 시작 안 됨.
- Fix: `[ -e /proc/$PID ] && ! grep -q \"^State.*Z\" /proc/$PID/status` 로 실제
  alive 체크.
- 교훈: long polling chain 에서 zombie 가 **silent failure mode** — race condition
  처럼 \"평소엔 안 보이는 버그\".

**Event 2: Variant A regression gate FAIL — recipe drift 발견 (2026-05-01)**:
- Setup: Variant A 학습 21K step 완료 후 m_lmswap_regression_eval.py 의 [0.03,
  0.50] gate.
- 관측: PMR_nolabel = 0.825 (step9000) / 0.869 (step21000) — gate 2 FAIL.
  Aggregate 0.4 too high.
- 분석: 그러나 (1) 480 stim 중 201 unique (image-blind 아님), (2) line→filled→
  shaded ramp 가 LLaVA-1.5 와 동일, (3) **step21000 line/blank/none 에서 PMR=0.000
  (cell-discrimination 학습)**.
- 재해석: Recipe drift — 우리 implementation 이 LLaVA-1.5 보다 더 physics-leaning
  으로 수렴. A↔B 통제는 양쪽에 동일 drift 적용 → 비교 의미 보존.
- 교훈: single gate (aggregate PMR) 만 보면 \"실패\" 처럼 보이지만, **per-cell
  pattern 을 봐야 진짜 status 가 보임**.

---

## Block J — Synthesis + 결론 (41-45)

### 슬라이드 41 — 5-fold downstream signature (+1 from B5)

**한 문장**: 5(+1) 개의 별도 측정 방식이 모두 같은 architectural clustering. paper
의 strongest claim.

| # | 지표 | non-CLIP cluster | CLIP cluster | 스토리 |
|---|---|---|---|---|
| 1 | PMR ceiling (M2 _nolabel) | [0.84, 0.92] | [0.14, 0.37] | 행동 1 — 5.5× separation |
| 2 | Decision-stability (RC) | ≥ 0.95 | ≤ 0.7 | 행동 2 — n=5 seed 안정성 |
| 3 | 픽셀 encodability (§4.6) | Qwen broad | weak (LLaVA-1.5 4/10) | 메커니즘 1 — pixel routability |
| 4 | LM logit-lens AUC (M4) | Qwen 0.96 / Idefics2 0.995 | 0.76-0.79 | 메커니즘 2 — LM probe |
| 5 | Encoder SAE break (M5b r1) | 3 of 3 break | 2 of 2 NULL on ball | 메커니즘 3 — feature ablation |
| **5+** | **Post-proj regime-cross (M5b r2)** | Qwen ★★★ + 32B + Idefics2 ★★ | LLaVA-1.5 ✦ baseline-abstract / LLaVA-Next ★★ partial | 메커니즘 4 — **오늘 추가 (B5)** |

Strongest claim: 5(+1) 개의 별도 측정이 모두 같은 architectural clustering 을
가리킨다 → 단일 architectural property 의 redundant manifestation. 단순 \"인코더
capacity\" 가설로는 설명 불가.

---

### 슬라이드 42 — Encoder ↔ LM dissociation map

**한 문장**: 5 모델 × 측정 표 — LM-side flip 가능 여부 + Encoder-side commitment
존재 여부의 2 × 2 분류.

| 모델 | M3 vision AUC (encoder) | M4 LM AUC | M5a steering (LM-side flip) | M5b encoder SAE | 분류 |
|---|---|---|---|---|---|
| Qwen2.5-VL | 0.99 | 0.96 | ✅ L10 α=40 | ✅ k=20 break | encoder + LM 둘 다 |
| Idefics2 | 0.93 | 0.995 | ✅ L25 α=20 | ✅ k=160 break | encoder + LM 둘 다 |
| InternVL3 | 0.89 | untestable | untestable (sat) | ✅ k=160 break | encoder + saturated |
| LLaVA-Next | 0.81 | 0.79 | ✅ L20+L25 | ✗ encoder NULL | LM only |
| LLaVA-1.5 | 0.73 | 0.76 | ✗ L25 α=0~60 | ✗ encoder NULL | neither (encoder bottleneck) |

구조:
- **Non-CLIP family (Qwen / Idefics2 / InternVL3)**: encoder 안에 ~30 SAE feature
  로 commitment 국소화. LM-side direction 도 작동.
- **LLaVA-Next (CLIP + Mistral + AnyRes)**: encoder NULL but LM 작동 — physics
  commitment 가 LM 안에서만 라우팅.
- **LLaVA-1.5 (CLIP + Vicuna)**: encoder NULL + LM 도 안 흔들림 — encoder 가 진짜
  bottleneck (행동 PMR 0.18 의 직접 원인).

---

### 슬라이드 43 — Limitations + paper plan

**한 문장**: 4 개 paper gap (G1-G4) 매핑 + 5 개 한계. Track B (ICLR 2027).

5 한계:
1. **Single-task evaluation (G1)**: next-state-prediction 만 검증. Counting /
   spatial / causality 등 미검증. M-MP Pillar A 가 부분 fix.
2. **Sparse non-Qwen (G2)**: 4 비-Qwen 모델만 (LLaVA-1.5/Next/Idefics2/InternVL3).
   Pixtral / Phi-3.5-V / GPT-4V 미테스트. → B4 (오늘).
3. **n=1 perceiver isolation (G3)**: Idefics2 가 유일 perceiver. controlled swap
   미검증 (M-PSwap NaN). → B2 fallback (literature-grounded).
4. **5-fold framing 명확성 (G4)**: paper §6 Marr 3-level 재구성 필요. → M-Marr
   (Pillar C).
5. **Human baseline 미수집**: Prolific 20 raters × 50 stim. paper-blocking 단계 (M7).

Track B (ICLR 2027) timeline:
- 선택 venue (2026-04-28 결정):
  - ICLR 2027 primary (deadline ~late Sep 2026)
  - NeurIPS 2027 secondary (~mid May 2027)
  - TMLR rolling fallback
- Framing: Production-VLM \"world-model commitment\" 3 Marr levels: Computational
  (PMR) → Representational (M3+M4) → Mechanistic (M5a+M5b). Connect: V-JEPA / RT-2
  / OpenVLA in §1+§9.
- Drop / defer rules:
  - week 8 Pillar B 결과 없으면 → M-PSwap drop, B2 lean.
  - week 14 still missing → ICLR 2027 → NeurIPS 2027 (+7 month).
  - Robust-not-flashy stance (2026-04-28).

---

### 슬라이드 44 — Next priorities (B4 / B1 / A1/A2)

**한 문장**: 다음 priorities — B4 (Pixtral) → B1 (multi-cell n) → A1/A2 결정.

**B4** — Pixtral chat template fix + M-Add6 6th model. ~30 min dev + 5 min infer.
- FB3 시도 시 jinja TypeError 발견. 수정 후 m_add6_pixtral_m8a inference + score.
  G2 (sparse non-Qwen) 의 6th data point.

**B1** — Multi-cell aggregation n=40 intervention (5-model). ~3-4 h GPU.
- filled+blank+{none, cast_shadow, motion_arrow, both} = 4 cells × 10 stim = 40
  stim per model. Regime-cross capacity ladder 의 statistical strength 강화.

**A1 vs A2** — Pillar B 결정 (Variant B 진행 vs A 재학습). User decision.
- step21000 baseline=0 PASS 가 A1 risk 줄임 (slide 36). Per-cell Δ-PMR 비교가
  의미 있는 측정.  A1 추천.

**C2-C7** — Doc consolidation 완료 ✅
- ✅ insight m5b_post_projection_cross_model.md / m5b_idefics2_non_monotonic.md /
  lmswap_a_recipe_drift.md
- ✅ hypotheses.md H-regime-cross 추가
- ✅ paper_gaps.md / roadmap.md / CHANGELOG.md 업데이트

---

### 슬라이드 45 — 결론 + Q&A

**한 문장**: 행동 → 메커니즘 → 픽셀의 3 차원으로 \"shortcut 의 위치\"를 좁혀
들어갔다. 추가로 Pillar B 통제 실험으로 \"LM identity 의 인과적 역할\" 이 직접
답해질 단계 직전.

4 conclusions:

1. **Architecture-level reframe** — 행동 PMR ceiling 은 인코더 표현력 단독으로
   결정 안 됨. 5-fold 시그니처가 동일 architectural property 의 redundant
   manifestation. CLIP 2-point 비교 (LLaVA-1.5 vs Next) 가 가장 깨끗한 disconfirmer.

2. **Causal localization (LM + Encoder)** — M5a steering: 3 of 4 모델 10/10 flip
   (Qwen L10, LLaVA-Next L20-25, Idefics2 L25). M5b SIP+MLP knockout: L9 MLP 가
   sufficient + necessary. M5b SAE: 비-CLIP 3/3 break, CLIP family encoder NULL on
   ball. → encoder ↔ LM dissociation map.

3. **Pixel-encodability + post-proj ladder** — §4.6 5-model: Qwen broad / LLaVA-
   Next L20+L25 / LLaVA-1.5 L25 only / Idefics2 0/9 (perceiver candidate) / InternVL3
   saturated. M5b round 2: regime-cross capacity ladder + 3 NULL phenomena 분해.

4. **Pillar B 진행 + 다음 단계** — M-LMSwap Variant A 학습 완료 (step21000 cell-
   discrimination 학습), Variant B 대기. M-PSwap NaN 으로 backlogged. B4 Pixtral +
   B1 multi-cell + M-Marr / M7 paper-blocker 단계.

---

## 부록: 발표 시간 가이드 (45 슬라이드 기준)

| 단락 | 슬라이드 | 권장 시간 |
|---|---|---|
| Setup (hook + journey + question + method) | 1–7 | 6 분 |
| Behavioral results | 8–14 | 9 분 |
| Encoder mechanism | 15–17 | 4 분 |
| LM mechanism | 18–21 | 6 분 |
| Triangulation (M5b) | 22–24 | 5 분 |
| Pixel + Idefics2 mystery | 25–27 | 5 분 |
| M5b round 2 + B5 today | 28–31 | 6 분 |
| Pillar B journey | 32–36 | 7 분 |
| Failures (M-PSwap, M-MP, infra) | 37–40 | 6 분 |
| Synthesis + 결론 + Q&A | 41–45 | 6 분 |
| **합계** |  | **~60 분** |

질의응답 포함 90 분 슬롯 가정. 시간 짧으면 38-39 (audit caveats), 13 (multilingual)
먼저 떨어뜨림. 16 (cross-encoder swap), 23 (MLP knockout), 27 (perceiver), 30 (3
NULL phenomena), 36 (decision matrix) 는 어떤 시간 제약에서도 살려두기 권장.

각 슬라이드의 raw data + reproducibility 는 `docs/insights/m{N}_*.md`,
`docs/CHANGELOG.md`, `references/roadmap.md` 참조.
