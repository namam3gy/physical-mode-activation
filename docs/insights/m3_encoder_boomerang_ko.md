# M3 (ST2) Insights — Encoder-Decoder Boomerang

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H4** — Open-ended vs. forced-choice PMR 간격은 language-prior ↔ visual-evidence 충돌의 안정적 signature.
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M3** — ST2 vision-encoder probing — encoder AUC ≈ 1.0 으로 factorial 축 자명 분리 ("boomerang").
> - **M4** — ST3 LM logit lens / per-layer probes — LM AUC 가 시각-토큰 위치에서 L5 부터 ~0.95 plateau.
> - **M5** — ST4 인과 localization (VTI steering / activation patching / SAE) — M5a, M5a-ext, M5b 참조.
> - **M6** — ST5 cross-model sweep — M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next) 참조.


**M2 자극에 적용된 encoder probe** — 아래 line/blank/none baseline. Vision encoder 는 factorial 셀을 AUC ≈ 1.0 으로 선형 분리; 같은 자극의 LM 행동 PMR 은 0.28–0.95 변동:

![M3 참조 자극: line / blank / none](../figures/01_line_blank_none.png)

Sub-task 2 (vision encoder probing) 결과의 심층 분석. Pilot 이 행동 수준의
gap(open vs forced-choice 등)을 보여줬다면, M3 는 그 gap 이 **어디서
발생하는지** 를 mechanism 수준에서 답한다.

원본 수치: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_vision/*.csv`
(생성 스크립트: `scripts/04_capture_vision.py`, `src/physical_mode/probing/vision.py`).

## 1. 한 문장 요약

**Qwen2.5-VL-7B 의 vision encoder 는 PhysCue factorial 의 모든 축을
완벽히 (AUC=1.00) 인코딩하지만, LM 의 행동은 그 정보를 부분적으로만
사용한다.** 즉 "encoder knows, decoder gates".

## 2. 무엇을 측정했나

### 2.1 Setup

- 대상: Qwen2.5-VL-7B 의 vision tower (`model.model.visual`, 32 블록).
- 자극: M2의 480개 stimuli (4 obj × 3 bg × 4 cue × 10 seeds × 1 event).
- Capture: `scripts/04_capture_vision.py` 로 forward-only 통과시켜
  `visual.blocks[3, 7, 11, 15, 19, 23, 27, 31]` 의 출력 저장.
  각 layer 의 출력 shape = `(1296, 1280)` (patch token 1296개, SigLIP dim 1280), bf16.
  총 12 GB.
- Probe: `sklearn.LogisticRegression`, 5-fold stratified CV, token-mean pool,
  StandardScaler 전처리. Implementation: `src/physical_mode/probing/vision.py`.

### 2.2 세 종류의 probe

1. **행동 PMR probe**: y = forced-choice PMR (각 stimulus 의 다수결 binary).
   "Encoder 가 어느 stimuli가 LM을 physics-mode 로 flip 시킬지 예측할 수
   있는가?"

2. **자극 속성 probe**: y = 자극 자체의 객관적 속성 (ground 유무, 3D 여부, cue 여부 등).
   "Encoder 가 stimulus 구성 요소를 알고 있는가?"

3. **Controlled subset probe**: cue=none 서브셋 120개에 대해 (1) 을 재실행.
   Red-arrow shortcut 이 제거된 조건에서도 encoder 가 behavior 를 예측하는가?

## 3. 핵심 수치

### 3.1 자극 속성은 layer 3 부터 완벽 인코딩

| target | L3 | L15 | L31 |
|---|---|---|---|
| y_bg_ground (bg!=blank) | 1.00 | 1.00 | 1.00 |
| y_bg_scene | 1.00 | 1.00 | 1.00 |
| y_obj_3d (shaded/textured) | 1.00 | 1.00 | 1.00 |
| y_obj_textured | 1.00 | 1.00 | 1.00 |
| y_cue_any | 1.00 | 1.00 | 1.00 |
| y_cue_shadow | 1.00 | 1.00 | 1.00 |
| y_cue_arrow | 1.00 | 1.00 | 1.00 |

→ 초기 블록 (layer 3) 부터 vision encoder 는 **모든** 축을 linear
하게 분리한다. 정보 bottleneck 은 0.

### 3.2 행동 PMR 은 그 정보를 부분만 사용

Forced-choice behavioral PMR 을 같은 stimulus 축으로 쪼갠 결과:

| axis | level | behavioral PMR | encoder signal |
|---|---|---|---|
| bg | blank / ground / scene | 0.51 / 0.71 / 0.77 | 1.00 for all |
| object | line / filled / shaded / textured | 0.58 / 0.65 / 0.71 / 0.71 | 1.00 for all |
| cue | none / cast_shadow / motion_arrow / both | 0.28 / 0.49 / 0.93 / 0.95 | 1.00 for all |

Encoder 가 cell 소속을 100% 정확히 알아도, LM 은 그 cell 에서 physics-mode
를 켜는 비율이 28%~95% 로 달라진다. **Gap 은 전적으로 LM 쪽에서 발생**.

### 3.3 Controlled probe — no-cue 서브셋

Cue=none 의 120 stimuli (red-arrow/shadow 없음):

| layer | encoder AUC (behavioral PMR 예측) | behavioral PMR |
|---|---|---|
| 3 | 0.793 | 0.28 |
| 15 | 0.863 | 0.28 |
| **19** | **0.890** | 0.28 |
| 31 | 0.859 | 0.28 |

해석:

- Encoder 는 120개 stimuli 중 **어느 것이 LM 을 physics-mode 로 flip 시킬지**
  layer 19 에서 AUC 0.89 으로 예측한다.
- 그러나 LM 은 120개 중 28% 만 실제로 flip.
- Encoder 가 구별해 둔 physics-leaning stimuli 의 대부분을 LM 이 **물리로 읽지 않고 있다**.

### 3.4 추상도가 올라갈수록 boomerang gap 이 커진다

Object_level 별 encoder AUC (layer 31) vs behavioral forced-choice PMR:

| object_level | encoder AUC | behavioral PMR | gap |
|---|---|---|---|
| line | 0.944 | 0.583 | **+0.361** |
| filled | 0.950 | 0.647 | +0.303 |
| shaded | 0.943 | 0.711 | +0.232 |
| textured | 0.952 | 0.714 | +0.238 |

**Encoder AUC 는 object_level 에 거의 불변 (0.94-0.95)** — 즉 visual encoder
는 stimulus 를 어떤 추상도 수준에서든 잘 표상한다. 반면 behavioral PMR 은
line → textured 에서 13pp 증가. Gap 은 **추상적일수록** 크다.

이는 H4 ("open vs forced-choice gap 은 언어 prior 대 시각 증거의 충돌;
추상도가 높을수록 언어 prior 가 지배한다") 의 **내부 메커니즘 증거** 다.
- 추상 stimulus 에서도 encoder 는 사실상 "physical-ness" 를 알고 있으나,
- LM 이 언어 prior (없는 ball 을 있는 ball 로 부르기 vs forced-choice D
  escape) 로 흐르는 빈도가 더 크다.

## 4. 이게 왜 중요한가

### 4.1 "Pixels to Principles" 주장의 직접적 강화

Ballout et al. 2025 가 사진 자극에서 "vision encoder 는 물리 cue 를 포착하나
LM 이 활용 못 함" 을 보였다. 본 연구는:

1. 동일 주장 (information gap) 을 **매개변수적 추상화 축** 에서 재현.
2. Encoder AUC 를 **stimulus ground truth** 와 비교해 "정말 encoder 에
   모든 정보가 있다" 를 수치로 증명.
3. Gap 이 **추상도와 양의 상관** 이라는 새 발견 — Pixels-to-Principles 는
   사진 자극만 써서 이 패턴을 볼 수 없었다.

### 4.2 논문 figure 후보

**Figure 2 후보: "Encoder knows, LM gates"**

X 축: stimulus property (bg, object, cue 각 수준)
Y 축 왼쪽: behavioral PMR (보통 0.2-0.95 범위)
Y 축 오른쪽: encoder AUC (모두 1.0, 수평선)
Bar chart + horizontal line overlay.

→ Visual 로 "encoder 가 알지만 LM 이 사용 안 함" 을 한 눈에 전달.

**Figure 3 후보: boomerang amplification**

X 축: object_level (line → textured)
Y 축: encoder AUC vs behavioral PMR, 두 line plot
→ Encoder AUC 평탄, behavioral PMR 우상향, gap shading.

### 4.3 Sub-task 4 (causal patching) 을 어디에 겨냥할지

Encoder 가 정보를 끝까지 가져오지만 LM 이 그걸 gating 한다 → intervention target
은 **vision-to-LM 전환 지점 (merger 근처)** 또는 **LM 초기 레이어** 일 가능성.
M5 activation patching 에서 clean/corrupted 쌍을 LM 레이어 **5 → 25** 으로
layer-sweep 하면 "어느 LM 층에서 encoder 정보가 사라지는가" 답 가능.

## 5. 방법론적 caveat

### 5.1 Encoder AUC = 1.0 은 자극이 프로그램적이기 때문에 trivial

512×512 크기에 명확한 color / gradient / discrete feature 가 있는 자극 (line
vs shaded, has ground vs blank 등) 은 **어떤** 합리적 representation 도
완벽 분리할 수 있다. Untrained ViT 도 비슷한 결과일 것.

→ 이 caveat 는 encoder-decoder gap 이 "encoder 가 뛰어나서" 가 아니라
"**decoder 가 정보를 버리고 있어서**" 임을 역으로 강화한다. 정보는 trivially
보존되는데도 LM 이 그걸 일관되게 활용하지 못한다.

### 5.2 Photorealistic stimulus 에서의 재검이 필요

연구계획 §4 (추가 아이디어 4.5) 와 M6 (cross-model sweep) 이 자연스러운
검증 경로:

1. FLUX/SDXL 로 axis A 에 photo 레벨 추가 → encoder AUC < 1.0 가 될 조건.
2. LLaVA-1.5 (CLIP backbone) vs Qwen2.5-VL (SigLIP) 에서 같은 실험 → encoder
   차이가 gap 에 영향을 주는지 확인.

### 5.3 Per-stimulus 단위의 probe 는 "cell" 을 학습하기 쉬움

같은 (obj, bg, cue) 조합의 10 seeds 는 근소하게 다른 이미지이지만
factorial 상 동일 cell 이다. 5-fold CV 에서 train/val 이 같은 cell 의 seeds 를
섞으면 "cell 학습 후 within-cell 예측" 이 쉬워질 수 있다.

→ 더 엄격한 protocol: **cell-level held-out CV** (한 cell 의 10 seeds 를 모두
train 혹은 val 로만 사용). 다음 라운드에서 확인 필요. 현재 결과의 **방향성**
은 바뀌지 않을 것으로 예상 (encoder 가 cell 을 학습하든 stimulus property 를
학습하든, gap 은 LM 쪽에서 온다는 결론은 같음).

## 6. 열린 질문 (M4 에서 답할 것들)

1. **Switching layer**: LM 의 어느 레이어에서 "물리 verb" 가 "기하 noun" 을
   logit 에서 앞서는가? Neo et al. 2024 는 LLaVA-1.5 에서 layer 15-24 로
   보고. Qwen2.5-VL-7B (28 layer LM) 에서 동일하게 중간 층일까?
2. **Per-layer LM probe AUC** 는 encoder AUC 1.0 에서 출발해 얼마까지 유지될까?
   급격히 떨어지는 layer 가 "information discard point" 후보.
3. 라벨 (axis D) 효과는 LM 안에서 어디서 발동하나? `planet`→orbital 서술의
   switch 는 logit lens 에서 "orbit" token 의 layer-wise 상승으로 보일까?

## 7. Paper framing 업데이트

Pilot + M2 + M3 결과를 정리하면:

- **Main claim (unified)**: "VLM 은 원을 공으로 본다 (encoder 가 안다), 하지만
  말할 때는 물리로 말하지 않는다 (LM 이 gate 한다)."
- **Figure 1**: Pilot cell (1) vs (2) 최소 쌍 (circle_blank vs circle_ground):
  behavioral flip 의 teaser.
- **Figure 2**: M3 boomerang — encoder AUC 평탄 1.00 vs behavioral PMR 가변.
- **Figure 3**: M3 per-object-level gap (abstraction × gap 양의 상관).
- **Figure 4** (M4 예정): logit-lens trajectory, switching layer.
- **Figure 5** (M4+M5 예정): LM layer 별 probe AUC × causal patching IE.
- **Figure 6** (optional M6): cross-model 재현.

EMNLP-grounding 프레이밍이 더 강화됨: "grounding failure 는 encoder 에서 오지
않는다; LM 이 grounding 을 leak 한다" — Pixels-to-Principles 의 다음 질문.
