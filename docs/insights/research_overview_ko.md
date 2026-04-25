---
type: 프로젝트 synthesis
date: 2026-04-25
status: M6 r6 + §4.2 + §4.10 시점 (논문 Section 4 lock)
audience: 논문 reviewer / 신규 협업자 / 미래 자신
---

# VLM 의 Physical-mode activation — 연구 종합

이 프로젝트가 무엇을 발견했고, 어떻게 발견했고, 무엇이 아직 열려있는지의
단일 문서 synthesis. 세션별 작업을 따라오지 않은 독자 대상.

## TL;DR

오픈소스 VLM 은 최소 합성 자극 (흰 배경의 검은 원) 을 물리 객체로 **읽지만**,
그 읽기는 encoder representational 수준이 아닌 **architecture 수준** (joint
vision-encoder + LM) 에서 결정된다. 5 테스트 모델 (Qwen2.5-VL, LLaVA-1.5,
LLaVA-Next, Idefics2, InternVL3) × 3 stim source (M8a 합성 도형 5, M8d
합성 카테고리 3, M8c 실사진 5) 에 걸쳐:

- 모든 encoder 가 physics-vs-abstract factorial 셀을 AUC = 1.0 으로
  linear separation — encoder 식별 능력 균일.
- 행동 PMR (모델이 physics-mode 언어에 commit 하는 비율) 은 같은 stim 에서
  joint architecture 에 따라 0.18–0.92 범위.
- 2-CLIP 점 비교 (LLaVA-1.5 PMR 0.18 vs LLaVA-Next PMR 0.70) 가 vision-
  encoder family 를 단독 driver 로 배제.
- 시각 token 이 입력의 79–98% 를 차지함에도 5 LM 모두 마지막 token
  attention 의 3–26% 만 시각 token 에 할당. 시각 attention 이 mid-layer
  에서 정점.
- 실사진은 인코더 갭을 **압축** (5 모델 모두 PMR [0.18, 0.67] 로 수렴) 하고
  라벨 기반 H7 효과를 **절반** 으로 (LLaVA-1.5 M8d +0.31 → M8c +0.10) —
  이미지가 풍부할 때 image-prior 가 label-prior 지배.

## 헤드라인 figure

![5-model × 3-stim PMR ladder](../figures/session_5model_cross_stim_pmr.png)

부트스트랩 CI 와 함께 5-모델 × 3-stim PMR 사다리. 합성 stim (M8a + M8d)
에서 인코더-family 분리, M8c 사진에서의 photo-collapse 가 두 개의 paper-
grade fact.

## 측정하는 것

**PMR (physics-mode reading rate)**: 모델 응답이 자극의 다음 상태를 물리적
용어로 묘사하는 비율 (예: "공이 떨어진다", "튀어 오른다", "굴러 내려간다").
규칙 기반 scorer, 손 라벨링 대비 ~5% 불일치. `docs/scoring_rubric.md` 참조.

변형:
- **PMR(_nolabel)** — 라벨 cue 없는 open-ended 프롬프트
  ("무엇이 보이는가? 다음에 무엇이 일어날까?"). joint encoder+LM 경향의
  직접 측정.
- **PMR(_physical)** — 같은 프롬프트 + 물리 역할 라벨 ("공이...", "차가
  ...", "사람이...").
- **PMR(_abstract)** — 같은 프롬프트 + 추상 역할 라벨 ("원이...", "실루엣
  이...", "stick figure 가...").
- **H7 paired-difference**: 도형별 PMR(_physical) − PMR(_abstract) 평균.
  라벨이 physics regime 을 선택하는지 검증.

부트스트랩 CI (5000 iter, shape × role 셀 내 prediction-level resampling)
가 M9 / M6 r6 / §4.2 에서 PASS/FAIL binarization 을 separability 검증으로
대체.

## 자극 설계

복잡도 순:

- **M8a — 합성 도형 (모델당 factorial 400 stim)**.
  5 도형 (원, 사각형, 삼각형, 육각형, 다각형) × 4 추상화 (선/채움/음영/
  텍스처) × 2 배경 (빈/땅) × 2 cue (없음/모두) × 5 seed. 단일 이벤트
  템플릿 (`fall`).
- **M8d — 합성 카테고리 (모델당 factorial 480 stim)**.
  3 카테고리 (car, person, bird) × 4 추상화 × 2 bg × 2 cue × 2 이벤트
  (`fall`, `horizontal`) × 5 seed. M8a 와 같은 프로토콜이지만 비-공
  물리 객체.
- **M8c — 실사진 (60 사진)**.
  카테고리당 12 사진 × 5 카테고리 (ball, car, person, bird, abstract)
  COCO 2017 + WikiArt 에서. 카테고리당 LABELS_BY_SHAPE 의 단일 라벨
  triplet.

각 labeled-arm run × 모델 × stim 이 ~400-1440 추론 생성; 각 label-free
arm 이 n=400 PMR(_nolabel) baseline 생성.

## 구축한 것 (M0–M5a 4 단계, 그 후 M6 + M8 + M9 4 단계)

### Phase 1: pilot + scoring (M0–M2)

- **M0**: PIL 의 프로그래밍 자극 생성 (`primitives.py`).
- **M1**: Qwen2.5-VL-7B-Instruct 의 pilot 추론, label-free arm. object_
  level / bg_level / cue_level 셀의 기본 PMR 사다리 확립. paired-delta
  (`physical_label` − `_nolabel`) 가 `textured/ground/both` 셀에서 0
  근처 (천장), `line/blank/none` 셀에서 강한 양수 (헤드룸).
  `docs/insights/m1_pilot.md`.
- **M2**: H1/H2/H4/H5/H6/H7 의 전체 PhysCue grid.
  - **H1 (ramp)**: PMR 이 object_level 와 함께 단조 상승.
  - **H2 (label)**: 물리 라벨이 PMR 상승 (나중에 "헤드룸이 있는 곳에서만
    PMR 상승" 으로 정련).
  - **H7 (label-selects-regime)**: 물리 라벨 (ball) 이 "fall/roll" 동사
    유발; 추상 라벨 (planet) 이 "orbit" 유발.

### Phase 2: probing (M3–M4)

- **M3 — vision encoder probe**: 풀링된 vision-encoder activation 의
  per-layer 로지스틱 회귀. Qwen SigLIP AUC = 0.99 모든 캡처 layer 에서 —
  encoder 가 physics-mode 와 abstract-mode stim 을 trivial 하게 linear
  separation. **H-boomerang** 발견: encoder AUC ≈ 1.0 인데 행동 PMR 이
  변동 — LM 이 encoder 신호를 "gate". `docs/insights/m3_encoder_boomerang.md`.
- **M4 — LM logit lens + per-layer probe**: LM activation 을 logit lens
  (vocabulary 로 projection) 에 공급. 라벨-physics margin 이 mid-layer
  에서 발달 (28 중 L20 정점). `docs/insights/m4_logit_lens.md`.
- **M4b — label-free H2 null test**: 라벨 없는 Qwen 이 `ball` 라벨 Qwen
  과 거의 같은 PMR (paired-delta +0.006). M2 의 "라벨이 PMR 상승" 패턴이
  **circle 억제**, ball 강화가 아니라고 귀속. `docs/insights/m4b_label_free.md`.
- **M4c — forced-choice label-free**: FC 채점이 M4b 의 null 재현 확인.
  LLaVA "A" 편향이 logit 수준 pathology, greedy 샘플링 artifact 가 아님.
  `docs/insights/m4c_fc_label_free.md`.

### Phase 3: causal steering (M5a)

- **M5a — VTI steering**: M3 도출 "object-ness" 방향으로 LM L10 인과
  개입. α = +40 가 10/10 `line/blank/none` Qwen 응답을 D (추상) 에서 B
  (정적 물리) 로 flip. `docs/insights/m5_vti_steering.md`.
- **M5a-ext**: −α steering 이 physics-mode 응답을 kinetic 에서 static
  으로 push, `v_L10` 이 **physics-mode 내 regime axis** (+α kinetic,
  −α static, baseline D 가 |α| 임계 아래) 임 확인.
  `docs/insights/m5a_ext_bidirection_and_label.md`.

### Phase 4: cross-model + cross-stim (M6 + M8 + M9)

- **M6 r1 — LLaVA-1.5-7B cross-model**: M2 + M4b 프로토콜 복제. LLaVA-1.5
  가 **원래 H2** 보임 (`ball` +0.475 vs no-label) — Qwen 의 "circle 억제"
  가 Qwen 특이적, encoder saturation 으로 추적. **Visual-saturation
  hypothesis** 도입. `docs/insights/m6_cross_model_llava.md`.
- **M6 r2 — 3-모델 확장 + capture + FC**: InternVL3 cross-model 행동
  (모든 라벨에서 paired-delta +0.010, super-saturated); LLaVA-1.5 vision
  encoder probe AUC ~0.73 (Qwen 0.99 보다 한참 아래). vision-encoder
  probe AUC 에 anchored 된 **H-encoder-saturation hypothesis**.
  `docs/insights/m6_r2_cross_model.md`.
- **M8a — 비-원형 합성 도형**: 5 도형 × Qwen + LLaVA, 사전 등록 엄격
  채점. **Qwen 1/4 PASS, LLaVA 4/4 PASS** — 비대칭이 H-encoder-saturation
  cross-shape 검증. H1 + H7 이 **unsaturated-only** 로 강등.
  `docs/insights/m8a_non_circle_shapes.md`.
- **M8d — 비-공 카테고리**: car/person/bird × 추상화 × bg × cue ×
  2 이벤트 × 5 seed. **LLaVA 3/3 H7 ✓** (프로젝트 최강 H7), Qwen 0/3
  binary (ceiling) 이지만 regime distribution 이 패턴 보존. 새 `classify_
  regime` 키워드 분류기 (5.6% 손 라벨링 오차).
  `docs/insights/m8d_non_ball_categories.md`.
- **M8c — 실사진**: 60 사진 × 5 카테고리. **사진이 Qwen PMR 18–48 pp
  감소** 카테고리 횡단 — 합성-stim 단순성이 행동 saturation 의 공동 인자,
  encoder 표현뿐 아니라. `docs/insights/m8c_real_photos.md`.
- **§4.5 — cross-encoder swap (Idefics2)**: Idefics2-8b (SigLIP-SO400M
  + Mistral-7B) on M8a — PMR(_nolabel) = 0.882, Qwen 0.838 일치 (vs
  LLaVA 0.175). H-encoder-saturation **encoder-family 수준 인과 확인**:
  SigLIP family 가 LM 무관하게 saturation. `docs/insights/encoder_swap_idefics2.md`.
- **M6 r3 — Idefics2 SigLIP-SO400M probe**: AUC 0.93. 3-점 AUC 사다리
  Qwen 0.99 / Idefics2 0.93 / LLaVA 0.73. `docs/insights/m6_r3_idefics2_probe.md`.
- **M6 r4 — InternVL3 InternViT probe + 4-모델 chain**: AUC 0.89 / PMR
  0.92. 동일 stim 에서 4-점 chain. **Stim-y check (후반 라운드 추가)**
  발견: 4 encoder 모두 stim-defined factorial 셀을 AUC = 1.0 분리.
  **H-encoder-saturation 을 architecture 수준 (encoder + LM 융합) 으로
  reframe**. `docs/insights/m6_r4_internvl3_probe.md`.
- **M6 r5 — M8c 사진 encoder probe**: cross-stim 4-모델. 사진에서 행동-y
  AUC 반전 (Qwen 0.88→0.44, 그러나 LLaVA 유지). Stim-y AUC 1.0 유지.
  architecture-level reframe 의 cross-stim 확인.
  `docs/insights/m6_r5_m8c_photo_probe.md`.
- **M9 — 일반화 audit / 논문 Table 1**: 부트스트랩 CI (5000 iter) 와 함께
  9 (모델 × stim) 셀. PASS/FAIL binarization 을 separability 검증으로
  대체. `docs/insights/m9_generalization_audit.md`.
- **M6 r6 — LLaVA-Next 5번째 모델 + cross-stim**: 2번째 CLIP 점 (CLIP-
  ViT-L + Mistral-7B + AnyRes). PMR(M8a) = 0.700 [0.65, 0.74], LLaVA-1.5
  바닥과 saturated cluster 사이. M8d 0.625, M8c 0.417 (= Idefics2). 3
  stim 모두에서 stim-y AUC = 1.0. **2번째 CLIP 점이 vision-encoder family
  를 단독 결정자로 배제**. 5×3 grid lock. `docs/insights/m6_r6_llava_next.md`.

### Phase 5: §4 add-on (이번 세션)

- **§4.2 — 실사진 역 프롬프팅**: 기존 M8c labeled-arm 데이터 재분석.
  실 물리 사진에서 image-prior 가 label-prior 지배: 5 모델 모두 phys − abs
  ≤ +0.146, vs LLaVA-1.5 M8d 합성 phys − abs +0.306. **라벨 지배는 이미지
  빈약을 요구**. `docs/insights/sec4_2_reverse_prompting.md`.

  ![§4.2 H7 효과가 사진에서 절반](../figures/session_image_vs_label_h7.png)

- **§4.10 — Attention 시각화 UI**: M8a subset 에서 5-모델 attention 캡처.
  **시각 token 이 입력의 79–98% 인데도 5 VLM 모두 마지막 token attention
  의 3–26% 만 시각 token 에 할당**. 시각 attention 이 mid-layer (15 또는
  20) 에서 정점. `docs/insights/sec4_10_attention_viz.md`.

  ![Cross-model 시각 token attention](../figures/session_attention_cross_model.png)

## 가설 상태 (M6 r6 + §4.2 + §4.10 후)

| 가설 | 상태 | 근거 |
|---|---|---|
| **H-encoder-saturation** (architecture-level) | ✅ 5 모델 점 × 3 stim source 에서 확인 | 5 stim-y AUC 모두 = 1.0; PMR 사다리는 downstream-conditional. CLIP+Vicuna 0.18 vs CLIP+Mistral+AnyRes 0.70 가 encoder family 를 단독 driver 로 배제. |
| **H1** (ramp) | ✅ unsaturated-only | LLaVA 5/5 M8a, Qwen 1/4 M8a (saturated). |
| **H7** (label-selects-regime) | ✅ unsaturated-only AND architecture-conditional | LLaVA-1.5 M8d +0.31 (프로젝트 max). LLaVA-Next M8d −0.05 — 동일 encoder family, architecture switch 가 H7 깸. |
| **H-direction-bidirectional** (M5a-ext) | ✅ 확인 | v_L10 이 physics-mode 내 regime axis (+α kinetic, −α static). |
| **H-boomerang** | ✅ Qwen-scoped (revised) | Qwen 에서 유지 (encoder AUC 0.99 + 행동 0.95). LLaVA-1.5 에서 반박 (encoder AUC 0.73 = bottleneck). |
| **H-LM-modulation** | ⚠ 시사만 | 두-Mistral M8d H7 ≈ 0 클러스터링 (Idefics2 +0.05 / LLaVA-Next −0.05) 이 multi-axis-confounded. 논문 옹호 불가. |
| **§4.2 image-dominates-label** | ✅ cross-stim 확인 | 합성 라벨 효과가 5 모델 모두 사진에서 ≤ +0.15 로 절반. |

## 방법론 기여 (논문 관련)

1. **PhysCue 자극 프로토콜** — 5 도형 × 4 추상화 × 2 bg × 2 cue factorial
   (M8a) + 3 카테고리 × 2 이벤트 확장 (M8d) + 60 실사진 (M8c). 프로그래밍,
   결정적, 재현 가능.
2. **PMR 채점 rubric** — 손 라벨링 검증과 함께 규칙 기반 (~5% 불일치).
   `docs/scoring_rubric.md`.
3. **Bootstrap CI 방법론 (M9)** — (shape × role) 셀 내 prediction-level 의
   5000-iter 재샘플링. PASS/FAIL binarization 을 separability 검증으로 대체.
4. **Stim-y vs 행동-y probe 구분 (M6 r4)** — encoder probe AUC 가 두 해석
   가짐: stim-defined y (factorial 셀) 로는 encoder discriminability 측정;
   behavioral y (모델의 PMR 분포) 로는 encoder-behavior 정렬 측정. 두
   측정이 sharply 발산 (1.0 vs 0.77–0.93), 그 발산이 H-encoder-saturation
   reframe.
5. **classify_regime 키워드 분류기 (M8d)** — 5.6% 손 라벨링 오차; binary-
   PASS saturation 아래의 physics-regime distribution 읽어서 H7 신호 구제.
6. **L10 의 VTI steering (M5a)** — `v_L10` 을 physics-mode regime axis
   로 인과 확인. α = ±40 가 응답을 physics-mode 내에서 static / kinetic
   사이 flip.

## 이 synthesis 가 다루지 않는 것

- **M5b — SIP / activation patching / SAE 특징 분해**. 다음 mechanism-level
  milestone, 미시작. 어떤 LM 특징이 physics-mode commit 를 carry 하는지
  인과적으로 위치시킴.
- **§4.6 — SAE / VTI 역방향 counterfactual stimulus 생성**. Adversarial
  physics-mode prompt 합성. 미시작.
- **§4.3 — 한국어 vs 영어 라벨 prior** (1시간 실험). 열림.
- **§4.4 — Michotte-style 2-frame causality**. 열림.
- **§4.7 — RC 재해석 per-axis 결정 안정성**. 열림.
- **§4.8 — PMR scaling** (Qwen 32B/72B). 열림.
- **§4.11 — 카테고리 H7 follow-up** (regime confusion matrix). 열림.
- **M7 — 논문 초고 + 인간 baseline** (Prolific 20 평가자 × 50 자극 +
  EMNLP/NeurIPS 초고). 열림.
- **Qwen2.5-VL / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 외부의
  external validity** — Pixtral, Phi-V, Gemini-VL, GPT-4V, Claude. 열림.

## 한계

1. **LLaVA-1.5 와 LLaVA-Next 간 multi-axis confound**. 0.18 → 0.70 PMR
   점프가 프로젝트의 가장 큰 행동 이동이지만 4 축 confound (AnyRes tiling,
   fusion projector, 학습 데이터 + 레시피, LM 계열). 시험된 어떤 모델도
   same-architecture LM-only swap 제공 안 함; 어느 축이 load 짊어지는지
   분리 불가.
2. **카테고리당 n=12 사진 on M8c** 가 H7 검출에 underpowered.
3. **합성 factorial 이 M8a/M8d-style** (선 / 빈 / 없음 ↔ 텍스처 / 땅 /
   모두). 실세계 stim 분포는 더 다양; M8c 가 이쪽으로의 작은 한걸음.
4. **Attention 귀속이 근사적**. §4.10 가 "마지막 token attention 의 시각
   token 비율" 측정 — visual-information access 의 거친 신호. 인과적
   주장에는 activation patching (M5b) 필요.
5. **인간 baseline 아직 없음**. PMR 은 모델-내부 metric; 인간 "physics-
   mode reading" 판단으로의 매핑은 M7 작업.

## 다음 연구 방향 (우선순위 순)

1. **M5b — SIP / activation patching / SAE**. 어떤 LM 특징이 physics-mode
   commit 를 carry 하는지 mechanism-level 증거.
2. **§4.6 — SAE counterfactual stimulus 생성**. Shortcut 해석 검증을 위한
   adversarial physics-mode prompt.
3. **M7 — 논문 초고**. Section 4 (encoder-saturation chain) 이 이제 paper-
   grade 완료; Section 5 (mechanism, M5b) 가 갭.
4. **External validity sweep** (Pixtral, Phi-V, Gemini-VL) — M5b 가 5-모델
   메커니즘 baseline 확립한 후.

## 재현 map (어디에 무엇이 있는지)

- **자극**: `inputs/m8a_qwen_*` (합성), `inputs/m8d_qwen_*` (카테고리),
  `inputs/m8c_photos_*` (실사진).
- **예측**: `outputs/<config_name>_<ts>_<hash>/predictions.{jsonl,parquet,csv}`.
- **Vision activation**: `outputs/encoder_swap_<model>_<stim>_vision_activations/*.safetensors`.
- **Probe 출력**: `outputs/encoder_swap_<model>_<stim>_probe{,_stim_y}/*.csv`.
- **집계 테이블**: `outputs/m9_audit/m9_{table1,summary}.csv`,
  `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`.
- **노트북**: `notebooks/encoder_saturation_chain.ipynb` (5-모델 ×
  3-stim chain + §4.2), `notebooks/attention_viz.ipynb` (§4.10).
- **Insight 문서**: `docs/insights/*.md` (per-milestone, ko/en 페어).
- **로드맵**: `references/roadmap.md` (상태 테이블, 가설 scorecard,
  추가 아이디어, 변경 이력).

## Per-milestone insight 문서 포인터

- M0: 자극 프로토콜 — `docs/stimulus_spec.md`
- M1: pilot — `docs/insights/m1_pilot.md`
- M2: PhysCue grid (H1/H2/H4/H5/H6/H7) — M3/M4 docs 에 implicit
- M3: encoder boomerang — `docs/insights/m3_encoder_boomerang.md`
- M4: LM logit lens — `docs/insights/m4_logit_lens.md`
- M4b: label-free null — `docs/insights/m4b_label_free.md`
- M4c: FC label-free — `docs/insights/m4c_fc_label_free.md`
- M5a: VTI steering — `docs/insights/m5_vti_steering.md`
- M5a-ext: bidirectional steering — `docs/insights/m5a_ext_bidirection_and_label.md`
- M6 r1 / r2: cross-model — `docs/insights/m6_cross_model_llava.md`,
  `docs/insights/m6_r2_cross_model.md`
- M6 r3 / r4 / r5 / r6: encoder probe chain —
  `docs/insights/m6_r{3,4,5,6}_*.md`
- M8a / M8c / M8d / M8e: stim 다양화 —
  `docs/insights/m8{a,c,d,e}_*.md`
- §4.5 / §4.5 ext: encoder swap — `docs/insights/encoder_swap_idefics2.md`
- M9: 일반화 audit — `docs/insights/m9_generalization_audit.md`
- §4.2: 역 프롬프팅 — `docs/insights/sec4_2_reverse_prompting.md`
- §4.10: attention viz — `docs/insights/sec4_10_attention_viz.md`
- 5-모델 synthesis (논문): `docs/insights/encoder_saturation_paper.md`
- 이번 세션 (2026-04-25): `docs/insights/session_2026-04-25_summary.md`
