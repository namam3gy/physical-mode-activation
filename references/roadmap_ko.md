# ROADMAP — Physical-Mode Activation

> **이 문서의 역할.** 이 프로젝트의 "지금 어디까지 왔고 다음 뭘 할지"를 한 곳에서 본다. 새 세션을 시작할 때 **이 파일부터 읽고**, 마일스톤이 끝날 때마다 §3의 상태를 갱신한다. 세부 내용은 각 doc / 코드로 링크한다.
>
> - 연구 철학·가설 원본: `research_plan.md` (한국어, 33k)
> - 아키텍처: `docs/00_architecture.md`
> - 자극 스펙: `docs/01_stimulus_spec.md` / 점수 기준: `docs/02_scoring_rubric.md`
> - 실제 실행 기록: `docs/03_run_log.md`
> - 다음 단계 코드 진입점: `docs/04_next_steps.md`
> - 최신 인사이트: `docs/05_insights.md`

---

## 1. 연구 정의

### 1.1 중심 질문

**어떤 시각적 단서가 임계치를 넘어서면 open-source VLM이 추상 도형(원)을 기하학적 객체에서 물리적 객체(공)로 "모드 전환"하여 처리하기 시작하는가?**

측정은 두 층위로 한다:

- **행동(behavior)**: next-state-prediction 프롬프트에 대한 응답의 PMR (physics-mode priming rate) / GAR (gravity-align rate) / RC (response consistency).
- **내부(mechanism)**: 시각 인코더의 선형 probe AUC, LM backbone의 층별 logit-lens 궤적, 활성화 패칭으로 드러나는 인과적 병목 층·head.

### 1.2 Sub-task 구성 (연구계획 §2)

| # | 제목 | 내용 | 입력 | 출력 |
|---|---|---|---|---|
| ST1 | **PhysCue** behavioral thresholds | 4-5개 축 factorial stimulus + 다음 상태 예측 프롬프트 | 프로그램/photo stimuli | PMR/GAR/RC 표, per-factor curves |
| ST2 | Vision-encoder probing | CLIP / SigLIP / InternViT 활성화에 linear probe + Gandelsman head decomposition + SAE | ST1 captured vision acts | layer×head AUC, monosemantic features |
| ST3 | LM backbone layer-wise emergence | Logit lens + per-layer probes at visual token positions (Neo et al. 2024 recipe) | ST1 captured LM acts | layer×token heatmap, switching-layer |
| ST4 | Causal localization | Semantic Image Pairs + activation patching + VTI steering + SAE intervention | pilot pairs | IE curve, steering vector, head ranking |
| ST5 | Cross-model + prompt-steering | LLaVA-1.5 / LLaVA-Next / Qwen2-VL / InternVL2에 같은 factorial + 프롬프트 steering (Gavrikov et al. 2024) | 확장된 EvalConfig | 모델 간 비교 표, prompt-bias curve |

### 1.3 가설 스코어카드

전체 스코어카드 (evidence chain 포함) 는 **`docs/hypotheses.md`** (영문 canonical). evidence column 이 길어져 2026-04-29 에 추출. 한 줄 요약:

| ID | 가설 (한 줄) | 상태 |
|---|---|---|
| **H1** | PMR 이 추상화 축에 따라 S자형 증가 | 지지, unsaturated-only AND 도형-축 특정 |
| **H2** | "ball" 라벨이 PMR 을 크게 증가 (언어 prior 독립 기여) | 완전 검증, 3-점 + encoder-anchored |
| **H3** | 장면 불일치가 RC 를 저하 | 미검증 |
| **H4** | Open vs FC PMR gap 이 언어-prior ↔ 시각-증거 충돌의 안정 signature | 지지 (Qwen-only — cross-model FC 미검증); v2 scorer 하 강화 |
| **H5** | 단일 ground line 이 텍스처 공 + no-ground 보다 더 큰 PMR 증가 | 혼재 |
| **H6** | arrow+shadow cue 포화는 cast shadow 단독으로도 일어남 | 지지 (수정 — arrow 가 dominant, shadow secondary) |
| **H7** | 라벨이 PMR 을 toggle 하는 것이 아니라 어떤 physics regime 을 선택 | 지지, unsaturated-only AND 카테고리 횡단 |
| **H-boomerang** | Encoder 는 알고, decoder 가 gate | Qwen 특이적 (수정; LLaVA-1.5 에서 반증) |
| **H-encoder-saturation** | 행동 PMR(_nolabel) 포화는 architecture level 에서 결정 | 5-모델 점에서 architecture-level 확인 + cross-stim 부트스트랩 검증 |
| **H-LM-modulation** | 인코더 포화 시 LM family 가 잔여 H7 측정 가능성 조절 | 시사만 — M8d CI 0 닿음 |
| **H-locus** | Bottleneck 은 LM mid-layer (Qwen L10 / LLaVA-Next L20-L25 / Idefics2 L25) | 지지, 세 번 정련 + cross-model 확인 |
| **H-direction-bidirectional** | `v_L10` 은 physics-mode 내부의 regime axis (+α kinetic, −α static) | 수정 — "bidirectional concept axis" 였음; baseline D 가 \|α\| 임계 아래 |
| **H-regime** | Steering direction 이 binary "object-ness", physics regime 은 label-driven | 원래 형태로 반증 (H7 qualifier 로 흡수) |
| **H-direction-specificity** | v_L 방향 픽셀-gradient ascent 가 PMR flip; 매칭 magnitude random 은 안 됨 | 5 architecture × 5 layer at n=10 에서 지지 |
| **H-shortcut** | Shortcut 해석이 이미지 자체에 인코드 가능 (픽셀 기반) | 지지, architecture-conditional; perceiver-resampler 가 leading remaining 메커니즘 후보 |

상세 evidence: `docs/hypotheses.md`. 가설 상태가 변경되거나 신규 evidence 추가 시: `docs/hypotheses.md` (full evidence) + 본 표 (한 줄 상태) 동시 갱신, `docs/CHANGELOG.md` 에 entry 추가.

### 1.4 Target 모델 & venue

- **1차(pilot / MVP-full)**: Qwen/Qwen2.5-VL-7B-Instruct — proven loader, 15 GB, H200에서 1.0 it/s.
- **ST5 확장**: LLaVA-1.5-7B, LLaVA-Next-7B, InternVL2-8B, (stretch) Qwen2-VL-7B (연구계획 §2.4가 명시한 모델과 layer-index align 용).
- **Venue**: EMNLP long (grounding failure / language-prior dominance angle)이 primary; NeurIPS main (ST3-4 mechanistic localization)이 stretch.

---

## 2. 마일스톤 전체 뷰

| # | 마일스톤 | 스코프 | 상태 | 완료일 |
|---|---|---|---|---|
| M0 | 인프라 스캐폴드 | Package layout, configs, scripts, tests, docs 기본 set | ✅ | 2026-04-24 |
| M1 | **ST1 Pilot** (Qwen2.5-VL-7B) | 240 stim × 2 prompts = 480 inferences; behavioral S-curve 1차 측정 | ✅ | 2026-04-24 |
| M2 | **ST1 MVP-full** (pilot 교훈 반영) | axis C 분해, axis D 확장, T=0.7, LM hidden-state capture, 2880 inferences (Qwen2.5-VL 만). Cross-model 확장은 **M6 r7** (2026-04-26) 에서 — 아래 참조. | ✅ | 2026-04-24 |
| **M6 r7** | **M2 cross-model — 5-model M2-stim apples-to-apples** | LLaVA-Next + Idefics2 + InternVL3 가 M2 stim 위 LM 활성화 캡처와 함께 추가 (Qwen + LLaVA-1.5 기존). 5-model PMR(_nolabel) 사다리: LLaVA-1.5 0.18 / LLaVA-Next 0.79 / Qwen 0.94 / Idefics2 0.97 / InternVL3 0.99. H1 ramp 는 LLaVA-1.5 만 깨끗 (+0.30 range). H2 paired-delta 가 3 가지 distinct 패턴 (LLaVA = 양수, Qwen/Idefics2 = circle override, InternVL3 ≈ 0). Per-model v_L10 추출; saturated 모델 (LLaVA-Next/Idefics2/InternVL3) 은 class-imbalanced (n_neg = 9/5/1, M2 만으로 v_L10 추출 불가). | ✅ | 2026-04-26 |
| **§4.6 cross-model** | **픽셀-인코드 가능성 cross-model layer sweep (5 architecture × n=10)** | 4 라운드: 2026-04-26 morning (Qwen+transfer test) → 2026-04-26 overnight (LLaVA-1.5 layer sweep) → 2026-04-27 afternoon (LLaVA-Next + Qwen layer sweep) → **2026-04-28 5-모델 n=10 chain** (Idefics2 + InternVL3 추가). 최종 그림: Qwen broad (5 shortcut layer, 모두 80 % +), LLaVA-Next L20+L25 (10/10), LLaVA-1.5 L25 만 (n=10 에서 40 % — n=5 보다 약함), **Idefics2 anomaly** (높은 AUC + 깨끗한 projection ascent 에도 깨끗한 shortcut 0), InternVL3 untestable (baseline=1.0). H-shortcut → architecture-conditional, model-specific shortcut profile; 이전 "capacity scales with saturation" reading 은 "CLIP/SigLIP+Qwen 부분집합 안의 패턴" 으로 격하 — Idefics2 perceiver-resampler / Mistral 상호작용 또는 미테스트 L26-31 이 anomaly 설명할 수도. | ✅ | 2026-04-28 |
| **§4.8** | **Qwen 7B vs 32B PMR scaling (M2 open prompt)** | 32B M2 inference (1440 stim × 1 prompt = H200 single-GPU bf16 에서 16 분 wall). **결과: aggregate PMR 0.926 ≈ 7B 0.931** — 5× 스케일링이 전체 PMR 못 움직임 (MechBench 식 "스케일이 grounding 못 고침" 지지). 유의미 shift: `abstract_reject` 32B 에서 35× 높음 (0.002 → 0.065, 대부분 cue=none cell), H2 label gap 절반 (`ball − circle` 7B +0.071 → 32B +0.010). 32B 가 더 cue-sensitive 이지만 언어-prior 지배 체제는 살아남음. Insight: `docs/insights/sec4_8_pmr_scaling.md` (+ ko). | ✅ | 2026-04-28 |
| M3 | **ST2 — Vision encoder probing** | Vision blocks capture (8 layers, 12 GB) + layer-wise linear probes. **Boomerang 확인**: encoder AUC=1.0 on every axis; behavioral PMR 0.28-0.95. | ✅ | 2026-04-24 |
| M4 | **ST3 — LM logit lens / layer-wise probe** | LM hidden @ visual tokens AUC 0.94-0.95 전 구간; L20 peak. Label prior 가 L5 부터 physics margin shift; object_level effect 는 7배 더 작음. | ✅ | 2026-04-24 |
| M5a | **ST4 Phase 1+2 — VTI steering** | 방향 추출 + residual-stream injection. **L10 α=40 이 10/10 D→B flip** — "physical object-ness" direction 인과 확인. | ✅ | 2026-04-24 |
| M5a-ext | **VTI 후속 (neg α, label swap, 양방향성 재검정)** | Exp 1-2 (2026-04-24): ceiling 에서 neg α + label=ball side-by-side. Exp 3 (2026-04-25): moderate baseline 에서 (α × label × obj) 그리드. **핵심 결과**: `v_L10` 은 physics-mode 내부의 regime axis — +α → A (falls), −α → B (stays still), baseline D 는 threshold 아래. | ✅ | 2026-04-25 |
| M4b | **Label-free prompt — H2 null test** | M2 자극에 `open_no_label` variant. **핵심 결과**: `ball` ≈ no-label; `circle` 이 PMR 을 6.5 pp 억제. 원래 H2 재해석: language prior 는 비대칭 — circle override, ball enhancement 아님. M4 visual-token capture 가 prompt-independent (구조적 artefact). | ✅ | 2026-04-25 |
| M6 r1 | **ST5 round 1 — LLaVA-1.5-7B cross-model** | M2 + M4b 프로토콜을 LLaVA-1.5-7B 에. **핵심 결과**: M4b 의 "circle suppression" 은 **Qwen 특이적** — LLaVA 는 *원래의* H2 (ball +47.5 pp, 모든 label 이 no-label baseline 대비 양수). 새 통합 가설: language prior 가 모든 label 에서 양수; Qwen 의 visual saturation 이 양의 기여를 mask 함. H7 cross-model 재현 (planet GAR << ball GAR 두 모델 모두). LLaVA 가 본 프로젝트 가장 깔끔한 H1 S-curve 제공. FC 제외 (LLaVA 가 모든 cell 에 "A" 반환). | ✅ | 2026-04-25 |
| M4c | **Forced-choice label-free** | 새 `forced_choice_no_label` variant ("the depicted object" antecedent 사용). Qwen 이 FC 하에서 M4b 의 H2 패턴 재현 + planet 억제 효과 추가 (옵션-셋 편향: orbital regime 이 D 로 collapse). LLaVA 의 "A" 편향이 re-template 에서도 유지 (477/480), 모델 수준 pathology 확인. | ✅ | 2026-04-25 |
| M6 r2 | **Cross-model round 2 (3-model + LLaVA captures + FC logit ratio)** | r2a: InternVL3-8B-hf cross-model 행동; r2b: LLaVA-1.5 activation captures + cross-model M3/M4 probing; r2c: 모든 FC run 의 first-token logit-ratio 채점. **핵심 결과**: visual-saturation 가설 3-of-3 모델 완전 검증; vision encoder probe AUC 에 뿌리 (Qwen 0.99, LLaVA 0.73). H-boomerang 을 Qwen-scoped 로 revised. 신규 **H-encoder-saturation** 가설. LLaVA "A" 편향이 logit-level (greedy-level 아님). | ✅ | 2026-04-25 |
| **M6 r3** | **Idefics2 비전-인코더 프로브 (AUC↔PMR 사슬 종결)** | 400 M8a stim × SigLIP-SO400M 4 레이어 (88 초 캡처), per-stim PMR 에 대한 per-layer 로지스틱 회귀 프로브. **평균 AUC 0.93 (레이어 9 에서 0.948 최고)** — SigLIP 클러스터가 인코더-family 수준에서 확인 (Qwen 0.99 + Idefics2 0.93 vs LLaVA 0.73). H-encoder-saturation 사슬 `인코더 family → AUC → PMR → H7` 이 메커니즘 수준에서 3 모델 점에 종결. | ✅ | 2026-04-25 |
| **M6 r4** | **InternVL3 비전-인코더 프로브 + Qwen/LLaVA M8a 재캡처 (동일 stim 4-모델 사슬, 비-CLIP-일반화)** | M8a 추론 (1200+400 12 분) + 400 stim × InternViT 4 레이어 (47 초 캡처) + per-layer 로지스틱 회귀 프로브. **InternVL3 평균 AUC 0.89 / PMR(_nolabel) 0.92.** 라운드에서 Qwen + LLaVA 도 M8a stim 에서 재캡처 — apples-to-apples 4-모델 AUC 표. **4-점 M8a 사슬: Qwen 0.88/0.84, LLaVA 0.77/0.18, Idefics2 0.93/0.88, InternVL3 0.89/0.92.** 3 개 별개의 비-CLIP 인코더 family (SigLIP, SigLIP-SO400M, InternViT) 가 AUC ≥ 0.88 클러스터; CLIP-ViT-L 만 0.77. 논문 주장이 "SigLIP 포화"에서 "비-CLIP 인코더 포화; CLIP 비포화 (이 표본)"로 일반화. InternVL3 `vision_tower.encoder.layer` (단수) 속성 인식 위한 vlm_runner 수정 필요. **Stim-y 검증 (후반 라운드 추가)**: 4 인코더 모두 stim-정의 factorial 셀을 AUC = 1.0 분리; 인코더 식별 능력은 family 횡단 균일. 사슬을 아키텍처 수준 (인코더 + LM 융합) 으로 재구성, 인코더-식별 능력 아님. | ✅ | 2026-04-25 |
| **M6 r5** | **M8c 사진 인코더 프로브 (4-모델 cross-stim)** | InternVL3 M8c 추론 (180+60 2분) + 4 모델 × 60 사진 캡처 (~3분) + 행동-y + stim-y 프로브. **행동-y AUC 가 cross-stim 역전**: Qwen 0.88→0.44, LLaVA 0.77→0.86, Idefics2 0.93→0.77, InternVL3 0.89→0.59. **Stim-y AUC 는 사진에서도 4 모델 모두 1.0 유지** (physical-shape vs abstract-shape). 인코더 식별 능력 균일은 cross-stim 에서 확인; 행동-y AUC 는 각 모델의 per-stim PMR 분포에 따라 변동하는 "인코더 ↔ 행동 정렬" 측정. 아키텍처 수준 재구성의 cross-stim 최종 확인. | ✅ | 2026-04-25 |
| **M6 r6** | **LLaVA-Next-Mistral 5번째 모델 점 (2번째 CLIP) + cross-stim 부록** | M8a (400 라벨 + 400 라벨-free + 400 stim × 4 레이어 × 5 타일 캡처). PMR(_nolabel) = **0.700, 95% CI [0.65, 0.74]**, LLaVA-1.5 바닥 [0.14, 0.21]과 saturated cluster [0.80, 0.92] 사이. 행동-y AUC 0.81; stim-y AUC = 1.0 4개 target 모두. **5-모델 M8a chain lock**. **Cross-stim** (1440 + 480 M8d + 180 + 60 M8c, ~16분): M8d PMR 0.625 [0.58, 0.67] mid-band 유지; M8c PMR 0.417 가 Idefics2 0.417 과 통계적 동일 (photo-collapse 일반화). **H7 cross-stim**: M8a +0.26 (5/5 PASS), M8d −0.05 (CI [−0.10, −0.01], noise floor), M8c +0.02 — 동일 encoder family architecture 스위치가 H7 약화시킴, PMR 헤드룸 있어도 (M8d PMR 0.625 천장 한참 아래). H-encoder-saturation **5 모델 점 + 2 CLIP 점 + 3 stim source 에서 architecture-level 확인**. H-LM-modulation 여전히 suggested-only (M8d 의 두-Mistral H7 ≈ 0 는 multi-axis-confounded). | ✅ | 2026-04-25 |
| **M8a** | **자극 다양화 — 비-원형 합성 shape** | Square / triangle / hexagon / irregular polygon × line/filled/shaded/textured × bg/cue grid; Qwen + LLaVA, labeled + label-free arms. **사전 등록 엄격 채점: Qwen 1/4, LLaVA 4/4** — 비대칭이 H-encoder-saturation 을 도형 간 검증. H1 + H7 은 unsaturated-only. Triangle (`wedge`) + polygon (`polygon`) 라벨 설계 약점으로 노출됨. | ✅ | 2026-04-25 |
| **M8d** | **자극 다양화 — 비-공 물리 객체 카테고리** | car / person / bird × line/filled/shaded/textured × bg/cue × `(fall, horizontal)` × 5 seeds. **사전 등록 엄격 채점: Qwen 0/3 H7 (binary, ceiling), LLaVA 3/3 H7 ✓ — 본 프로젝트 가장 강력한 카테고리 횡단 H7 증거.** Qwen 천장 아래에서 regime 분포는 17.5 % static (figurine) / 22.5 % static (statue). H1 양 모델 모두 실패 (도형-축 특정). H-encoder-saturation 카테고리 횡단 검증. 새 `classify_regime` keyword 분류기 (5.6 % 손 라벨링 오차). | ✅ | 2026-04-25 |
| **M8c** | **자극 다양화 — 실사진** | 60 사진 (12 × {ball, car, person, bird, abstract}) from COCO 2017 + WikiArt. **핵심 결과**: 사진이 Qwen PMR(_nolabel) 을 카테고리에 걸쳐 18-48 pp 낮춤 — 시각-포화 가설 정제: 행동 PMR 포화는 인코더 표현 신뢰 AND 입력-맥락 단순성의 결합. LLaVA H7 부분적 성립 (2/4 binary). LLaVA person 사진 PMR 합성 대비 +39 pp 상승 (인코더가 마침내 사람 인식). | ✅ | 2026-04-25 |
| **4.5** | **Cross-encoder swap (CLIP / SigLIP / DINOv2)** | H-encoder-saturation 의 인과적 counterfactual: LLaVA 의 CLIP-ViT-L 을 SigLIP 으로 교체 (Qwen 의 경우 그 반대). Encoder probe AUC 가 saturation 수준의 *원인* 인지 가장 깔끔한 검증. **§4.5 에서 promotion**. | ▶ **PRIORITY 4 (다음)** | — |
| **4.6** | **VTI 역방향 counterfactual 자극 생성** | Qwen2.5-VL post-processor `pixel_values` 위 픽셀-공간 gradient ascent 로 `<h_L10[visual], v_L10>` 를 최대화. **ε = 0.05 에서 5/5 v_L10 flip; 매칭 ε = 0.1 의 random direction 0/15 flip** — 방향 특이성이 "어떤 perturbation 이든" 대안을 falsify. Random-control 응답이 과도하게 허용적인 scorer 를 노출 (비대칭 수정 검증: v_L10 응답 0/20 이 새 abstract marker 와 매칭). v_L10 은 이미지에 인코드 가능; M5a shortcut 은 픽셀 기반 경로 위에 존재. | ✅ | 2026-04-26 |
| **4.10** | **Attention visualization UI** | Qwen2.5-VL 초기 릴리스 (notebook + heatmap overlay) + 5 VLMs 전체로 cross-model 확장 (동일 M8a stim). Visual token 에 대한 last-token attention 이 architecture-level 차이 표시 (Qwen ~17%, LLaVA-1.5 ~7%, Idefics2 ~30%, ...). | ✅ | 2026-04-25 |
| M5b | ST4 Phase 3 — SIP + patching + knockout + SAE | (1) Qwen2.5-VL 에서 SIP + activation patching: 20 SIP 페어 × 28 LM layer — sharp L10 boundary, IE=+1.0 at L0-L9, +0.6 at L10-L11, 0 at L14+. (2) LLaVA-1.5 cross-model SIP+patching (n=15 × 32 layer): lock-in L20 시작 (62.5% relative depth) vs Qwen L10 (36%) — 곡선 모양 재현, locus 가 shift. (3) Layer-level attention + MLP knockout: attention 모든 layer 에서 IE=0; **L9 MLP IE=+1.0 (유일 necessary)**; partial ring L8 +0.4 / L10 +0.6 / L11 +0.4 / L14 +0.4. (4) Per-head attention knockout (2026-04-27): 20 stim × 7 layer × 28 head = 196 (L,h) 모두 IE=0 — attention 이 *layer 와 head resolution 모두에서* fully redundant 확인. (5) **SAE intervention on Qwen vision encoder L31 (2026-04-27, 이번 라운드)**: 622K visual token 위 5120-feature SAE 학습; top-20 physics-cue feature (delta = mean_phys − mean_abs) 빼면 **PMR 완전 break (0/20 retain physics)**; matched-size random k=20 control (3 set) 가 **PMR 보존 (1/1/1.0)** — encoder level 의 깨끗한 direction-specificity; single top-1 feature dispensable; non-monotone mid-range (k=10 이 1.0 으로 회복, k=5 의 0.6 과 k=20 의 0.0 사이) 미해결이지만 random-control headline 흔들지 않음. Triangulation 완성: encoder ~20 SAE feature → L0-L9 visual token → L9 MLP construction → L10 read-out → letter. | complete ✅ (5개 하위 작업 모두 완료) | 2026-04-26..27 |
| M6 r3+ | ST5 round 3+ — encoder counterfactuals + LLaVA-Next | LLaVA-Next, InternVL3 captures, scale 변종 (Qwen 32B/72B), 다른 VLM family (Pixtral / Phi-V). | optional | — |
| **M9** | **일반화 audit — 논문 Table 1 (3 모델 × 3 stim, 부트스트랩 CI)** | M8a (5 도형) + M8d (3 카테고리) + M8c (5 사진 카테고리) × {Qwen, LLaVA, Idefics2} 를 9 (모델 × stim) 셀로 통합, 평균 PMR(_nolabel) + 평균 H7 차이에 95% 부트스트랩 CI. **헤드라인**: (1) 인코더 family 가 합성-stim PMR 천장 인과 driver (CI 완전 분리, 0.84-0.89 vs 0.18-0.33); (2) 사진이 인코더 갭 압축 (3 모델 모두 0.28-0.55 로 수렴); (3) H7 비포화 체제에서만 견고 (LLaVA M8a + M8d, CI > 0); (4) 포화 시 LM-modulation 시사만 (Idefics2 M8d CI 0 닿음). | ✅ | 2026-04-25 |
| **M-MP** (Pillar A) | **Multi-prompt cross-task generalization** | Track B Pillar A (`references/submission_plan.md`). **Behavioral coverage = 5-model × 4-prompt** (Phase 1 + Phase 2 + audit follow-up MCQ Phase 1 smoke + Phase 2 full = 28,800 추론). **Causal coverage = 2-model × 3 new prompts** (Qwen on `describe_scene` + `meta_phys_yesno` + `meta_phys_mcq`; Idefics2 on `describe_scene` + `meta_phys_yesno` + cell-2 SAE). **Behavioral 헤드라인 (5-model × 4-prompt, audit follow-up 2026-04-28 evening)**: H2 paired-delta 가 **19/20 (model × prompt) cell** 에서 양수; **Qwen × MCQ 가 예외** (Δ = −0.050, label-image-mismatch 상호작용). 7200 MCQ 추론 모두 100% parse rate. Saturation × prompt 상호작용 유지 (saturated → describe 가 가장 informative; unsaturated → open). **Causal 헤드라인 (2026-04-28 evening, post-MCQ + 2nd-cell)**: (i) **Qwen × MCQ 의 cross-method split** (이전: 단일 generative-vs-categorical 경계): audit-pinned cell 에서 M5a method 는 null (yesno 와 매칭: 0/10 flip vs describe + open 10/10) 그러나 M5b method 는 break (describe 와 매칭: 10/10 break under top-k=20 SAE ablation; random retains baseline). 통일된 framing 이 split — yes/no 의 M5b immunity 는 **yes/no-prompt-specific** 으로 기술 (n=1 categorical-binary prompt). (ii) **Idefics2 single-cell 부분 lift**: 2nd-cell test (`textured/ground/cast_shadow ball`) 의 baseline 이 이미 suspended ("in the air"), kinetic 아님 — top-k SAE ablation no-op (specificity 확인: SAE feature 가 kinetic-verb production 인코드). 그러나 cell-1 framing-shift 를 architecture-level 로 lift 하지는 못함. Phase 1+2+3 + audit MCQ + 2nd-cell 모두 ✅. | ✅ behavioral 5-model × 4-prompt, ✅ causal 2-model refined | 2026-04-28 (모든 phase) |
| **M-PSwap** (Pillar B) | **Idefics2 controlled projector-swap LoRA** | Track B Pillar B / G3 fix. Idefics2 perceiver-resampler 를 MLP projector 로 교체 (LoRA rank-32, encoder + LM 고정). §4.6 + M5b 재실행. **예측**: perceiver-resampler 가 인과적이면 Idefics2-MLP 가 §4.6 에서 LLaVA-Next 처럼 flip. **상태 (2026-04-29)**: feasibility spike 완료 — bypass 만으로는 FAIL (`38302ec` / `10bafd3`), full LoRA training 필수. LoRA infra 구축 (`d35d512` / `69634e7`): `src/physical_mode/lora/{idefics2_mlp_resampler.py, load_swapped.py}` + `scripts/m_pswap_{train,smoke,regression_eval,post_training,diagnose_nan,repro_nan_batch,discriminator}.py`. Smoke (50 step) PASS. Full training **step 1000 에서 NaN-blocked** (run `outputs/mpswap_run_20260429-033238/step1000`); diagnostic suite (`m_pswap_diagnose_nan_v2.py`) WIP. Fallback: B2 only + 문헌 기반 이론 주장. | **infra built / training NaN-blocked WIP** | week 4–5 |
| M7 | 인간 baseline + 논문 작성 | Prolific 20명 × 50 stim + ICLR 2027 / NeurIPS 2027 초안 | optional | — |

---

## 3.X Track B 우선순위 (현 진행 작업, 2026-04-28+)

Track B (ICLR 2027 primary, NeurIPS 2027 secondary) 4개 paper gap (G1-G4) → 3 pillar (Pillar A multi-prompt / Pillar B controlled architectural counterfactuals / Pillar C Marr 3-level reframing). 자세한 일정은 `references/submission_plan.md`, gap 매핑은 `references/paper_gaps.md` 참조.

| Pillar | 마일스톤 | 상태 | 주차 | 해결하는 gap |
|---|---|---|---|---|
| A | **M-MP Phase 1+2** Multi-prompt behavioral PMR (5 모델 × 3 prompt × 480 stim) | ✅ 2026-04-28 | 1–2 | G1 (single-task) — behavioral evidence |
| A | **M-MP Phase 3** Cross-prompt M5a + M5b (Qwen + Idefics2 × 3 prompt) — *causal* test | ✅ 2026-04-28; **Mixed (Qwen-specific gen-vs-cat)** | 3 | G1 — causal evidence |
| A | **M-MP audit follow-up** MCQ probe + Idefics2 2nd-cell test | ✅ 2026-04-28 evening; cross-method split (Qwen × MCQ M5a− M5b+); 2nd-cell partial lift | 3 | G1 audit caveats |
| B | **M-PSwap** Projector-swap LoRA on Idefics2 (perceiver → MLP) | **infra built; training NaN-blocked WIP (2026-04-29)** — feasibility spike 가 bypass-only FAIL 확인; full LoRA pipeline step 1000 에서 NaN; 진단 WIP | 4–5 | G3 (n=1 perceiver) |
| B | **M-LMSwap** LM-only-swap LoRA (CLIP+ViT-L × Vicuna-7B vs Mistral-7B) | planned | 5–6 | G2 (sparse non-Qwen) + G3 |
| C | **M-Marr** 논문 §6 을 3 Marr level (Computational PMR / Representational M3+M4 / Mechanistic M5a+M5b) 로 재구성 | planned (paper-side) | 7–8 | G4 (5-signature framing) |

성공 기준 / fallback 트리거: `submission_plan.md` §"Risk register" 참조. 핵심: 8주차 종료까지 Pillar B 결과 미산출 시 → M-PSwap drop, M-LMSwap 유지, M-Add6 + 문헌에 의존.



---

## 3. 진행 상태 (세부)

마일스톤별 deep dive 는 `docs/insights/m{N}_*.md` (영문). 본 섹션은 ✅ 완료 마일스톤마다 한 줄 요약 + insight 포인터, 그리고 forward-looking 항목의 짧은 plan 만 유지. 마일스톤 완료 시: `docs/insights/` 에 본문 작성 → `docs/CHANGELOG.md` 에 entry 추가 → 본 표 row 추가/갱신.

EN canonical: `references/roadmap.md` §3. 본 KO mirror 는 헤드라인 / 포인터 동기화만 수행한다.

### 완료 마일스톤

| 마일스톤 | 날짜 | 헤드라인 | Insight |
|---|---|---|---|
| **M0** Infra scaffold | 2026-04-24 | 패키지/스크립트/config/test/docs scaffold; uv sync + pytest pass | (코드 only) |
| **M1** ST1 Pilot (Qwen2.5-VL-7B) | 2026-04-24 | 480 추론; ground 효과 +36 pp (단일 인자 최대); FC 하 언어-prior 지배; H1 부분 / H2 강함 | `m1_pilot.md` |
| **M2** ST1 MVP-full (Qwen) | 2026-04-24 | 5축 factorial × 480 stim × 3 label × 2 prompt = 2880 추론; H1 monotone-but-saturated; H7 promoted; LM hidden capture for M4 | (`m4_logit_lens.md`, `m4b_label_free.md` 에 흡수) |
| **M3** ST2 Vision-encoder probing | 2026-04-24 | 8-layer SigLIP capture; **encoder boomerang** — encoder AUC ≈ 1.0 모든 layer (Qwen 행동 PMR 0.28-0.95 임에도) | `m3_encoder_boomerang.md` |
| **M4** ST3 LM logit lens / per-layer probe | 2026-04-24 | LM AUC 0.94-0.95 모든 layer (peak L20 = 0.953); label 이 L5 부터 physics-margin 주도 | `m4_logit_lens.md` |
| **M4b** Label-free prompt H2 null test | 2026-04-25 | Paired PMR(ball) − PMR(_nolabel) ≈ 0; circle = −0.065 → **H2 revised** (Qwen 비대칭 "circle override") | `m4b_label_free.md` |
| **M4c** Forced-choice label-free | 2026-04-25 | Qwen 이 FC 하에서 M4b 재현; open-vs-FC paired delta no-label = −0.131 (label-free H4 측정 가능) | `m4c_fc_label_free.md` |
| **M5a** ST4 VTI steering | 2026-04-24 | L10 α=40 이 `line/blank/none` 10/10 D→B flip — "object-ness" direction 인과 확인 | `m5_vti_steering.md` |
| **M5a-ext** VTI 후속 (neg α, label swap, 양방향성) | 2026-04-24/25 | Exp 1-2 ceiling artifact + label=ball B→A; **Exp 3** 가 `v_L10` 을 regime axis 로 reframe (+α kinetic / −α static / baseline D 임계 아래) | `m5a_ext_bidirection_and_label.md` |
| **M6 r1** Cross-model — LLaVA-1.5-7B | 2026-04-25 | ball +0.475, planet +0.244, circle +0.173 (모두 양수); H2 가 **visual-saturation 가설** 로 재개정 | `m6_cross_model_llava.md` |
| **M6 r2** 3-모델 + LLaVA captures + FC logit ratio | 2026-04-25 | InternVL3 saturated; LLaVA encoder AUC ~0.73 vs LM AUC ~0.75 — encoder 가 bottleneck. **신규 H-encoder-saturation 가설** | `m6_r2_cross_model.md` |
| **M6 r3** Idefics2 vision-encoder probe | 2026-04-25 | SigLIP-SO400M AUC 0.93 (peak L9 = 0.948) — 3-점 AUC↔PMR 사슬 종결 | `m6_r3_idefics2_probe.md` |
| **M6 r4** InternVL3 InternViT probe → 4-모델 사슬 | 2026-04-25 | InternVL3 AUC 0.89 / PMR 0.92; 비-CLIP cluster ≥ 0.88, CLIP 0.77 (apples-to-apples M8a-stim) | `m6_r4_internvl3_probe.md` |
| **M6 r5** M8c photo encoder probe — 4-모델 cross-stim | 2026-04-25 | 행동-y AUC cross-stim 역전; stim-y AUC 1.0 유지 — encoder 식별 능력 균일, architecture-level reframe | `m6_r5_m8c_photo_probe.md` |
| **M6 r6** LLaVA-Next-Mistral 5번째 모델 점 (2번째 CLIP) | 2026-04-25 | PMR 0.700 [0.65, 0.74] — LLaVA-1.5 floor 와 saturated cluster 사이; 5-모델 M8a chain lock | `m6_r6_llava_next.md` |
| **M6 r7** M2 cross-model — 5-모델 M2-stim apples-to-apples | 2026-04-26 | 5-model PMR(_nolabel) 사다리 0.18 → 0.99; H2 paired-delta 3가지 패턴; per-model `v_L10` 추출 | `m2_cross_model.md` |
| **M8a** 비-원형 합성 도형 | 2026-04-25 | 5 도형 × Qwen + LLaVA 엄격 채점: Qwen 1/4 PASS, LLaVA 4/4 — H1/H7 unsaturated-only | `m8a_non_circle_shapes.md` |
| **M8c** 실사진 | 2026-04-25 | 60 사진; **사진이 Qwen PMR(_nolabel) 을 18-48 pp 낮춤** — 합성-stim 단순성이 saturation 의 공동 인자 | `m8c_real_photos.md` |
| **M8d** 비-공 물리 객체 카테고리 | 2026-04-25 | car/person/bird × 추상화 × bg × cue: LLaVA 3/3 H7 ✓ (가장 강력한 카테고리 횡단 H7); Qwen 0/3 (천장) | `m8d_non_ball_categories.md` |
| **M8e** Cross-source paired analysis | 2026-04-25 | M8a + M8d + M8c 통합 heatmap (논문 Table 1 후보); cross-source PMR shift 확인 | `m8e_cross_source.md` |
| **§4.5** Cross-encoder swap (Idefics2) | 2026-04-25 | Idefics2-8b on M8a: Qwen 0.838 / LLaVA 0.175 / Idefics2 0.882 — encoder type 이 LM 무관하게 PMR 천장 결정. **H-encoder-saturation 인코더-family 인과 확인** | `encoder_swap_idefics2.md` |
| **M9** 일반화 audit (논문 Table 1) | 2026-04-25 | 9 (모델 × stim) 셀 × 부트스트랩 CI; 비-CLIP [0.80, 0.92] vs CLIP [0.14, 0.37] 합성에서 완전 분리; 사진 [0.18, 0.67] 수렴 | `m9_generalization_audit.md` |
| **§4.6** Counterfactual stim via VTI reverse | 2026-04-26 | 5/5 v_L10 flip at ε=0.05; 0/15 random-direction at matched ε=0.1 — **`v_L10` 이 이미지에 인코드 가능** | `sec4_6_counterfactual_stim.md` |
| **§4.6 cross-model** Pixel-encodability layer sweep | 2026-04-26 → 2026-04-28 | 5-model n=10 chain lock; **Idefics2 anomaly L5-L31 에서 해결** — perceiver-resampler 가 leading remaining 메커니즘 후보 | `sec4_6_cross_model_revised.md` |
| **§4.7** Decision-stability boundary | 2026-04-26 | `cue_level=both` 가 dominant 결정 stabilizer; saturated 모델은 ceiling 효과 | `sec4_7_rc_per_axis.md` |
| **§4.8** Qwen 7B vs 32B PMR scaling | 2026-04-28 | Aggregate PMR 0.926 ≈ 7B 0.931 — 5× 스케일링이 전체 PMR 못 움직임; `abstract_reject` (0.002 → 0.065) + label gap 절반 만 shift | `sec4_8_pmr_scaling.md` |
| **§4.10** Attention visualization UI (5-model) | 2026-04-25 | Last-token attention to visual tokens 가 architecture-level 차이 (Qwen 17%, LLaVA-1.5 7%, Idefics2 30%, ...) | `sec4_10_attention_viz.md` |
| **§4.11** H7 follow-up — regime distribution | 2026-04-26 | LLaVA-Next intermediate 3-way split on `person × exotic`; 5-model gradient = M9 H7 의 granular form | `sec4_11_regime_distribution.md` |
| **M5b** ST4 Phase 3 — SIP + patching + knockout + SAE | 2026-04-26..28 | (1) Qwen SIP+patching: sharp L10 boundary; (2) LLaVA-1.5 SIP locks at L20; (3) attention/MLP knockout: **L9 MLP uniquely necessary** (IE=+1.0); (4) per-head knockout: 196 (L,h) all IE=0; (5) SAE intervention: top-20 vision-encoder physics-cue feature → 0/20 retain physics. Round-2 cross-model SAE: Qwen + Idefics2 + InternVL3 break; LLaVA family NULL — encoder-vs-LM 메커니즘 dissociation lock. | `m5b_*.md` (7개 파일) |
| **M-MP** Multi-prompt cross-task generalization (Pillar A) | 2026-04-28 | Phase 1+2+3 + audit follow-up MCQ + Idefics2 2nd-cell. Behavioral 19/20 cell 양수; causal **cross-method split at Qwen × MCQ** (M5a− M5b+) | `m_mp_phase3_followup_2026-04-28.md` (+ Phase 1+2+3 experiments) |

### Forward-looking (남은 작업)

- **M-PSwap** (Pillar B) — Idefics2 projector-swap LoRA (perceiver → MLP). 상태: infra 구축 완료, training step 1000 NaN-blocked, diagnostic suite WIP. §3.X 행 + `submission_plan.md` §B + `paper_gaps.md` G3 참조.
- **M-LMSwap** (Pillar B) — LM-only-swap LoRA (Vicuna-7B vs Mistral-7B on CLIP+ViT-L). Planned, 5-6주차.
- **M-Marr** (Pillar C) — 논문 §6 을 3 Marr level 로 재구성. Paper-side, 7-8주차.
- **M7** — 인간 baseline (Prolific 20 raters × 50 stim) + ICLR 2027 / NeurIPS 2027 초안. Track B 14주차로 deferred. Plan: `docs/m7_human_baseline_plan.md`.
- **§4.4** — Michotte 2-frame causality (2-image prompt 지원 필요; 현재 out of scope).
- **H3 Scene-inconsistency × RC** — focused mini-experiment, M2 가 axis E 를 drop 한 후 미진행.

---

## 4. 원래 계획에 없던 추가 아이디어

연구 진행 중 떠올랐거나 `references/project.md` §2 에 없는 확장.

**다음-tier priority 로 promotion** (작업 상세는 §3 의 해당 섹션 참조):
- **4.5** Cross-encoder swap — M8a/c/d 후 priority 4 (H-encoder-saturation 의 인과 검증).
- **4.6** VTI 역방향 counterfactual 자극 생성 — ✅ 2026-04-26 (ε=0.05 에서 v_L10 5/5 flip; random 0/15; v_L10 픽셀에 인코드).
- **4.10** Attention visualization UI — priority 6.

나머지는 여전히 optional / 열린 아이디어.

Pilot 에서 떠오른, 혹은 연구계획 §2 에 없는 확장 방향. 선택적 — 각각 1-2 주 작업.

### 4.1 Block-stack을 별도 "abstract-physical" 경로로

현재 코드 (`primitives.py::_draw_block_stack`) 는 있으나 pilot 에서 미사용. 블록은 "추상적인 기하 + 명백한 물리" 조합이라 **원-공 축과 다른 질문**을 묻는다: "도형은 추상인데 구성(stacking)은 물리인 자극에서 VLM 이 어느 쪽으로 가는가?" → 예상: PMR 높음 + abstract_reject 낮음. 원-공 축의 컨트롤로 유용.

### 4.2 역과제 (reverse prompting) ✅ (2026-04-25)

프롬프트에 `"abstract"` 라벨을 *실제* 공 사진에 붙였을 때 PMR 떨어지는가?
H2 (언어 prior 지배) 의 counterfactual. **2026-04-25 기존 M8c labeled-arm
데이터 재사용으로 완료** (5 모델 × 3 라벨 역할 × 4 물리 사진 카테고리 ×
12 seed = 모델당 720). **헤드라인**: 실 물리 사진에서 image-prior 가
label-prior 를 지배 — 5 모델 모두 phys_minus_abs ≤ +0.146, vs LLaVA-1.5
M8d 합성 phys_minus_abs +0.306 (사진에서 라벨 효과 절반). **LLaVA-Next
phys − abs = 0.000** on 물리 사진: 실 공을 `"circle"` 이라 부르는 것이
`"ball"` 보다 PMR 낮추지 않음. 이미지 vs 라벨 trade-off 가 입력 측에서
본 saturation 효과: 풍부한 이미지 → 이미지 지배; 빈약한 이미지 → 라벨
지배. 전체 문서: `docs/insights/sec4_2_reverse_prompting_ko.md`.

### 4.3 Label 언어 전환 ✅ (2026-04-26, 5-model)

한국어 `"공"` vs 영어 `"ball"` 같은 stimulus 에서 PMR 차이? Qwen2.5-VL
multilingual; cross-model 이 일반화 검증.

**2026-04-26 5 VLMs × M8a circle 에서 완료 (모델 × 언어 × 라벨당
n=80)**: Qwen-only 헤드라인이 아래, cross-model 확장이 그 후.

Qwen2.5-VL (원래):

| Role | EN PMR | KO PMR | Δ |
|------|-------:|-------:|---:|
| ball / 공 | 0.81 | 0.85 | +0.04 |
| circle / 원 | 0.80 | 0.76 | −0.04 |
| planet / 행성 | 0.96 | 0.88 | −0.09 |

Cross-model EN→KO Δ (KO − EN), Korean-aware scorer (원래 영어-키워드
scorer 가 조용히 누락한 12개 한국어-only 응답 추가):

| Model | physical | abstract | exotic | mean |Δ| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | +0.04 | −0.04 | −0.09 | 0.06 |
| LLaVA-1.5  | **−0.19** | **+0.13** | +0.01 | 0.11 |
| LLaVA-Next | −0.05 | +0.04 | −0.04 | 0.04 |
| Idefics2   |  0.00 | **+0.11** | −0.05 | 0.05 |
| InternVL3  |  0.00 | −0.03 | −0.03 | 0.02 |

**헤드라인 (5-model)**:
1. **Cross-label ordering 4/5 모델에서 보존** (Qwen, LLaVA-1.5,
   LLaVA-Next, InternVL3). Idefics2 가 예외: KO 순서 `공 > 원 > 행성`
   vs EN `ball > planet > circle` — `행성` rank 가 `원` 아래로 떨어짐.
2. **LLaVA-1.5 swing 최대** (avg |Δ|=0.11; Vicuna LM 이 약한 한국어
   SFT). **InternVL3 swing 최소** (avg |Δ|=0.02; 천장 + 강한
   InternLM3 한국어 coverage).
3. 원래 Qwen-only multilingual 주장 살아남지만, cross-model 그림이
   **language-prior 축** 추가: LM 한국어 fluency 가 영어 라벨 prior 가
   얼마나 transfer 되는지 modulate, vision encoder 와 독립. 같은
   encoder + 다른 LM → 다른 한국어 magnitude.

문서: `docs/insights/sec4_3_korean_vs_english_ko.md`. Figures:
`docs/figures/sec4_3_korean_vs_english.png` (Qwen-only) +
`docs/figures/sec4_3_korean_vs_english_cross_model.png` (5-model).

**Japanese 확장 (2026-04-26, 같은 날)**: 같은 5-model 디자인을
Japanese 라벨 (ボール / 円 / 惑星) 로 추가. Korean 과 다른 mechanism
드러남: Qwen2.5-VL 가 Japanese 라벨 85-91% 유지 (진짜 Japanese
engagement); LLaVA-1.5 / LLaVA-Next / InternVL3 가 대부분 kanji 를
영어로 내부 번역; **Idefics2 가 `惑星` 에서 19/80 응답을 Chinese 로
fallback** (Mistral-7B 가 제한된 Japanese SFT 지만 kanji 를 Chinese
惑星 = planet 로 인식).

| Model | physical | abstract | exotic | mean |Δ| |
|-------|---------:|---------:|-------:|---------:|
| Qwen2.5-VL | **+0.13** | 0.00 | −0.01 | 0.05 |
| LLaVA-1.5  | −0.05 | +0.04 | +0.05 | 0.05 |
| LLaVA-Next | −0.03 | +0.10 | +0.04 | 0.05 |
| Idefics2   | −0.01 | +0.06 | +0.05 *| 0.04 |
| InternVL3  |  0.00 | −0.01 | −0.03 | 0.01 |

\* Idefics2 exotic +0.05 는 새 `CHINESE_PHYSICS_VERB_STEMS` lexicon 으로
정확히 점수된 Chinese-fallback 응답에서 옴. Fix 없으면 보이는 Δ 가
**−0.15** — 순수 scorer artifact.

Cross-language 요약:
- Korean 이 **language-fluency-bottleneck** 테스트 (Hangul 고립이
  engagement 강제; 4/5 ordering 보존; LLaVA-1.5 swing 0.11 이
  Vicuna-Korean 약점 진짜 측정).
- Japanese 가 **kanji-as-bridge** 테스트 (bootstrap noise 안에서 5/5
  ordering 보존, 그러나 mixed path 통해: 진짜 engagement (Qwen), 내부
  번역 (LLaVA-1.5), 또는 cross-script fallback (Idefics2)).
- LLaVA-1.5 ↓Korean / ≈Japanese 비대칭 (0.11 vs 0.05) 이 *Vicuna-
  Japanese 가 Vicuna-Korean 보다 강함의 증거가 아님* — script 의
  번역 가능성을 반영, LM SFT 깊이 아님.

**여전히 열림**: 중국어 / 스페인어; 완전 target-language 프롬프트
(단지 영어 템플릿에 라벨 삽입이 아닌).

### 4.4 Video frame pair → Michotte-style causality

두 프레임 (t=0, t=1) 에 객체 위치만 달라지는 쌍을 주고 "launched by X?" 질문. Michotte (1946) launching effect 가 VLM 에 나타나는가? 동영상 모델 필요 없이 2-image prompt 로 proxy 가능.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2) ⭐ promoted

"Vision encoder 가 CLIP 이면 안 보이는 cue 를 DINOv2 기반 모델은 본다" 가설. Eyes Wide Shut (Tong et al. 2024) MoF 제안의 연속선. 단, standalone encoder 는 LLaVA-1.5 (CLIP-ViT-L/14) vs Qwen2.5-VL (SigLIP) 의 자연스러운 비교로 이미 M6 에 포함.

**상태 (2026-04-25)**: 다음-tier priority 로 promotion — H-encoder-saturation (M6 r2) 이 현재 3-model correlational; 이건 인과 counterfactual. 작업 상세 §3 위 참조.

### 4.6 Activation 기반 counterfactual 자극 생성 ✅ (2026-04-26)

Qwen2.5-VL post-processor `pixel_values` (T_patches × 1176, patch-flattened normalized 표현) 위 픽셀-공간 gradient ascent 로 `<mean(h_L10[visual]), v_L10>` 최대화. 미분 불가능한 PIL → patch 전처리를 우회하면서, inverse permute + de-norm 으로 이미지 복원 가능. **ε = 0.05 에서 5/5 v_L10 flip** (사전 등록 ≥ 3/5); **매칭 ε = 0.1 의 random-direction 0/15 flip** — 방향 특이성이 "어떤 perturbation 이든 PMR 을 뒤집는다" 를 falsify. `v_L10` 은 이미지에 인코드 가능 — M5a shortcut 은 픽셀 기반 경로 위에 존재하지, runtime hidden-state injection 만의 속성이 아님. 깊이 분석: `docs/insights/sec4_6_counterfactual_stim.md` (+ ko).

### 4.7 결정 consistency 의 경계 측정 ✅ (2026-04-26)

Pilot 에서 T=0 이라 RC 측정 못했음. M2 RC (T=0.7 하) 를 **axis 별 결정
안정성** 으로 해석. **2026-04-26 5-모델 M8a label-free 에서 완료**.

**헤드라인**: `cue_level=both` 가 지배적 결정 안정자 (+9–16 pp RC), 3
saturated 모델 (Qwen 0.84→1.00, Idefics2 0.88→0.99, InternVL3 0.89→0.98).
LLaVA-1.5 + LLaVA-Next (CLIP encoder) 에서 반전/사라짐. bg_level=ground
가 보조 안정자 (+3–8 pp). object_level 가 가장 약한 안정자.

**해석**: saturation 이 단지 행동 PMR ceiling 만이 아니라 **결정-안정성
ceiling** 이기도 함 — non-CLIP 모델이 cue fire 시 5 seed 모두 같은 PMR
call 로 수렴. CLIP-기반 모델이 강한 cue 에서도 seed-level variance 보유.
H-encoder-saturation reframe 의 별도 시그니처.

문서: `docs/insights/sec4_7_rc_per_axis_ko.md`. Figure:
`docs/figures/sec4_7_rc_per_axis.png`.

### 4.8 PMR 스케일링

H-class (Qwen2.5-VL-7B/32B/72B), LLaVA-1.5-7B/13B 에서 모델 크기별 PMR. MechBench (Zhang et al. 2024) 의 "scale doesn't help" 주장이 PMR 에도 성립하는가? H6 에 강한 interpretability 함의.

### 4.9 Label 없이 프롬프트

`"What do you see? What might happen next?"` — "ball" 단어 **없이** 질문. H2 의 언어 prior 기여를 null-hypothesis 형태로 측정. 쉬운 추가 — `prompts.py` 에 `open_no_label` variant.

### 4.10 Attention visualization UI ✅ (2026-04-25)

Captured attention 으로 interactive heatmap (노트북 기반). Per-stimulus,
per-layer, per-head 의 visual token attention. 논문 appendix figure 용.

**2026-04-25 완료** (초기 release, Qwen2.5-VL 만):
- 신규 `configs/attention_viz_qwen.py` (M8a stim, limit=20, layer (5,15,20,25),
  `capture_lm_attentions=True`).
- `src/physical_mode/models/vlm_runner.py` — `attn_implementation` 가 캡처
  시 자동으로 `"eager"` 전환 (SDPA 는 attention weight 반환 안함).
- `notebooks/attention_viz.ipynb` — 6 섹션 인터랙티브 노트북: 캡처 로드,
  per-layer heatmap, 이미지 overlay, physics-vs-abstract 비교, per-head
  미세 구조, attention-entropy 집계.
- 캡처 비용: ~30초 + stim 당 ~7 MB.
- 전체 문서: `docs/insights/sec4_10_attention_viz_ko.md`.

LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3 으로 확장은 follow-up; 디스크
비용 ~2 GB (5 모델 × 60 record).

### 4.11 H7 follow-up — label-regime 범주 주석 ✅ (2026-04-26)

"라벨이 물리 regime 을 선택한다" 의 체계적 검증. **2026-04-26 완료
(5-모델 M8d, classify_regime 으로 kinetic / static / abstract /
ambiguous)**.

**헤드라인**: LLaVA-1.5 가 라벨로 regime 깔끔하게 선택 (`person × no
label` 40% kinetic + 40% static; `person × physical` 62% kinetic;
`car × abs / silhouette` 28% kinetic + 70% ambiguous). Qwen + Idefics2
+ InternVL3 는 saturated kinetic 모든 곳에서, `person × exotic`
(statue) 만 예외: Qwen ~30% static, **InternVL3 ~65% static** (프로젝트
에서 가장 강한 단일 라벨-driven static commit — 라벨이 uniquely
disambiguate 할 때 saturated-encoder architecture 도 언어에 deferred).
LLaVA-Next intermediate, `person × exotic` 에서 3-way split (30%
kinetic + 25% static + 25% abstract). 5-모델 gradient 가 M9 H7 finding
의 granular form.

문서: `docs/insights/sec4_11_regime_distribution_ko.md`.
Figure: `docs/figures/sec4_11_regime_distribution_5model.png`.

**여전히 열림**: M2 circle-only 데이터의 5-카테고리 fine-grained regime
(gravity-fall / gravity-roll / orbital / inertial / static); M8a
5-도형 classify_regime 확장 (도형별 신규 키워드 셋 필요).

---

## 5. 작업 시 참조 규칙

**각 새 세션 시작 시** (future Claude 또는 user):

1. 이 파일 (`ROADMAP.md`) 을 제일 먼저 읽어 "지금 어느 M에 있나" 확인.
2. 현재 M 의 **성공 기준** 과 **블로킹 이슈** 를 체크.
3. 세부 기술 질문은 `docs/00_architecture.md` → `docs/04_next_steps.md` 순서로 drilldown.
4. 최신 실험 결과가 필요하면 `docs/03_run_log.md` (수치) + `docs/0X_*.md` (각 milestone 심층 인사이트).

**Insights 파일 규칙**: 각 주요 마일스톤 완료 시 `docs/0X_<milestone>_insights.md` 를 한 개 작성한다 — 한국어, 원본 수치 링크 + 해석 + 가설 스코어카드 업데이트 + paper implications. 현재 트리:
- `docs/05_insights.md` — M1 pilot insights (legacy 이름)
- `docs/06_m3_insights.md` — M3 encoder boomerang
- `docs/07_m4_insights.md` — M4 LM logit lens
- `docs/08_m5_insights.md` — M5 VTI steering causal intervention
- (M5b, M6 ... 추가 예정)

**마일스톤 하나를 완료할 때마다**:

- §2 표의 상태 컬럼 업데이트 (▶ → ✅) + 완료일 기록.
- §3 해당 마일스톤 섹션에 "검증된 사실 / 블로킹 해결 / 새 가설" 적는다.
- 새 가설이 도출되면 §1.3 스코어카드에 H* 추가.
- `docs/03_run_log.md` 에도 run 단위 entry 추가 (이 파일과 별개).

**가설이 반박되거나 수정될 때**:

- §1.3 에서 상태 변경 + 이유 한 줄.
- 심한 수정은 `research_plan.md` 원본이 아니라 이 ROADMAP 에만 기록 (원본은 읽기 전용 스펙).

**새 아이디어가 떠오를 때**:

- 즉시 §4 에 번호 붙여 추가. 이후 M2-M6 중 어디에 끼울지 / 독립 M 으로 띄울지는 나중에.

---

## 6. 변경 이력

전체 chronological changelog (project-level — 마일스톤, 가설 flip, 우선순위 재배치, load-bearing infra) 는 **`docs/CHANGELOG.md`** (영문 canonical). 신규 entry 는 거기에 append 하고 본 파일에는 추가하지 않음.

작업 시: `docs/CHANGELOG.md` 마지막에 (날짜, 한 단락 요약, commit hash) row 추가. 관련 `docs/insights/m{N}_*.md` 와 cross-link, 마일스톤 완료면 본 파일 §3 표에도 row 추가/갱신.
