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

연구계획 §2.2의 원래 H1-H3 + pilot 에서 도출된 H4-H6. pilot 결과는 `docs/05_insights.md` 근거.

| ID | 가설 | 상태 (post-M5a-ext recheck) | 근거 / 다음 검증 |
|---|---|---|---|
| **H1** | PMR이 추상화 축(line → textured)에 따라 S자형 증가; 3D 음영·지면 도입이 가장 큰 단계 증가. | **지지, unsaturated-only AND 도형-축 특정 (M8a + M8d)** | M2 (Qwen): 4개 object_level monotone (0.744 → 0.832) 이지만 saturated. M6 (LLaVA-1.5, 2026-04-25): 깔끔한 S-curve 0.51 → 0.81. **M8a (2026-04-25)**: 도형 간 엄격 채점 — Qwen 3/5 (square/triangle 실패; ceiling-effect 압축), LLaVA 4/5 (polygon 만 filled→shaded 역전으로 실패). **M8d (2026-04-25)**: 카테고리 간 엄격 채점 — Qwen 0/3 (천장), LLaVA 0/3 (비단조). H1 은 기하-도형 ↔ 명명-객체 축의 속성: car/person/bird 의 모든 추상화 레벨이 이미 카테고리 인식 가능 → 시각 디테일이 affordance 를 바꾸지 않음. ramp 는 unsaturated 인코더 AND 추상-도형 ↔ 물리-객체 축일 때만 측정 가능. |
| **H2** | "ball" 라벨은 선화에서도 PMR을 크게 증가시킨다 → 언어 prior 독립 기여. | **세 점으로 완전 검증 + encoder-anchored** | Qwen (saturated, M4b): ball/planet ≈ 0, circle = −0.065. LLaVA (unsaturated, M6 r1): ball +0.475, planet +0.244, circle +0.173. InternVL3 (super-saturated, M6 r2a): 모든 label +0.010 ≈ noise. 3-model paired-delta 패턴이 encoder-saturation 예측과 일치. M6 r2b 가 saturation 차이가 vision encoder probe AUC (Qwen 0.99 vs LLaVA 0.73) 에 뿌리내림을 보임. M4b 의 "circle suppression only" 패턴은 encoder 가 이미 saturated 인 Qwen 특이적 증상. |
| **H3** | 장면 불일치는 RC를 저하시킨다. | **미검증** | axis E 는 M2에서 빠짐 (complexity); 별도 mini-실험으로 처리. RC 인프라는 M2에서 검증됨 (103/288 cells RC<1). |
| **H4** (pilot-derived) | Open vs forced-choice PMR gap 은 **언어 prior ↔ 시각 증거** 충돌의 안정적 signature다. | **지지 — 확장** | M2: gap이 모든 object_level에 존재 (line 32pp → textured 22pp). 추상도 ↑ 일수록 gap ↑ — abstraction 이 vision 증거를 약화시켜 언어가 더 지배한다는 structural prediction. 다음 검증: ST5 cross-model. |
| **H5** (pilot-derived) | 지면 한 줄(ground line) 단독이 텍스처 공 + no ground 보다 **더 큰** PMR 증가를 만든다. | **혼재** | M2: bg delta (blank 0.67 → scene 0.88 = +21pp) > object delta (line 0.74 → textured 0.83 = +9pp). 방향은 맞음; 단 scene 이 ground 를 또 넘음. |
| **H6** (pilot-derived) | arrow+shadow cue의 포화는 **cast shadow 단독**으로도 일어나며, arrow는 annotation에 가깝다. | **지지 (수정)** | M2 분해: cast_shadow 단독 = +17.5 pp above none (Kersten 지면 부착 cue 확인); **그러나 arrow 도 단독으로 0.96 에 saturate** — "arrow 는 annotation" 부분은 반증. Arrow 가 dominant cue, shadow 가 secondary. |
| **H7** (M2-derived) | 라벨은 PMR 을 toggle 하는 것이 아니라 **어떤 물리 regime** 을 선택한다. | **지지, unsaturated-only AND 카테고리 횡단 (M8a + M8d)** | M2 GAR: ball 0.79 / circle 0.70 / planet 0.48. M5a-ext Exp 2: `line/blank/none × +α=40` 에서 label flip 으로 B↔A swap. M6 r1 + r2a cross-model: `planet GAR << ball/circle GAR` 가 Qwen (0.32 vs 0.71/0.75), LLaVA-1.5 (0.07 vs 0.36/0.15), InternVL3 (0.43 vs 0.82/0.79) — circle-only. **M8a (2026-04-25)**: 도형 간 role-PMR 엄격 채점 — Qwen 1/5 (square 만; 나머지 -0.10 ~ +0.075 = ceiling-flat), LLaVA 4/5 (triangle 만 +0.025 실패; `wedge` 가 약한 physical 라벨, 도형 실패가 아님). H7-GAR 엄격: Qwen 1/5, LLaVA 5/5. **M8d (2026-04-25)**: 카테고리 횡단 role-PMR 엄격 채점 — LLaVA **3/3** (car +0.525, person +0.138, bird +0.550 PMR_regime physical−abstract; 본 프로젝트 가장 강력한 카테고리 횡단 H7 증거). Qwen 0/3 binary (천장) 이지만 regime 분포가 동일 패턴: figurine 17.5 % static, statue 22.5 % static. label-selects-regime 클레임이 이제 카테고리 일반, circle-특이가 아님. |
| **H-boomerang** | Encoder 는 알고, decoder 가 gate: vision encoder 가 physics-mode class 를 linear 로 분리하지만 behavior 는 실패. | **Qwen 특이적 (revised)** | Qwen2.5-VL 에서 성립: M3 encoder AUC ~0.99 모든 layer; M4 LM AUC ~0.94 visual tokens; behavioral PMR ~0.93 — 작은 "encoder knows, decoder mildly gates" gap. M5a: L10 causal intervention 이 behavior flip. **LLaVA-1.5 에서 반증** (M6 r2b): vision encoder AUC ~0.73, LM AUC ~0.75, behavioral ~0.78 — pipeline 평탄, encoder 가 bottleneck. Boomerang 현상은 encoder saturation 필요. |
| **H-encoder-saturation** (M6 r2 도출; M8c 정련; §4.5 인과; M9 부트스트랩; M6 r4 + apples-to-apples M8a-stim; **stim-y 검증이 좌표를 아키텍처 수준 (인코더 + LM) 으로 이동**; **M6 r6 가 2번째 CLIP 점 추가**) | 합성 stim 의 행동 PMR(_nolabel) 포화는 **아키텍처 수준** (joint 인코더 + LM) 에서 결정, 인코더 표상 능력 수준 아님. **Stim-정의 y 검증 (2026-04-25)**: 5 테스트 인코더 모두 (SigLIP, CLIP-ViT-L ×2, SigLIP-SO400M, InternViT) factorial 셀을 AUC = 1.0 으로 선형 분리 (rendered_vs_line, physics_cell_vs_abstract_cell, within-object-level 최소-짝). 인코더 식별 능력은 균일; 차이는 인코더 출력의 LM-side 소비. 5-모델 행동 PMR 사다리 (비-CLIP: 0.84-0.92; CLIP-LLaVA-1.5: 0.18; CLIP-LLaVA-Next: 0.70) 와 행동-y probe AUC (0.77-0.93) 는 각 LM 의 인코더 출력 → physics-mode 신호 읽기를 반영 — *downstream-conditional*, 인코더-정보 아님. **M9 부트스트랩 CI** (3 모델 × 3 stim): 비-CLIP CI [0.80, 0.92] vs CLIP [0.14, 0.37] 합성에서 — 완전 분리; 사진에서 3 모두 [0.18, 0.67] 로 수렴. **M6 r6 (2026-04-25)**: 2번째 CLIP 점 (LLaVA-Next, Mistral LM, AnyRes) PMR 0.70 [0.65, 0.74] — CLIP-as-encoder 설명 배제; LLaVA-1.5 → LLaVA-Next 0.52-PMR 점프는 4축 confound (AnyRes / projector / training / LM family) — architecture-level reframe 과 부합하나 LM 분리 불가. | **5 모델 점 (3 비-CLIP + 2 CLIP) 에서 architecture-level 확인 + cross-stim 부트스트랩 검증 (M9 + M6 r3-r6)** | M6 r2b: Qwen vision AUC 0.99 / behavioral PMR(_nolabel) 0.95; LLaVA-1.5 AUC 0.73 / behavioral 0.38; InternVL3 미캡처지만 behavioral PMR(_nolabel) 0.99 (포화 프로파일 일치). **M8a (2026-04-25)**: 도형 간 paired-delta — Qwen 5/5 0 근처/음수; LLaVA 5/5 ≥+0.125. **M8d (2026-04-25)**: 카테고리 간 paired-delta — Qwen +0.000 / +0.025 / +0.125; LLaVA +0.275 / -0.100 / +0.262. **M9 (2026-04-25)**: 3-모델 × 3-stim 부트스트랩 CI: 비-CLIP [0.800, 0.917] CLIP [0.140, 0.371] 분리; 사진 [0.183, 0.667] 수렴. **M6 r3 (2026-04-25)**: Idefics2 SigLIP-SO400M probe AUC 0.93 (4 레이어 평균). **M6 r4 (2026-04-25)**: InternVL3 InternViT probe AUC 0.89 / PMR 0.92 — 4-점 사슬이 포화를 비-CLIP-일반화. **M6 r6 (2026-04-25)**: LLaVA-Next CLIP-ViT-L 행동-y AUC 0.81, stim-y AUC 1.0, PMR 0.70 [0.65, 0.74] — 5-모델 chain lock. |
| **H-LM-modulation** (M9-derived, 2026-04-25) | 인코더 포화 시 LM family 가 잔여 H7 측정 가능성을 조절할 수 있음 — Mistral-7B (Idefics2) 가 M8d 에서 H7 평균 +0.048 [+0.000, +0.094] 보임 vs Qwen2-7B (Qwen) 동 stim 평균 +0.008 [−0.033, +0.052]. | **시사만 — M8d CI 0 닿음, M8a/M8c 완전 중첩** | M9 부트스트랩 (5000 iter × 9 셀): Idefics2 M8d H7 CI 가 0 에 겨우 닿음; Qwen M8d H7 CI 는 0 가로지름. 33-pt PASS 율 갭 (0.667 vs 0.333) 은 단일 도형 (`car`: +0.025 vs +0.094) 가 strict 임계 가로지른 결과. 현 데이터로는 논문 옹호 불가; 동일-인코더 LM 스왑 또는 3-5× 더 많은 도형 필요. |
| **H-locus** (M4-derived) | Bottleneck 은 LM final layers + decoding head 에 있으며 그 이전은 아님. | **지지 (early-mid sweet spot)** | M5a: L10 α=40 은 10/10 abstract → physical 응답을 flip; 후반 layer 들은 움직이지 않음. M5a-ext Exp 3: L10 regime-flip (sign 으로 A vs B) 이 모든 cell 에서 성립. Basu et al. 2024 의 early-layer constraint-storage 결과와 정합. |
| **H-direction-bidirectional** (M5a-ext, 2026-04-24; 개정 2026-04-25) | `v_L10` 은 단순 bidirectional concept axis 로, −α 가 physics-mode 를 abstract 로 억제한다. | **revised — physics-mode 내부의 regime axis** | Exp 1 (textured/ground/both ceiling): −α 효과 없음 → 초기 "one-way activator" 프레이밍. Exp 3 (textured/blank/none moderate, 2026-04-25): −α=40 이 (line, textured) × (ball, circle) 모두에서 D → B ("stays still") 를 균일하게 유도. α 의 sign 이 regime 을 선택 (+kinetic / −static); baseline D 는 \|α\| threshold *아래* 에 위치 (axis endpoint 가 아님). |
| **H-regime** (M5a-derived) | Steering direction 은 binary "object-ness"이고 physics regime 은 label 이 결정. | **원래 형태로는 반증** | Kinetic vs static 이 이미 sign-selected regime 이어서 H-direction-bidirectional 해석으로 대체. `line/blank/none × +α=40` 의 좁은 label-driven flip (Exp 2) 은 H7 qualifier 로 흡수. |

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
| M2 | **ST1 MVP-full** (pilot 교훈 반영) | axis C 분해, axis D 확장, T=0.7, LM hidden-state capture, 2880 inferences | ✅ | 2026-04-24 |
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
| **4.6** | **SAE / VTI 역방향 counterfactual 자극 생성** | 학습된 방향 (M5a v_L10 또는 SAE feature) 을 사용하여 모델 시각으로 physics-mode 를 maximize 하는 자극을 gradient-ascent 로 합성. M5a 의 adversarial / shortcut-revealing 확장. **§4.6 에서 promotion**. | ▶ **PRIORITY 5 (다음)** | — |
| **4.10** | **Attention visualization UI** | Qwen2.5-VL 초기 릴리스 (notebook + heatmap overlay) + 5 VLMs 전체로 cross-model 확장 (동일 M8a stim). Visual token 에 대한 last-token attention 이 architecture-level 차이 표시 (Qwen ~17%, LLaVA-1.5 ~7%, Idefics2 ~30%, ...). | ✅ | 2026-04-25 |
| M5b | ST4 Phase 3 — SIP + patching + SAE | Semantic Image Pairs + activation patching (attention 필요 → re-capture) + SAE feature decomposition. | optional | — |
| M6 r3+ | ST5 round 3+ — encoder counterfactuals + LLaVA-Next | LLaVA-Next, InternVL3 captures, scale 변종 (Qwen 32B/72B), 다른 VLM family (Pixtral / Phi-V). | optional | — |
| **M9** | **일반화 audit — 논문 Table 1 (3 모델 × 3 stim, 부트스트랩 CI)** | M8a (5 도형) + M8d (3 카테고리) + M8c (5 사진 카테고리) × {Qwen, LLaVA, Idefics2} 를 9 (모델 × stim) 셀로 통합, 평균 PMR(_nolabel) + 평균 H7 차이에 95% 부트스트랩 CI. **헤드라인**: (1) 인코더 family 가 합성-stim PMR 천장 인과 driver (CI 완전 분리, 0.84-0.89 vs 0.18-0.33); (2) 사진이 인코더 갭 압축 (3 모델 모두 0.28-0.55 로 수렴); (3) H7 비포화 체제에서만 견고 (LLaVA M8a + M8d, CI > 0); (4) 포화 시 LM-modulation 시사만 (Idefics2 M8d CI 0 닿음). | ✅ | 2026-04-25 |
| M7 | 인간 baseline + 논문 작성 | Prolific 20명 × 50 stim + EMNLP/NeurIPS 초안 | optional | — |

---

## 3. 진행 상태 (세부)

### M0 — 인프라 스캐폴드 ✅ (2026-04-24)

완료 항목:
- `src/physical_mode/` 모듈 (config, utils, stimuli, models, inference, metrics, probing scaffold).
- `scripts/0{1,2,3}_*.py` argparse runner.
- `configs/{pilot,mvp_full}.py` — config-as-code.
- `tests/` — 35 개 (stimulus determinism + PMR scoring regression).
- `docs/00-05` — architecture / spec / rubric / run log / next-steps / insights.
- `notebooks/demo.ipynb` — 32-cell walkthrough with cached outputs.
- CLAUDE.md, README.md, pyproject.toml (cu130 index), .gitignore, uv.lock.
- 프로젝트 repo: private https://github.com/namam3gy/physical-mode-activation.

성공 기준 (모두 충족):
- `uv sync` 성공, `uv run python -m pytest` 통과.
- `scripts/01_generate_stimuli.py --config configs/pilot.py --limit 10` 성공.
- 파일럿 inference + score pipeline end-to-end.

### M1 — ST1 Pilot (Qwen2.5-VL-7B) ✅ (2026-04-24)

실행: `uv run python scripts/02_run_inference.py --config configs/pilot.py`.
출력: `outputs/pilot_20260424-072418_2c16efb6/` — 480 predictions, 8 분 wall clock.

**헤드라인 발견** (`docs/05_insights.md` §2, §3):

| 관찰 | 수치 | 함의 |
|---|---|---|
| Ground 유무 효과 | blank 0.49 → ground 0.85 (+36pp) | **단일 최대 요인**. 지면이 가장 저렴하고 강한 physics trigger. |
| Abstraction endpoints | line 0.58 → textured 0.81 | H1 부분 지지; 중간 2 수준 tie. |
| Arrow+shadow cue | 1.000 (포화) | 측정 불가 cell — MVP-full에서 분해 필요. |
| Wind cue | 0.513 ≈ none 0.500 | VLM에 안 읽힘 — 자극 교체 필요. |
| Open vs forced-choice | PMR 0.80 vs 0.54, abstract_reject 0.00 vs 0.45 | **언어 prior 지배성** — H2 강지지. |

**가설 스코어**: H1 부분, H2 강지지, H3 미검증, H4-H6 후보 도출.

**검증된 인프라 특성**:
- `PhysModeVLM`이 `AutoModelForImageTextToText` generic 로더로 Qwen2.5-VL에서 작동. ST5의 모델 스왑이 config 변경만으로 가능.
- `predictions.jsonl` streaming flush가 crash-safe.
- Factorial 축 중 **event_template**이 behavioral output에 영향 없음 → MVP-full에서 downgrade.

### M2 — ST1 MVP-full ✅ (2026-04-24)

실행: `uv run python scripts/02_run_inference.py --config configs/mvp_full.py`.
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/` — 2880 predictions, 55 분 wall clock, 5.2 GB LM activations.

**성공 기준 결과** (상세: `docs/03_run_log.md` M2 항목):

| criterion | status |
|---|---|
| Monotone S-curve over object_level (forced-choice) | ✅ 0.583 < 0.647 < 0.711 < 0.714 |
| Open-vs-forced gap at every object_level | ✅ 22-32 pp, abstraction 과 양의 상관 |
| cast_shadow alone > none + 20 pp | ✅ +17.5 pp 평균 (blank 조건에서 +23) |
| RC < 1 cells exist (T>0 확인) | ✅ 103/288 (36%) cells RC<1 |
| `outputs/*/activations/` 채워짐 | ✅ 480 safetensors, 5 layers, bf16 hidden states |

**새로 도출된 헤드라인**:
1. 추상화 axis monotone S-curve 이제 깨끗히 확인 (H1).
2. 라벨이 물리 regime 을 선택 — 같은 이미지에 `circle/ball/planet` → static / rolls / orbits the Sun (H7, 신규).
3. cast_shadow 단독이 +17.5 pp; arrow 가 dominant cue (H6 수정).
4. Open-ended PMR 0.93, abstract_reject 0.002 (3/1440) — 언어 prior 지배성 재확인 + 확장 (H4).

**블로킹 해결**: 
- primitives motion-trail 은 결국 미사용 (axis C 재설계가 wind 축 자체를 대체).
- axis E (scene consistency) 는 M2에서 제외; `docs/04_next_steps.md` 에 별도 mini-experiment 로 이관 예정.
- `capture_lm_attentions=False` 플래그가 성공 — disk 가 예상의 1/3 (5.2 GB vs 15+ GB if attentions on).

### M3 — ST2 Vision encoder probing ▶ (다음 마일스톤)

**설정 변경** (`configs/mvp_full.py`를 pilot 기반으로 재작성):

1. **Axis C 재설계**: `("none", "cast_shadow", "motion_arrow", "both")` — arrow/shadow 분해로 H6 검증.
2. **Wind cue 교체**: `draw_wind_marks` 를 `draw_motion_trail` (blurred afterimage 또는 dust trail)로 재구현. 새 primitive 추가.
3. **Axis D 확장**: `("circle", "ball", "planet")` — H2/H4 정량화.
4. **Event template 축 접기**: `fall` 1개로 고정 (behavior 차이 없음). 그 용량을 seeds로 재투자.
5. **Temperature 0.7**: RC 측정 가능하게. seeds_per_cell ≥ 10.
6. **Activation capture 활성화**: `capture_lm_layers=(5, 10, 15, 20, 25)`. `capture_vision_layers` 은 `vlm_runner.py`에 아직 없으므로 ST2 전까지 LM 만.
7. **Axis E (scene consistency)** 최소 2 수준: 일관 vs 불일치(예: 사진 공 + 선화 배경). H3 검증.

**스코프 예산**: 4 object × 3 bg × 4 cue × 3 label × 10 seeds × 2 prompt ≈ 2880 cells × 2 prompts ≈ **5 760 inferences**. axis E 추가 시 ×2 = **11 520**. H200에서 1.0-1.5 it/s 이면 3-4 시간. Activation capture 포함 시 5-6 시간 + 디스크 ~8 GB.

**성공 기준**:
- [x 체크 예정] behavioral S-curve (forced-choice) 가 4개 수준에서 monotone 증가 (H1 깨끗히 검증).
- [체크 예정] Open vs forced gap 이 모든 object_level 에서 나타남 (H4 확인).
- [체크 예정] `cast_shadow` 단독이 PMR > `none` + 0.2 이상 (H6 최소 조건).
- [체크 예정] RC < 1 인 cell 존재 (T>0 이 제대로 작동).
- [체크 예정] `outputs/mvp_full_*/activations/` 에 LM hidden states 저장 확인.

**블로킹 이슈**:
- 현재 `src/physical_mode/stimuli/primitives.py` 에 motion-trail drawer 없음 → 추가 필요.
- 현재 `FactorialSpec` 에 axis E (scene_consistency) 없음 → 추가 필요.
- 기존 pilot config 와 호환성은 신경 쓸 필요 없음 (새 config 하나 만들면 끝).

### M3 — ST2 Vision encoder probing ✅ (2026-04-24)

실행: `uv run python scripts/04_capture_vision.py --stimulus-dir inputs/mvp_full_... --output-dir outputs/mvp_full_.../vision_activations --layers 3,7,11,15,19,23,27,31`
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/vision_activations/` (480 safetensors, 12 GB) + `probing_vision/*.csv`

**핵심 발견** (상세: `docs/03_run_log.md` M3 항목):

- **Encoder AUC = 1.00 on every factorial axis, from layer 3 onward**. Vision 인코더는 bg/object/cue 모든 속성을 완벽히 인코딩. 정보 병목 없음.
- **Behavioral forced-choice PMR은 0.28 (cue=none) ~ 0.95 (both)** 범위. LM 의 gating 이 gap을 만든다.
- **Controlled no-cue subset (120 stimuli)**: encoder AUC 0.89 vs behavioral PMR 0.28 → encoder 가 "어떤 cells 가 physics-mode 를 trigger할지" 를 알지만 LM 은 일부만 통과시킴.
- **Per-object-level encoder AUC ~0.95 constant while behavior 0.58-0.71**: gap 은 line (가장 추상) 에서 +36 pp 로 최대 — H4 (추상도 ↑ ⇒ 언어 prior ↑) 의 내부 메커니즘 증거.

**가설 업데이트**:
- H-boomerang (원래 §1.4 의 "encoder knows, decoder doesn't"): **지지 (증거 포화)**.
- H4, H6 모두 mechanism-level 증거 확보.

**블로킹 해결 / 소득**:
- `PhysModeVLM.capture()` 에 vision hook 구현 완료 (`_resolve_vision_blocks` 헬퍼가 Qwen/LLaVA/InternVL 모두 커버).
- 프로그램 자극이 encoder AUC 1.0 을 trivially 만든다는 methodological caveat 기록. 포토리얼 stimulus 확장이 검증 단계로 필요 — §4 연동.

### M4 — ST3 LM backbone logit lens / layer-wise probing ✅ (2026-04-24)

실행: `uv run python scripts/05_lm_probing.py --run-dir outputs/mvp_full_20260424-094103_8ae1fa3d`.
출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_lm/*.{csv,parquet}`.
심층 인사이트: `docs/07_m4_insights.md`.

**핵심 결과**:
- LM per-layer probe AUC (forced-choice PMR) = **0.94-0.95 across all layers**, peak L20 = 0.953.
- Logit lens: physics logit > geometry logit from L5 onwards because "ball" label primes the LM.
- Object_level 효과 (L25 line 3.76 vs textured 4.35, margin +0.6) 는 **label effect (+4.0 전체 shift)** 의 ~14%.
- Switching-layer 메트릭은 label-primed 프롬프트에서 무력화됨 (모두 L5) → §4.9 "label 없는 프롬프트" 테스트를 M5 전 mini-실험으로 승격.

**Boomerang 정확한 위치**: vision encoder (0.94-1.0) → LM hidden (0.95) 은 정보 보존. Decoding 단계에서 ~29 pp accuracy 손실 발생. ST4 의 개입 우선순위는 LM final layers + logit head.

### M5a — ST4 Phase 1+2 VTI steering ✅ (2026-04-24)

실행:
- Phase 1: inline Python driver, `compute_steering_vectors` from `src/physical_mode/probing/steering.py`
- Phase 2: `uv run python scripts/06_vti_steering.py --run-dir outputs/mvp_full_... --test-subset line/blank/none --label circle --layers 10,15,20,25 --alphas 0,5,10,20,40`

출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/` (vectors) + `steering_experiments/` (intervention predictions). 심층 인사이트: `docs/08_m5_insights.md`.

**핵심 결과**:
- Steering vectors `v_L = mean(h|PMR=1) - mean(h|PMR=0)`. Norm 이 layer 통해 5× 증폭 (L5: 5.9 → L25: 31).
- Projection@L20 이 factorial cue axis 와 정렬 (none 22.3 → both 42.7).
- **Layer 10 α=40 주입 → 10/10 `line/blank/none` 응답이 "D: abstract" → "B: stays still" 로 flipping**. L15/L20/L25 는 같은 α 로 flipping 없음.
- Intervention 은 "abstract → physical object" 의 binary shift 를 만듦. "A: falls" 아닌 "B: stays" 로 감 → direction 은 object-ness, not gravity. H7/H-regime 일관.

**가설 업데이트**:
- H-boomerang: 확장 + **인과 지지**
- H-locus: **지지 (early-mid layer L10)**
- H-regime (신규): **후보** — steering direction 은 coarse "object-ness", regime 선택은 label-driven.

### M5a-ext — VTI 후속 ✅ (2026-04-24, 2026-04-25)

실행: `uv run python scripts/06_vti_steering.py` + `--output-subdir` flag 로
같은 M2 output tree 안에 sub-experiment 를 분리.

출력: `outputs/mvp_full_20260424-094103_8ae1fa3d/steering_experiments/{neg_alpha_textured_ground_both, ball_line_blank_none, bidirectional_recheck_*}/`.
심층 인사이트: `docs/insights/m5a_ext_bidirection_and_label_ko.md`. 원자료:
`docs/experiments/m5a_ext_neg_alpha_and_label_ko.md`.

**핵심 결과**:
- Exp 1 (ceiling): `textured/ground/both × circle` 에 `-α · v_L10` → first-letter
  10/10 A 유지. 초기 "one-way activator" 해석 — 재검정 (Exp 3) 에서 ceiling
  artifact 로 확인됨.
- Exp 2 (label swap): `line/blank/none × ball` 에 `+α=40 · v_L10` → 10/10 A
  ("falls"), label=`circle` (M5a) 에서는 10/10 B ("stays still"). Label-driven
  regime flip 의 causal 시연.
- Exp 3 (양방향성 재검정, 2026-04-25): `{line, textured} × blank × none` 에서
  완전한 (α × label × obj) 그리드. **신규 발견**: `-α=40` 이 4개 (obj × label)
  조건 모두에서 10/10 B ("stays still") 로 **균일하게** flip. 따라서 `v_L10` 은
  physics-mode 내부의 regime axis (+α kinetic, −α static) 이지 physics-vs-abstract
  activator 가 아니다. Baseline D 는 axis 한쪽 끝이 아니라 |α| activation
  threshold *아래* 에 위치.
- H7 qualifier: `textured/blank/none` 의 +α=40 은 label 무관 A; image 가
  physical-object signal 을 지니면 label 단독 regime flip 은 실패. Regime 은
  joint (image, label, α sign) 함수로 결정.

**가설 업데이트**:
- H-direction-bidirectional (신규): **2026-04-25 revised** — regime axis
  해석이 이전의 "one-way activator" 프레이밍을 대체.
- H-regime: **원래 형태로는 반증** — label 단독 regime flip 은 일반화되지
  않음; H-direction-bidirectional + H7 qualifier 로 흡수.
- H-locus: **unchanged (강화)** — L10 regime-flip 이 Exp 3 모든 4 cell 에서
  성립.

### M4b — Label-free prompt H2 null test ✅ (2026-04-25)

실행: `uv run python scripts/02_run_inference.py --config configs/label_free.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` 후 `scripts/03_score_and_summarize.py` 와 `scripts/05_lm_probing.py --sources open_no_label`.

출력: `outputs/label_free_20260425-031430_315c5318/` — 480 predictions + 480 activation safetensors. 심층 인사이트: `docs/insights/m4b_label_free_ko.md`. 원자료: `docs/experiments/m4b_label_free_ko.md`.

**핵심 결과**:
- Label-free baseline 과의 paired PMR delta (480 matched seed): ball +0.006,
  planet +0.006, **circle −0.065**. M2 에서 보고된 "ball vs circle" gap 은
  실제로는 circle 억제이지 ball 증가가 아니다.
- Cell 별 구조: circle 이 추상 이미지에서 더 강하게 억제 (line: −9.2 pp;
  filled: −4.2 pp); `motion_arrow` cue 가 circle 억제를 완전 override
  (+0.000); `none` cue 가 최대 억제 (−15.0 pp).
- `line/blank/none` 4-label 표가 label 기여를 깔끔하게 분리: ball (regime
  shift, kinetic→static), circle (full suppression, PMR 0.40 → 0.10),
  planet (+30 pp PMR — orbital prior 때문에 시각 default 위에 physics 를
  *진짜로* 추가하는 유일한 label).
- Label-free activation 에 M4 재실행 → M2 의 physics-margin 표 bit-for-bit
  재현. visual-token capture 가 prompt-independent 임을 확인 (image token
  이 question text 보다 앞에 있고 causal attention). L5 의 collapsed
  switching-layer 는 capture 지점의 구조적 artefact 이지 label-independent
  한 LM commitment 의 증거가 아님.

**가설 업데이트**:
- H2: **revised** — ball ≈ no-label; circle 이 suppressive override.
  Per-label 기여가 비대칭.
- H-boomerang: **강화** — visual-token hidden states 가 prompt-independent
  이므로 L5 의 physics bias 는 image-only 기원.
- H-locus: **unchanged** — label 의 행동적 효과는 visual-token 위치 하류에
  localize, M5a 의 image-preceding trajectory 에서의 L10 효과성과 일관.
- H4: **refined** — circle 억제 강도가 이미지 추상도와 함께 증가 — 추상도 →
  language-prior-gap scaling 의 image-side dual.

### M6 round 1 — LLaVA-1.5-7B cross-model ✅ (2026-04-25)

실행: `uv run python scripts/02_run_inference.py --config configs/cross_model_llava{,_label_free}.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` (두 번 pass), 후 `scripts/03_score_and_summarize.py` 각각.

출력:
- `outputs/cross_model_llava_20260425-035506_7ff0256b/` — labeled (1440 rows).
- `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/` — label-free (480 rows).

심층 인사이트: `docs/insights/m6_cross_model_llava_ko.md`. 원자료: `docs/experiments/m6_cross_model_llava_ko.md`.

**핵심 결과**:
- Label-free baseline 과의 paired PMR delta (LLaVA): ball **+0.475**,
  planet +0.244, circle +0.173 — 원래의 H2 패턴, Qwen 의 M4b 와 정반대.
  H2 reframing 은 Qwen 특이적이었음.
- Visual-saturation 가설: Qwen 의 `PMR(_nolabel)` 이 object level 전반에서
  0.93–0.98; LLaVA 는 0.14–0.59. Qwen 의 labeled run 은 headroom 이 없어
  양의 label 기여를 보이지 못하지만, LLaVA 는 분명히 보임.
- H1 (S-curve) 가 LLaVA 에서 가장 깔끔 (line → textured 0.51 → 0.81),
  Qwen 에서는 보이지 않음 (saturated 0.93).
- H7 cross-model 재현: `planet GAR << ball/circle GAR` 가 두 모델 모두에서
  성립, `planet` 이 physics 서술을 orbital / cosmic event 로 route ("orbit
  around the sun", "consumed by a black hole").
- FC 제외: LLaVA 가 모든 (image, label) FC 자극에 "A" 반환 (smoke 12/12).
  Pathological 모델 편향, 여기서 prompt 변경으로 해결 불가. Round 2 에서
  FC redesign 또는 first-letter-token-probability 채점 필요.

**가설 업데이트**:
- H2: **재개정 — visual-saturation 가설** 이 M4b 와 M6 를 단일 statement
  로 통합 (모든 label / 모델에서 양의 language-prior 기여 존재; visual
  saturation 이 mask).
- H1: **지지, LLaVA 에서 더 sharp** — canonical figure 는 LLaVA 의 monotone
  curve 권장.
- H7: **지지, cross-model** — orbital-routing dissociation 재현.
- H4 / H-boomerang / H-locus / H-direction-bidirectional: **round 1
  cross-model 미검정** — FC 실패가 H4 차단; LLaVA activation capture 부재가
  나머지 차단. Round 2.

### M4c — Forced-choice label-free ✅ (2026-04-25)

실행: `uv run python scripts/02_run_inference.py --config configs/fc_label_free_{qwen,llava}.py --stimulus-dir inputs/mvp_full_20260424-093926_e9d79da3` (두 번 pass), 후 `scripts/03_score_and_summarize.py` 로 채점.

출력:
- `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/` — Qwen, 480 rows.
- `outputs/fc_label_free_llava_20260425-044517_81ae56d5/` — LLaVA, 480 rows (degenerate).

심층 인사이트: `docs/insights/m4c_fc_label_free_ko.md`. 원자료: `docs/experiments/m4c_fc_label_free_ko.md`.

**핵심 결과**:
- Qwen FC label-free 가 FC 하에서 M4b 의 H2 reframing 재현: `ball − _nolabel
  = +0.013` (≈ 0), `circle − _nolabel = −0.208` (M4b 의 −0.065 보다 강함),
  `planet − _nolabel = −0.263` (신규 — orbital regime 이 FC 의 gravity-centric
  옵션 셋 하에서 D 로 collapse).
- Qwen 의 no-label 동일 자극에서의 open-vs-FC paired delta: **−0.131** —
  FC 가 일관되게 보수적; label confounding 없는 H4 cross-format 측정 가능.
- `line/blank/none` 하 FC: 모든 label condition 이 D=10/10 (또는 `_nolabel`
  의 9/10) 으로 collapse. FC 의 D 옵션이 완전 모호 이미지에서 모든 label
  condition 을 abstract reject 로 끌어당기는 "abstract sink".
- LLaVA FC label-free: 477/480 = 99.4 % `A`. "the depicted object" 로
  re-template 해도 M6 r1 의 편향 완화되지 않음. 모델 수준 pathology 확인,
  prompt-fixable 아님.

**가설 업데이트**:
- H2: **추가 강화** — Qwen FC 가 다른 prompt format 에서 M4b 재현. "planet
  suppression" 발견은 nuance 추가: Qwen 의 per-label suppression 이 부분적
  으로 옵션-셋 artefact 임을 visual-saturation 프레이밍을 지지.
- H4: **Qwen no-label 에서 측정 가능** (paired delta = −0.131); cross-model
  H4 는 LLaVA FC 편향에 막힘.
- H7: **caveat 추가** — regime 구별은 narrative latitude 가 허용되는 prompt
  에서만 보임. FC 하에서 모든 non-gravity regime (orbital, "consumed by
  black hole" 등) 이 D 로 collapse 되어 H7 mask. 확장 FC 옵션 셋이 FC-side
  H7 에 필요.
- LLaVA FC pathology: **확정** — round-2 아이디어는 greedy argmax 대신
  first-token logit 비율 사용.

### M6 round 2 — Cross-model 확장 ✅ (2026-04-25)

세 sub-deliverable. Run dirs:

- `outputs/cross_model_internvl3_20260425-051009_fc710e85/` — InternVL3 labeled (1440 rows).
- `outputs/cross_model_internvl3_label_free_20260425-053116_ea0a07c5/` — InternVL3 label-free (480 rows).
- `outputs/cross_model_llava_capture_20260425-054821_65214a5d/` — LLaVA captured (1440 rows + 14 GB activations + probing_vision/ + probing_lm/).

심층 인사이트: `docs/insights/m6_r2_cross_model_ko.md`. 원자료: `docs/experiments/m6_r2_cross_model_ko.md`.

**핵심 결과**:
- **r2a (InternVL3 행동)**: paired delta vs label-free 가 모든 label 에서 +0.010 — InternVL3 의 PMR(_nolabel) = 0.99 이라 headroom 없음. 3-model paired-delta 패턴 (Qwen ≈ 0, LLaVA 강한 양수, InternVL3 ≈ 0) 이 encoder-saturation 예측과 일치.
- **r2b (LLaVA captures)**: vision encoder AUC ~0.73 (vs Qwen ~0.99); LM AUC ~0.75 (평탄 — boomerang recovery 없음); behavioral PMR ~0.78. 모델 간 saturation 차이가 vision encoder 에 뿌리내림.
- **r2c (FC logit ratio)**: LLaVA FC 편향이 underlying logit 수준 (90% rows 가 top_p=0.95 에 `A` 만 통과). Greedy → logit-ratio rescue 실패. Qwen 의 경우 logit-argmax 가 text-PMR 보다 깨끗한 FC metric (greedy formatting drift 로 잃은 ~14 pp 신호 회복).

**가설 업데이트**:
- H-boomerang: **Qwen-scoped** — encoder-knows / decoder-gates gap 이 Qwen 에 존재, LLaVA 에는 없음 (encoder 가 bottleneck, gate 아님).
- H-encoder-saturation (신규): **3 모델 점에 걸쳐 제안 + 지지** — vision encoder probe AUC 가 `PMR(_nolabel)` 와 per-label paired delta 방향 모두 예측.
- H2: **세 모델 점에서 visual-saturation 가설 하 완전 검증**; paired-delta 패턴이 encoder-saturation 예측과 일치.
- H7: **3-of-3 cross-model** — orbital-routing dissociation 이 보편적.
- H4: **InternVL3 + LLaVA 미검정** — FC 제외 (cost) / 차단 (LLaVA "A" 편향).

### M8a — 비-원형 합성 shape ✅ (2026-04-25)

**결과**: 5 도형 × 4 추상화 × 2 bg × 2 cue × 5 시드 = 400 자극; Qwen + LLaVA × labeled + label-free arms = 약 3200 추론, 단일 (혼잡한) GPU 1 에서 약 43 분.

**사전 등록 엄격 채점**:

| 기준 | Qwen | LLaVA |
|---|---|---|
| H1 ramp (PMR(textured)−PMR(line) ≥ 0.05; 역전 >0.05 없음) | 3/5 ✗ | 4/5 ✓ |
| H7 (PMR(physical)−PMR(abstract) ≥ 0.05) | 1/5 ✗ | 4/5 ✓ |
| H7-GAR (GAR(physical) ≥ GAR(abstract)) | 1/5 ✗ | 5/5 ✓ |
| Visual-saturation Δ on physical role | 3/5 ✓ borderline | 5/5 ✓ |

**헤드라인 해석**: 포화된 모델은 4개 중 3개 기준에서 실패; 포화되지 않은 모델은 4개 모두 통과. 그 비대칭이 **바로** H-encoder-saturation 의 도형 간 검증 — 가설에 의해 예측됨, 단순히 일치 아님. PMR(_nolabel) 베이스라인: Qwen 도형 간 0.78–0.93 (천장), LLaVA 0.075–0.288 (headroom).

**도형별 주목할 발견**:
- Qwen `square` paired-delta -0.20 / -0.28 / -0.21 — M4b 의 circle "라벨-억제-물리" 효과의 깔끔한 도형 간 재현.
- LLaVA `triangle` paired-delta 가 +0.125 / +0.10 / +0.10 만 — `wedge` 가 약한 physical-object 라벨 (PMR(wedge)=0.20 vs PMR(ball/brick/nut/rock)≈0.7). 라벨 설계 caveat, 도형 효과 아님.
- LLaVA `polygon abstract` paired-delta -0.05 — 음수가 된 유일한 LLaVA paired-delta. "Polygon" 이 수학 용어로 읽힘; 일반 어휘 기하 명사 없는 도형에 대해 role taxonomy 가 샘.

**로드맵 영향**:
- H1 + H7 은 이제 *unsaturated-only*; 논문에 caveat 추가.
- H-encoder-saturation 이 3-model correlational 에서 3-model + 5-shape 으로 이동, H1/H7 실패를 예측.
- Triangle / polygon 라벨 품질 문제는 M8c 사진 큐레이션과 M8d 비-공 카테고리에 피드 (더 강한 physical 라벨 사용).

**아티팩트**: `docs/insights/m8a_non_circle_shapes.md`, `docs/experiments/m8a_non_circle_shapes.md`, `docs/figures/m8a_{shape_grid,full_scene_samples,pmr_ramp,pmr_by_role,paired_delta}.png`, `notebooks/m8a_non_circle_shapes.ipynb`, `outputs/m8a_*` (4 run dirs).

### M8c — 실사진 — 작업 상세 ▶ priority 2

**동기**: M2 자극은 프로그램 합성. "encoder probe AUC" 발견이 합성 패턴에 overfit 일 수 있음. 사진은 visual-saturation 의 out-of-distribution 검증.

**Sub-tasks**:
1. 50-100 사진 큐레이션: 공 (농구공, 축구공, 볼링공, 탁구공, 테니스공, 당구공), 다른 잡을 수 있는 객체 (사과, 캔, 머그, 책), 추상 사진 (도면, 다이어그램). License 허용 source (PEXELS / Unsplash / ImageNet 하위 클래스 등).
2. 별도 `inputs/real_photo_<ts>/` manifest 에 동일 column schema (`sample_id`, `image_path`, label-axis fields, plus `source_type ∈ {synthetic, photo}` column).
3. 캡처된 모델 (Qwen + LLaVA + InternVL3) 에 동일 prompt 프로토콜 (open / open_no_label / forced_choice) 실행.
4. 직접 비교: paired (synthetic-textured-ball vs photo-ball) PMR delta. Photo-realism 이 encoder 를 더 saturate 시키는가? LLaVA gap 을 좁히는가?

**성공 기준**:
- Photo PMR(_nolabel) ≥ synthetic textured PMR(_nolabel) 각 모델별 — 방향성 확인.
- 최소 Qwen + LLaVA 에서 photo vs synthetic encoder probe AUC 비교 가능.

**예상 소요**: 4-6 시간 (사진 큐레이션이 slow step).

### M8d — 비-공 물리 객체 카테고리 ✅ (2026-04-25)

실행: GPU 0 (단일 H200) 에서 `bash scripts/m8d_run_all.sh`.

출력: 4 run 디렉터리 (Qwen labeled / Qwen label-free / LLaVA labeled / LLaVA label-free) → 3840 추론을 **31.9 분** wall clock.

자극 디렉터리: `inputs/m8d_qwen_20260425-151543_19e1fcd0/` — 480 자극 (3 카테고리 × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds).

심층: `docs/insights/m8d_non_ball_categories_ko.md`. 숫자: `docs/experiments/m8d_non_ball_categories_ko.md`.

**사전 등록 엄격 채점**:

| 기준              | Qwen | LLaVA |
|-------------------|------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| 시각 포화 delta    | 1/3 (bird) | 2/3 (car, bird; person 음수로 flip) |

**헤드라인 해석**:

- **H7 LLaVA에서 3/3 카테고리 횡단 일반화** — 본 프로젝트 가장 강력한 카테고리 횡단 H7 증거. car +0.525, person +0.138, bird +0.550 PMR_regime(physical) − PMR_regime(abstract) on horizontal 부분 집합. label-selects-regime 클레임이 이제 카테고리 일반.
- **Qwen H7 strict 실패 (binary, 천장)** 이지만 regime 분포가 동일 패턴: figurine 17.5 % static, statue 22.5 % static (physical 라벨의 ~5 % 대비). 새 방법론적 발견: **regime 분포가 binary saturation 에서 H7 신호를 구제**.
- **H1 양 모델 모두 실패** — Qwen 천장, LLaVA 비단조. 추상화 ramp 는 기하-도형 ↔ 명명-객체 축의 속성, 일반적 시각-디테일 → 물리-prior 메커니즘이 아님. M8a 가 H1-unsaturated-only 확립; M8d 가 H1-도형-축-특정으로 추가 한정.
- **H-encoder-saturation 카테고리 횡단 검증** — Qwen car/person 천장 (0.97-1.0), LLaVA 범위 0.55-0.84.

**코드 변경**:
- `stimuli/primitives.py`: 12 개 새 draw 함수 (car / person / bird × line / filled / shaded / textured).
- `stimuli/scenes.py`: `horizontal` 이벤트의 ground-bound shape positioning.
- `inference/prompts.py::LABELS_BY_SHAPE`: car / person / bird 트리플.
- `metrics/lexicons.py`: `CATEGORY_REGIME_KEYWORDS` + `UNIVERSAL_KINETIC_STEMS` (카테고리 공통 gravity-fall 동사).
- `metrics/pmr.py`: `classify_regime(category, text) → {kinetic, static, abstract, ambiguous}`.
- `configs/m8d_*.py`, `scripts/m8d_*.py`. 123 단위 테스트 통과.

**분류기 검증**: `scripts/m8d_hand_annotate.py --mode score` 가 54 stratified 행에 적용 → **5.6 % 오차율** (paper-ready 15 % 임계값 이하). 미스매치 3개는 "no movement" / "pulled away" 패턴의 stem-matching false-positive.

**로드맵 함의**:
- H7 "circle-only" 에서 "cross-category" 로 승격 (논문 헤드라인).
- H1 "기하-도형 ↔ 명명-객체 축" 으로만 한정.
- regime-distribution 이 논문 방법론적 기여로 (binary saturation 에서 H7 구제).
- M8c 강하게 동기화 — 사진 사실성이 LLaVA 의 인코더 갭을 닫는가?
- Round-2 개선: `duck` 을 비행 못 하는 새 (penguin / ostrich / chicken) 로 교체하여 더 깨끗한 H7 exotic 역할.

**아티팩트**: `docs/insights/m8d_non_ball_categories.md` (+ `_ko`), `docs/experiments/m8d_non_ball_categories.md` (+ `_ko`), `docs/figures/m8d_{shape_grid,full_scene_samples,pmr_ramp,pmr_by_role,paired_delta,regime_distribution}.png`, `notebooks/m8d_non_ball_categories.ipynb`, `outputs/m8d_summary/` (모델별 rollup + 결합 주석 parquet).

### M6 r4 — InternVL3 비전-인코더 프로브 (4-모델 사슬) ✅ (2026-04-25)

심층: `docs/insights/m6_r4_internvl3_probe.md`.

**범위**: H-encoder-saturation 사슬에 네 번째 점 추가 — InternVL3-8B (InternViT + InternLM2-7B). M8a 행동 (1200 + 400 추론을 12 분에) + 비전 캡처 (400 stim × 4 InternViT 레이어를 47 초에) + 선형 프로브.

**헤드라인**: M8a 합성 stim 의 4-점 AUC ↔ 행동 PMR(_nolabel) 사슬 (4 인코더 모두 M8a stim 산출 — apples-to-apples):

```
인코더 family             AUC      PMR(_nolabel)
─────────────             ────     ─────────────
SigLIP    (Qwen)          0.88     0.84
SigLIP-SO400M (Idefics2)  0.93     0.88
InternViT (InternVL3)     0.89     0.92     ← M6 r4
CLIP-ViT-L (LLaVA)        0.77     0.18
```

**3 개 별개의 비-CLIP 인코더 family** (SigLIP, SigLIP-SO400M, InternViT) 모두 AUC ≥ 0.88 / PMR ≥ 0.84 도달. **CLIP-ViT-L 만 포화 미달** (0.77 / 0.18). 4 LM family 횡단 (Qwen2-7B, Mistral-7B, InternLM2-7B, Vicuna-7B), 인코더 family 가 통합 포화 driver.

**가설 업데이트**: H-encoder-saturation 이 "SigLIP 포화"에서 "비-CLIP 인코더 포화; CLIP 비포화 (이 표본)"으로 일반화. 논문급 헤드라인: 비전-인코더 family 가 합성-stim 천장 체제를 인과적으로 결정. 비선형 AUC → PMR 매핑 (≈0.10 AUC 갭 → 0.65 PMR 갭) 은 AUC 0.85 부근의 포화 임계와 일치.

**구현 노트**: InternVL3 의 `vision_tower.encoder.layer` (복수가 아닌 단수) 인식 위해 `_resolve_vision_blocks` 수정 필요했음. 본 라운드는 또 Qwen + LLaVA 를 M8a stim 에서 재캡처하여 4 AUC 수치가 동일 factorial 산출; M6 r2 수치 (Qwen 0.99 / LLaVA 0.73) 는 M2 stim 의 bimodal line/ground-cells 분포 (M8a 의 더 넓은 per-cell PMR 범위보다 더 분리 가능) 를 반영.

**산출물**: `docs/insights/m6_r4_internvl3_probe.md` (+ `_ko`), `docs/figures/encoder_chain_4model.png` (논문 헤드라인 그림), `docs/figures/encoder_swap_internvl3_probe.png`, `outputs/encoder_swap_internvl3_probe/{layer_sweep,by_object_level,by_shape}.csv`, `outputs/encoder_swap_probe_summary/encoder_chain_table.csv`, `scripts/encoder_swap_probe.py` (모델-무관), `scripts/encoder_swap_probe_summary.py`.

### M6 r3 — Idefics2 비전-인코더 프로브 ✅ (2026-04-25)

심층: `docs/insights/m6_r3_idefics2_probe.md`.

**범위**: Idefics2 SigLIP-SO400M 비전 활성화를 M8a stim 에서 캡처 (400 자극 × 4 레이어, GPU 0 에서 88 초) + Idefics2 자체 per-stim 라벨링 PMR 을 y 타겟으로 layer-wise 선형 프로브. 세 번째 SigLIP 점에서 AUC ↔ 행동 PMR 사슬 종결.

**헤드라인**: Idefics2 비전-인코더 probe AUC = **0.93** 레이어 간 (레이어 9 에서 0.948 peak). Qwen SigLIP (M3 / M6 r2: 0.99) 와 LLaVA CLIP-ViT-L (M6 r2: 0.73) 사이. 3-모델 AUC ↔ 행동 PMR 사슬:

```
인코더 family             AUC      M8a PMR(_nolabel)
─────────────             ────     ─────────────────
SigLIP    (Qwen)          0.99     0.84
SigLIP-SO400M (Idefics2)  0.93     0.88     ← M6 r3
CLIP-ViT-L (LLaVA)        0.73     0.18
```

**가설 업데이트**: H-encoder-saturation 가 *메커니즘* 수준에서 완전 종결: 인코더 family → 인코더 probe AUC → 행동 PMR(_nolabel) → H7 측정 가능성, 4 노드 모두 3 모델 점에서 경험적 지지. 업데이트된 논문 주장: "인코더 family 가 비전-인코더 probe AUC 포화 야기, 이것이 행동 PMR(_nolabel) 포화 야기, 이것이 H7 측정 가능성 게이팅".

**주의**: per-shape AUC 분산 큼 (`polygon` AUC 가 깊은 레이어에서 0.5 미만 — n-불균형 아티팩트, 진짜 역전 아님). InternVL3 캡처 미실행 — 4 점 probe 표를 위한 자연스러운 다음 단계.

**산출물**: `docs/insights/m6_r3_idefics2_probe.md` (+ `_ko`), `docs/figures/encoder_swap_idefics2_probe.png`, `outputs/encoder_swap_idefics2_probe/{layer_sweep,by_object_level,by_shape}.csv`, `scripts/encoder_swap_idefics2_probe.py`.

### M9 — 일반화 audit (논문 Table 1) ✅ (2026-04-25)

심층: `docs/insights/m9_generalization_audit.md`.

**범위**: M8a (5 도형) + M8d (3 카테고리) + M8c (5 사진 카테고리) × {Qwen2.5-VL-7B (SigLIP), LLaVA-1.5-7B (CLIP), Idefics2-8b (SigLIP-SO400M)} 를 단일 9-셀 논문 Table 1 으로 통합, 평균 PMR(_nolabel) + 평균 H7 paired-difference 에 **95% 부트스트랩 CI** (5000 iter).

**헤드라인**:

1. **인코더 family 가 합성-stim PMR(_nolabel) 천장의 인과 driver** — 견고. SigLIP CI [0.800, 0.917] vs CLIP CI [0.140, 0.371] M8a + M8d 에서; CI 완전 분리.
2. **사진이 인코더 갭을 압축** — 견고. 3 모델 모두 M8c 에서 [0.183, 0.667] 로 수렴. 5× 비율이 ~1.5-2× 로 축소.
3. **H7 측정 가능성은 LLaVA-on-synthetic 에서만 견고** — LLaVA M8a CI [+0.30, +0.42], M8d CI [+0.25, +0.36] 0 과 분리; LLaVA M8c CI [−0.03, +0.23] 0 가로지름 (n=12 underpowered). Qwen + Idefics2 H7 CI 모두 0 가로지름, Idefics2 M8d CI [+0.000, +0.094] 만 겨우 닿음.
4. **포화 시 LM-modulation** — *시사만*. Idefics2 M8d H7 CI 0 위에 살짝 vs Qwen M8d H7 CI 0 가로지름; PASS-rate 갭 (0.667 vs 0.333) 은 단일 도형 (`car`) 가 strict 임계 가로지른 결과. 향후 작업 플래그 강등; 깔끔한 검증은 동일-인코더 LM 스왑 필요.

**통계 방법론 기여**: PASS/FAIL binarization (M8a, §4.5) 을 평균 H7 차이의 부트스트랩 CI 로 대체. `Qwen 1/5 PASS` 패턴이 noise-floor binarization (실제 평균 H7 = 0 with CI 0 가로지름) 임을 드러냄.

**가설 업데이트**:
- H-encoder-saturation: *강화* — 논문 주장 이제 "인코더 family 가 합성-stim PMR 천장 유발; 포화가 H7 측정 가능성 게이팅" (cross-stim 부트스트랩 검증).
- H7: *범위 명확화* — 인코더 headroom 남기는 곳에서만 견고 (LLaVA on synthetic).
- 신규 **H-LM-modulation**: 시사만 그러나 현 데이터로 옹호 불가.

**한계**: M8c n=12/카테고리 H7 underpowered; M8d 3 도형 cross-shape 분산에 얇음; Idefics2 인코더 프로브 AUC 아직 없음.

**아티팩트**: `docs/insights/m9_generalization_audit.md` (+ `_ko`), `docs/figures/m9_summary.png`, `docs/figures/m9_table1_heatmap.png`, `outputs/m9_audit/m9_{table1,summary}.csv`, `scripts/m9_generalization_audit.py`.

### 4.5 Cross-encoder swap — 작업 상세 ▶ priority 4 (promoted)

**동기**: H-encoder-saturation 가 현재 3-model correlational (M6 r2). 인과 검증은 모든 것을 고정한 채로 encoder 만 *swap*.

**Sub-tasks**:
1. SigLIP encoder swap LLaVA-1.5-7B (HF community port 존재: 예: `google/siglip-base-patch16-224` projector retraining). 또는 SigLIP 을 encoder 로 사용한 LLaVA-style from-scratch 학습.
2. 가장 깔끔한 대안: 이미 encoder swap 한 LLaVA-1.5 파생 family (예: ShareGPT4V, Bunny). 행동 run 만 — `PMR(_nolabel)` 와 encoder AUC 가 함께 움직이는지 확인.
3. Stretch: minimal projector swap 학습 (~few hr GPU) 으로 LLaVA-1.5 의 CLIP ↔ SigLIP swap.

**예상 소요**: 4-7 시간 (기존 swap variant 사용; +다수 시간 if fresh swap 학습).

### 4.6 SAE / VTI 역방향 counterfactual 자극 생성 — 작업 상세 ▶ priority 5 (promoted)

**동기**: "Adversarial physics-mode" 자극이 모델이 무엇을 physical 로 보는지 드러냄. 합성된 자극이 사람에게는 추상으로 보이지만 모델에는 physical 로 읽히면, 깔끔한 shortcut-interpretation 발견.

**Sub-tasks**:
1. M5a steering direction `v_L10` 또는 학습된 SAE feature 를 가져옴.
2. Image space 에서 gradient-ascent (PIL / torch differentiable) 로 활성화의 `v_L10` 사영 maximize.
3. 결과 자극 시각 검사 + PMR 측정.

**예상 소요**: 6-10 시간 (Qwen 파이프라인 통한 image differentiability 가 non-trivial).

### 4.10 Attention visualization UI — 작업 상세 ▶ priority 6 (promoted)

**동기**: Cross-axis (layer × head × visual-token-position) attention map 이 어떤 head 가 어떤 cue 에 attend 하는지를 정성적으로 드러냄. 논문 부록 figure + patching target 발견에 유용.

**Sub-tasks**:
1. `capture_lm_attentions=True` 로 subset (~20-50 stim) 을 Qwen + LLaVA 에 재실행.
2. Notebook UI 구축: stimulus 선택 → layer → head → image 위에 overlay attention heatmap, 추가로 label token 에 대한 attention.
3. 5-10 illustrative cell 큐레이션.

**예상 소요**: 5-7 시간.

### M5b — ST4 Phase 3 (SIP patching + SAE) — 작업 상세

**작업 분할**:
1. `PhysModeVLM.capture()` 에 `capture_vision_layers` 경로 구현 (Qwen2.5-VL 은 `model.visual.blocks[i]`; LLaVA 는 `model.vision_tower.vision_model.encoder.layers[i]`).
2. `src/physical_mode/probing/vision.py` 에 `train_probes(X_per_layer, y_pmr) -> dict[int, Probe]` 작성 (sklearn LogisticRegression, 5-fold stratified).
3. 추가 MVP-full 재실행 (capture_vision_layers 켜고) — 또는 M2에서 이미 켰으면 스킵.
4. Figure: layer-wise probe AUC × object_level → "encoder knows" 증명.
5. **Stretch**: Gandelsman head decomposition (CLIP 전용). Qwen2.5-VL 은 SigLIP 기반이라 Balasubramanian 적응 필요.

**가설 검증**:
- 시각 인코더 AUC 의 S-curve 기울기가 행동 PMR S-curve 보다 **가파르다** (encoder-decoder boomerang).
- 특정 head 또는 feature direction 이 "physical-ness" 축에 특화된다.

**성공 기준**:
- 최소 1 개 layer 에서 PMR probe AUC > 0.75.
- Encoder AUC 와 behavioral PMR 의 per-cell gap 이 유의미 (paired t-test / bootstrap).

### M5 — ST4 Causal localization

**전제**: M3-M4 로 어떤 layer/head 가 후보인지 파악됐음.

**작업 분할**:
1. Semantic Image Pair 생성: pilot factorial 에서 single-axis-differ 쌍 뽑아 `sip_manifest.parquet` 작성.
2. **Activation patching** (TransformerLens 또는 raw PyTorch hooks):
   - Clean/corrupted forward 쌍 캡처.
   - Layer-sweep: 각 layer 의 visual token 위치 활성화를 clean → corrupted 로 교체, PMR 확률 회복량 측정.
3. **Attention knockout**: 특정 head 의 visual-to-last-token attention 을 0 으로 → PMR 변화.
4. **VTI steering**: `v_layer = mean(h_clean) - mean(h_corrupted)`. Test-time 에 `alpha * v` 를 residual stream 에 추가 → line 원이 물리 모드로 flipping 되는지.
5. **SAE** (stretch): vision-encoder activations 에 SAE 학습 → monosemantic "shading" / "ground" feature 식별 + intervention.

**헤드라인 claim 후보**: "LLaVA layer 19 head 14 의 knockout 이 PMR 을 50pp 감소; 동일 head 만 보존하고 나머지 visual attention 다 끊으면 PMR 유지" 같은 문장 (연구계획 §3.2).

### M6 — ST5 Cross-model sweep

작업:
1. `configs/cross_model.py` — `model_id` list + 같은 factorial.
2. `scripts/02_run_inference.py` 수정: list of model_ids 를 순회.
3. LLaVA-1.5-7B (~13 GB), LLaVA-Next-7B (~14 GB), InternVL2-8B (~16 GB), (stretch) Qwen2-VL-7B (~15 GB). 총 다운로드 ~60 GB.
4. 각 모델에서 behavioral 표 + (가능하면) ST3/4 축약 버전.
5. **Prompt steering**: `system_prompt_override` 로 `"treat this as an abstract geometric shape"` vs `"treat this as a physical object subject to gravity"` → PMR shift 측정.

**가설**: 지면 효과 (H5) 는 모든 open-source VLM 에서 재현; open-vs-forced gap (H4) 는 모델별 크기 차이.

### M7 — 인간 baseline (optional) + 논문

- Prolific 20 명 × 50 stimuli (random subset) × open-ended 프롬프트.
- Human PMR 과 VLM PMR 의 per-cell alignment 분석.
- EMNLP long (primary) / NeurIPS main (stretch) 초안.

---

## 4. 원래 계획에 없던 추가 아이디어

연구 진행 중 떠올랐거나 `references/project.md` §2 에 없는 확장.

**다음-tier priority 로 promotion** (작업 상세는 §3 의 해당 섹션 참조):
- **4.5** Cross-encoder swap — M8a/c/d 후 priority 4 (H-encoder-saturation 의 인과 검증).
- **4.6** SAE/VTI 역방향 counterfactual 자극 생성 — priority 5.
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

**여전히 열림**: 다른 언어 (일본어 / 중국어 / 스페인어); 완전 한국어
프롬프트 (단지 영어 템플릿에 한국어 라벨 삽입이 아닌).

### 4.4 Video frame pair → Michotte-style causality

두 프레임 (t=0, t=1) 에 객체 위치만 달라지는 쌍을 주고 "launched by X?" 질문. Michotte (1946) launching effect 가 VLM 에 나타나는가? 동영상 모델 필요 없이 2-image prompt 로 proxy 가능.

### 4.5 Cross-encoder swap (SigLIP vs CLIP vs DINOv2) ⭐ promoted

"Vision encoder 가 CLIP 이면 안 보이는 cue 를 DINOv2 기반 모델은 본다" 가설. Eyes Wide Shut (Tong et al. 2024) MoF 제안의 연속선. 단, standalone encoder 는 LLaVA-1.5 (CLIP-ViT-L/14) vs Qwen2.5-VL (SigLIP) 의 자연스러운 비교로 이미 M6 에 포함.

**상태 (2026-04-25)**: 다음-tier priority 로 promotion — H-encoder-saturation (M6 r2) 이 현재 3-model correlational; 이건 인과 counterfactual. 작업 상세 §3 위 참조.

### 4.6 Activation 기반 counterfactual 자극 생성 ⭐ promoted

SAE 또는 VTI 로 찾은 steering vector 를 반대로 써서 "VLM 이 보기엔 '물리 모드' 를 최대화하는 자극" 을 gradient ascent 로 합성. **adversarial physics-mode prompt** → 오픈소스 VLM 의 shortcut 해석 증거.

**상태 (2026-04-25)**: 다음-tier priority 로 promotion. M5a `v_L10` 방향이 잘 특성화됨 (M5a-ext) — 역방향 합성이 자연스러운 확장. 작업 상세 §3 위 참조.

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

| 날짜 | 변경 | commit |
|---|---|---|
| 2026-04-24 | 최초 작성 — M0/M1 완료, M2 준비 상태까지 반영 | `23171b6` |
| 2026-04-24 | M2 완료 반영: 가설 스코어카드 (H1→지지, H2→정량화, H4→지지, H5→혼재, H6→지지 수정, H7 신규), M3 를 다음 마일스톤으로, §4 에 H7 follow-up 추가 | `1d17252` |
| 2026-04-24 | M3 완료: vision encoder probing — boomerang 확인 (encoder AUC=1.0 / behavioral 0.28-0.95), M4 를 다음 마일스톤으로. | `1205821` |
| 2026-04-24 | M4 완료: LM logit lens + per-layer probe. LM AUC 0.94-0.95 전 구간 (peak L20=0.953); label 이 L5 부터 physics margin 주도. M5 를 다음 마일스톤으로. | `2abdc32` |
| 2026-04-24 | M5a 완료 (VTI steering): L10 α=40 이 "line/blank/none" 10/10 을 D(abstract) → B(physical-static) flip. "object-ness" direction 인과 확인. M5b (SIP+SAE), M6 이 남음. | `61ffd29` |
| 2026-04-24 | M5a-ext Exp 1+2 완료: ceiling 에서 negative α (null — 이후 ceiling artifact 로 판명) + label=ball swap on line/blank/none (clean B→A flip). H-direction-bidirectional 신규 (초기엔 "one-way activator"), H-regime 을 "지지" 로 격상. | `9a0ed86` (merge) |
| 2026-04-25 | M5a-ext Exp 3 (`textured/blank/none` moderate baseline 에서 양방향성 재검정): −α=40 → (line/textured) × (ball/circle) 모두에서 10 B 를 균일하게 유도. H-direction-bidirectional 을 "physics-mode 내부의 regime axis" 로 개정 (+α kinetic, −α static, baseline D 는 threshold 아래). H-regime 원래 형태 반증 후 H7 qualifier 로 축소. | `f8f0fdd` |
| 2026-04-25 | M4b 완료: M2 자극에 label-free prompt 를 H2 null test 로 적용. Paired PMR(ball) − PMR(_nolabel) = +0.006 ≈ 0; PMR(circle) − PMR(_nolabel) = −0.065. **H2 revised** — language prior 는 비대칭 (circle override, ball enhancement 아님). M4 visual-token capture 가 prompt-independent (causal-attention artefact); switching-layer 의 붕괴는 구조적 현상. | `e97db16`, `990ddf7` |
| 2026-04-25 | M6 round 1 완료 (LLaVA-1.5-7B cross-model): paired PMR delta vs label-free → ball +0.475, planet +0.244, circle +0.173 (모두 양수). **H2 재개정 — visual-saturation 가설**: M4b 의 "circle suppression only" 은 Qwen 특이적; LLaVA 는 visual prior 가 unsaturated 라서 원래 H2 보여줌. H1 S-curve 가 LLaVA 에서 가장 깔끔 (0.51 → 0.81). H7 cross-model 재현 (planet GAR << ball GAR 두 모델 모두). FC 제외 — LLaVA 가 모든 cell 에 "A" 반환. | `c1b885f` |
| 2026-04-25 | M4c 완료 (forced-choice label-free): 새 `forced_choice_no_label` variant. Qwen 이 FC 하에서 M4b 재현 (ball ≈ no-label, circle 이 더 강하게 억제, planet 이 FC gravity-centric 옵션 셋으로 인해 새로 억제). Qwen 의 no-label 에서 open-vs-FC paired delta = −0.131 (label confound 없는 H4 측정 가능). LLaVA "A" 편향이 re-template 에서도 유지 (477/480) — 모델 수준 pathology 확인. | `70dc39c` |
| 2026-04-25 | M6 round 2 완료 (3 sub-deliverable): r2a InternVL3 cross-model (모든 label paired delta +0.010, fully saturated), r2b LLaVA-1.5 captures (vision encoder AUC ~0.73, LM AUC ~0.75 — boomerang gap 이 Qwen 특이적, LLaVA encoder 가 bottleneck), r2c FC logit-ratio (LLaVA "A" 편향이 logit 수준 — top_p 에 90% rows 가 A 만 통과, greedy 가 아님). **신규 H-encoder-saturation 가설** 이 3-model H2 패턴을 vision encoder probe AUC 에 anchor. | `47f4b18` |
| 2026-04-25 | Roadmap 우선순위 재배치: depth 보다 외부 타당성 (external validity) 우선. 신규 마일스톤 **M8a (비-원형 합성 shape), M8c (실사진), M8d (비-공 물리 객체 카테고리)** 를 최우선 추가. **§4.5 (encoder swap), §4.6 (counterfactual 자극 생성), §4.10 (attention viz UI)** 를 다음-tier priority 로 promotion. M5b (SIP+SAE) 와 M6 r3+ 는 M8 + 4.5/6/10 결과 이후의 optional 로 강등. M9 (일반화 audit) 신규 추가 — M8 + M6 r3+ 후의 통합 마일스톤. | `cfbe5a2` |
| 2026-04-25 | **M8a 완료 (비-원형 합성 shape)**: 5 도형 × Qwen + LLaVA, 4 추론 config (모델당 labeled + label-free), 400 자극, ~43 분. 사전 등록 채점 **엄격**: Qwen 1/4 PASS (visual-saturation Δ borderline), LLaVA 4/4 PASS. 비대칭이 *바로* H-encoder-saturation 가설의 도형 간 검증: 포화된 인코더 → ceiling effect → ramp/라벨/중력 사전분포가 작동할 headroom 없음; 포화되지 않은 인코더 → 4가지 모두 측정 가능. H1 (ramp) 와 H7 (label-role) 을 **unsaturated-only** 로 개정 (LLaVA-clean, Qwen-suppressed). H-encoder-saturation 이 도형 간 검증됨 (이전엔 3-model correlational). Triangle 의 `wedge` 와 polygon 의 `polygon` 이 라벨 설계 약점으로 노출; M8c follow-up 에 flag. | `a83267c` |
| 2026-04-25 | **M8d 완료 (비-공 물리 객체 카테고리)**: 3 카테고리 (car/person/bird) × 4 추상화 × 2 bg × 2 cue × **2 events** × 5 seeds = 480 자극; Qwen + LLaVA, labeled + label-free arms = 3840 추론을 GPU 0 에서 **31.9 분**. 사전 등록 엄격: Qwen 0/3 H7 binary (천장), LLaVA **3/3 H7 ✓** (car +0.525 / person +0.138 / bird +0.550 PMR_regime physical−abstract). Qwen 천장 아래에서 regime 분포가 figurine 17.5 % static / statue 22.5 % static. H1 양 모델 모두 실패 (ramp 가 도형-축-특정). H-encoder-saturation 카테고리 횡단 검증. 새 `classify_regime` keyword 분류기 (5.6 % 손 라벨링 오차, 15 % 임계값 이하). H7 을 "circle-only" 에서 "cross-category" 로 승격 (논문 헤드라인); regime-distribution 이 binary saturation 에서 H7 신호를 구제하는 방법론적 기여. | `f7d0375` |
| 2026-04-25 | **M8c 완료 (실사진)**: 60 사진 (12 × {ball, car, person, bird, abstract}) from COCO 2017 + WikiArt; Qwen + LLaVA × labeled + label-free = 480 추론을 GPU 0 에서 **5 분**. **핵심 발견**: 사진이 Qwen PMR(_nolabel) 을 카테고리 4 개에 걸쳐 18-48 pp 낮춤 — 합성-stim 단순성이 행동 포화의 공동 인자, 인코더 표현뿐 아니라. LLaVA H7 사진에서 부분적 성립 (2/4 binary). LLaVA person 사진 PMR 합성 대비 +39 pp 상승 (인코더 사람 인식). H-encoder-saturation 정제 — "인코더 표현 포화 AND 입력-맥락 단순성"; M6 r2 linear-probe AUC 가 인코더-포화 마커로 여전히 유효, 그러나 행동 PMR(_nolabel) 은 더 이상 순수 인코더 readout 아님. | `c568497` |
| 2026-04-25 | **M8e 완료 (cross-source paired analysis)**: M8a + M8d + M8c 를 단일 (모델 × 카테고리 × source_type) 뷰로 통합. 헤드라인 그림 `m8e_cross_source_heatmap.png` 가 논문 Table 1 후보. cross-source PMR shift 확인. | `87c990c` |
| 2026-04-25 | **§4.5 cross-encoder swap 완료 (Idefics2)**: Idefics2-8b (SigLIP-SO400M + Mistral-7B) 를 M8a stim 에 → 1600 추론 GPU 0 에서 8 분. **5 도형 평균 PMR(_nolabel): Qwen 0.838 / LLaVA 0.175 / Idefics2 0.882.** Idefics2 가 PMR + H7 (1/5 vs 1/5 strict) 에서 Qwen 과 동일 패턴. LLaVA 만 outlier. **H-encoder-saturation 이 인코더-family 수준에서 인과 확인** — 인코더 type (SigLIP vs CLIP) 이 LM (Qwen2-7B vs Mistral-7B) 와 무관하게 PMR 천장 결정. | `304e927` |
| 2026-04-25 | **§4.5 ext: Idefics2 on M8d + M8c**: Idefics2 추가 4 config (M8d labeled + label-free, M8c labeled + label-free) → 2160 추론을 GPU 0 에서 11 분. Idefics2 M8d 평균 PMR(_nolabel) **0.890** Qwen **0.869** 와 일치 (vs LLaVA 0.331); Idefics2 M8c **0.417** vs Qwen **0.550** vs LLaVA **0.283** — 3 모델 모두 사진에서 압축. cross-stim 인코더-스왑이 SigLIP 합성 포화, CLIP 비포화, 사진에서 수렴 확인. | `3503cd3` |
| 2026-04-25 | **M9 완료 (일반화 audit / 논문 Table 1)**: 9 (모델 × stim) 셀 × 부트스트랩 CI (5000 iter). **견고 헤드라인**: (1) 인코더 family 가 합성-stim 천장 인과 (SigLIP CI [0.80, 0.92] vs CLIP [0.14, 0.37] 완전 분리); (2) 사진이 인코더 갭 압축 (3 모델 → [0.18, 0.67]); (3) H7 LLaVA-on-synthetic 에서만 견고. **시사 (강등)**: Idefics2 M8d H7 CI [+0.000, +0.094] 가 0 에 닿음 — LM-modulation 가능하나 논문 옹호 불가. PASS/FAIL binarization 을 부트스트랩 CI 로 대체 — "Qwen 1/5 PASS" 패턴이 noise-floor artifact 임을 드러냄. 신규 **H-LM-modulation** 가설 flag. | `6210b13` |
| 2026-04-25 | **M6 r3 완료 (Idefics2 비전-인코더 프로브가 AUC↔PMR 사슬 종결)**: 400 M8a stim × 4 SigLIP-SO400M 레이어를 88 초에 캡처; per-stim PMR 에 대한 layer-wise 로지스틱 회귀 프로브가 **AUC 0.93** 산출 (레이어 9 에서 0.948 peak). 3-점 AUC 사다리 = **Qwen 0.99 / Idefics2 0.93 / LLaVA 0.73** — SigLIP family 가 포화에 클러스터, CLIP 은 headroom. H-encoder-saturation 사슬 `인코더 family → AUC → PMR → H7` 가 4 노드 × 3 모델 점 모두에서 경험적으로 지지. | `1a4313a` |
| 2026-04-25 | **M6 r4 완료 (InternVL3 InternViT 프로브 → 4-모델 사슬)**: 1200+400 M8a 추론 12 분 + 400 × InternViT 4 레이어 47 초 캡처 + 프로브 → **InternVL3 AUC 0.89 / PMR(_nolabel) 0.92**. 4-점 사슬: Qwen 0.99/0.84 → LLaVA 0.73/0.18 → Idefics2 0.93/0.88 → InternVL3 0.89/0.92. 3 개 별개 비-CLIP 인코더 family (SigLIP, SigLIP-SO400M, InternViT) 가 AUC ≥ 0.89 클러스터링; CLIP-ViT-L 만 미달. H-encoder-saturation 이 **비-CLIP-일반** 으로 일반화 (M6 r3 에서 SigLIP 특이적이었음). InternVL3 의 `vision_tower.encoder.layer` (단수) 속성 위한 `_resolve_vision_blocks` 수정 필요. | `d3840ac` |
| 2026-04-25 | **Apples-to-apples 4-모델 M8a-stim AUC**: Qwen + LLaVA 를 M8a stim 에서 재캡처 (4-모델 사슬의 M2-stim M6 r2 AUC 수치 대체). 업데이트 사슬: Qwen 0.88/0.84, LLaVA 0.77/0.18, Idefics2 0.93/0.88, InternVL3 0.89/0.92. M6 r2 의 0.99/0.73 수치는 M2 의 bimodal line/ground 분리 (M8a 의 더 넓은 per-cell PMR 범위보다 probe-friendly) 반영. 헤드라인 (비-CLIP 클러스터 ≥ 0.88, CLIP 0.77) 이 두 stim 출처에서 모두 유지; AUC↔PMR 매핑은 비선형 (≈0.10 AUC 갭 → 0.65 PMR 갭) — AUC 0.85 부근 포화 임계와 일치. | `7e7f101` |
| 2026-04-25 | **M6 r5 완료 (M8c 사진 인코더 프로브 — 4-모델 cross-stim)**: InternVL3 M8c 추론 (180+60 2분) + 4 모델 × 60 사진 캡처 (~3분) + 행동-y + stim-y 프로브. 행동-y AUC 가 cross-stim 역전 (Qwen 0.88→0.44, LLaVA 0.77→0.86, Idefics2 0.93→0.77, InternVL3 0.89→0.59), 그러나 stim-y AUC 는 4 모델 모두 사진에서 1.0 유지. 인코더 식별 능력 균일을 cross-stim 에서 확인 — architecture-level reframe 의 cross-stim 최종 확인. | `166c053` |
| 2026-04-25 | **M6 r6 완료 (LLaVA-Next-Mistral 5번째 모델 점, 2번째 CLIP)**: LLaVA-v1.6-Mistral-7b on M8a (400 라벨 + 400 라벨-free + 400 stim × 4 레이어 × 5 타일 캡처). 평균 PMR(_nolabel) **0.700, 95% CI [0.65, 0.74]**, LLaVA-1.5 바닥 [0.14, 0.21]과 saturated cluster [0.80, 0.92] 사이. 행동-y AUC 0.81; stim-y AUC = 1.0 4개 target 모두. **5-모델 M8a chain lock** (Qwen / LLaVA-1.5 / LLaVA-Next / Idefics2 / InternVL3). 2번째 CLIP 점이 vision-encoder 계열을 PMR 단독 결정자로 배제. LM-controlled counterfactual 아닌 5번째 관측치로 보고: 4개 architecture 축이 동시 변경 (AnyRes tiling, fusion projector, 학습, LM 계열). H-encoder-saturation **5 모델 점 + 2 CLIP 점에서 architecture-level 확인**. | `b2434d4` |
| 2026-04-25 | **M6 r6 cross-stim 부록**: LLaVA-Next 를 M8d + M8c (1620 추론, GPU 0 ~16분). M8d PMR 0.625 [0.58, 0.67] mid-band 유지; M8c PMR 0.417 가 Idefics2 0.417 과 통계적 동일 (photo-collapse 가 5번째 모델에 일반화). **H7 cross-stim**: M8a +0.26 (5/5 PASS, mid-strong), M8d −0.05 (CI [−0.10, −0.01], noise floor), M8c +0.02. M8d H7 collapse 는 saturation 효과가 **아님** (PMR 0.625 천장 한참 아래); 동일 encoder family architecture 스위치가 PMR 헤드룸 있어도 H7 약화. H-encoder-saturation reframe 5 모델 × 3 stim 에서 유지. H-LM-modulation 여전히 suggested-only (두-Mistral M8d H7 ≈ 0 클러스터링은 advisor 지적대로 multi-axis-confounded). | `524e32b` |
