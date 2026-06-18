# Physical-Mode Activation 연구 리뷰 (2026-04-25)

> **작성 목적.** `references/`와 `docs/`에 누적된 연구 방향, 가설, 실험 결과가 서로 일관되는지 점검하고, 현재 진척이 NeurIPS / EMNLP 같은 일류 학회에 채택될 가능성이 어느 정도인지, 어떤 방향으로 보강하면 채택률을 높일 수 있는지를 정리한다. 마지막으로 비슷한 시기에 등장한 경쟁/유사 연구를 살펴 차별점을 점검한다.
>
> **약어 풀이.** VLM = Vision-Language Model(시각-언어 모델), LM = Language Model(언어 모델), PMR = Physics-Mode Priming Rate(물리 모드 점화율), GAR = Gravity-Align Rate(중력 정렬률), RC = Response Consistency(응답 일관성), VTI = Visual-Textual Intervention(시각-언어 개입, 헛소리 억제용 잠재공간 스티어링), SAE = Sparse Autoencoder(희소 자기부호화기), SIP = Semantic Image Pairs(의미적 이미지 쌍, 잡음 없는 인과 매개 분석용), AUC = Area Under the ROC Curve(분류 성능 지표), IE = Indirect Effect(간접 효과, 활성치 패칭 측정값), MLLM = Multimodal Large Language Model(멀티모달 대형 언어 모델), VLA = Vision-Language-Action(시각-언어-행동 로봇 모델), CLIP / SigLIP / DINOv2 = 이미지 인코더 백본, ARR = ACL Rolling Review(ACL 계열 학회의 표준 리뷰 사이클).

---

## 0. TL;DR

| 항목 | 평가 |
|---|---|
| **연구 방향과 데이터의 일관성** | ✅ 매우 일관됨. project.md의 H1-H7 및 파생 가설(H-boomerang, H-locus, H-regime)이 단계적으로 검증되며 누적된다. |
| **방법론 진척** | M0-M5a 완료(5개 sub-task 중 ST1-ST3과 ST4 Phase 1+2). ST4 Phase 3(SIP + SAE)과 ST5(cross-model)가 미완. |
| **EMNLP 2026 long 채택 가능성** | **borderline → 충분히 가능.** 단, ① cross-model 1개 이상, ② photorealistic 자극, ③ label-free prompt, ④ SIP 패칭 중 최소 두 가지를 더하면 합격선에 진입한다. |
| **NeurIPS 2026 main 채택 가능성** | **현재로서는 낮음.** SAE 기반 해석 + cross-model 일반화 + human baseline까지 완성해야 main의 mech-interp 트랙 컷에 든다. |
| **EMNLP 2026 Findings 또는 Workshop** | 현재 패키지 그대로도 통과 가능성이 높다(소규모 의의를 갖는 short/long-style). |
| **가장 중요한 보강** | ① cross-model 스윕(M6) ② photorealistic 자극(axis A 확장) ③ SIP 활성치 패칭(M5b) ④ label-free prompt 미니 실험 |

핵심 요약: **연구가 한 줄로 요약되는 단계까지 도달했다.** 즉 *"Qwen2.5-VL의 시각 인코더는 원이 공이 될 조건을 완벽히 알지만, 언어 모델은 그 정보를 일부만 해석에 반영한다 — 그리고 우리는 layer 10에 α=40 벡터를 더해 그 게이트를 강제로 열 수 있다."* 이 한 문장이 **EMNLP 2026 long 트랙의 메인 클레임으로 충분히 매력적이다.** 다만 single-model + 프로그램 자극 한계가 약점이므로 보강이 필요하다.

---

## 1. 연구 흐름 점검 — project.md, roadmap.md, 실험 결과의 정합성

### 1.1 한눈에 보는 milestone × 가설 매트릭스

| milestone | 검증된 가설 | 새로 도입된 가설 | 한 줄 결론 |
|---|---|---|---|
| **M1 — Pilot** (480 추론) | H2 강한 지지, H1 부분 지지 | H4(open–forced 격차), H5(ground 효과), H6(arrow vs shadow) 후보 | 행동 베이스라인. ground 1줄이 PMR을 +36%p 끌어올린다는 가장 깨끗한 단일 효과 발견. |
| **M2 — MVP-full** (2880 추론) | H1, H2, H4, H6 정량화 | H7(label = physics regime) | S-curve 단조 회복. "circle / ball / planet" 라벨이 같은 그림에서 정적 / 굴러내림 / 공전 응답을 각각 만들어내는 결정적 사례. |
| **M3 — Vision encoder probing** | H-boomerang 강한 지지 | (-) | **인코더 AUC = 1.00 vs 행동 PMR 0.28-0.95.** "인코더는 알지만 디코더가 차단" 패턴을 메커니즘 수준에서 정량화. |
| **M4 — LM logit lens + per-layer probe** | H-boomerang 확장, H7 추가 지지 | H-locus (LM 후반 + decoding 병목) | **LM hidden state AUC = 0.94-0.95 (모든 층 일정)**. 정보가 LM 끝까지 보존되며 decoding 단계에서 손실됨. label이 L5부터 physics margin을 끌어올림. |
| **M5a — VTI causal steering** | H-boomerang 인과 지지, H-locus 지지(L10) | H-regime ("object-ness binary") | **L10에 α=40 v 주입 → 10/10이 D(추상) → B(정적 물체)로 뒤집힘.** 다른 층에서는 동일 α로 변화 없음. |

→ **모든 가설이 데이터에 의해 직접 갱신되며 연결된다.** project.md의 H1-H3 → 파생된 H4-H7 → 메커니즘 수준 H-boomerang / H-locus / H-regime으로 자연스럽게 확장되었고, 새로운 발견(H7 = label이 어떤 물리 체계를 호출할지를 정함)이 모순 없이 누적된다.

### 1.2 하나의 그림으로 보는 정보 흐름

```
[stimulus]
    │  (axis A: line → filled → shaded → textured)
    │  (axis B: blank → ground → scene)
    │  (axis C: none → cast_shadow → motion_arrow → both)
    │  (axis D: circle / ball / planet)
    ▼
시각 인코더 (Qwen2.5-VL의 SigLIP, 32 블록)
    │  ─── 모든 axis에 대해 AUC = 1.00 (M3)
    │  ─── 정보 손실 0
    ▼
LM hidden states (28-layer Qwen 디코더)
    │  ─── 모든 captured 층에서 AUC = 0.94-0.95 (M4)
    │  ─── physics margin이 L5부터 양수, L25에 +4.0으로 폭주
    │  ─── label 효과는 object_level 효과의 7배
    ▼
decoding (token sampling, T=0.7)
    │  ─── forced-choice 정확도 0.66 ↘ ~30%p 손실
    │  ─── 여기가 "boomerang의 끝"
    ▼
behavioral PMR
    │
    │  CAUSAL INTERVENTION (M5a)
    │  → L10에 α=40 v 주입 시 D → B로 일관 전환
```

이 그림이 **paper의 Figure 1 후보**다.

### 1.3 일관성에서 나타나는 약점 4가지

| # | 약점 | 영향 | 현재 대응 |
|---|---|---|---|
| 1 | 단일 모델 검증 (Qwen2.5-VL-7B 단독) | "이건 Qwen-specific 현상 아닌가" 리뷰어 자동 반사 | M6 (cross-model) 미완 |
| 2 | **프로그램 자극만 사용 → 인코더 AUC = 1.00 trivial** | reviewer가 "인코더가 정말 잘하는 게 아니라 자극이 너무 단순한 것" 지적 가능. 실제로 m3 insights §5.1에서 인정함. | photorealistic 변형(FLUX/SDXL/Blender) 미실시 |
| 3 | switching-layer metric이 label-prime 때문에 degenerate (모든 샘플이 L5에서 switch) | M4의 핵심 그래프 중 하나가 무의미해짐 | label-free prompt 미니 실험 미실시 (`docs/next_steps.md`에 등재) |
| 4 | M5a 인과 결과 표본이 작음 (10 자극 × 1 cell) | "L10 sweet spot 일반화" 클레임 약함 | 다른 추상 baseline 셀(`filled+blank+none`, `line+ground+none`)에 대해 동일 검증 필요 |

→ 1번과 2번은 **반드시 다음 라운드에서 해결해야 venue 통과한다.** 3번과 4번은 추가 1-2일짜리 실험이라 즉시 처리 가능.

### 1.4 project.md ↔ roadmap.md ↔ 실험 결과 사이의 모순 여부

| 점검 항목 | 결과 |
|---|---|
| project.md §2.2 H1-H3 | M1-M2 데이터로 직접 채점됨. H1 supported / H2 quantified / H3 untested. |
| project.md §2.2의 5축 factorial 설계 | M2에서 axis E(scene consistency) 명시적으로 deferred. 나머지 4축은 그대로 실시. event_template은 pilot 데이터로 downgrade(정당한 reallocation). |
| project.md §2.4 cue 위계 가설(3D shading > shadow + ground > material > perspective > motion blur) | M2 데이터로 부분 검증. shadow 단독 +17.5%p (Kersten cue 확인). object_level의 단조 +9%p보다 ground/cue 효과가 큼. |
| project.md §3.2 헤드라인 후보 4개 | 1, 2, 4번은 데이터로 강화. 3번("layer 19 head 14")은 attention 캡처가 없어 미검증. |
| roadmap.md §1.3 hypothesis scorecard | §1.1 매트릭스와 합치. 각 milestone 직후 즉시 갱신되어 drift 없음. |
| 모순/누락 | **없음.** 단, project.md의 §2.5 "구체적 attention head 지목" 클레임은 아직 인과 테스트가 빈 상태이므로 paper에 적기 전에 M5b 완성 필요. |

→ **문서 자체에는 내적 모순이 없다.** 다만 paper로 변환할 때는 "M5b가 빠진 상태의 클레임"과 "M5b까지 끝낸 상태의 클레임"이 다르다는 점을 분명히 인지하고 venue를 골라야 한다.

---

## 2. 연구 진행 방향 권고

### 2.1 그대로 진행해도 되는가?

**그대로 진행해도 무방하다.** 단, 채택 가능성을 위해 다음 4가지 보강을 우선순위 순으로 권장한다.

### 2.2 우선순위별 보강안

| 순위 | 보강 항목 | 노력(시간) | 채택률 기여 | 비고 |
|---|---|---|---|---|
| 1 | **M5b — SIP + activation patching** | 1-2주 | ⭐⭐⭐⭐ | "L10 sweet spot"을 "특정 (layer × head)가 인과적으로 필요"로 격상시킴. NeurIPS 자격선 근접. attentions 재캡처 + ~120 SIP 쌍 + TransformerLens hooks. |
| 2 | **M6 — cross-model 일부 (LLaVA-1.5 + InternVL2)** | 2-3주 | ⭐⭐⭐⭐ | "Qwen-specific 아닌가" 리뷰어 차단. 동일 sweet-spot이 30-40% LM depth에 모이면 일반화 클레임 가능. |
| 3 | **Axis A 확장: photorealistic FLUX/SDXL 자극** | 1주 | ⭐⭐⭐ | 인코더 AUC=1.0 caveat 해결. 추상 ↔ 사진 사이의 진짜 transition을 측정. |
| 4 | **Label-free prompt 미니 실험** | 1일 | ⭐⭐ | M4의 switching-layer metric 살리기. `prompts.py`에 `open_no_label` 추가만 하면 됨. |
| 5 | M5a 확장: 다른 abstract baseline 셀에서 L10 α=40 검증 | 1일 | ⭐⭐ | "10/10 flip"의 일반성 검증. PMR scorer를 first-letter 기반으로 보강하면 동시 해결. |
| 6 | Negative α (abstract-ness 방향) 테스트 | 0.5일 | ⭐⭐ | reviewer가 무조건 묻는 "역방향 작동하는가" 즉시 답변. |
| 7 | **M7 — Human baseline (50 stim × 20 Prolific raters)** | 1-2주 + ~$300 | ⭐⭐⭐ (NeurIPS 한정) | "사람도 같은 cue에 같은 방식으로 반응하는가" 답변. NeurIPS scope에서 차별 포인트. |
| 8 | SAE 학습 (stretch) | 2-3주 | ⭐⭐⭐ (NeurIPS 한정) | 헤드라인 후보 #4. monosemantic "ground / shading / object-ness" feature 발견 시 mech-interp 트랙 컷 진입. |

### 2.3 "EMNLP 2026 long을 노린다면" 최소 패키지

- 현재 M5a까지 + **M6 (cross-model 1개 추가)** + **label-free 미니 실험** + **SIP 패칭 1개 layer만이라도 검증**
- 데드라인: ARR 2026 May cycle = **2026-05-25** (UTC-12). 약 한 달 남음.
- 이 패키지는 완성 가능. 단, M6가 시간 빡빡함.

### 2.4 "NeurIPS 2026 main을 노린다면" 풀 패키지

- 위 EMNLP 패키지 + **M5b SIP 완전판 (attention knockout 포함)** + **SAE feature 발견 1개 이상** + **human baseline**
- NeurIPS 2026 main 데드라인: 보통 5월 중순. EMNLP보다 약간 빠름.
- 현실적으로는 EMNLP 2026 commitment를 1차 목표로 하고, 동일 버전을 NeurIPS 2027 또는 ICLR 2027에 풀 패키지로 보내는 2단계 전략이 안전.

### 2.5 venue 선택 권고

| venue | 적합성 | 권고 |
|---|---|---|
| **EMNLP 2026 long** (Interpretability & Analysis of Models for NLP) | ⭐⭐⭐⭐ | **1순위 타깃.** Pixels-to-Principles, NOTICE, BlindTest가 모두 NAACL/EMNLP/ACCV 계열이라 자연스러운 follow-up 위치. |
| **EMNLP 2026 long** (Multimodality and Language Grounding) | ⭐⭐⭐⭐ | 동급 1순위. "grounding failure" 프레임이 정확히 부합. |
| **NeurIPS 2026 main** (Interpretability + Eval & Analysis) | ⭐⭐ | M5b + SAE + cross-model 완성 시. 현재 패키지론 어려움. |
| **NeurIPS 2026 D&B Track** (PhysCue 데이터셋) | ⭐⭐⭐ | 데이터셋 단독 contribution 가능. 단, 메인 페이퍼와 venue 충돌. |
| **EMNLP 2026 Findings** | ⭐⭐⭐⭐ | 현재 패키지로 거의 확실. 안전판. |
| **ICLR 2027 main** | ⭐⭐⭐ | NeurIPS 2026이 안 되면 풀 패키지로 재시도. |
| **ACL 2026 / NAACL 2027** | ⭐⭐ | 가능하지만 EMNLP가 분야별로 더 적합. |

### 2.6 연구 방향 자체의 수정 권고

**대폭 수정은 권장하지 않는다.** 가설/방법론은 이미 데이터로 충분히 검증되어 일관성이 있고, 헤드라인 클레임도 명확하다. 다만 **paper narrative 재구성**은 권장한다.

**현재 narrative 안:**
"VLM은 추상→구체 시각 cue 임계점에서 mode switching이 일어나며, 우리는 그 임계 layer를 인과적으로 발견하고 steering으로 조작했다."

**개선된 narrative 안 (M3-M5a 결과 반영):**
"VLM의 시각 인코더는 추상 vs 물리적 객체를 완벽히 구분하지만, 언어 모델은 그 정보의 일부만 해석에 반영한다 — 우리는 그 게이트를 layer 10의 단일 직선 방향에 국한할 수 있고, 거기에 α=40 만큼 더하면 Qwen2.5-VL이 빈 원을 정적 물체로 인식하기 시작한다. 또한 라벨('circle' / 'ball' / 'planet')은 어떤 물리 체계가 호출될지를 정한다 — visual cue와 language prior는 독립적으로 mode를 결정한다."

→ "boomerang"과 "label-as-regime"을 paper의 두 축으로 재구성. M3+M4+M5a를 묶어 **하나의 인과 chain (correlation → causation)** 으로 제시할 수 있다.

---

## 3. 비슷한 시기 발표된 경쟁/유사 연구 매핑

채택 가능성 분석에서 가장 중요한 것은 **이미 비슷한 페이퍼가 나왔는지**다. 2025-2026년 검색을 통해 다음을 확인했다.

### 3.1 직접 경쟁 (가장 가까운 work)

| 논문 | venue | 위협도 | 우리의 차별점 |
|---|---|---|---|
| **Pixels to Principles** (Ballout et al., arXiv:2507.16572, 2025-07) | preprint | 🔴 직접 경쟁 | (a) parametric abstraction axis 사용, (b) next-state prediction(우리는 verb 분포; 그쪽은 plausibility 판단), (c) **인과 개입 + steering** 추가. |
| **Can you SPLICE it together?** (Ballout et al., **EMNLP 2025 Findings**) | EMNLP 2025 | 🟡 경쟁 가능성 | 같은 저자가 EMNLP에 후속 게재 중 — paper congestion. 단 우리는 mechanistic interpretation 깊이가 다름. |
| **Vision Language Models are Biased** (Vo et al., arXiv:2505.23941, **ICLR 2026 accepted**) | ICLR 2026 | 🟡 직접 영감원 | counterfactual counting(다리 5개 개) — memorized prior 우세 패턴 입증. 우리는 "abstract → physical mode" 차원으로 이전 + 인과 메커니즘 추가. |
| **Vision Language Models Are Blind** (Rahmanzadehgervi et al., ACCV 2024 Oral, arXiv:2407.06581) | ACCV 2024 | 🟡 패턴 직접 경쟁 | "encoder knows decoder doesn't" 패턴 같음. physics 차원이 우리만의 것. |

### 3.2 방법론 도구 (선행 연구 — 우리가 사용하는 방법)

| 논문 | venue | 위협도 | 사용 방식 |
|---|---|---|---|
| **Towards Interpreting Visual Information Processing in VLMs** (Neo et al., **ICLR 2025**) | ICLR 2025 | 🟢 도구 | 우리 M4의 정확한 방법론적 청사진. layer 15-24가 LLaVA-1.5 sweet spot이라는 발견. Qwen2.5-VL에서 동일 패턴 확인 → 일반화 증거. |
| **What Do VLMs NOTICE?** (Golovanevsky et al., **NAACL 2025**) | NAACL 2025 | 🟢 도구 | M5b에서 사용할 SIP 방법. NOTICE 파이프라인 그대로 적용 예정. |
| **The Hidden Life of Tokens (VTI)** (Liu et al., **ICLR 2025**) | ICLR 2025 | 🟢 도구 | 우리 M5a의 직접 기반. 헛소리 억제용으로 발표된 방법을 "physics mode 강제 점화"로 전용. |
| **Sparse Autoencoders Learn Monosemantic Features in VLMs** (Pach et al., **NeurIPS 2025**) | NeurIPS 2025 | 🟢 도구 | M5b stretch — SAE로 "shading", "ground" feature 분해. |
| **Mechanistic Interpretability Meets VLMs** (ICLR 2025 Blogpost) | ICLR 2025 | 🟢 메타 | "mech interp toolset이 physics에 거의 적용 안 됨" 명시 → 우리 work의 niche 정당화. |

### 3.3 대척점 또는 새로 등장한 위협 (2025년 4분기 이후)

| 논문 | venue | 위협도 | 우리에게 미치는 영향 |
|---|---|---|---|
| **LLMs Can Compensate for Deficiencies in Visual Representations** (arXiv:2506.05439) | preprint | ⚠️ **반대 발견** | "LM이 visual 결함을 *보완*한다"는 정반대 클레임. 우리는 "LM이 게이트를 *닫는다*"고 주장. 둘이 양립 가능(여기는 generative ↔ 거기는 classification)임을 paper에서 명시 필요. |
| **Rethinking Visual Information Processing in MLLMs** (arXiv:2511.10301, 2025-11) | preprint | 🟡 직접 경쟁 가능성 | 매우 최신. M4의 layer-wise 분석을 별도 각도에서 본다. 우리 paper에서 cite + 차별점(physics 차원) 제시 필요. |
| **Mechanistic Interpretability for Steering VLA Models** (arXiv:2509.00328) | preprint | 🟢 다른 도메인 | 같은 도구를 로봇 VLA에 적용. 우리는 정적 이미지의 "physics mode prediction" → 직접 충돌 없음. |
| **Steering Vector Fields** (Cho et al., 2026-02) | preprint | 🟡 도구 진화 | static vector → 미분가능 scoring function. M5a보다 진보된 방법. paper에서 "future work"로 인정 필요. |
| **Venkatesh & Kurapath** (steering vectors are non-identifiable, 2026-02) | preprint | 🔴 방법론 비판 | "여러 다른 vector가 동일 행동을 만든다"는 증명. 우리 L10 v_unit가 unique하지 않을 수 있다는 약점 — paper에서 명시 + IE 검증으로 보강해야 함. |
| **Causal Interpretation of SAE Features in Vision** (arXiv:2509.00749) | preprint | 🟢 도구 | M5b stretch에서 SAE 후속 분석에 사용 가능. |
| **Interpreting CLIP with Hierarchical SAE** (PatchSAE) | OpenReview | 🟢 도구 | M5b stretch — patch-level interpretability. |
| **Interpreting VLMs with VLM-LENS** (EMNLP 2025 demos) | EMNLP 2025 | 🟢 도구 | 우리 코드 베이스를 VLM-LENS에 호환되게 만들면 reviewer에게 사용성 어필. |

### 3.4 전체 정리 표

| 카테고리 | 논문 수 | 대표 | 우리 위치 |
|---|---|---|---|
| 우리와 같은 질문, 같은 모델 | 1 | Pixels-to-Principles | 직접 follow-up 위치(차별 명확) |
| 우리와 같은 패턴, 다른 도메인 | 3 | VLMs Biased / Blind | "physics 차원 + causal" 차별 |
| 우리가 빌려온 방법론 | 5 | Neo / NOTICE / VTI / SAE-VLM | 정통 사용자 |
| 새로 등장한 위협 (반례 가능) | 1 | LLMs Compensate | paper에서 변별 분석 필요 |
| 동시기 등장한 직접 경쟁 가능성 | 2 | Rethinking VIP / SPLICE | cite + 차별 명시 |
| 방법론 진화 (우리 한계 명시) | 2 | Steering Vector Fields / non-identifiability | future work에 등재 |

→ **종합: 직접 동일 페이퍼는 발견되지 않았다.** Pixels-to-Principles가 가장 가깝지만 (a) parametric abstraction axis, (b) 인과 개입 (M5a), (c) label × physics regime 발견(H7) 모두 우리만의 차별 포인트로 살아있다.

---

## 4. 채택 가능성 정량 분석

### 4.1 venue별 채택률 추정 매트릭스

| venue | 현재 (M5a) | + M6 (cross-model) | + M5b (SIP+패칭) | + photorealistic | + human baseline | + SAE |
|---|---|---|---|---|---|---|
| **EMNLP 2026 Findings** | 🟢 70% | 🟢 80% | 🟢 85% | 🟢 90% | 🟢 92% | 🟢 95% |
| **EMNLP 2026 long (Interpretability)** | 🟡 30% | 🟢 50% | 🟢 60% | 🟢 70% | 🟢 75% | 🟢 80% |
| **NeurIPS 2026 main** | 🔴 5% | 🔴 15% | 🟡 30% | 🟡 40% | 🟢 55% | 🟢 65% |
| **NeurIPS 2026 D&B** (데이터셋 단독) | 🟡 30% | 🟢 45% | — | 🟢 55% | 🟢 60% | — |
| **ICLR 2027 main** | 🔴 5% | 🟡 25% | 🟡 35% | 🟢 50% | 🟢 60% | 🟢 70% |

(채택률은 정성 추정치다. 색깔: 🔴 어려움, 🟡 가능성 있음, 🟢 합리적 기대.)

### 4.2 권장 전략

**Plan A (현실적 1순위):** 한 달 내에 M6 + label-free + SIP 부분 검증 → ARR 2026 May cycle (5-25 마감) → EMNLP 2026 long commit. 실패 시 자동으로 EMNLP 2026 Findings 또는 ARR 다음 사이클로 연결.

**Plan B (스트레치):** Plan A + photorealistic 자극 + SAE 1개 feature 발견 → 동시에 NeurIPS 2026 main 제출 (5월 중순 마감). 두 venue 동시 제출은 venue 정책상 불허이므로 EMNLP 또는 NeurIPS 한쪽만 선택해야 함.

**Plan C (안전판):** 현재 M5a 결과만으로 NeurIPS Mech Interp Workshop 또는 ICLR 2026 Workshop 제출 → 피드백 수집 → 풀 페이퍼는 ICLR 2027 또는 EMNLP 2027.

### 4.3 채택률을 결정짓는 4가지 변수

| 변수 | 영향 | 현재 상태 | 권장 액션 |
|---|---|---|---|
| **cross-model 일반화 여부** | EMNLP main에서 가장 큰 변수 | 미검증 | M6 최소 1개 추가 |
| **자극의 ecological validity** | reviewer의 "synthetic only" 비판 차단 | 프로그램 자극만 | photorealistic 추가 |
| **인과 증거의 다층화** | "VTI 단일 결과로는 약함" 차단 | M5a 1개 layer × 1개 cell | SIP + attention knockout |
| **headline claim의 명료성** | 모든 venue 공통 | 강함 ("L10 α=40 flips 10/10") | 문장 그대로 paper 제목·abstract에 |

---

## 5. paper narrative 권장안

### 5.1 두 가지 가능한 frame

**Frame 1 — "Boomerang" (M3-M4 중심) → EMNLP 적합**

| 섹션 | 내용 |
|---|---|
| Abstract | "VLM의 시각 인코더는 알지만 언어 모델은 말하지 않는다 — 그리고 우리는 이를 인과적으로 뒤집을 수 있다." |
| Figure 1 | M1 cell (1) vs (2) 미니멀 페어 (line+blank vs line+ground) |
| Figure 2 | M3 boomerang 차트 (encoder AUC 1.0 vs behavioral PMR 0.28-0.95) |
| Figure 3 | M4 information trajectory (per-layer probe AUC + physics margin) |
| Figure 4 | M5a causal steering result (L10 α=40 flips 10/10) |
| Figure 5 | M6 cross-model replication |
| Figure 6 (optional) | M5b SAE features OR human baseline |

**Frame 2 — "Mechanism" (M5a-M5b 중심) → NeurIPS 적합**

| 섹션 | 내용 |
|---|---|
| Abstract | "우리는 VLM의 'abstract vs physical object' 결정을 LM의 layer 10에 있는 단일 직선 방향으로 국한했다." |
| Figure 1 | PhysCue stimulus grid (M2 axis A × C) |
| Figure 2 | layer-wise probe heatmap (M3 + M4 통합) |
| Figure 3 | activation patching IE curve (M5b) |
| Figure 4 | steering vector intervention table (M5a) |
| Figure 5 | SAE monosemantic feature ablation (M5b stretch) |
| Figure 6 | cross-model L10-equivalent localization |

→ 현재 데이터로는 Frame 1이 자연스럽다. M5b/SAE 완성 시 Frame 2로 전환.

### 5.2 라이벌과의 차별 1줄 sentence (paper 도입부 후보)

> "Pixels-to-Principles (Ballout et al., 2025) demonstrated that the bottleneck for VLM physics reasoning is at the vision-language interface; we extend their finding to (i) a parametric abstraction axis, (ii) a causal mechanism (Layer 10 steering vector), and (iii) a label-driven physics regime (the same image becomes static when called *circle*, rolls when called *ball*, and orbits when called *planet*)."

---

## 6. 즉시 실행 가능한 7가지 액션

| # | 액션 | 소요 | 결과 |
|---|---|---|---|
| 1 | `prompts.py`에 `open_no_label` 변형 추가 + 240 자극 재추론 | 0.5일 | M4 switching-layer metric 부활 |
| 2 | M5a를 다른 abstract baseline (`filled+blank+none`, `line+ground+none`)에 동일 적용 | 1일 | "L10 sweet spot 일반성" 미니 검증 |
| 3 | Negative α 테스트 (textured+ground+both → L10에 -α v 주입) | 0.5일 | 역방향 작동 확인, reviewer 차단 |
| 4 | PMR scorer에 first-letter 모드 추가 (`forced-choice 시 PMR 대신 A/B/C/D`) | 0.5일 | M5a의 forced-choice false positive 해결 |
| 5 | LLaVA-1.5-7B만 다운로드 + 같은 factorial 480 자극 1차 추론 | 3-4일 | M6 부분 (cross-model 1개) |
| 6 | SIP 매니페스트 자동 생성 (M2 factorial에서 single-axis-differ pair 추출) | 1일 | M5b 첫 단계 |
| 7 | FLUX.1-schnell로 textured_photo 50개 생성 + axis A level 5 확장 | 2-3일 | photorealistic caveat 해결 |

→ **한 달 내 1+2+3+4+5+6 모두 가능.** 7번은 GPU 필요 시간이 있어 별도 일정.

---

## 7. 결론

### 7.1 한 줄 답변

| 질문 | 답 |
|---|---|
| 연구 방향이 일관되는가? | ✅ 매우 일관됨. project.md → roadmap.md → 실험 결과의 chain이 깨끗하게 연결됨. |
| 이대로 진행해도 되는가? | ✅ 그렇다. 단, cross-model + photorealistic 보강이 EMNLP/NeurIPS 채택의 임계점이다. |
| 수정 방향은? | 가설/방법론 자체는 그대로. **paper narrative를 "boomerang + label-as-regime"로 재구성**하고, M6 + M5b + label-free + photorealistic 보강에 한 달 투자. |
| EMNLP 2026 채택 가능한가? | EMNLP Findings는 현재 패키지로 거의 확실. EMNLP long은 M6 + label-free 추가 시 50% 이상. |
| NeurIPS 2026 채택 가능한가? | 현재로선 어렵다. 풀 패키지 (M5b + SAE + cross-model + human baseline) 완성 시 50%대로 진입. |
| 비슷한 논문이 있는가? | Pixels-to-Principles가 가장 가까우나 우리만의 (a) parametric abstraction axis, (b) 인과 개입, (c) label-as-regime이라는 3개 차별점 모두 살아있다. |

### 7.2 가장 중요한 한 가지 권고

**ARR 2026 May cycle (마감 5-25)에 EMNLP 2026 long을 노리고, 한 달 내에 M6 부분(LLaVA-1.5 1개 모델) + label-free 미니 실험 + SIP 패칭 1개 layer 검증을 추가하라.** 이 패키지면 EMNLP long Interpretability 트랙에서 채택률 50% 이상이 합리적 기대치다. NeurIPS는 풀 패키지 완성 후 다음 라운드(2027 ICLR 또는 NeurIPS 2027)로 미루는 것이 현실적이다.

---

## 부록 A — 검색에 활용한 출처

### A.1 직접 경쟁 / 동일 영역
- [Pixels to Principles: Probing Intuitive Physics Understanding in Multimodal Language Models (Ballout et al., 2025)](https://arxiv.org/abs/2507.16572)
- [Can you SPLICE it together? (Ballout et al., EMNLP 2025 Findings)](https://aclanthology.org/2025.findings-emnlp.604/)
- [Vision Language Models are Biased (Vo et al., 2025, ICLR 2026 accepted)](https://arxiv.org/abs/2505.23941)
- [Vision Language Models Are Blind (Rahmanzadehgervi et al., ACCV 2024 Oral)](https://arxiv.org/abs/2407.06581)

### A.2 방법론 기반
- [Towards Interpreting Visual Information Processing in VLMs (Neo et al., ICLR 2025)](https://arxiv.org/abs/2410.07149)
- [What Do VLMs NOTICE? (Golovanevsky et al., NAACL 2025)](https://aclanthology.org/2025.naacl-long.571.pdf)
- [The Hidden Life of Tokens / VTI (Liu et al., ICLR 2025)](https://arxiv.org/abs/2502.03628)
- [Sparse Autoencoders Learn Monosemantic Features in VLMs (Pach et al., NeurIPS 2025)](https://github.com/ExplainableML/sae-for-vlm)
- [Mechanistic Interpretability Meets VLMs (ICLR 2025 Blogpost)](https://iclr-blogposts.github.io/2025/blog/vlm-understanding/)
- [Interpreting VLMs with VLM-LENS (EMNLP 2025 demos)](https://aclanthology.org/2025.emnlp-demos.68.pdf)

### A.3 신규 위협 / 동시기 경쟁
- [LLMs Can Compensate for Deficiencies in Visual Representations (arXiv:2506.05439)](https://arxiv.org/html/2506.05439)
- [Rethinking Visual Information Processing in MLLMs (arXiv:2511.10301, 2025-11)](https://arxiv.org/html/2511.10301)
- [Mechanistic Interpretability for Steering VLA Models (arXiv:2509.00328)](https://arxiv.org/html/2509.00328)
- [Causal Interpretation of SAE Features in Vision (arXiv:2509.00749)](https://arxiv.org/html/2509.00749v1)
- [Interpreting CLIP with Hierarchical SAE (PatchSAE)](https://openreview.net/forum?id=5MQQsenQBm)

### A.4 venue 정보
- [Call for Main Conference Papers — EMNLP 2026](https://2026.emnlp.org/calls/main_conference_papers/)

### A.5 cognitive-science / 영감원
- [Intuitive physics understanding emerges from self-supervised pretraining on natural videos / V-JEPA (Garrido et al., 2025)](https://arxiv.org/html/2502.11831v1)
- [Spelke "What babies know" Vol. 1 (2022)](https://onlinelibrary.wiley.com/doi/10.1111/mila.12482)

---

## 부록 B — 자극 예시 (M1 pilot 캡처에서)

| 셀 | 이미지 | 역할 |
|---|---|---|
| `line + blank + none` | <img src="docs/figures/01_line_blank_none.png" width="180"> | 가장 추상 — 행동 베이스라인 |
| `line + ground + none` | <img src="docs/figures/02_line_ground_none.png" width="180"> | 추상 원 + 바닥 1줄 — ground 효과의 미니멀 페어 |
| `shaded + ground + none` | <img src="docs/figures/03_shaded_ground_none.png" width="180"> | 3D shading + 바닥 — 정통 "공-떨어짐" 자극 |
| `textured + ground + arrow_shadow` | <img src="docs/figures/04_textured_ground_arrow_shadow.png" width="180"> | 최대 cue — 텍스처 공 + 바닥 + 화살표 + 그림자 |
| `filled + blank + wind` | <img src="docs/figures/05_filled_blank_wind.png" width="180"> | wind cue 무인지 사례 — 좌측 회색 호를 VLM이 안 읽음 |
| `textured + blank + none` | <img src="docs/figures/06_textured_blank_none.png" width="180"> | 3D 텍스처 + 바닥 없음 — object-level 효과 단독 |

---

## 부록 C — 가장 강한 정량 결과 모음 (paper의 abstract용)

| # | 결과 | 출처 |
|---|---|---|
| 1 | **single ground line이 PMR을 +36%p 끌어올린다** (blank 0.49 → ground 0.85) | M1 pilot |
| 2 | **open-ended PMR 0.93 vs forced-choice PMR 0.66** — open에서는 model이 자기 자극을 abstract라고 부르는 비율 0.2% (3/1440) | M2 |
| 3 | **같은 이미지에서 라벨만 바꾸면** circle → 정적 / ball → 굴러내림 / planet → 공전 | M2 정성 |
| 4 | **인코더 AUC = 1.00 vs 행동 PMR 0.28-0.95** — encoder는 모든 axis를 완벽 분리하지만 LM은 일부만 사용 | M3 |
| 5 | **추상도가 높을수록 boomerang gap이 크다** (line +36%p > textured +24%p) — 메커니즘 수준의 H4 증거 | M3 |
| 6 | **LM의 모든 captured 층에서 PMR probe AUC = 0.94-0.95** — 정보가 LM 끝까지 살아있음 | M4 |
| 7 | **physics margin이 L25에 +4.0으로 amplify되지만 decoding은 그 신호를 부분만 사용** | M4 |
| 8 | **L10에 α=40 v 주입 → 10/10이 D(추상) → B(정적 물체)로 일관 전환** — 다른 layer에서는 동일 α로 변화 없음 | M5a |
| 9 | **steering 방향이 "object-ness" binary** (B로만 가고 A로 안 감) — physics regime 선택은 라벨 주도 | M5a |

이 9개가 paper의 abstract에 들어가야 할 정량 결과 후보다.
