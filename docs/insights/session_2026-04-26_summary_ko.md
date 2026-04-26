---
session: 2026-04-26 자율 (이어짐)
date: 2026-04-26
status: complete
scope: §4.7 (RC per-axis 안정성) + §4.11 (categorical H7 regime 분포)
commits: 309bdf6 → bbf01f9
---

# 세션 2026-04-26 — §4.7 + §4.11

## 이 세션의 산출물

기존 M8d / M8a label-free 데이터를 재사용하는 §4 add-on 2 개, 새 추론
없음. §4 backlog 의 분석-only 항목 마무리.

1. **§4.11 — categorical H7 regime 분포** (commit `309bdf6`).
   `classify_regime` 을 4-모델 M8d label-free + labeled run (Qwen /
   LLaVA-1.5 / LLaVA-Next / Idefics2) 에 적용. 4×3×4 stacked-bar matrix
   가 (모델 × 카테고리 × label_role) 별 kinetic / static / abstract /
   ambiguous 비율 보임.

2. **§4.7 — per-axis RC 결정 안정성** (commit `bbf01f9`).
   T=0.7 하의 RC 를 M8a label-free 에서 per-axis 결정 안정성으로 재해석.
   5 모델 × 3 축 (object_level / bg_level / cue_level) × {각 2-4 level}.

## 헤드라인 발견

### §4.11 — regime 분포가 binary H7 가 가린 모델 차이를 드러냄

![§4.11 4-모델 M8d regime 분포](../figures/sec4_11_regime_distribution_4model.png)

- **Qwen + Idefics2**: 모든 곳에서 saturated kinetic (~95%). Qwen
  `person × exotic` (statue) 만 ~30% static.
- **LLaVA-1.5**: 가장 regime-discriminative. `car × abs` (silhouette) 가
  kinetic 을 28% 로 낮추고 ambiguous 70%.
- **LLaVA-Next**: intermediate. `person × exotic` (statue) 가 3-way split
  보임 (30% kinetic + 25% static + 25% abstract) — LLaVA-1.5 에 없는
  multi-axis architectural twist.

`person × abs` (stick figure) 의 4-모델 gradient:
| Model | % kinetic |
|---|---:|
| Qwen | 91 |
| Idefics2 | 99 |
| LLaVA-Next | 80 |
| LLaVA-1.5 | 58 |

M9 H7 finding 의 granular form. Categorical 뷰가 commit 의 *종류* 를
드러냄, 단지 commit *여부* 가 아님.

### §4.7 — saturated 모델에 cue_level 이 지배적 결정 안정자

![§4.7 5-모델 per-axis RC](../figures/sec4_7_rc_per_axis.png)

| model | cue=none → cue=both | bg=blank → bg=ground |
|-------|---------------------|----------------------|
| Qwen2.5-VL | 0.84 → **1.00** (+0.16) | 0.88 → 0.96 (+0.08) |
| Idefics2 | 0.88 → 0.99 (+0.11) | 0.92 → 0.95 (+0.03) |
| InternVL3 | 0.89 → 0.98 (+0.09) | 0.92 → 0.96 (+0.04) |
| LLaVA-1.5 | 0.85 → 0.85 (0) | 0.88 → 0.82 (**−0.06**) |
| LLaVA-Next | 0.78 → 0.78 (0) | 0.77 → 0.80 (+0.03) |

**해석**: saturation 이 단지 행동 PMR ceiling 만이 아니라 **결정-안정성
ceiling** 이기도 함. Non-CLIP 모델이 cue fire 시 5 seed 모두 같은 PMR
call 로 수렴; CLIP-기반 모델이 강한 cue 에서도 seed-level variance 보유.
H-encoder-saturation reframe 의 별도 시그니처.

## 가설 상태 업데이트

- **H7** — 이미 "unsaturated-only AND architecture-conditional"; §4.11
  가 **categorical** 차원 추가 (binary→regime 분포), binary H7 수치가
  muted 인 곳에서도 LLaVA-1.5 의 라벨-disambiguate 가 regime 수준에서
  작동함을 보임.
- **H-encoder-saturation** — 이미 "5 모델 점 × 3 stim source 에서
  architecture-level 확인"; §4.7 가 **결정-안정성 차원** 추가:
  saturation 이 cue 하의 seed-level commit 도 lock. 같은 architectural
  속성의 두 별도 시그니처.

## 이어지는 한계

1. **§4.11 InternVL3 누락**: M8d 가 InternVL3 에 미실행 (M6 r5 round
   에서 deferred). 5-모델 그림 닫을 수 있음.
2. **§4.11 5-카테고리 fine-grained 분류기** (gravity-fall / gravity-roll
   / orbital / inertial / static) M2 circle-only 데이터에 여전히 열림 —
   신규 키워드 셋 + `classify_regime` 의 `circle` 도형 확장 필요.
3. **§4.7 n_seeds=5** 가 RC 의 최소. ≥10 pp 차이 robust; <5 pp 차이
   시사적.
4. **§4.7 단일 arm (label-free)**. Labeled arm 은 다른 RC 구조 보일 수
   있음, 라벨 자체가 commit 안정화 하므로.

## 산출물

### Commit (이 세션, 2 substantive)

- `309bdf6` — §4.11 4-model M8d regime distribution
- `bbf01f9` — §4.7 per-axis RC stability

### 신규 figure

- `docs/figures/sec4_11_regime_distribution_4model.png`
- `docs/figures/sec4_7_rc_per_axis.png`

### 신규 insight 문서

- `docs/insights/sec4_11_regime_distribution.md` (+ ko)
- `docs/insights/sec4_7_rc_per_axis.md` (+ ko)
- `docs/insights/session_2026-04-26_summary.md` (이 문서, + ko)

### 신규 script

- `scripts/sec4_11_regime_distribution.py`
- `scripts/sec4_7_rc_per_axis.py`

### Roadmap

- §4.11 가 "partial complete" 표시 (4-모델 M8d 완료; M2 fine-grained
  여전히 열림)
- §4.7 가 "complete" 표시

## 이 세션 후 통합 backlog

열린 §4 항목:
- §4.3 — 한국어 vs 영어 라벨 prior (1시간, 그러나 PMR scorer 가
  English-only, scorer 확장 필요)
- §4.4 — Michotte 2-frame causality (2-이미지 prompt 지원 필요)
- §4.6 — SAE counterfactual stim 생성 (복잡, 4-6 시간)
- §4.8 — PMR scaling (Qwen 32B/72B — 새 대형 모델 로드 필요)

주요 milestone:
- **M5b** — SIP / activation patching / SAE 특징 분해 (mechanism-level
  증거, 다음 논문-section 갭)
- **M7** — 논문 초고 + Prolific 인간 baseline

## 세션 누계 (2026-04-25 + 2026-04-26)

- M6 r6 시작 이래 총 commit: ~17 substantive
- 총 insight 문서: 16 (영어) + 16 (한국어) = 32 페어 문서
- 총 figure: 30+ (프로젝트 횡단); 이 자율 run 에서 5 추가
- 총 notebook: 13 (프로젝트 횡단); 1 신규 (attention_viz.ipynb) +
  1 확장 (encoder_saturation_chain.ipynb)
- pytest: 123/123 (regression 없음)
