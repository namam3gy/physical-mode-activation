# M5a 확장 실험 — 양방향성과 label 상호작용

`m5_vti_steering.md`의 후속 실험. §7에서 제기된 두 가지 한계를
다룬다: negative-α bidirectionality와 label × steering-direction 상호작용.

Raw numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.
Implementation: `scripts/06_vti_steering.py` (extended), `src/physical_mode/metrics/first_letter.py`.

## 1. 한 줄 요약

L10 direction은 **one-way activation이지 bidirectional axis가 아니다** —
negative α는 physics-mode를 억제하지 않는다 — 그러나 **label-composable하다**:
동일한 `+α · v_L10`은 label `ball`에서 "falls" (A)로, label `circle`에서
"stays still" (B)로 라우팅된다. Object-ness는 steering이, regime은 label이
인과적으로 결정한다.

## 2. 양방향성 테스트 (Exp 1)

### 2.1 실험 설정

Run log §Experiment 1 참조.

### 2.2 결과

| α | A | other | D | PMR |
|---|---|---|---|---|
| 0   | 9  | 1 | 0 | 1.0 |
| -5  | 10 | 0 | 0 | 1.0 |
| -10 | 10 | 0 | 0 | 1.0 |
| -20 | 10 | 0 | 0 | 1.0 |
| -40 | 10 | 0 | 0 | 1.0 |

### 2.3 해석 — 방향은 one-way 이다

physics-mode baseline에 `-α · v_L10`을 주입해도 physics-mode가 억제되지
않는다: α ∈ {0, -5, -10, -20, -40} 전 구간에서 first-letter 분포가 ≥ 9/10 A를
유지한다. α=0에서의 1개 "other"는 오히려 negative α에서 A로 집중된다.
이 null result와 일관된 두 가지 해석이 있다:

- **(a) Ceiling effect**: `textured/ground/both`는 α=0에서 이미 PMR=1이므로
  physics-mode 증가를 관찰할 여지가 없다; 다만 D 방향으로의 이동이 없다는 것은
  -v가 physics-mode를 *제거*하지도 않는다는 의미다.
- **(b) Inherent asymmetry**: 해당 direction은 physical-object 개념을
  activate하지만 동일한 residual direction에 "abstract-mode" 반대극이 존재하지
  않는다. Physics-mode를 억제하려면 다른 메커니즘(다른 direction, 다른 layer,
  또는 다른 feature에 대한 negative sign)이 필요하다.

어느 쪽이든 **M5a 결과는 one-way activator로 프레이밍해야 한다**:
M5a는 positive α가 abstract → physical을 push한다는 것을 보였고,
Exp 1은 negative α가 physical → abstract를 push하지 않는다는 것을 보인다.
이 direction은 VTI가 가정하기도 하는 bidirectional concept axis(예: LLM
문헌의 gender, truthfulness direction)가 아니다.

중요한 caveat: baseline이 이미 physics-mode ceiling에 있기 때문에,
더 깔끔한 bidirectionality 테스트는 +α와 -α 모두에 측정 여지가 있는
중간 baseline PMR(약 0.5)의 `textured/blank/none` stimulus를 사용해야 한다.
이는 차후 과제로 남긴다.

## 3. Label × steering 상호작용 (Exp 2)

### 3.1 실험 설정

Run log §Experiment 2 참조.

### 3.2 결과

| α | A | D | PMR |
|---|---|---|---|
| 0  | 0  | 10 | 0.0 |
| 5  | 0  | 10 | 0.0 |
| 10 | 0  | 10 | 0.0 |
| 20 | 0  | 10 | 0.0 |
| 40 | 10 | 0  | 1.0 |

### 3.3 해석 — regime 은 label 이, activation 은 steering 이 결정

label=`ball`에서 α=40은 10/10을 A ("falls")로 flip한다. label=`circle`
(M5a)에서는 동일한 intervention이 10/10을 B ("stays still")로 flip했다.
이것은 **clean dissociation**이다: image, stimulus cell, steering vector,
magnitude를 모두 고정하고 오직 prompt token만 바꿨을 때 flip target이
B에서 A로 바뀐다. 이는 **H7 (label이 physics regime을 선택한다)**에 대한
강력한 인과적 증거다: steering direction은 "physical object-ness"를 coarse
binary로 activate하고, label prior가 모델이 서술하는 구체적인 physics를
결정한다.

부수적 관찰: ball+line+blank+none의 α=0 baseline은 10/10 D이다 —
label prior만으로는 steering 스크립트의 forced-choice 프롬프트 하에서
이 abstract stimulus를 physics-mode로 만들지 못한다. 이는 M5a에서
`circle+line+blank+none × α=0`도 10/10 D였다는 관찰(`docs/insights/m5_vti_steering.md`
§3.2)과 일치한다. 둘을 합치면 steering 스크립트의 forced-choice 프롬프트
템플릿이 M2의 기본 프롬프트보다 더 보수적임을 시사한다 — 향후 프롬프트
variant 감사에서 조정이 필요하지만 현재 결과에 대한 위협은 아니다.

## 4. 가설 스코어카드 업데이트

| H | Pre-M5a-ext | Post-M5a-ext | Change |
|---|---|---|---|
| H-boomerang | extended + causal | **extended + causal (unchanged)** | Exp 2가 인과적 근거를 강화한다 (label × steering composability를 보여주는 메커니즘 수준의 intervention이 하나 더 추가됨). |
| H-locus | supported (early-mid) | **unchanged** | Exp 2가 label swap에서도 L10이 여전히 유효한 site임을 확인한다. |
| H-regime | candidate | **supported (causally)** | Exp 2: label swap만으로 동일한 intervention이 A vs B flip을 만든다. 동일한 steering 하에서 label이 regime을 선택한다. |
| **H-direction-bidirectional** (new) | — | **refuted (as bidirectional), supported (as one-way activator)** | Exp 1: `textured/ground/both`에서 `-α · v_L10`이 D 방향으로 shift하지 않는다. 이 direction은 object-ness를 activate하지만 동일한 residual-space direction에서 억제하지는 않는다. |

## 5. 논문 기여

- **Figure 6 (M5a causal steering)에 companion figure 추가**: Exp 2
  side-by-side — 동일한 image, 동일한 α, label만 다름 → A vs B — 는
  `steering = object-ness; label = regime` 분해의 가장 명확한 인과적 시연이다.
  하나의 side-by-side 패널로 논문의 H7 주장이 자기완결적이 된다.
- H-direction-bidirectional의 refutation은 과도한 주장을 억제한다:
  `v_L10`을 막연히 "physics-mode direction"이라 부르지 말고
  "physical-object-ness activator"라 불러야 한다. LLM interp의 VTI-style
  vector는 종종 bidirectional concept axis로 묘사되기 때문에 정확한 언어가
  중요하다.
- M5b 우선순위: ROADMAP M5b의 SAE 분해가 이제 더 compelling해진다 —
  `v_L10`이 one-way activator라면, SAE features는 (a) activating sub-features와
  (b) 단순한 mean-difference VTI가 놓치는 별도의 suppressive direction을
  모두 드러낼 수 있다.

## 6. 남은 한계

- Exp 1 baseline이 ceiling(α=0에서 PMR=1.0)에 있어, "bidirectional effect 없음"이
  "효과를 관찰할 여지 없음"과 교란된다. 중간 PMR baseline
  (예: `textured/blank/none`)으로 구별할 수 있다.
- α=40은 여전히 magic number다; 임계값을 정확히 파악하기 위한 finer sweep
  (30/35/40/45/50)은 차후 과제다.
- 두 실험 모두 L10만 테스트했다. L15/L20/L25에서의 label swap은 재현되지 않았다.
- SAE / patching / cross-model은 다루지 않는다 — 이는 M5b/M6에 해당한다.
