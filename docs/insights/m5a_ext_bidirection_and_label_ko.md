# M5a 확장 실험 — 양방향성과 label 상호작용

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택 (ball → 동적 / circle → 정적 / planet → 궤도).
> - **H-boomerang** — Vision encoder 가 행동이 실패하는 곳에서도 physics-mode class 를 선형 분리 — encoder 는 알고 decoder 가 gate. (Qwen 한정: LLaVA-1.5 에서는 CLIP encoder 자체가 bottleneck 이라 반박.)
> - **H-direction-bidirectional** — v_L10 은 physics-mode 안의 regime 축 (+α → 동적, −α → 정적); 초기 "one-way activator" framing 에서 수정됨.
> - **H-locus** — Bottleneck 은 LM 중간 레이어 (특히 L10) 에 있음 — 더 이른 곳도, decoding head 도 아님.
> - **H-regime** — Steering 방향은 binary "object-ness" — 반박됨; H-direction-bidirectional 로 대체.
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M5a** — ST4 VTI steering — LM L10 시각 토큰에 +α·v_L10 더하면 line/blank/none 이 "정지" → physics-mode 로 뒤집힘.
> - **M5b** — ST4 Phase 3 (SIP + activation patching + SAE 특징 분해) — 보류 / optional.
> - **M6** — ST5 cross-model sweep — M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next) 참조.
> - **v_L10** — M5a class-mean diff (physics − abstract) 에서 유도된 layer 10 LM hidden space (dim 3584) steering 방향. Unit norm.


**Steering 대상 자극** — `line / blank / none` (M5a Exp 1) 과 `textured / blank / none` (Exp 3). Exp-3 baseline 은 |α| 임계값 *아래* 에 있어 −α 로 정적, +α 로 동적 flip 가능:

![M5a-ext 참조 자극: line / blank / none](../figures/01_line_blank_none.png)

`m5_vti_steering.md`의 후속 실험. §7에서 제기된 두 가지 한계를
다룬다: negative-α bidirectionality와 label × steering-direction 상호작용.

Raw numbers: `docs/experiments/m5a_ext_neg_alpha_and_label.md`.
Implementation: `scripts/06_vti_steering.py` (extended), `src/physical_mode/metrics/first_letter.py`.

> **2026-04-25 개정**: Exp 1 의 "negative α 효과 없음" 은 ceiling artifact 로
> 확인되었다. Moderate baseline 에서 실시한 Exp 3 은 negative α가 실제로는
> 행동 shift 를 만들어냄을 보였다 — physics-mode 를 abstract 로 억제하는 것이
> 아니라 "stays still" (B) regime 으로 뒤집는다. §1, §2.3, §3.3, §4 를 이에
> 맞춰 재작성했고; §2.2 는 Exp 1 원래 수치를 그대로 보존해 두었다.

## 1. 한 줄 요약

L10 direction 은 **physics-mode 내부의 regime axis** 로, physics-vs-abstract
activator 가 아니다: L10 에서 큰 `|α|` 는 모델을 baseline "abstract / won't
move" (D) 응답에서 physics-mode 로 끌어내고, α 의 *sign* 이 모델이 서술할
**어떤** physics regime 을 선택한다 — `+α` 는 kinetic "falls" (A) 쪽으로,
`-α` 는 static "stays still" (B) 쪽으로 push 한다. Label 과 image prior 는
positive-α target 을 수정한다 (label=ball 또는 obj=textured 만 있어도 +α=40
에서 A 유도; 둘 다 abstract 면 +α=40 에서도 B), 그러나 `-α=40` 은 label/object
에 무관하게 B 를 균일하게 유도한다.

## 2. 양방향성 테스트

### 2.1 Exp 1 설정 (ceiling baseline — 2026-04-24)

Run log §Experiment 1 참조.

### 2.2 Exp 1 결과

| α | A | other | D | PMR |
|---|---|---|---|---|
| 0   | 9  | 1 | 0 | 1.0 |
| -5  | 10 | 0 | 0 | 1.0 |
| -10 | 10 | 0 | 0 | 1.0 |
| -20 | 10 | 0 | 0 | 1.0 |
| -40 | 10 | 0 | 0 | 1.0 |

### 2.3 Exp 3 재검정 (moderate baseline — 2026-04-25)

Exp 1 의 `textured/ground/both` baseline 은 PMR=1 ceiling 이었다; negative
α 의 null-movement 는 (a) ceiling artifact 또는 (b) 내재적 asymmetry 중
어느 쪽이어도 해석이 가능했다. Exp 3 은 `textured/blank/none` (α=0
baseline ≈ 10/10 D, floor 근접) 으로 이동하고 α ∈ {−40, −20, −10, −5, 0,
5, 10, 20, 40} 을 `label ∈ {ball, circle}` 및 `obj ∈ {line, textured}`
양쪽에 걸쳐 sweep — L10, T=0 에서 완전한 (α × label × obj) 그리드 확보.
원자료는 run log §Experiment 3.

|α|=40 에서의 교차 요약:

| obj × label | α=−40 | α=0 | α=+40 |
|---|---|---|---|
| line × circle     | **10 B** | 10 D | 10 B |
| line × ball       | **10 B** | 10 D | 10 A |
| textured × circle | **10 B** | 10 D | 10 A |
| textured × ball   | **10 B** |  9 D + 1 B | 10 A |

- **`-α=40` → 4개 조건 모두 10 B** (label, object 무관).
- **`+α=40` → 4개 중 3개에서 10 A** (textured/*, line/ball); `line × circle`
  만이 +α=40 에서 10 B (원래 M5a 결과).
- `α=0` 은 항상 10 D (Exp 3a 의 한 stimulus 만 1 B / 9 D 예외).

### 2.4 재해석 — physics-mode 내부의 regime axis

Exp 1 의 "효과 없음" 은 **ceiling artifact** 였지 내재적 asymmetry 가 아니었다.
Baseline 이 D floor 로 내려오면 L10 에서의 큰 |α| 가 모델을 D 에서 physics-mode
응답으로 끌어내며, **sign** 이 어떤 physics regime 을 선택한다:

- **`+α · v_L10`** → "falls" (A) — kinetic / gravity-active regime. +α=40 에서
  image 에 physical 외형 (textured) 이 있거나 label 에 physical prior (ball) 가
  있으면 A 가 유도된다. Image 와 label 둘 다 abstract 인 경우 (line + circle)
  에만 +α=40 에서 B ("stays still") 로 default 한다.
- **`-α · v_L10`** → "stays still" (B) — static / gravity-passive regime.
  Exp 3 의 4개 (obj × label) 조건 모두에서 −α=40 target 은 B; label 도 image
  prior 도 이 target 을 옮기지 못한다.

따라서 `v_L10` 은 "physics vs abstract" axis 도 "object-ness on/off activator"
도 아니다. **Physics-mode subspace 내부의 axis** 이며, 두 endpoint 에 대립되는
regime (kinetic vs static) 이 있고 baseline D ("won't move — abstract 니까")
응답은 axis 의 한쪽 끝이 아니라 |α| threshold *아래* 에 위치한다.

이는 M5a causal story 를 "object-ness 가 켜진다 / 꺼진다" 에서 "모델이 `v_L10`-
정렬 physics-mode subspace 를 가지고 있음을 확인했고, sign 선택으로 모델이
서술할 regime 을 인과적으로 고를 수 있다 — 단 |α| 가 ~15-20 을 넘었을 때만"
으로 바꾼다. D 응답은 subspace 바깥: α 의 어느 방향이든 모델을 D 로 되돌리지
못한다.

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

### 3.3 해석 — label 은 +α regime target 에만 작용, −α regime target 에는 무관

`line/blank/none × +α=40` 에서 regime target 은 label 에 따라 달라진다:
- label=`circle` (M5a) → 10/10 B.
- label=`ball` (Exp 2) → 10/10 A.

image, stimulus cell, steering vector, magnitude 를 고정하고 label 만 바꾸면
flip target 이 B 에서 A 로 옮겨간다. 이는 H7 (label 이 physics regime 을
선택한다) 에 대한 dissociation 증거 — 단 Exp 3 의 정보를 반영하면 H7 주장은
원래보다 **좁게** 제한된다:

- **Label-selects-regime 의 적용 범위**: image 자체가 abstract (`line`) 일 때만
  성립. `textured/blank/none × +α=40` 에서는 label 둘 다 A 를 주는 것으로
  관찰된다 (Exp 3a: ball → 10 A; Exp 3b: circle → 10 A) — image-level
  physical signal 이 일단 존재하면 label 은 완전히 dominated.
- **−α regime target 은 label 무관**: `−α=40` 은 Exp 3 의 4개 (obj × label)
  조건 모두에서 B 를 유도한다. Label 은 static regime 을 disambiguate 하지
  못한다.

재서술: regime 은 label 단독이 아니라 **joint** (image, label, α sign) 함수로
선택된다. Label-driven regime-flipping 은 다른 두 channel 이 약할 때 (abstract
image, moderate |α|) 에만 관찰된다. Exp 2 의 가장 깔끔한 발견 — "동일한
steering, label 만 다름, regime 이 flip" — 은 유효한 causal 시연이지만 그
generalization 은 abstract-image 영역으로 제한된다.

부수적 관찰: ball+line+blank+none의 α=0 baseline은 10/10 D이다 —
label prior만으로는 steering 스크립트의 forced-choice 프롬프트 하에서
이 abstract stimulus를 physics-mode로 만들지 못한다. 이는 M5a에서
`circle+line+blank+none × α=0`도 10/10 D였다는 관찰(`docs/insights/m5_vti_steering.md`
§3.2)과 일치한다. 둘을 합치면 steering 스크립트의 forced-choice 프롬프트
템플릿이 M2의 기본 프롬프트보다 더 보수적임을 시사한다 — 향후 프롬프트
variant 감사에서 조정이 필요하지만 현재 결과에 대한 위협은 아니다.

## 4. 가설 스코어카드 업데이트

| H | Pre-M5a-ext | Post-M5a-ext (2026-04-24) | Post-recheck (2026-04-25) |
|---|---|---|---|
| H-boomerang | extended + causal | unchanged | **unchanged** — causal leg 강화 (Exp 2 + Exp 3 그리드). |
| H-locus | supported (early-mid) | unchanged | **unchanged** — L10 이 모든 Exp 3 조건에서 유효. |
| H-regime | candidate | supported (causally) | **supported but narrower** — label 단독으로 regime 을 flip 하는 것은 image 가 abstract 일 때에만 성립; textured image 에서는 +α=40 이 label 무관하게 A 를 준다. Regime 은 (image, label, α sign) joint 함수로 선택. |
| **H-direction-bidirectional** (new 2026-04-24) | — | refuted as bidirectional, supported as one-way activator | **revised**: `v_L10` 은 **physics-mode 내부의 regime axis** — +α → kinetic/falls (A), −α → static/stays-still (B), baseline D 는 \|α\| activation threshold 아래에 위치. 이전의 "one-way activator" 프레이밍 자체가 Exp 1 의 ceiling artifact 였음. |

## 5. 논문 기여

- **Figure 6 (M5a causal steering)에 multi-panel companion 추가**: Exp 2 +
  Exp 3 의 (α × label × obj) 그리드는 "동일한 image, 동일한 α sign, label 만
  flip" 및 "동일한 image, 동일한 label, α sign 만 flip" 시연의 2×2 를
  제공한다. 가장 깔끔한 단일 figure 는 `-α=40` 행이다: 네 개의 서로 다른
  (obj, label) 조건 모두가 prior signal 에 무관하게 B 로 collapse — 이는
  논문에서 가장 강력한 isolated causal effect.
- **본문에서의 언어 규율**: `v_L10` 은 "physics-mode 내부의 regime axis" 로
  기술해야 하며, "physics-mode activator" (Exp 3 을 고려할 때 너무 좁음) 나
  "physics-vs-abstract direction" (negative α 가 D 를 복구하지 못하므로 단순
  오류) 로 부르면 안 된다. 논문은 또한 baseline D 응답이 axis 의 한쪽 끝이
  아니라 activation threshold 아래에 위치함을 명시해야 한다 — 그렇지 않으면
  VTI 에 익숙하지 않은 독자는 보통 "bidirectional concept direction" 프레임을
  기본값으로 떠올릴 것이기 때문.
- **M5b (SAE) 동기가 강화된다**: regime-axis 구조는 단순 activator 보다 SAE
  decomposition 에 더 compelling. 기대 SAE features: (a) "kinetic / falls"
  feature, (b) "static / stays" feature, (c) physics-mode axis 바깥에 있는
  별도의 "abstract / not a physical object" feature. SAE 가 mean-diff
  direction 에 반대 부호로 loading 되는 (a) + (b) 를 복원한다면 behavioral
  steering 이 닿지 않는 더 미세한 단위에서 regime-axis 해석이 검증된다.
- H7 은 논문에서 qualifier 가 필요하다: "label 이 regime 을 선택한다 — **image
  가 abstract 일 때**" — 전역적 주장이 아니다. Exp 3 데이터에 따르면 image 가
  physical-object signal (textured) 을 가지면 label dominance 는 실패한다.

## 6. 남은 한계

- |α| activation threshold 는 Exp 3a 의 α=+10 (7A+1B+2D) / α=+20 (10A)
  transition 로 |α|∈[15, 25] 구간으로 locate 되었다. 더 촘촘한 sweep (10, 12,
  15, 18, 22, 25) 로 threshold 를 정확히 핀다는 과제는 deferred.
- Exp 3 에서 L10 만 테스트했다. L15 / L20 / L25 에서의 α-sign regime split
  은 재현되지 않았다; M5a single-layer 결과로 볼 때 late layer 들은 regime
  axis 에 참여하지 않을 수도 있다.
- Exp 3 은 (line, textured) × (ball, circle) 만 다룬다; 전체 M2 axis 는 4 × 3
  (obj × label) + `planet`. `planet` 및 `shaded / filled` 는 미검증. M2 데이터는
  `planet` 이 baseline 에서 orbital physics (gravity-fall 아님) 를 activate 함을
  시사한다 — `planet` 에서도 `-α=40` 이 B 로 push 하는지는 미확인.
- Prompt template 은 여전히 `forced_choice` @ T=0. Prompt 감사 (open-ended /
  T>0 / label-free §4.9) 는 deferred.
- SAE / patching / cross-model 은 다루지 않는다 — M5b/M6 해당.
