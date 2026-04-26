# M6 Round 1 — LLaVA-1.5-7B 에서의 Cross-Model H2 Fork

> **이 문서에서 쓰는 코드 한 줄 recap** (전체 정의는 `references/roadmap.md` §1.3 + §2 참조)
>
> - **H1** — PMR 이 abstraction 축을 따라 S 모양으로 상승 (line → filled → shaded → textured); ground 도입이 가장 큰 단일 jump.
> - **H2** — label (ball / circle / planet) 자체가 PMR 을 독립적으로 끌어올림 — 시각 증거를 넘는 language-prior 기여.
> - **H4** — Open-ended vs. forced-choice PMR 간격은 language-prior ↔ visual-evidence 충돌의 안정적 signature.
> - **H7** — Label 은 PMR 을 toggle 하지 않음 — 어느 물리 regime 이 적용되는지 선택 (ball → 동적 / circle → 정적 / planet → 궤도).
> - **H-boomerang** — Vision encoder 가 행동이 실패하는 곳에서도 physics-mode class 를 선형 분리 — encoder 는 알고 decoder 가 gate. (Qwen 한정: LLaVA-1.5 에서는 CLIP encoder 자체가 bottleneck 이라 반박.)
> - **H-direction-bidirectional** — v_L10 은 physics-mode 안의 regime 축 (+α → 동적, −α → 정적); 초기 "one-way activator" framing 에서 수정됨.
> - **H-locus** — Bottleneck 은 LM 중간 레이어 (특히 L10) 에 있음 — 더 이른 곳도, decoding head 도 아님.
> - **M2** — ST1 MVP-full — 5축 factorial (2880 stim); H1 monotone S-curve, H7 등장.
> - **M4b** — M4 + label-free 프롬프트로 H2 null test; Qwen 에서 H2 가 비대칭 (circle 억제, ball 증강 아님).
> - **M6** — ST5 cross-model sweep — M6 r1 (LLaVA-1.5), r2 (InternVL3 + LLaVA capture + FC ratio), r3 (Idefics2), r4 (InternVL3 probe), r5 (M8c photo probe), r6 (LLaVA-Next) 참조.

M4b 의 H2 reframing (`ball ≈ no-label`, `circle = suppressor`) 이 Qwen2.5-VL
에서 두 번째 open-source VLM 으로 일반화되는지 검증. Round 1 은 LLaVA-1.5-7B-hf
만 다룸; LLaVA-Next, InternVL2, Qwen2-VL 는 deferred.

원자료: `docs/experiments/m6_cross_model_llava_ko.md`.
Configs: `configs/cross_model_llava.py`, `configs/cross_model_llava_label_free.py`.

## 1. 한 줄 요약

M4b 의 reframing 은 **Qwen 특이적**이지, 일반 open-source-VLM 속성이 아니다.
LLaVA-1.5-7B 는 *원래의* H2 패턴을 보임: 모든 label 이 no-label baseline 대비
PMR 을 enhance (`ball +47.5 pp`, `planet +24.4 pp`, `circle +17.3 pp`).
원인은 LLaVA-1.5 의 시각적 physics prior 가 훨씬 약하다는 것 — M2 자극에서
Qwen2.5-VL 처럼 physics-mode 로 default 하지 않으므로, label 이 Qwen 에서
visual content 가 담당하는 활성화 작업을 떠맡는다.

## 2. The fork, in one comparison

480 matched `(obj, bg, cue, seed)` tuple 에 대한 평균 paired delta
`PMR(label) − PMR(_nolabel)`, T=0.7, open prompt:

| label  | Qwen2.5-VL-7B | LLaVA-1.5-7B |
|---|---|---|
| ball   | +0.006 | **+0.475** |
| planet | +0.006 | **+0.244** |
| circle | **−0.065** | **+0.173** |

세 가지 관찰:

- Qwen 에서 non-trivial signed effect 를 만드는 label 은 `circle` 만이며,
  그 효과는 **음수** — 이미 saturate 된 visual default 를 label 이 억제.
  M4b 의 reframing.
- LLaVA 에서는 **세 label 모두 강하게 양수**, `circle` 도 마찬가지. 모델은
  physics 로 commit 하기 위해 label 이 필요하며, abstract-prime 인 `circle`
  조차 no-label 보다 더 많은 physics-mode 를 활성화.
- Per-label rank order (`ball > planet > circle`) 는 두 모델에서 보존.
  Language-prior contribution 의 *모양* 은 동일; *zero point* (`_nolabel`
  의 위치) 가 극적으로 다르다.

## 3. 메커니즘 — language-prior 비대칭이 아니라 visual default 차이

Object level 별 `PMR(_nolabel)`:

| object   | Qwen | LLaVA |
|---|---|---|
| line     | 0.942 | 0.142 |
| filled   | 0.933 | 0.317 |
| shaded   | 0.942 | 0.592 |
| textured | 0.975 | 0.483 |

Qwen 의 visual-only PMR 은 **모든 object level 에서 이미 ceiling** (0.93–
0.98). Qwen 에서는 language prior 가 headroom 이 없음 — `ball` 추가로 위로
push 못하고, `circle` 로만 아래로 (그것도 약하게) 끌어내림. LLaVA 는 visual-
only PMR 이 훨씬 낮고 (0.14–0.59), object 추상도와 monotone 도 아니라서,
language prior 가 unsaturated 한 dynamic range 위에서 작동하여 행동을
실질적으로 shift.

이는 M4b finding 이 원래의 H2 와 모순되는 게 아니라, **M4b 가 Qwen2.5-VL
의 visual saturation 을 드러낸 것**이며, 이것이 그 language-prior contribution
이 비대칭으로 (circle 만의 효과로) 보이는 구조적 원인. LLaVA-1.5 의 더 낮은
visual prior 는 모든 label 에서 language-prior contribution 의 full
visibility 를 부여 — "원래의" H2 reading.

두 모델 모두를 포괄하는 가장 단순한 H2 statement:

> **Qwen2.5-VL 과 LLaVA-1.5 양쪽에서, 시험된 모든 label 의 language prior
> 는 양수 (또는 0) 이며, vision prior 가 강한 모델에서는 visual saturation
> 이 이를 mask 한다. Qwen 에서의 "circle 이 visual default 이하로 억제" 효과
> 는 saturation 의 결과이다: PMR ≈ 0.95 에서는 양의 language-prior 기여가
> 표출될 공간이 없으므로 음의 방향 (`circle` 의 끌어내림) 만 측정 가능.
> LLaVA 에서는 같은 `ball` 과 `planet` prior 가 unmasked 되어 의미 있게
> 양의 기여를 한다.**

## 4. H1 (S-curve) cross-model

Object 별 PMR (label 평균, open prompt):

| object   | Qwen labeled | LLaVA labeled |
|---|---|---|
| line     | 0.906 | 0.508 |
| filled   | 0.933 | 0.656 |
| shaded   | 0.933 | 0.753 |
| textured | 0.950 | 0.806 |

LLaVA-1.5 가 본 연구에서 측정된 가장 깔끔한 S-curve 를 보임: 4 개 object
level 전반에서 monotone, line → textured 30 pp 상승. Qwen 의 S-curve 는
labeled run 이 이미 ceiling 이라 보이지 않음; gradient 는 M2 의 *forced-
choice* 수치 (0.74 → 0.83) 에 존재했지만 open-prompt 비교 창에서는 측정
불가.

H1 verdict: **모델이 visual saturation 에 있지 않을 때만** cross-model 에서
지지됨. 본 연구에서 가장 깔끔한 H1 evidence 는 Qwen2.5-VL 이 아니라 LLaVA-1.5
에 있다.

## 5. H7 (label selects regime) cross-model

Label 별 GAR (open prompt):

| label  | Qwen | LLaVA |
|---|---|---|
| ball   | 0.706 | 0.356 |
| circle | 0.753 | 0.153 |
| planet | 0.319 | 0.072 |

`planet GAR << ball/circle GAR` 의 qualitative 패턴이 두 모델 모두에서 재현
— `planet` label 이 physics 서술을 gravity-aligned motion 대신 orbital /
cosmic event 로 route. 샘플 LLaVA `planet × line/blank/none` 응답: "the
planet will continue to spin and orbit around the sun", "the planet will be
consumed by a black hole" — 명시적 non-gravitational physics.

H7 verdict: **supported (cross-model)**. Label-selects-regime 메커니즘은
Qwen 특이적이 아님. 크기는 다르지만 qualitative gradient 는 보존.

## 6. 가설 스코어카드 업데이트

| H | Pre-M6 | Post-M6 |
|---|---|---|
| **H1** (S-curve) | supported | **supported, LLaVA 에서 더 sharp** — Qwen 은 saturation; LLaVA 가 가장 깔끔한 monotone object-level gradient. LLaVA S-curve 를 canonical figure 로 보고 권장. |
| **H2** (language prior 가 PMR 을 높인다) | revised (M4b: ball ≈ no-label, circle suppression) | **재개정 — visual-saturation 가설** — M4b 의 "circle suppression only" 패턴은 Qwen 의 visual ceiling 의 결과. LLaVA 는 visual prior 가 약해서 원래 H2 를 회복 (`ball +47.5 pp, planet +24.4 pp, circle +17.3 pp`). 통합 statement: language prior 는 두 모델 양쪽에서 label 전반에 양수로 기여; visual saturation 이 양의 기여를 mask 하고 음의 signal 만 남길 수 있음. |
| **H4** (open-FC gap) | supported (Qwen) | **cross-model 미검정** — LLaVA-1.5 는 모든 FC 자극에 "A" 반환 (smoke 12/12) → FC PMR 정보 없음. Round 2 에서 다른 FC template 또는 first-letter probability 채점으로 deferred. |
| **H7** (label selects regime) | supported (Qwen) | **supported, cross-model** — `planet GAR << ball/circle GAR` 가 Qwen (0.32 vs 0.71/0.75) 과 LLaVA (0.07 vs 0.36/0.15) 양쪽에서 성립. 크기는 다르지만 gradient 보존. |
| **H-boomerang / H-locus / H-direction-bidirectional** | various | **M6 round 1 미검정** — LLaVA 에 activation capture 와 steering 미실행. Round 2 에서 LLaVA 활성화 재캡처 후 boomerang AUC 와 L? regime axis 비교 권장. |

## 7. 논문 기여

- M4b 의 "circle suppression / ball ≈ no-label" reframing 은 cross-model
  claim 이 아니라 Qwen2.5-VL 특이 관찰로 보고. 가장 깔끔한 단일 statement
  는 **visual-saturation 가설**: VLM 의 visual physics prior 가 이미 PMR
  ceiling 에 있을 때 language prior 의 *양의* 기여는 보이지 않고 *음의*
  기여 (`circle` 의 끌어내림) 만 측정 가능.
- Cross-model H1 figure 는 Qwen 의 saturated table 이 아니라 LLaVA-1.5 의
  S-curve 여야. "추상도가 중요" 의 더 sharp 한 검증이며, 두 모델의 S-curve
  간 격차 자체가 saturation 해석의 evidence.
- H7 cross-model 재현이 본 연구에서 가장 깔끔한 단일 양의 cross-model
  claim. `planet GAR < ball GAR` dissociation 이 두 모델에서 같은 방향으로
  성립.
- "model-specific visual prior" plot: object level 별 `PMR(_nolabel)`,
  Qwen vs LLaVA side-by-side — saturation 설명을 직접 시각화.

## 8. 한계

- LLaVA 의 FC "A" 편향 으로 이번 round 에서 FC 기반 metric 사용 불가. Round 2
  에서 re-template FC ("Choose the most plausible outcome") 또는 greedy
  argmax 대신 first-letter-token probability score 필요.
- LLaVA 의 no-label run 이 non-monotone object-level PMR 을 보임 (shaded >
  textured). 7B-scale artefact 이거나 LLaVA 가 M2 textured 자극 (bowling-
  ball 텍스처 사용) 을 읽는 방식의 특이성일 가능성; round 2 에서 LLaVA-Next
  또는 다른 텍스처 set 으로 재검증 권장.
- 이번 round 에 LLaVA activation capture 없음 — H-boomerang / H-locus
  cross-model evidence 미수집.
- 단일 cross-model 점. "visual-saturation 가설" 검증을 위해 양쪽 방향 중
  최소 한 모델 추가 필요 (예: 더 작거나 큰 Qwen variant; LLaVA-Next;
  InternVL2).
- M2 자극은 Qwen 에 맞춰 튜닝됨 — LLaVA 의 no-label PMR collapse 가 부분적
  으로는 자극이 다른 vision encoder 를 가진 모델에 self-identifying physics
  scene 으로 안 보일 수도 있음. Model-agnostic 자극 protocol 이 모델 별로
  다른 no-label baseline 을 산출할 수 있음.
