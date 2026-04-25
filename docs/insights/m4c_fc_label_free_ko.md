# M4c — Forced-Choice Label-Free Prompt

M4b 의 companion. H2 reframing (`ball ≈ no-label`, `circle = suppressor`)
이 open-ended 에서 forced-choice prompt 로 옮겨가도 살아남는지 검증;
또한 FC 옵션의 label antecedent 를 제거 (`The depicted object falls down...`)
하면 M6 r1 에서 관찰된 LLaVA-1.5 의 pathological "A" 편향이 완화되는지 검증.

원자료: `docs/experiments/m4c_fc_label_free_ko.md`.
Code: `src/physical_mode/inference/prompts.py` (신규 `forced_choice_no_label`
variant), `configs/fc_label_free_{qwen,llava}.py`.

## 1. 한 줄 요약

Qwen2.5-VL 에서 M4b 의 H2 발견이 **FC 하에서 강화** — `ball` 은 여전히 ≈
no-label (Δ=+0.013), `circle` 은 더 강하게 억제 (Δ=−0.208 vs M4b open
Δ=−0.065), `planet` 은 새로 억제 (Δ=−0.263) — FC 옵션 셋이 gravity-centric
이라 orbital physics 가 D ("abstract") 싱크로 collapse 되기 때문.
LLaVA-1.5 의 FC "A" 편향은 re-templated prompt 에서도 **유지** (477/480 =
99.4 % `A`) — 편향은 모델 수준이며 FC 는 prompt 설계와 무관하게 LLaVA-1.5
에서 사용 불가.

## 2. Qwen FC label-free — M4b H2 의 확인 + 강화

### 2.1 방향이 일관적

| comparison | M4b open | M4c FC |
|---|---|---|
| `PMR(ball) − PMR(_nolabel)`   | +0.006 | +0.013 |
| `PMR(circle) − PMR(_nolabel)` | −0.065 | **−0.208** |
| `PMR(planet) − PMR(_nolabel)` | +0.006 | **−0.263** |

`ball ≈ no-label` 이 prompt-format 변경에도 깨끗이 유지. `circle` 은 FC 하에서
더 음수로 이동. `planet` 은 "≈ no-label" 에서 강한 음수로 flip.

### 2.2 FC 의 "abstract sink" — circle 과 planet 이 FC 하에서 더 음수로 가는 이유

FC 옵션 A-C 는 모두 gravity-centric: A "falls down", B "stays still", C
"moves sideways". D 는 abstract escape: "This is an abstract shape — nothing
physical happens." 이미지가 모호하고 label 이 {falls, stays, moves sideways}
에 안 맞는 regime 을 evoke 할 때:

- Open prompt 는 모델이 선호하는 narration 을 작성하게 함 (`circle` →
  "becomes smaller", `planet` → "orbits the sun") — verb-PMR lexicon 이
  일부 응답을 physics-mode (PMR = 1) 로, 다른 응답은 그렇지 않게 채점.
- FC prompt 는 모델이 {A, B, C, D} 중 하나로 commit 하게 강제. Orbital
  physics 는 native FC 옵션이 없으므로 `planet` → D. Abstract geometry 도
  native 옵션이 없으므로 `circle` → D. Per-label D 비율이 sharply 상승하여
  text-PMR 을 끌어내림.

`line/blank/none` 에서의 수치 예시:

| condition       | first_letter | text PMR |
|---|---|---|
| open × _nolabel | n/a (open)   | 0.40 |
| FC × _nolabel   | D=9, B=1     | 0.00 |
| FC × ball       | D=10         | 0.00 |
| FC × circle     | D=10         | 0.00 |
| FC × planet     | D=10         | 0.00 |

완전 추상 이미지에서 FC 가 모든 condition 을 label 무관하게 D 로 collapse —
모델은 image content 가 gravity-centric reading 을 적극 지원하지 않는 한
"abstract shape" 을 적용 가능한 것으로 해석. 이는 FC 의 옵션-셋 편향이지
label 자체에 대한 행동적 statement 가 아님.

### 2.3 H2 reading 에 대한 함의

Visual-saturation 가설 (M6 r1) 이 강화: M4b 의 "circle suppression only" 는
open prompt 하에서 Qwen 의 PMR ceiling 의 Qwen 특이적 결과; FC 하에서는 모델이
abstract 옵션에 명시적으로 접근 가능하므로 suppressive 방향이 label 전반에서
더 잘 보임. 통합 statement 가 성립: **language prior 는 모든 label 에 대해
양수 (또는 0) 로 기여**, Qwen 에서의 명백한 음의 기여는 FC 의 옵션-셋 편향 +
Qwen 의 visual saturation 이지 실제 음의 semantic prior 가 아님.

## 3. LLaVA FC label-free — pathology 유지, FC 는 이 모델에서 죽음

| `first_letter` | count |
|---|---|
| A | 477 |
| B |   3 |
| C |   0 |
| D |   0 |

LLaVA 는 label antecedent 를 제거한 re-templated prompt 에서도 ("the ball
falls down" 대신 "the depicted object falls down") 477/480 자극에 `A` 반환.
편향은 다음과 무관:
- Image content (`line/blank/none` 과 `textured/ground/both` 모두 100% A),
- Label (M6 labeled smoke 12/12; label-free 477/480),
- Prompt antecedent (`It` vs `the {label}` vs `the depicted object`).

따라서 LLaVA FC 편향은 모델 수준. 가설: LLaVA 의 instruction-tuning 데이터가
MCQ 형식에서 "A" 답을 over-weighting 했거나, FC 지시가 start-of-answer
token 통계와 상호작용하여 첫 글자를 lock. 논문의 cross-model 섹션에 sticky
한 model-property 발견 — 이번 round 에서 fix 불가능.

## 4. 가설 스코어카드 업데이트

| H | Pre-M4c | Post-M4c |
|---|---|---|
| **H2** | visual-saturation 가설 (M6 r1) 하에 revised | **추가 강화** — Qwen FC label-free 가 다른 prompt format 에서 M4b 의 "ball ≈ no-label, circle suppresses" 패턴 재현. "planet" 의 FC sink 는 Qwen 의 per-label suppression 이 *부분적으로* 옵션-셋 artefact 임을 드러내며, 특정 label 에 대한 literal "abstract override" claim 보다 visual-saturation 프레이밍을 지지. |
| **H4** (open vs FC gap) | supported (Qwen) | **Qwen 에서 cross-format 측정 가능** — 동일 자극 (no-label) 의 paired open-vs-FC delta = **−0.131**. FC 가 open 보다 일관되게 보수적. LLaVA FC 편향이 해결되면 cross-model H4 검증 준비됨. |
| **H7** (label selects regime) | 지지 but narrower; cross-model 재현 (M6 r1) | **caveat 추가** — regime 구별 (planet → orbital, ball → gravity) 은 narrative latitude 가 허용되는 prompt 에서만 보임. FC 하에서 모든 non-gravity regime 이 D 로 collapse 되어 H7 mask. 현재의 FC 옵션 셋은 gravitational physics 로 편향; H7 을 FC 하에서 검증하려면 확장 옵션 셋 ("D) The depicted object orbits or rotates", "E) Other") 필요. |
| **LLaVA FC bias** (M6 r1 발견) | 관찰 | **모델 수준 pathology 확인** — FC antecedent re-templating 으로 LLaVA 가 A default 에서 안 벗어남. FC 는 LLaVA-1.5 에서 사용 불가; cross-model H4 test 가 여기서 완전 차단. |

## 5. 논문 기여

- M4b 의 "ball ≈ no-label, circle = suppressor" 발견은 **Qwen 에서 prompt-
  robust** — open prompt 와 FC prompt 양쪽에서 등장, FC 버전은 더 강한
  circle suppression 과 옵션-셋 artefact 인 새로운 "planet" suppression 보유.
- FC 옵션 셋이 gravity-centric 이라 non-gravitational regime 을 D 로 끌어
  당김. 방법론적 함의: regime-flexible label (예: `planet`) 전반의 FC 기반
  PMR 비교는 orbital-routed label 의 physics-mode commitment 를 underestimate
  할 것. Methods 섹션에 한 줄 note 로 flag 권장.
- LLaVA-1.5 FC 편향은 모델 수준 pathology. Cross-model 방법론 포인트로
  reportable: forced-choice answer selection 에 의존하는 행동 metric 은
  cross-model 비교를 confound 시키는 큰 prompt-side scaffolding (다른 옵션
  셋, 다른 instruction style) 없이는 LLaVA-1.5 로 portable 하지 않음.
  Cross-model 표준으로 open-prompt protocol 권장.
- Open-vs-FC gap (H4) 가 no-label 에서 측정 가능: Qwen 에서 동일 자극의
  paired PMR **−0.131**, label confounding 없이 M2 H4 패턴 깨끗이 확인.

## 6. 한계

- LLaVA FC degenerate. Round-2 아이디어: greedy first-letter argmax 대신
  FC 단계 전체를 **first-token logit 비율** (P(`A`) vs P(`B`) vs P(`C`) vs
  P(`D`)) 로 교체. Bias 가 temperature-sampling 단계에 있으면 우회 가능,
  underlying logits 에 있으면 여전히 발생.
- FC 하에서 planet 의 collapse 는 옵션-셋 driven 이지 underlying physics-mode
  commitment 의 변화는 아님. 분리하려면 확장 FC 옵션 셋 (orbit / rotate /
  other) 필요.
- 이번 round 에서 M2 의 FC labeled PMR 을 `first_letter` 로 재산출하지
  않음 (M6 r1 advisor 권고). Cross-model first-letter PMR 비교는 여전히 TODO;
  M4c 가 Qwen FC label-free first-letter 표를 시작점으로 제공.
- 단일 FC prompt 설계. 다른 re-wording ("Choose the most plausible outcome",
  "Pick A/B/C/D", "Answer with a single letter" 지시 제거) 에 대한 sensitivity
  미검정.
