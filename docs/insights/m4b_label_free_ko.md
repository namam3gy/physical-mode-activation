# M4b — H2 Null-Hypothesis Test 로서의 Label-Free Prompt

ROADMAP §4.9 ("H2 null-hypothesis test 로서의 label-free prompt") 를 실행.
부산물로 M4 의 switching-layer degeneracy 를 재검토.

Raw numbers: `docs/experiments/m4b_label_free_ko.md`.
Code: `configs/label_free.py`, `src/physical_mode/inference/prompts.py`
(`open_no_label` variant), `src/physical_mode/probing/lm.py`
(generic `pmr_source`).

## 1. 한 줄 요약

원래 명시된 H2 는 **misattribution** 이었다: M2 에서 관찰된 +15 pp "ball vs
circle" gap 은 ball-enhancement 효과가 아니라 **circle suppression** 효과.
Label-free baseline 과의 paired comparison 에 따르면 `ball ≈ no-label`
(+0.6 pp), `circle − no-label = −6.5 pp`. Label prompt 없이도 Qwen2.5-VL 의
시각 시스템은 M2 의 자극에 대해 이미 physics-mode 로 default 하며; "ball"
label 은 redundant, "circle" label 만이 그 default 를 능동적으로 override 한다.

## 2. Pairing 설계

M2 는 `(image × label × open)` 및 `(image × label × forced_choice)` 를
{circle, ball, planet} 전반에 걸쳐 실행했다. Label-free run 은 label token 을
포함하지 않는 prompt 하에서 동일한 이미지 (M2 stimulus dir) 를 재노출:

> "What do you see in the image? What might happen next? Answer in one short sentence."

T=0.7, seed 10개/cell 동일 → 480 matched `(obj, bg, cue, seed)` tuple 에서
동일 seed 에 대한 stochastic-to-stochastic paired comparison 이므로,
PMR delta 는 고정 이미지에서의 per-label 기여를 noise-filtered 로 추정한다.

이번 회차에서는 설계상 forced-choice 를 제외: FC 의 A-D 선택지는 "It" 의
antecedent 가 없어 label 없이 rendering 하려면 다른 문구가 필요. 향후로 deferred.

## 3. 행동적 발견 — language-prior attribution flip

### 3.1 Paired delta

| M2 label | 480 pair 평균 `PMR(label) − PMR(_nolabel)` |
|---|---|
| ball   | **+0.006** |
| planet | **+0.006** |
| circle | **−0.065** |

- `ball` 과 `planet` 은 전체 수준에서 no-label 과 통계적으로 구별 불가능.
  M2 의 자극에서 physics-mode 진입에 "ball" token 이 *필요*하지 않음 — 시각
  content 만으로 충분하다.
- `circle` 은 평균 ~6.5 pp PMR 을 감소시킨다. "circle" token 은
  **override token** 이다: 이미지가 physics 를 지원할 때에도 semantic 으로
  abstract-geometry reading 을 강제한다.

### 3.2 Cell-level 구조

두 축이 circle-suppression 을 modulate 한다:

- **Object 추상도**: `circle` 이 `line` 을 9.2 pp vs `filled` 를 4.2 pp 억제.
  이미지가 추상적일수록 label override 가 더 강해진다.
  이는 H4 (추상도 ↑ → language prior gap ↑) 의 symmetric dual: label
  변경에 따라 language prior 의 *방향* 이 뒤집힘 — physical prior (ball) 은
  이미지와 align, abstract prior (circle) 는 이미지와 대립하며, 이미지가
  추상적일수록 circle override 가 PMR 을 끌어내릴 여지가 커진다.
- **Cue 강도**: `motion_arrow` 는 circle-label 억제를 0.000 으로 만든다.
  Arrow 는 label override 를 override 할 만큼 강력한 visual signal. `none`
  (cue 없음) 이 최대 억제 (−15 pp) — label 이 유일한 text-side signal 이므로.

### 3.3 완전 추상 cell — `line/blank/none`

M2 는 3개 label 을 돌렸고 label-free 가 4번째 condition 을 추가하므로, 가장
모호한 cell 에 대해 4-point view 를 얻는다:

| label    | PMR  | hold_still |
|---|---|---|
| _nolabel | 0.40 | 0.20 |
| ball     | 0.40 | 0.60 |
| circle   | 0.10 | 1.00 |
| planet   | 0.70 | 0.30 |

세 가지 서로 다른 label effect — 이 셀에서만 드러나는 이유는 시각 모호성이
language prior 를 최대로 노출시키기 때문이다:

- `ball` 은 PMR 은 그대로 두고 regime 만 swap — 응답이 더 이상 fall (kinetic)
  이 아니라 stay still (static). "ball" label 이 physics-mode 로 route 하지만
  추상 이미지 위에서는 regime 을 static 으로 redirect 한다. 이는 M5a-ext Exp 3
  가 VTI-steering 측에서 관찰한 것과 동일한 패턴 (moderate baseline 에서의
  static regime).
- `circle` 은 즉시 abstract-override: PMR 0.10, hold_still 1.00.
- `planet` 은 **no-label baseline 대비 PMR 을 올리는 유일한 label** — 그것도
  +30 pp. "planet" prior 는 시각 content 위에 진정으로 additive, 전체 GAR 0.32
  가 이미지 단독으로는 불러오지 못하는 orbital physics 를 대표함과 일관된다.

### 3.4 H2 재서술

개정된 H2:

> **Label 은 시각-단독 baseline 대비 PMR 을 균일하게 올리지 않는다. M2 의
> 프로그램적 자극에서 `ball` 은 시각적으로 redundant (≈ no-label), `circle`
> 은 abstract override (−6.5 pp vs no-label, 추상 이미지에서 증폭), `planet`
> 은 추상 이미지에서만 보이는 orbital-physics prior 를 가볍게 추가한다.**

원래 H2 방향 (language 가 중요) 은 보존하지만 per-label 기여를 재분배:
M2 의 "ball > circle" gap 은 이제 `circle < visual-default` 이며,
`ball > visual-default` 가 아니다.

## 4. 메커니즘적 발견 — visual-token capture 는 label 을 관찰하지 못한다

Label-free activations 에 M4 재실행 (`scripts/05_lm_probing.py --sources open_no_label`)
을 돌리면 M2 와 같은 physics-margin 표와 같은 collapsed switching-layer
(480 샘플 모두 L5) 가 재현된다. 두 run 의 activation 을 layer {5, 10, 15, 20, 25}
에서 bit-for-bit diff 한 결과: **visual-token hidden states 가 prompt 무관하게
동일**. `input_ids` 와 `visual_token_mask` 만 길이가 다르다.

이는 Qwen2.5-VL chat template 의 구조적 귀결: image token 이 user message 의
question text 보다 앞에 있으므로, causal attention 하에서 question text
(label 포함) 는 visual-token 위치로 역전파되지 않는다. 따라서 현재 설계의
M4 logit lens 및 per-layer PMR probe 는 이미지 측 정보 흐름만 capture 하며,
label 기여를 측정할 수 없다.

함의:

- "switching layer 가 condition 무관하게 L5" 라는 M4 주장은 참이지만 underdetermined
  — 이는 prompt 간 수렴 행동이 아니라 capture 지점의 구조적 특성을 반영한다.
- LM 내부에서 label 효과를 localize 하려면, probe 가 label token *이후* 의
  text 위치 (예: last question-text token, start-of-answer token) 에서 capture
  해야 한다. Late-prompt 위치에 second capture hook 을 추가하면 prompt 별로
  실제로 다른 hidden states 가 나올 것이다.
- 반대로, image-side hidden states 가 prompt 전반에 걸쳐 일정하다는 사실은
  H-boomerang 에 대한 독립 증거 (vision encoder 가 pixel 만으로 physics-vs-abstract
  를 encoding; LM 의 label-induced 행동 변조는 모두 visual-token 하류).

## 5. 가설 스코어카드 업데이트

| H | Pre-M4b | Post-M4b |
|---|---|---|
| **H2** (language prior 가 PMR 을 높인다) | quantified | **revised** — ball ≈ no-label, planet ≈ no-label (추상 이미지 제외), circle = 억제자. Per-label 기여가 뒤집힘: "language prior" 는 비대칭이며, 주로 `circle` 이 주도하는 *음의* 효과. |
| **H-boomerang** | supported + causal | **강화** — visual-token hidden states 가 prompt-independent, 따라서 L5 부터 존재하는 physics-bias 는 이미지 단독 기원. |
| **H-locus** | supported (early-mid L10) | **unchanged** — label 의 행동적 효과는 image token 이후 의 text 위치에 localize (여기서 capture 되지 않음); M5a 의 L10 중심 intervention 이 image-preceding trajectory 에서 효과적이었던 것과 일관. |
| **H4** (추상도 → gap) | supported, extended | **refined** — label override 강도 (`circle − no-label`) 가 추상 이미지에서 더 크며, 이는 M4 의 language-prior-per-abstraction 결과의 image-side dual. |

## 6. 논문 기여

- 논문은 **정정된 H2** 를 앞세워야 한다: "ball label 이 PMR 을 크게 높인다" 를
  "circle label 이 시각 default baseline 이하로 PMR 을 크게 낮춘다" 로 교체.
  이는 "VLM 이 physics 를 활성화하려면 language 가 필요하다" 에서 "VLM 은
  physical-looking 자극에서 physics 로 default 하며, 명시적 abstract label
  (`circle`) 만이 그 default 에 적극 대립한다" 로 서사를 바꾼다.
- `line/blank/none` 4-label 표 (§3.3) 가 이 주장의 가장 깔끔한 figure: 각
  label 기여를 직교적으로 localize 한다 (`ball` → regime, `circle` → 억제,
  `planet` → orbit-prior).
- 방법론적으로 §4 의 activation-capture diff note 가 M4 섹션에서 표시되어야
  한다 — 원래 switching-layer 주장을 완화하고 capture 된 hidden states 가
  측정할 수 있는/없는 것을 명확히 한다.

## 7. 남은 한계

- Forced-choice label-free 는 deferred: FC 선택지 ("It falls / stays / ...")
  는 label 없이 해석되려면 주어를 다시 써야 한다. `this` 또는 `the object`
  같은 단어 placeholder 가 다음에 검증할 설계.
- Last-question-token / start-of-answer 위치에서의 capture 는 LM 내부의
  label downstream effect 를 측정할 수 있다. 여기서는 수행하지 않음.
- 단일 prompt text; sensitivity audit (예: "Describe what is in the image..."
  vs "Is there a ball / object / shape?...") 가 wording 효과를 core
  label-vs-no-label contrast 와 분리할 것이다.
- Cross-model replication 은 M6 scope. 같은 자극에 LLaVA-1.5 가 physics-mode
  진입에 `ball` 이 필요하다면, 여기서의 "시각 default" 는 Qwen 특이적.
