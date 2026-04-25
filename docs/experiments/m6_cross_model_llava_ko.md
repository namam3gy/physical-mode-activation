# M6 Round 1 — LLaVA-1.5-7B Cross-Model 실행 로그

M2 + M4b 를 동일한 자극 위에 LLaVA-1.5-7B-hf 로 재실행. 핵심 질문:
M4b 의 H2 reframing (`ball ≈ no-label`, `circle = suppressor`) 이 두 번째
open-source VLM 에도 일반화되는가, 아니면 Qwen 특이적인가?

Round 1 은 LLaVA-1.5-7B-hf 만; LLaVA-Next, InternVL2, Qwen2-VL 는 deferred.

실행일: 2026-04-25.

## 설정

- Configs:
  - `configs/cross_model_llava.py` — labels `(circle, ball, planet)` × `prompt_variants=("open",)`.
  - `configs/cross_model_llava_label_free.py` — `labels=("_nolabel",)` × `prompt_variants=("open_no_label",)`.
- Stimuli: M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/` 재사용 (480 stim).
- 모델: `llava-hf/llava-1.5-7b-hf`, bf16, sdpa attention.
- 생성: T=0.7, top_p=0.95, max_new_tokens=96.
- Forced-choice 제외 (아래 "Smoke test → FC 편향" 참조).
- Activation capture 비활성 (round 1 은 행동 측정만).
- 출력:
  - `outputs/cross_model_llava_20260425-035506_7ff0256b/` — labeled (1440 rows).
  - `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/` — label-free (480 rows).

## Smoke test → FC 편향

4-cell × 3-label × 3-variant smoke (36 inferences) 결과: LLaVA-1.5-7B 의
`forced_choice` 가 **모든** (image, label) 조합에서 first_letter `A` 를 반환
(`line/blank/none`, `textured/blank/none`, `textured/ground/both`,
`line/blank/both` × {ball, circle, planet} 중 12/12). 편향은 T=0.7 에서
non-stochastic, 자극 내용 무관.

결정: 이번 round 에서 FC 완전 제외. Open prompt 는 다양하고 label/image
민감한 응답을 주어 cross-model H1, H2, H7 검증에 활용 가능. H4 (open-FC gap)
는 이번 round 에서 미검정; deferred.

## 행동 결과 — LLaVA labeled (open)

### 전체 (label 전반 평균)

| metric | 값 |
|---|---|
| n | 1440 |
| PMR | 0.681 |
| GAR | 0.194 |
| hold_still | 0.010 |
| abstract_reject | 0.001 |

### Label 별

| label  | PMR  | GAR  | hold_still |
|---|---|---|---|
| ball   | 0.858 | 0.356 | 0.012 |
| circle | 0.556 | 0.153 | 0.010 |
| planet | 0.627 | 0.072 | 0.006 |

### object_level × label (PMR)

| object | ball  | circle | planet |
|---|---|---|---|
| line     | 0.833 | 0.275 | 0.417 |
| filled   | 0.825 | 0.575 | 0.567 |
| shaded   | 0.875 | 0.658 | 0.725 |
| textured | 0.900 | 0.717 | 0.800 |

### cue_level × label (PMR)

| cue          | ball  | circle | planet |
|---|---|---|---|
| both         | 0.883 | 0.608 | 0.658 |
| cast_shadow  | 0.858 | 0.658 | 0.592 |
| motion_arrow | 0.842 | 0.508 | 0.692 |
| none         | 0.850 | 0.450 | 0.567 |

## 행동 결과 — LLaVA label-free

### 전체

| metric | 값 |
|---|---|
| n | 480 |
| PMR | 0.383 |
| GAR | 0.181 |
| hold_still | 0.123 |
| abstract_reject | 0.006 |

### object_level 별

| object | PMR | hold_still |
|---|---|---|
| line     | 0.142 | 0.058 |
| filled   | 0.317 | 0.067 |
| shaded   | 0.592 | 0.225 |
| textured | 0.483 | 0.142 |

(주의: shaded > textured 는 anomaly; LLaVA-1.5 가 "rendered 3D shading" 을
"photorealistic texture" 보다 더 physical-content-laden 으로 읽는 것 같다.
Round 2 에서 flagging 필요하지만 blocking 은 아님.)

### cue_level 별

| cue          | PMR | hold_still |
|---|---|---|
| both         | 0.442 | 0.067 |
| cast_shadow  | 0.508 | 0.167 |
| motion_arrow | 0.292 | 0.083 |
| none         | 0.292 | 0.175 |

## H2 cross-model — paired PMR delta

M4b 와 동일한 `(obj, bg, cue, seed)` 페어링. 각 cell 은 10 seeds × T=0.7 로
평균.

### LLaVA-1.5 (이 run)

| label  | mean `PMR(label) − PMR(_nolabel)` | n_pairs |
|---|---|---|
| ball   | **+0.475** | 480 |
| planet | **+0.244** | 480 |
| circle | **+0.173** | 480 |

### Qwen2.5-VL-7B (M4b, 비교용)

| label  | mean `PMR(label) − PMR(_nolabel)` | n_pairs |
|---|---|---|
| ball   | +0.006 | 480 |
| planet | +0.006 | 480 |
| circle | **−0.065** | 480 |

## Cross-model S-curve — object_level 별 PMR

| object   | Qwen labeled | Qwen no-label | LLaVA labeled | LLaVA no-label |
|---|---|---|---|---|
| line     | 0.906 | 0.942 | 0.508 | 0.142 |
| filled   | 0.933 | 0.933 | 0.656 | 0.317 |
| shaded   | 0.933 | 0.942 | 0.753 | 0.592 |
| textured | 0.950 | 0.975 | 0.806 | 0.483 |

- Qwen labeled 은 모든 object level 에서 0.93 근방에 saturate — S-curve 보이지 않음.
- Qwen no-label 도 동일한 flat 패턴 (visual saturation).
- LLaVA labeled 은 분명한 monotone S-curve (0.51 → 0.81).
- LLaVA no-label 은 non-monotone (shaded > textured) — 위 anomaly 참조.

## H7 cross-model — label 별 GAR (open prompt)

| label  | Qwen | LLaVA |
|---|---|---|
| ball   | 0.706 | 0.356 |
| circle | 0.753 | 0.153 |
| planet | **0.319** | **0.072** |

H7 패턴 (`planet` GAR << `ball`/`circle` GAR) 이 두 모델 모두에서 재현됨 —
`planet` label 이 physics 서술을 gravity-aligned motion 대신 orbital / cosmic
event 로 route. 샘플 LLaVA `planet × line/blank/none` 응답: "the planet
will continue to spin and orbit around the sun", "the planet will be
consumed by a black hole" — 명시적 non-gravitational physics.

H7 verdict: **supported (cross-model)**. Label-selects-regime 메커니즘은
Qwen 특이적이 아님. 크기는 다르지만 qualitative dissociation 은 보존.

## Cell 별 — `line/blank/none` × label (LLaVA)

| label    | PMR | hold_still |
|---|---|---|
| _nolabel | 0.0 | 0.0 |
| ball     | 0.9 | 0.0 |
| circle   | 0.0 | 0.0 |
| planet   | 0.5 | 0.0 |

(GAR 계산 불가; bg_level=blank → ground 없음.)

`line/blank/none × ball` 의 LLaVA 는 9/10 을 physics 로 commit ("ball will
fall", "ball will roll off the edge"), Qwen 의 동일 cell 4/10 commit (그 중
1개는 hold_still) 과 대조. 이는 cross-model H2 fork 의 가장 깔끔한 표현:

- **Qwen**: 이미지가 완전 추상 → visual evidence 약함 → prior 가 모델을
  강하게 push 하지 않음 → physics-mode + hold-still spread, 전체 PMR ~0.40.
- **LLaVA**: 이미지가 완전 추상 → LLaVA 의 visual evidence 는 더욱 약함 →
  no-label baseline 이 0 으로 collapse; ball label 이 *강한* physics
  activator → 9/10 commit.

## 샘플 raw 응답

`line/blank/none` (완전 추상) cell:

- LLaVA × ball:
  - "The ball will roll off the edge of the circle."
  - "The ball will fall."
  - "The ball will fall into a hole."
- LLaVA × circle:
  - "The circle will become more visible."
  - "The circle is going to get smaller and smaller."
  - "The circle will become a dot."
- LLaVA × planet:
  - "In the next moment, the planet will continue to spin and orbit around the sun."
  - "The planet will be consumed by a black hole."
- LLaVA × _nolabel:
  - "A round object is in the middle of the image, and the next part of the image is white."
  - "A black circle is seen in the background and the center of the circle is white."

`textured/blank/none` (physical-looking object, no scene):

- LLaVA × ball:
  - "The ball is in the air and about to fall."
  - "The ball is suspended in midair... it is likely that it will remain in th[e air]."
  - "The ball will roll."
- LLaVA × circle:
  - "The circle will roll on the ground."
  - "The circle is going to be on the ground."
  - "The circle will spin and change direction."

(주의: textured-ball 이미지에서 `circle` label 이라도 LLaVA 는 종종 "the
circle" 에 physics-mode 술어를 적용한다. 따라서 LLaVA 는 circle → abstract
로 strict 라우팅하지 않음; 이미지 + label 의 joint 작용으로 라우팅하며,
이미지가 충분한 physical signal 을 가질 때 이미지가 우세.)

## 원본 artifacts

- `outputs/cross_model_llava_20260425-035506_7ff0256b/predictions{_scored,}.{jsonl,parquet,csv}` — 1440 rows.
- `outputs/cross_model_llava_label_free_20260425-040821_39e68cd4/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/cross_model_llava_*/summary_*.csv` — factor-level rollups.
