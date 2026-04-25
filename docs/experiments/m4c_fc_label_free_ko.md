# M4c FC Label-Free Prompt — 실행 로그

M4b 의 companion — H2 null-hypothesis test 를 open-ended 에서 forced-choice
prompt 로 확장. 새 `forced_choice_no_label` variant (4-way MCQ, 옵션 A-C 가
"the depicted object" 를 antecedent 로 사용) 추가하여 Qwen2.5-VL-7B 와
LLaVA-1.5-7B 양쪽에 동일 M2 자극에서 실행.

실행일: 2026-04-25.

## 설정

- 새 prompt variant: `forced_choice_no_label` — `forced_choice` 와 동일한
  A/B/C/D 옵션이지만 A/B/C 의 주어가 "It"/"the {label}" 대신 "the depicted
  object". D 는 그대로 유지 ("This is an abstract shape — nothing physical
  happens.") — depicted object 를 참조하지 않으므로.
- Configs:
  - `configs/fc_label_free_qwen.py` — Qwen2.5-VL-7B-Instruct.
  - `configs/fc_label_free_llava.py` — llava-hf/llava-1.5-7b-hf.
- 자극: M2 manifest `inputs/mvp_full_20260424-093926_e9d79da3/`.
- 생성: T=0.7, top_p=0.95, max_new_tokens=96.
- Activation capture 비활성.
- 출력:
  - `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/` — 480 rows.
  - `outputs/fc_label_free_llava_20260425-044517_81ae56d5/` — 480 rows.

## Prompt template

```
Look at the image. Which option best describes what will happen next?
A) The depicted object falls down.
B) The depicted object stays still.
C) The depicted object moves sideways.
D) This is an abstract shape — nothing physical happens.
Answer with a single letter (A, B, C, or D), then briefly justify.
```

## Qwen FC label-free — 전체

| metric | 값 |
|---|---|
| n | 480 |
| PMR (text-based) | 0.817 |
| GAR | 0.781 |
| hold_still | 0.225 |
| abstract_reject | 0.135 |

### First-letter 분포

| letter | count | %    |
|---|---|---|
| A | 292 | 60.8 |
| B |  77 | 16.0 |
| D |  57 | 11.9 |
| other | 54 | 11.3 |

(C 는 전혀 등장하지 않음, M2 FC labeled 와 일관.)

### object_level 별 first-letter

| object   | A  | B  | D  | other |
|---|---|---|---|---|
| filled   | 67 | 25 | 17 | 11 |
| line     | 60 | 16 | 30 | 14 |
| shaded   | 80 | 18 |  4 | 18 |
| textured | 85 | 18 |  6 | 11 |

### cue_level 별 first-letter

| cue          | A  | B  | D  | other |
|---|---|---|---|---|
| both         | 98 |  0 |  0 | 22 |
| cast_shadow  | 70 | 23 | 27 |  0 |
| motion_arrow | 88 |  0 |  0 | 32 |
| none         | 36 | 54 | 30 |  0 |

### Qwen open label-free (M4b) 와의 paired delta (`sample_id` 동일)

| comparison | mean Δ |
|---|---|
| `PMR(FC, _nolabel) − PMR(open, _nolabel)` | **−0.131** |

같은 (이미지, label-free) cell 에서 FC 가 open 보다 보수적 — image 가 모호할 때
더 많은 응답을 D (abstract reject) 로 라우팅.

### Qwen FC labeled (M2) 와의 paired delta (label 별)

| labeled M2 | mean Δ `PMR(FC, label) − PMR(FC, _nolabel)` |
|---|---|
| ball   | **+0.013** |
| circle | **−0.208** |
| planet | **−0.263** |

- `ball − _nolabel` ≈ 0 — M4b open prompt 의 null-finding 과 일치: ball 은
  no-label baseline 위로 enhance 하지 않음.
- `circle − _nolabel` = −0.208 — M4b open prompt delta −0.065 보다 훨씬 큼.
  FC 의 D 옵션이 깨끗한 abstract outlet 을 제공하여 circle 이 더 적극적으로 활용.
- `planet − _nolabel` = −0.263 — **FC 에서의 새 발견**. M4b open prompt 에서
  `planet ≈ no-label` 이었던 이유는 planet 의 orbital 응답 ("orbits the sun",
  "consumed by black hole") 이 verb lexicon 에서 physics-mode 로 채점되었기
  때문. FC 하에서는 옵션 셋이 gravity-centric (falls/stays/moves sideways)
  이라 같은 orbital 직관이 표현될 수 없고 D ("abstract shape — nothing
  physical happens") 로 collapse, PMR 억제.

## Qwen — `line/blank/none` cell (완전 추상, FC vs open vs labeled)

| condition          | PMR  | hold_still | abstract_reject | first_letter (n=10) |
|---|---|---|---|---|
| open × _nolabel    | 0.40 | 0.20 | 0.00 | n/a |
| FC × _nolabel      | 0.00 | 0.40 | 1.00 | D=9, B=1 |
| FC × ball          | 0.00 | 0.60 | 1.00 | D=10 |
| FC × circle        | 0.00 | 1.00 | 1.00 | D=10 |
| FC × planet        | 0.00 | 0.10 | 1.00 | D=10 |

이 완전 추상 cell 에서 FC 하에 **4개 label condition 모두 D 로 collapse**.
First-letter 표는 no-label run 이 labeled run 들 (D=10) 보다 약간 덜 collapse
됨 (D=9, B=1) — no-label 이 모델에 약간의 유연성을 주지만 FC 옵션 셋의 D escape
가 지배적. 완전 추상 이미지에서 FC 의 "abstract sink" pull 의 가장 깔끔한 시연.

주의: PMR=0.00 이지만 hold_still=0.60 / 1.00 인 것은 모순이 아님 — 모델이 D 를
선택하고 "this is a static representation" 같은 justification 을 작성; "static"
이 hold_still 을 트리거하지만 응답은 여전히 D 선택.

## LLaVA FC label-free — degenerate

전체 480 자극에 대한 `first_letter` 분포:

| letter | count |
|---|---|
| A | 477 |
| B | 3 |
| C | 0 |
| D | 0 |

M6 round 1 에서 관찰한 "A" 편향 (4-cell smoke 에서 labeled FC 12/12) 이
re-templated label-free prompt 에서도 **유지**. "It falls down" / "the
{label} falls down" → "the depicted object falls down" 재작성으로 LLaVA 가
A default 에서 벗어나지 않음. 480 응답 중 3개만 B 선택; C 와 D 는 0.

이는 LLaVA FC 편향의 원인으로서 prompt-template 가설을 배제. 편향은 모델
수준 — LLaVA-1.5 의 학습 데이터가 MCQ 맥락에서 "A" 를 선호하거나 FC choice
generation 단계에서의 tokenization artefact. 어느 쪽이든 FC 는 verb-PMR 또는
first-letter metric 으로 LLaVA-1.5 에서 사용 불가.

LLaVA FC label-free 샘플 응답:

| cell | 응답 |
|---|---|
| line/blank/none | `'A'` (× 9), 나머지도 `'A'` |
| textured/ground/both | `'A'` (× 10) |
| textured/blank/none | `'A'` (× 10) |

(480 자극 중 3개 `B` 응답은 cell 패턴 없이 산재 — random noise.)

## 원본 artifacts

- `outputs/fc_label_free_qwen_20260425-042817_eec92f1a/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows.
- `outputs/fc_label_free_llava_20260425-044517_81ae56d5/predictions{_scored,}.{jsonl,parquet,csv}` — 480 rows (degenerate; H2 testing 에 사용 안 함).
- `outputs/fc_label_free_*/summary_*.csv` — factor-level rollups.
