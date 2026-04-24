# 점수 기준 — PMR / GAR / RC

`references/project.md` §2.2 의 세 가지 behavioral metric 의 정확한 정의.
구현: `src/physical_mode/metrics/pmr.py`.

## PMR — Physics-Mode Priming Rate

응답당 binary. **PMR = 1 iff**:

1. lowercased 응답에 `ABSTRACT_MARKERS` 의 phrase 가 *전혀* 없음
   (예: "this is just a circle", "won't move", "abstract shape"); **AND**
2. 응답의 whitespace-separated word 중 최소 하나가 `PHYSICS_VERB_STEMS` 의 stem
   으로 시작 (예: "falls" 가 `fall` 매치, "rolling" 이 `roll` 매치, "accelerates"
   가 `accelerat` 매치).

abstract-reject gate 가 **먼저** 적용되어 "this is an abstract shape — it won't
move" 같은 응답은 "move" 가 `mov` stem 에 매치되어도 PMR = 0 으로 점수.

### Lexicon 확장

`predictions_scored.csv` 검사에서 false negative (명백한 physical 응답이 0
으로 점수) 가 발견되면 `src/physical_mode/metrics/lexicons.py` 의
`PHYSICS_VERB_STEMS` 에 누락된 stem 추가 + `tests/test_pmr_scoring.py::PMR_POSITIVE`
에 assertion 추가. False positive 의 경우 반대로. Lexicon 은 canonical 이 아닌
living 으로 취급.

## GAR — Gravity-Align Rate

Ternary: **1 / 0 / None**. 다음 조건에서만 정의:

- `bg_level ∈ {ground, scene}` (지면이 존재해야 함), **AND**
- `event_template ∈ {fall, roll_slope}` (중력이 salient force)

GAR 가 정의된 응답에서 **GAR = 1 iff** 응답이 `DOWN_DIRECTION_PHRASES` 의
phrase 를 포함 (예: "down", "to the ground", "onto the floor"). Aggregation
은 `None` row 를 skip.

## RC — Response Consistency

각 factorial cell (object × bg × cue × event × label × prompt_variant) 에 대해,
RC 는 seed 간 majority PMR value 의 비율:

    RC(cell) = max(count(PMR=1), count(PMR=0)) / n(cell)

RC ∈ [0.5, 1.0]. 낮은 RC 는 모델이 *동일 factor level* 에서 physics-mode 와
abstract-mode 를 flip 한다는 뜻 — prompt 불안정성 또는 borderline cue 의 신호.
주의: temperature = 0 에서는 RC 가 degenerate (항상 1.0); M2 에서는 RC 가
informative 하도록 temperature 를 0.7 로 높임.

## 보조 컬럼

Scorer 가 또한 emit:

- `hold_still` — "stay / remain / rest / sit" verb 가 fire 하면 1.
  PMR = 0 과 co-occur 하면 "명시적 no-motion" 케이스.
- `abstract_reject` — `ABSTRACT_MARKERS` phrase 가 매치하면 1. PMR gate 와
  동등하지만 분석에서 "abstract 로 거부됨" 과 "아무것도 예측 안 함" 을 구분
  하기 위해 surface.

## Cell-level 예상 outcome (sanity check 용)

| Object | Bg | Cue | Event | 예상 PMR | 예상 GAR |
|---|---|---|---|---|---|
| `line` | `blank` | `none` | `fall` | low (< 0.3) | N/A |
| `textured` | `ground` | `none` | `fall` | high (> 0.7) | high (> 0.7) |
| `line` | `blank` | `wind` | `horizontal` | mid (wind cue 단독이 motion language 유발할 수 있음) | N/A |
| `shaded` | `ground` | `arrow_shadow` | `fall` | high | high |

이는 사전 기대치이지 ground truth 가 아니다. 이 표에서 크게 벗어나면 *과학적*
finding 이지 bug 가 아님 — 적절한 `docs/experiments/m{N}_*.md` 에 sample model
output 과 함께 기록.

## 알려진 scoring artifact

Forced-choice 응답은 종종 선택되지 않은 option 을 enumerate ("D — it *cannot
fall, move, or change direction*"); enumerate 된 verb 가 `PHYSICS_VERB_STEMS`
를 trigger 해서 false PMR=1 를 produce. Forced-choice run 을 평가할 때 PMR 과
응답의 **첫 글자** (A/B/C/D) 를 함께 봐야 함. 첫 글자 신호가 더 깨끗한 causal
indicator; PMR 은 open-ended 응답용 적절 metric.
