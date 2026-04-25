# M8d — 비공(非球) 물리 객체 카테고리 (외적 타당성 라운드 2)

**상태**: 사전 등록 완료 (측정 전 기준 잠금). 결과 섹션은 실행 후 추가.

## 동기

M8a의 도형 스윕 (정사각형 / 삼각형 / 육각형 / 불규칙 다각형) 결과
시각 포화 가설이 검증되었고 H1 + H7은 *비포화 전용*
(LLaVA-clear / Qwen-suppressed) 으로 한정되었다. 그러나 모든 도형이
여전히 2-D 기하 프리미티브이며 라벨 트리플이 자기 자신의
언어 별칭 (`circle ↔ ball ↔ planet`,
`polygon ↔ rock ↔ boulder`) 으로 구성된다.
M8d는 인접한 일반화 질문을 던진다: H7 (라벨이 regime을 선택) 과
H1 (객체 추상화 ramp) 결과가 **비공 물리 객체 카테고리** —
자동차 / 사람 / 새 — 로 확장되는가? 이 카테고리들의 regime 구분은
중력 낙하가 아니라 운동 동작 (drives / walks / flies) 대
묘사 스타일 추상 라벨 (silhouette, stick figure) 이다.

H7이 일반화되면 발표 가능한 주장이
"라벨이 중력 family 안의 특정 물리 regime을 디스패치한다
(circle / ball / planet)" 에서
"라벨이 객체 종류에 걸쳐 *카테고리*에 적합한 물리 regime을
디스패치한다" 로 확장된다. 이는 더 큰 기여이며 M5a / M5a-ext steering
결과를 circle 한정 메커니즘 진술에서 카테고리 일반 진술로 끌어올린다.

## 자극 설계

3 카테고리 × 4 추상화 레벨 × 2 배경 × 2 큐 조건 × **2 이벤트** ×
5 시드 = **480개 자극**. M8a 대비 이벤트 축이 두 배가 된다 — `fall`은
M8a 비교용 중력 스트레스 테스트이고 `horizontal`은 regime 선택이
가장 깨끗한 자연 이벤트 셀 (자동차는 수평 운전, 사람은 수평 보행,
새는 수평 비행) 이다.

```
shapes:         car, person, bird
object_levels:  line, filled, shaded, textured
bg_levels:      blank, ground
cue_levels:     none, both     (none = 노출; both = cast_shadow + motion_arrow)
event:          fall, horizontal
seeds:          5
```

카테고리별 라벨 트리플 (physical / abstract / exotic) 은 프롬프트
시점에 `LABELS_BY_SHAPE` 를 통해 디스패치된다:

| 카테고리 | physical | abstract       | exotic   |
|----------|----------|----------------|----------|
| car      | car      | silhouette     | figurine |
| person   | person   | stick figure   | statue   |
| bird     | bird     | silhouette     | duck     |

`abstract` 역할은 비공 카테고리에 자연스러운 기하 클래스 명칭이
없으므로 강제 기하 클래스가 아니라 묘사 스타일 라벨 (silhouette /
stick figure) 을 사용한다. `exotic` 역할은 각 카테고리의 기본 운동
이벤트로부터 regime을 옮긴다: `figurine` → 진열용 정적 토이,
`statue` → 정적 석상 인물, `duck` → 혼합 (수영 / 뒤뚱뒤뚱 / 걷기 +
기본 비행). 알려진 약점: `bird/duck` 은 부분적 (오리는 비행하는 새이며
exotic shift는 "수영/뒤뚱"이지만 비행도 regime 분포에 남음). 비행하지
않는 새 (펭귄 / 타조) 가 더 깨끗한 신호를 주지만 저해상도에서
인식 가능하게 그리기 어렵다.

3×4 시각 그리드 (`docs/figures/m8d_shape_grid.png`) 는 각
카테고리-레벨 셀이 시각적으로 구분 가능함을 확인한다; M8a 스타일의
풀 신 그리드 (`docs/figures/m8d_full_scene_samples.png`) 는
카테고리 × 이벤트 조합을 보여준다. `horizontal` 이벤트의 자동차 /
사람은 자연 운동 의미 (drives / walks) 가 cast shadow와 기하학적으로
일치하도록 ground line **위에** 배치된다; 새는 공중 (자연 비행 의미)
배치를 유지한다.

## Regime 분류기

표준 PMR (중력 동사 편향) 은 자동차 (`drives`) 와 사람 (`walks`,
`runs`) 응답이 생성하는 운동 동사를 체계적으로 과소 카운트하므로,
M8d는 카테고리 인식 분류기 `classify_regime(text, category) →
{kinetic, static, abstract, ambiguous}` 를 병행 정의한다
(`metrics/lexicons.py::CATEGORY_REGIME_KEYWORDS`,
`metrics/pmr.py::classify_regime`). 어휘는 수작업 큐레이션:

```
car kinetic    : driv roll spee mov race accel trav head
car static     : park stop stay still stationary display remain
person kinetic : walk run jog step stride mov march pace
person static  : stand stay still stationary motionless frozen sit rest remain
bird kinetic   : fly fli flew flown swim swam soar waddl mov glid flap hop
bird static    : perch sit stay still stationary rest remain
```

`abstract` 마커는 카테고리 공통 (M8a와 공유되는 기존 `ABSTRACT_MARKERS`
세트 — "abstract", "geometric", "drawing", "diagram", "won't move",
"nothing happens" 등). 분류기는 kinetic / static 검사 전 abstract 마커에
서 단락(short-circuit)하므로 "this is just a drawing of a car — it
stays parked" 같은 응답은 `static`이 아니라 `abstract`로 분류된다.

`metrics/pmr.py` 의 원본 PMR / GAR 채점기는 **변경되지 않으므로** M8a
분석은 비트 단위로 동일하다. M8d 분석 코드는 `m8d_analyze.py::annotate`
를 통해 `classify_regime`을 직접 호출한다.

이진 PMR 스타일 분석을 위해 M8d는 다음을 정의한다:

```
PMR_regime = 1  iff  classify_regime(text, category) ∈ {kinetic, static}
PMR_regime = 0  iff  regime ∈ {abstract, ambiguous}
```

이는 "어떤 regime이 발화되든 모델이 객체를 물리적으로 다루었는가"
를 포착하며 원본 PMR과 같은 정신이다. H1 ramp는 이벤트 union 위에서
`PMR_regime` 을 사용; H7 paired-delta는 `horizontal` 부분 집합 (자연
이벤트 가장 깨끗한 셀) 에서 `PMR_regime`을 사용한다.

## 사전 등록된 성공 기준 (측정 전 2026-04-25 잠금)

세 가지 기준, M8a와 평행하게, 명시된 대로 충족 시 카테고리 횡단 복제
선언; 그렇지 않으면 실패 (또는 부분 복제) 보고. 기준은 의도적으로
타이트 — 거품 도장이 아니라 진짜 테스트.

### H1 — object_level 추상화 ramp 일반화 (카테고리별)
각 카테고리에 대해 라벨 `open` 프롬프트 하에서 모든 (bg, cue,
label_role, event, seed) 셀을 평균하여
PMR_regime(line / filled / shaded / textured) 계산.
**복제**: 2/3 이상의 카테고리가
`PMR_regime(textured) − PMR_regime(line) ≥ 0.05`
를 만족하고, line→filled→shaded→textured 시퀀스에서 0.05 초과의
내부 역전이 없을 것.

### H7 — label-role이 PMR_regime을 구동 (physical > abstract)
각 카테고리에 대해 `horizontal` 부분 집합 (자연 이벤트 가장 깨끗한
셀) 에서 `label_role` 별 평균 PMR_regime 계산.
**복제**: 2/3 이상의 카테고리가
`PMR_regime(physical) − PMR_regime(abstract) ≥ 0.05` 만족.

### 시각 포화 paired-delta 일반화
각 카테고리 × 모델에 대해 `horizontal` 부분 집합에서 시드별로
paired-delta = PMR_regime(label_role) − PMR_regime(_nolabel) 계산
(카테고리 내 평균).
**복제**: 모델당 2/3 이상의 카테고리가 M6 r2 / M8a 선례 방향을 보일 것
— Qwen은 0 근처 (포화) 또는 노이즈로 축소; LLaVA는 대부분의
카테고리에서 `physical` 역할 ≥+0.05.

### 실패 모드
- H1이 대부분의 카테고리에서 실패 → ramp가 도형 (또는 카테고리)
  특이 — 그 자체로 발표 가능한 뉘앙스. null 결과를 부끄러워하지 않는다.
- 어느 카테고리에서 H7 실패 → 라벨 태깅이 시각 콘텐츠를 압도하기에
  너무 약할 수 있음 (M8a의 `wedge` / `polygon` 약 라벨과 같음). 관련
  카테고리별 라벨을 표기하고 논문에서 라벨 디자인 caveat을 논의한다.

## 셋업

```bash
# 자극 (한 번 실행; 4개 추론 config가 재사용).
uv run python scripts/01_generate_stimuli.py --config configs/m8d_qwen.py

# 추론 (단일 GPU 0, 순차).
M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_qwen.py            --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_qwen_label_free.py --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_llava.py           --stimulus-dir "$M8D_DIR"
CUDA_VISIBLE_DEVICES=0 uv run python scripts/02_run_inference.py \
    --config configs/m8d_llava_label_free.py --stimulus-dir "$M8D_DIR"

# 동등하게:
bash scripts/m8d_run_all.sh

# 분석 + 그림.
uv run python scripts/m8d_analyze.py \
    --qwen-labeled  outputs/m8d_qwen_<ts>/predictions.jsonl \
    --qwen-nolabel  outputs/m8d_qwen_label_free_<ts>/predictions.jsonl \
    --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \
    --llava-nolabel outputs/m8d_llava_label_free_<ts>/predictions.jsonl \
    --out-dir       outputs/m8d_summary
uv run python scripts/m8d_figures.py --summary-dir outputs/m8d_summary

# 손 라벨링 분류기 검증 (50 stratified 행).
uv run python scripts/m8d_hand_annotate.py --mode sample \
    --qwen-labeled  outputs/m8d_qwen_<ts>/predictions.jsonl \
    --llava-labeled outputs/m8d_llava_<ts>/predictions.jsonl \
    --out           docs/experiments/m8d_hand_annotate.csv
# ... hand_regime 컬럼 채우기 ...
uv run python scripts/m8d_hand_annotate.py --mode score \
    --csv           docs/experiments/m8d_hand_annotate.csv
```

## 결과

_실행 완료 후 추가 (~30분 wall-clock, H200 GPU 0).
`scripts/m8d_analyze.py` 헤드라인 + 그림으로 채움._

### 사전 등록된 채점 요약

| 기준              | Qwen  | LLaVA |
|-------------------|-------|-------|
| H1 ramp           | _TBD_ | _TBD_ |
| H7 (phys>abs)     | _TBD_ | _TBD_ |
| 시각 포화 delta    | _TBD_ | _TBD_ |

### PMR_regime(_nolabel) 베이스라인

| 카테고리 | Qwen  | LLaVA |
|----------|-------|-------|
| car      | _TBD_ | _TBD_ |
| person   | _TBD_ | _TBD_ |
| bird     | _TBD_ | _TBD_ |

### 헤드라인 해석

_채워질 예정._

### 카테고리별 노트

_채워질 예정. 예상:_
- _car: 가장 깨끗 (시각적으로 가장 구분되며 운동 동사 응답이 풍부)._
- _person: 중간 (라벨 `person`이 약간 덜 구체적일 수 있음)._
- _bird: H7 가장 잡음 (`duck` exotic 역할이 부분적 — 비행이 완전히
  억제되지 않음)._

## 가설 업데이트 (결과 후 채움)

- H1 — _TBD_
- H7 — _TBD_ (카테고리 횡단)
- H-encoder-saturation — _TBD_

## 로드맵 함의

_결과 후 채움._
