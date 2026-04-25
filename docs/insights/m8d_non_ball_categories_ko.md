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

## 결과 (2026-04-25)

총 wall clock: H200 GPU 0에서 **32분** (Qwen labeled 12.6분, Qwen
label-free 6.2분, LLaVA labeled 8.8분, LLaVA label-free 4.3분). 480
자극 × 4 추론 config = 3840 추론.

분류기 검증: **오차율 5.6 %** (54개 손 라벨링 행 중 3개가 keyword
분류기와 불일치; paper-ready 15 % 임계값 이하). 3개 미스매치 모두
"no movement" / "pulled away" 패턴의 stem-matching false-positive — 알려진 한계.

### 사전 등록된 채점 요약

| 기준              | Qwen  | LLaVA |
|-------------------|-------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| 시각 포화 delta   | 1/3 (bird) | 2/3 (car, bird; person 음수로 flip) |

### PMR_regime(_nolabel) 베이스라인 (event union)

| 카테고리 | Qwen  | LLaVA |
|----------|-------|-------|
| car      | 1.000 | 0.450 |
| person   | 0.988 | 0.744 |
| bird     | 0.831 | 0.706 |

(`horizontal` 부분 집합만의 PMR_regime(_nolabel): Qwen car 1.000 /
person 0.975 / bird 0.862; LLaVA car 0.550 / person 0.838 / bird 0.688.)

Qwen은 카테고리에 걸쳐 0.83-1.00 — 포화. LLaVA는 0.45-0.74 범위 —
인코더가 binary 메트릭에 라벨이 움직일 여지를 부여.

### 헤드라인 해석

**H7 (라벨이 regime을 선택) 결과가 LLaVA에서 카테고리 횡단 3/3
일반화.** 프로젝트에서 가장 강력한 카테고리 횡단 H7 증거. LLaVA
`horizontal` 부분 집합:

| 카테고리 | physical PMR | abstract PMR | physical − abstract |
|----------|--------------|--------------|---------------------|
| car      | 0.825        | 0.300        | **+0.525** |
| person   | 0.738        | 0.600        | +0.138 |
| bird     | 0.950        | 0.400        | **+0.550** |

car와 bird는 abstract 라벨 (silhouette) 이 물리 모드 응답을 강하게
억제. person은 억제가 더 작음 (`stick figure` 라벨이 LLaVA에서
걷는 응답을 여전히 허용).

**Qwen의 H7은 사전 등록 strict에서 실패** — 왜냐하면 PMR_regime이
모든 (label_role × category) 셀에서 0.95-1.0 천장 포화이기 때문.
이 천장은 정확히 M8a 스타일 시각 포화 패턴: Qwen 인코더가 카테고리를
인식하면 자동으로 물리 모드를 트리거 → 라벨 억제가 binary 메트릭을
움직일 수 있는 여지(headroom)가 없음.

**Qwen 천장 아래에서 regime 분포는 H7 신호가 카테고리 수준에서 살아있음을 보여줌**:

| category × label | kin_frac | static_frac |
|---|---|---|
| car/physical (car)       | 0.944 | 0.050 |
| car/exotic (figurine)    | 0.806 | **0.175** |
| person/physical (person) | 0.912 | 0.081 |
| person/exotic (statue)   | 0.750 | **0.225** |
| bird/physical (bird)     | 0.981 | 0.012 |
| bird/exotic (duck)       | 0.962 | 0.006 |

statue와 figurine 라벨은 Qwen에서 static regime을 17.5 % / 22.5 %
주입 (physical 라벨의 ~5 % 대비). duck exotic은 그렇지 않음 — 사전
등록된 약점 (duck still flies) 확인.

번역하면: **이진 "physics-mode active" 측정은 Qwen에서 인식 가능한
카테고리에 포화하지만, 카테고리 regime 선택 (kinetic vs static) 은
라벨 주도 조작에서 살아남음.** M2의 "라벨이 regime을 선택" 결과의
더 미세한 버전 — 이진 physics 축이 포화돼도 regime 축은 robust.

### 시각 포화 paired-delta (horizontal 부분 집합)

| (카테고리, 역할)   | Qwen Δ | LLaVA Δ |
|--------------------|--------|---------|
| car / physical     | +0.000 | +0.275  |
| car / abstract     | -0.012 | **-0.250** |
| car / exotic       | -0.025 | +0.000  |
| person / physical  | +0.025 | -0.100  |
| person / abstract  | +0.012 | -0.238  |
| person / exotic    | +0.012 | -0.288  |
| bird / physical    | +0.125 | +0.262  |
| bird / abstract    | +0.088 | **-0.288** |
| bird / exotic      | +0.138 | +0.238  |

LLaVA의 car/abstract = -0.250와 bird/abstract = -0.288가 M8d에서
가장 강한 cell-단위 억제 — silhouette 라벨이 PMR_regime을 자동차에서
~25 pp, 새에서 ~29 pp 줄임. Qwen은 모두 0 근처 (포화). bird는
Qwen에서 +0.09~+0.14의 작은 양수 delta가 일관되게 나타남 — 인코더가
세 카테고리 중 bird에서 가장 덜 포화 (PMR_regime(_nolabel)
horizontal 베이스라인 0.862 대 car/person ≈ 0.97-1.0).

LLaVA의 person은 특이한 패턴: **세 역할 모두 음수 paired-delta**
(−0.10, −0.24, −0.29). person `_nolabel` 베이스라인 (PMR_regime
0.838) 이 비정상적으로 높음 — 라벨 없는 응답이 stick figure나 person
silhouette에 대해 자주 "the person walks forward"를 출력하기 때문.
person은 가장 nuanced한 M8d 셀이며 가장 잡음이 많은 카테고리.

### 카테고리별 노트

- **Car** — 가장 깨끗한 H7 LLaVA 결과 (PMR +0.525; Qwen kin_frac
  +0.138). figurine은 Qwen에서 17.5 % static — "진열용 정적 토이"
  읽기가 작동.
- **Person** — 가장 잡음. person/`_nolabel` LLaVA 베이스라인이
  높음 (시각적 기본 읽기가 "the person walks") → 라벨이 억제만
  가능하고 추가는 불가능. Qwen에서 statue가 가장 강한 static 주입
  (22.5 %).
- **Bird** — LLaVA에서 가장 강한 cell-단위 PMR delta (+0.550 physical
  − abstract). duck exotic이 kinetic 유지 — 사전 등록된 약 라벨
  노트의 cross-validation: duck flies as much as bird flies, M8d
  round 2에는 비행 못 하는 새가 필요.

### H1 ramp 실패 분석

H1은 두 모델 모두에서 실패 — Qwen 천장 (0.97 → 0.98), LLaVA 비단조
(car 평탄 ~0.50, person 0.68 → 0.43 → 0.64, bird 0.74 → 0.84 → 0.68
→ 0.73). 비단조성 자체가 정보적:

- circle/ball (M2) 의 경우, `line`은 *abstract reject 영역* — 시각
  기본 읽기가 "this is just a circle". 시각 디테일 추가가 점진적으로
  물리 모드 활성화.
- car/person/bird의 경우, 모든 추상화 레벨이 *이미 카테고리 인식
  가능*: line car는 바퀴가 있고, line person은 stick figure이며,
  line bird는 부리가 있다. 카테고리는 line 레벨만으로도 식별되므로
  ramp 할 게 없음 — 시각 디테일은 의미(affordance)를 바꾸지 않고
  표면 사실성만 바꿈.

이것이 **shape-vs-category 분리**: H1 ramp는 추상-도형 ↔ 물리-객체
축의 속성 (라벨이 명확화) 이지, 명명된 카테고리에 대한 추상화-레벨
간의 객체 인식 가능성의 일반적 속성이 아님.

## 가설 업데이트

- **H1** — *추상-도형 ↔ 물리-객체 축에 추가로 한정*. M8a는 H1을
  비포화 전용 (LLaVA-clean / Qwen-suppressed) 으로 발견. M8d는 H1이
  카테고리 명명 객체에서 LLaVA에서도 실패함을 발견 — 카테고리
  인식이 이진 측정을 시각 추상화와 무관하게 포화시키기 때문. H1은
  추상-도형 ↔ 물리-객체 축의 특정 속성이며, 일반적 시각-사실성 →
  물리-prior 결합이 아님.
- **H7** — **LLaVA에서 카테고리 횡단 3/3 복제, 추가로 Qwen에서
  regime-수준 복제**. 라벨이 카테고리에 적합한 regime을 디스패치한다는
  가장 강력한 증거. 구체적으로:
  - LLaVA: physical − abstract paired delta on PMR_regime (이진)
    car/person/bird 전반 ≥ +0.138.
  - Qwen: paired delta on KIN_fraction (4-class regime)
    physical − exotic ≥ +0.138 for car (figurine) and person (statue);
    이진 측정은 포화.
- **H-encoder-saturation** — **카테고리 횡단으로 추가 검증**. Qwen의
  car/person 천장 (0.97-1.0) 은 M8a 천장 패턴 복제; LLaVA의 PMR_regime
  범위 (0.55-0.84) 는 headroom 보유; 비대칭이 H7 측정 가능성을 예측
  → 우리가 관측하는 그대로 (Qwen H7 0/3 binary, LLaVA H7 3/3 binary).

## 로드맵 함의

1. **H7 일반화** — 논문 헤드라인을 "circle-only"에서 "cross-category"로 승격.
2. **H1은 도형 특이** — 논문 클레임을 "추상화 ramp는 기하 도형 ↔
   명명된 객체 축의 속성이지 일반적 시각 디테일 → 물리 prior
   메커니즘이 아님"으로 정제.
3. **regime-distribution 이 더 미세한 도구** — 이진 PMR_regime이
   포화할 때 regime 분포 안의 kinetic vs static split은 여전히
   label_role과 함께 움직임. 논문 방법론적 기여로 추가: "regime
   distribution 이 binary saturation에서 H7 신호를 구제".
4. **M8c (실제 사진)** 는 이제 강하게 동기화 — 사진 사실성이
   LLaVA의 인코더 갭을 닫는가? Qwen을 더 천장으로 밀어붙이는가?
   사전 등록된 M8c 기준은 M8d-검증된 regime 분류기 사용 가능.
5. **Round-2 약 라벨 수정** — `duck`은 `penguin` / `ostrich` /
   `chicken` (비행 못 하는 또는 지상 새) 으로 변경하여 더 깨끗한
   H7 exotic 역할.
6. **메커니즘 작업 (encoder probing, LM probing) 은 M8d에 불필요**
   — 행동 신호가 충분히 깨끗. 메커니즘은 M8c (논문 figure 1) 또는
   §4.5 (encoder swap) 용으로 보관.
