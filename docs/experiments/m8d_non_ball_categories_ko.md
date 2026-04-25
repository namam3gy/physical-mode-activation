# M8d — 비공 물리 객체 카테고리 (실행 로그)

외적 타당성 라운드 2, 2026-04-25 실행.

## 셋업

- 3 카테고리 × 4 obj_levels × 2 bg_levels × 2 cue_levels × 2 events × 5 seeds
  = **480 자극** (한 개의 공통 자극 디렉터리; 4개 추론 config 모두에서
  재사용).
- 자극 디렉터리: `inputs/m8d_qwen_<ts>_<hash>/`.
- 샘플링: T=0.7, top_p=0.95, max_new_tokens=96 (M6 r1 / r2 / M8a와 일치).
- 단일 GPU 0, 순차. 총 wall clock: **~30-35 분**.

## Configs

| Config | Run dir | n | wall |
|---|---|---|---|
| `m8d_qwen.py` | `outputs/m8d_qwen_20260425-151811_6c200dc8/` | 1440 (480 × 3 roles) | 12.6 분 |
| `m8d_qwen_label_free.py` | `outputs/m8d_qwen_label_free_20260425-153049_e1f19e0d/` | 480 | 6.2 분 |
| `m8d_llava.py` | `outputs/m8d_llava_20260425-153701_ea751428/` | 1440 | 8.8 분 |
| `m8d_llava_label_free.py` | `outputs/m8d_llava_label_free_20260425-154549_16bc0be7/` | 480 | 4.3 분 |

총: H200 GPU 0에서 **31.9 분** wall clock (15:18:07 → 15:50:03).

## 코드 변경

- `src/physical_mode/stimuli/primitives.py`: 3 카테고리 (car / person /
  bird) × 4 추상화 모드 = 12개 새 draw 함수 (`_draw_*_car`,
  `_draw_*_person`, `_draw_*_bird` line / filled / shaded / textured).
  `Shape` literal 확장.
- `src/physical_mode/stimuli/scenes.py`: ground-bound shape (car,
  person) 은 `horizontal` 이벤트에서 ground 위에 위치하여 자연 운동
  의미가 cast shadow와 기하학적으로 일치; 새는 공중 배치 유지.
- `src/physical_mode/config.py`: `Shape` literal 확장; `Label` literal
  확장 — `(car, silhouette, figurine)`, `(person, "stick figure",
  statue)`, `(bird, duck)`. silhouette은 car와 bird 사이 재사용.
- `src/physical_mode/inference/prompts.py`: `LABELS_BY_SHAPE` 에 car /
  person / bird 트리플 추가.
- `src/physical_mode/metrics/lexicons.py`: car / person / bird용
  `CATEGORY_REGIME_KEYWORDS` 추가 (kinetic + static 단어 stem). 기존
  `PHYSICS_VERB_STEMS`, `HOLD_STILL_STEMS`, `DOWN_DIRECTION_PHRASES`,
  `ABSTRACT_MARKERS` 는 변경 없음 — M8a 채점이 비트 단위 동일하게 유지.
- `src/physical_mode/metrics/pmr.py`: `classify_regime(category, text)
  → {kinetic, static, abstract, ambiguous}` 추가. 원본 `score_pmr` /
  `score_gar` 변경 없음.
- `scripts/m8d_spot_check.py`, `scripts/m8d_run_all.sh`,
  `scripts/m8d_analyze.py`, `scripts/m8d_figures.py`,
  `scripts/m8d_hand_annotate.py` — 드라이버 + 분석 유틸리티.
- `tests/test_m8d_labels.py`, `tests/test_m8d_primitives.py`,
  `tests/test_m8d_regime.py`. 123 단위 테스트 통과.

## 시각 검증

- `docs/figures/m8d_shape_grid.png` — 480-stim 세트 생성 전 3×4
  spot-check 렌더링. 각 (카테고리 × 추상화-레벨) 셀 시각적으로 구분 가능.
- `docs/figures/m8d_full_scene_samples.png` — 3 카테고리 × 4 추상화 ×
  2 이벤트 풀 신 샘플.

## 사전 등록된 기준 채점 (최종)

| 기준              | Qwen  | LLaVA |
|-------------------|-------|-------|
| H1 ramp           | 0/3 ✗ | 0/3 ✗ |
| H7 (phys>abs)     | 0/3 ✗ | **3/3 ✓** |
| 시각 포화 delta   | 1/3 (bird) | 2/3 (car, bird; person 음수로 flip) |

카테고리별 세부: `docs/insights/m8d_non_ball_categories_ko.md` §결과 참조.

## 헤드라인 숫자

`PMR_regime(_nolabel)` 베이스라인 by (model × category) on **horizontal** subset:

| 카테고리 | Qwen  | LLaVA |
|----------|-------|-------|
| car      | 1.000 | 0.550 |
| person   | 0.975 | 0.838 |
| bird     | 0.862 | 0.688 |

`PMR_regime` paired-delta `physical − _nolabel` on `horizontal` subset:

| 카테고리 | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.000 | **+0.275** |
| person   | +0.025 | -0.100 |
| bird     | +0.125 | **+0.262** |

`PMR_regime` ramp `(textured − line)` per category × model (event union):

| 카테고리 | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.008 | -0.033 |
| person   | -0.009 | -0.033 |
| bird     | +0.008 | -0.017 |

H7 paired-difference `PMR_regime(physical) − PMR_regime(abstract)` on `horizontal` subset:

| 카테고리 | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | +0.012 | **+0.525** |
| person   | +0.012 | +0.138 |
| bird     | +0.038 | **+0.550** |

H7 paired-difference (kinetic-fraction 수준; Qwen 천장 우회):

| 카테고리 | Qwen Δ kin_frac (physical − abstract) | Qwen Δ kin_frac (physical − exotic) |
|----------|--------|--------|
| car      | +0.063 | **+0.138** |
| person   | -0.013 | **+0.162** |
| bird     | +0.106 | +0.019  |

LLaVA `physical − exotic` kin_frac 차이 (완성도 위한):

| 카테고리 | LLaVA Δ kin_frac (physical − exotic) |
|----------|--------|
| car      | +0.262 |
| person   | +0.138 |
| bird     | +0.062 |

## 분류기 검증 (54-stim 손 라벨링)

`scripts/m8d_hand_annotate.py --mode sample --n-per-cell 3 --seed 42`
는 **54개 stratified 행** (model × category × role × 3 = 54) 을 샘플링.
손 라벨링은 keyword 분류기보다 더 풍부한 영어 어휘 — kinetic 동사,
static 상태, abstract-reject 구문 — 를 적용하여 진정한 인간 읽기를 모방.

- 손 라벨링된 행 수: 54
- 결합 오차율: **0.056** (3개 미스매치; 임계값 < 0.150 — **PASS**)
- regime별 precision / recall:
  - kinetic:  precision 0.949 / recall 0.974
  - static:   precision 1.000 / recall 0.778
  - abstract: precision NaN / recall NaN  (샘플에 abstract 응답 없음)
  - ambiguous: precision 0.875 / recall 1.000

미스매치 (3/54):
- 2 × Qwen person/abstract: "stick figure will *remain stationary*, no
  indication of *movement*" — keyword 분류기는 `mov` (kinetic) 와
  `remain`/`stationary` (static) 둘 다 보고 kinetic-first 로 해결;
  인간은 static 으로 해결. Stem-matching 한계.
- 1 × LLaVA person/exotic: "the statue will be *pulled* away from the
  line" — `pull`은 PHYSICS_VERB_STEMS 에는 있지만 카테고리별 kinetic
  세트에는 없음, 따라서 classify_regime이 ambiguous 반환. 카테고리별
  kinetic 어휘 확장 가능하지만 임계값보다 충분히 낮음.

CSV: `docs/experiments/m8d_hand_annotate.csv` (54 행, predicted +
hand 컬럼).

## 파일

- `outputs/m8d_summary/` — 모델별 rollup + 결합된 주석 parquet
  (`m8d_qwen_annotated.parquet`, `m8d_llava_annotated.parquet`).
- `docs/figures/m8d_*.png` — 5개 그림.
- `notebooks/m8d_non_ball_categories.ipynb` — 셀별 재현.
