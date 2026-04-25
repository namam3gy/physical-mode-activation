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

| Config | Run dir | n |
|---|---|---|
| `m8d_qwen.py` | `outputs/m8d_qwen_<ts>_<hash>/` | 1440 (480 × 3 roles) |
| `m8d_qwen_label_free.py` | `outputs/m8d_qwen_label_free_<ts>_<hash>/` | 480 |
| `m8d_llava.py` | `outputs/m8d_llava_<ts>_<hash>/` | 1440 |
| `m8d_llava_label_free.py` | `outputs/m8d_llava_label_free_<ts>_<hash>/` | 480 |

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

_실행 완료 후 채움:_

| 기준              | Qwen  | LLaVA |
|-------------------|-------|-------|
| H1 ramp           | _TBD_ | _TBD_ |
| H7 (phys>abs)     | _TBD_ | _TBD_ |
| 시각 포화 delta    | _TBD_ | _TBD_ |

카테고리별 세부: `docs/insights/m8d_non_ball_categories_ko.md` §결과 참조.

## 헤드라인 숫자

_실행 완료 후 채움._

`PMR_regime(_nolabel)` 베이스라인 by (model × category):

| 카테고리 | Qwen  | LLaVA |
|----------|-------|-------|
| car      | _TBD_ | _TBD_ |
| person   | _TBD_ | _TBD_ |
| bird     | _TBD_ | _TBD_ |

`PMR_regime` paired-delta `physical − _nolabel` on `horizontal` subset:

| 카테고리 | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | _TBD_  | _TBD_  |
| person   | _TBD_  | _TBD_  |
| bird     | _TBD_  | _TBD_  |

`PMR_regime` ramp `(textured − line)` per category × model:

| 카테고리 | Qwen   | LLaVA  |
|----------|--------|--------|
| car      | _TBD_  | _TBD_  |
| person   | _TBD_  | _TBD_  |
| bird     | _TBD_  | _TBD_  |

## 분류기 검증 (50-stim 손 라벨링)

_`m8d_hand_annotate.py` 가 예측에 적용된 후 채움._

- 손 라벨링된 행 수: _TBD_
- 결합 오차율: _TBD_ (paper-ready 신호 임계값: < 0.150)
- regime별 confusion: `docs/experiments/m8d_hand_annotate.csv` 참조.

## 파일

- `outputs/m8d_summary/` — 모델별 rollup + 결합된 주석 parquet
  (`m8d_qwen_annotated.parquet`, `m8d_llava_annotated.parquet`).
- `docs/figures/m8d_*.png` — 5개 그림.
- `notebooks/m8d_non_ball_categories.ipynb` — 셀별 재현.
