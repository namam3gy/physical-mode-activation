# M8a — 원형이 아닌 합성 도형 (런 로그)

외적 타당성 라운드 1, 2026-04-25 실행.

## 셋업

- 5 도형 × 4 obj_levels × 2 bg_levels × 2 cue_levels × 1 event × 5 시드
  = **400 자극** (공통 stim 디렉터리; 4개 추론 config 모두 재사용).
- 자극 디렉터리: `inputs/m8a_qwen_20260425-091713_8af4836f/`.
- 샘플링: T=0.7, top_p=0.95, max_new_tokens=96 (M6 r1/r2와 동일).
- 단일 GPU 1, 순차. 총 wall clock: **~43분**.

## Configs

| Config | Run dir | n |
|---|---|---|
| `m8a_qwen.py` | `outputs/m8a_qwen_20260425-092423_bf03832e/` | 1200 (400 × 3 roles) |
| `m8a_qwen_label_free.py` | `outputs/m8a_qwen_label_free_20260425-094239_26c66949/` | 400 |
| `m8a_llava.py` | `outputs/m8a_llava_20260425-095133_a2b5f318/` | 1200 |
| `m8a_llava_label_free.py` | `outputs/m8a_llava_label_free_20260425-100253_99a20dd8/` | 400 |

## 코드 변경

- `src/physical_mode/stimuli/primitives.py`: `Shape` Literal 추가 +
  4개 새 도형 클래스 × 4개 추상화 모드
  (`_draw_square / _draw_triangle / _draw_hexagon / _draw_polygon`,
  각각 line / filled / shaded / textured). 비-구체 도형에 대해 방향성
  음영 (Lambert 풍, 좌상단 광원)이 radial shading을 대체. Polygon vertex는
  시드 기반 (5–7 vertices, jittered radii / angles).
- `src/physical_mode/stimuli/scenes.py`: `draw_object`에 `shape` 전달.
- `src/physical_mode/config.py`: `Shape` Literal,
  `StimulusRow.shape` 필드 (default `"circle"` — backward compat),
  `FactorialSpec.shapes` 축 추가. Sample-id는 `len(shapes) > 1`일 때만
  shape를 포함하므로 기존 단일-도형 config는 변경 없이 재현된다.
- `src/physical_mode/inference/prompts.py`: `LABELS_BY_SHAPE` dict +
  `labels_for_shape()` helper 추가. 도형별 트리플렛
  `(physical, abstract, exotic)`. Advisor 검토 후 polygon의 exotic이
  `shape` (기하 클래스보다 더 추상적이므로 역할 순서를 역전시킬 수 있음)
  → `boulder`로 변경됨.
- `src/physical_mode/inference/run.py`: `len(cfg.factorial.shapes) > 1`이면
  `cfg.labels`는 리터럴 라벨이 아닌 *역할 이름*
  (`physical / abstract / exotic`) 리스트로 해석된다. Per-row 리터럴
  라벨은 `LABELS_BY_SHAPE`를 통해 디스패치된다.
- `src/physical_mode/metrics/pmr.py`: per-axis summary loop에 `shape` 추가.
- `scripts/m8a_spot_check.py`, `scripts/m8a_run_all.sh`,
  `scripts/m8a_analyze.py`, `scripts/m8a_figures.py` — driver + 분석
  유틸.

50개 단위 테스트는 여전히 통과 (`uv run python -m pytest`).

## 시각 검증

- `docs/figures/m8a_shape_grid.png` — 400-stim 세트 생성 전 5×4
  spot-check.
- `docs/figures/m8a_full_scene_samples.png` — 5개 대표 full-scene
  셀 (textured + ground + arrow + cast shadow).

## 사전 등록 기준 채점 (최종)

| 기준                  | Qwen | LLaVA |
|-----------------------|------|-------|
| H1 ramp               | 3/5 ✗ | 4/5 ✓ |
| H7 (phys>abs)         | 1/5 ✗ | 4/5 ✓ |
| H7-GAR                | 1/5 ✗ | 5/5 ✓ |
| Visual-saturation Δ   | 3/5 ✓ borderline | 5/5 ✓ |

도형별 상세: `docs/insights/m8a_non_circle_shapes_ko.md` §결과 참조.

## 헤드라인 숫자

PMR(_nolabel) baseline by (model × shape):

| shape    | Qwen  | LLaVA |
|----------|-------|-------|
| circle   | 0.825 | 0.288 |
| square   | 0.925 | 0.088 |
| triangle | 0.788 | 0.075 |
| hexagon  | 0.875 | 0.150 |
| polygon  | 0.775 | 0.275 |

Qwen은 도형 간 0.78–0.93 — 비전 인코더가 이미 physics-mode에 commit한다.
LLaVA는 0.075–0.288 — 라벨이 행동의 대부분을 한다.

Paired-delta `PMR(physical) − PMR(_nolabel)`:

| shape    | Qwen   | LLaVA  |
|----------|--------|--------|
| circle   | -0.013 | +0.575 |
| square   | -0.200 | +0.625 |
| triangle | -0.025 | +0.125 |
| hexagon  | -0.125 | +0.550 |
| polygon  | +0.025 | +0.487 |

LLaVA의 `physical` 라벨은 모든 도형에서 **+0.125 ~ +0.625** PMR 부스트를
준다. Qwen의 `physical` 라벨은 **near-zero 또는 음수** — 인코더가 이미
물리적 해석을 인코딩하고 있다.

## 도형별 주목할 발견

- **Qwen `square`**: paired-delta -0.200 / -0.275 / -0.212 — M4b의
  circle "라벨이 물리를 억제"한 효과의 깔끔한 도형 간 재현. 라벨이
  시각이 이미 제공한 것에 어떤 것도 추가하지 않으며, *그리고* physics-mode
  언어를 약간 억제한다 (모델이 "the brick will fall"보다 "the brick is on
  the ground, gray, weathered"를 더 자주 쓴다).
- **LLaVA `triangle`**: paired-delta가 +0.125 / +0.100 / +0.100에
  불과하며, PMR(physical=wedge) = 0.200 vs ball/brick/nut/rock의 ~0.7.
  거의 확실히 라벨 품질 문제: "wedge"는 약한 물리적 객체 큐다. 향후
  런은 triangle에 대한 대안 physical 라벨 (`pyramid`, `sandbag`,
  `ramp`)을 시험해야 한다.
- **LLaVA `polygon` abstract = -0.050**: 음수가 된 유일한 LLaVA
  paired-delta. "Polygon"은 물리적 묘사가 아닌 수학 용어로 읽힌다.
  일반 어휘 기하 명사가 없는 불규칙 도형에 대해 role taxonomy가 샌다.
- **Qwen `circle` planet (exotic) → GAR = 0.175**: 반면 GAR(ball) =
  0.675, GAR(circle) = 0.700. "Planet" 사전 분포가 떨어지는 언어를
  활성적으로 *억제*한다 ("orbits the sun, rotates on axis"), 그리고
  모델의 `_nolabel` 베이스라인은 planet 해석보다 ball/circle 해석에
  가깝다. 도형 간에서는 nut / coin / boulder가 planet만큼 강한
  non-falling 사전 분포를 갖지 않으므로 이 효과가 깔끔하게
  나타나지 않는다.
- **Qwen `square` GAR**: physical=0.475, abstract=0.500, exotic=0.525
  — Qwen의 near-saturation에 의해 추동된 평탄한 행. 인코더가 이미
  LM에 "이것은 지면이 아래 있을 때 떨어지는 무언가다"를 말하고 있다;
  역할명은 그것을 움직이지 않는다.

## 원시 아티팩트

- `outputs/m8a_qwen_*/predictions{.jsonl,.parquet,.csv}`,
  `m8a_pmr_by_shape_*.csv`, `m8a_ramp_per_shape.csv`.
- `outputs/m8a_paired_deltas.csv` — Qwen + LLaVA × 5 도형 × 3 역할.
- `outputs/m8a_run_all.log` — 전체 4-run 로그 (~487 KB).
- `inputs/m8a_qwen_20260425-091713_8af4836f/` — 400개 PNG 자극 +
  manifest.parquet.
