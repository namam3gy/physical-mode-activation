# M8c — 실사진 (실행 로그)

외적 타당성 라운드 3, 2026-04-25 실행.

## 셋업

- 60 사진 (12 × {ball, car, person, bird, abstract}).
- 자극 디렉터리: `inputs/m8c_photos_20260425-162031/`.
- 출처:
  - COCO 2017 validation (`phiyodr/coco2017`) — ball / car / person / bird.
  - WikiArt (`huggan/wikiart`) — abstract.
- 샘플링: T=0.7, top_p=0.95, max_new_tokens=96.
- 단일 GPU 0, 순차. 총 wall clock: **5분**.

## Configs

| Config | Run dir | n | wall |
|---|---|---|---|
| `m8c_qwen.py` | `outputs/m8c_qwen_20260425-162502_13738370/` | 180 (60 × 3 roles) | 1.4 분 |
| `m8c_qwen_label_free.py` | `outputs/m8c_qwen_label_free_20260425-162628_b8060cda/` | 60 | 1.2 분 |
| `m8c_llava.py` | `outputs/m8c_llava_20260425-162739_48498b56/` | 180 | 1.5 분 |
| `m8c_llava_label_free.py` | `outputs/m8c_llava_label_free_20260425-162909_6ca82730/` | 60 | 1.0 분 |

총: 480 추론을 5.1 분.

## 코드 변경

- `src/physical_mode/inference/prompts.py::LABELS_BY_SHAPE`:
  `"ball": ("ball", "circle", "planet")` 추가 (사진-ball 자극용으로
  circle 트리플 재사용) 및 `"abstract": ("object", "drawing", "diagram")`
  추가.
- `src/physical_mode/config.py`: `Shape` literal 에 `ball`, `abstract`
  확장. `Label` literal 에 `drawing`, `diagram` 추가.
- `scripts/m8c_curate_photos.py` — 사진 큐레이션 driver.
- `scripts/m8c_run_all.sh`, `scripts/m8c_analyze.py`, `scripts/m8c_figures.py`.
- `configs/m8c_*.py` — 4 개 configs.

123 단위 테스트 통과.

## 큐레이션 방법론

- COCO 카테고리 (ball/car/person/bird): 캡션을 카테고리 키워드로 필터
  (ball 은 `["basketball", "soccer ball", "baseball", ...]`). 카테고리당
  최대 12 사진을 seed=42 + cat 이니셜로 랜덤 샘플링.
- Abstract: WikiArt shard 0 (1132 행), `style ∈ {0, 1, 4, 5, 6, 25}` 로
  필터 (휴리스틱 abstract-related 스타일).
- 모든 사진은 흰색 사각 패딩 후 512×512 으로 리사이즈.
- 라이선스는 `photo_metadata.csv` 에 사진별 기록.

알려진 큐레이션 caveat:
- COCO 사진은 scene-rich (ball 이미지가 격리된 공이 아니라 야구 선수를
  보여주는 경우 많음). M8c 가설 (사진 사실성이 PMR 을 어떻게 바꾸는가)
  검증에는 허용 가능하지만, 합성 stim 과의 깨끗한 객체-단위 비교를
  제한.
- WikiArt shard 0 은 abstract 보다 figurative 회화가 많음; abstract
  카테고리는 "purely abstract" 보다 "다양한 회화 스타일" 로 읽는 것이
  적절.

## 헤드라인 숫자

`PMR(_nolabel)` 베이스라인 by (model × category):

| 카테고리 | Qwen photo | LLaVA photo |
|----------|-----------:|------------:|
| ball     | 0.667      | 0.500       |
| car      | 0.500      | 0.000       |
| person   | 0.667      | 0.417       |
| bird     | 0.417      | 0.500       |
| abstract | 0.500      | 0.000       |

`PMR(_nolabel)` 합성-텍스처 베이스라인 (M8a circle + M8d 에서):

| 카테고리 | Qwen synth-textured | LLaVA synth-textured |
|----------|--------------------:|---------------------:|
| ball (circle) | 0.900 | 0.450 |
| car           | 0.975 | 0.375 |
| person        | 0.850 | 0.025 |
| bird          | 0.875 | 0.600 |

합성 − 사진 `Δ` (카테고리별 paired):

| 카테고리 | Qwen Δ | LLaVA Δ |
|----------|-------:|--------:|
| ball     | −0.233 | +0.050 |
| car      | −0.475 | −0.375 |
| person   | −0.183 | +0.392 |
| bird     | −0.458 | −0.100 |

`PMR` by (category × label_role) — Qwen:

| 카테고리 | physical | abstract | exotic |
|----------|---------:|---------:|-------:|
| ball     | 0.667 | 0.583 | 0.583 |
| car      | 0.667 | 0.833 | 0.500 |
| person   | 0.750 | 0.250 | 0.500 |
| bird     | 0.750 | 0.750 | 0.583 |
| abstract | 0.500 | 0.500 | 0.417 |

`PMR` by (category × label_role) — LLaVA:

| 카테고리 | physical | abstract | exotic |
|----------|---------:|---------:|-------:|
| ball     | 0.667 | 0.500 | 0.667 |
| car      | 0.083 | 0.083 | 0.333 |
| person   | 0.333 | 0.583 | 0.417 |
| bird     | 0.833 | 0.167 | 0.500 |
| abstract | 0.000 | 0.083 | 0.167 |

H7 paired-difference `physical − abstract` on 사진:

| 카테고리 | Qwen | LLaVA |
|----------|-----:|------:|
| ball     | +0.083 | +0.167 |
| car      | −0.167 | 0.000  |
| person   | **+0.500** | **−0.250** |
| bird     | 0.000 | **+0.667** |
| abstract | 0.000 | −0.083 |

## 파일

- `outputs/m8c_summary/` — 모델별 rollup + 결합 주석 parquet.
- `outputs/m8c_summary/m8c_synthetic_baseline.csv` — 합성 베이스라인.
- `outputs/m8c_summary/m8c_synthetic_vs_photo.csv` — paired delta.
- `docs/figures/m8c_{photo_grid,pmr_by_category,paired_synthetic_vs_photo}.png`.
- `notebooks/m8c_real_photos.ipynb` — 셀별 재현.
