# M8c — Real photographs (run log)

External-validity round 3, executed 2026-04-25.

## Setup

- 60 photos (12 × {ball, car, person, bird, abstract}).
- Stimulus directory: `inputs/m8c_photos_20260425-162031/`.
- Sources:
  - COCO 2017 validation (`phiyodr/coco2017`) for ball / car / person / bird.
  - WikiArt (`huggan/wikiart`) for abstract.
- Sampling: T=0.7, top_p=0.95, max_new_tokens=96.
- Single GPU 0, sequential. Total wall clock: **5 minutes**.

## Configs

| Config | Run dir | n | wall |
|---|---|---|---|
| `m8c_qwen.py` | `outputs/m8c_qwen_20260425-162502_13738370/` | 180 (60 × 3 roles) | 1.4 min |
| `m8c_qwen_label_free.py` | `outputs/m8c_qwen_label_free_20260425-162628_b8060cda/` | 60 | 1.2 min |
| `m8c_llava.py` | `outputs/m8c_llava_20260425-162739_48498b56/` | 180 | 1.5 min |
| `m8c_llava_label_free.py` | `outputs/m8c_llava_label_free_20260425-162909_6ca82730/` | 60 | 1.0 min |

Total: 480 inferences in 5.1 min.

## Code changes

- `src/physical_mode/inference/prompts.py::LABELS_BY_SHAPE`: added
  `"ball": ("ball", "circle", "planet")` (reuses circle's triplet for
  photo-ball stimuli) and `"abstract": ("object", "drawing", "diagram")`.
- `src/physical_mode/config.py`: extended `Shape` literal with `ball`,
  `abstract`. Added `drawing`, `diagram` to `Label` literal.
- `scripts/m8c_curate_photos.py` — photo curation driver.
- `scripts/m8c_run_all.sh`, `scripts/m8c_analyze.py`, `scripts/m8c_figures.py`.
- `configs/m8c_*.py` — 4 configs.

123 unit tests pass.

## Curation methodology

- COCO categories (ball/car/person/bird): captions filtered by category
  keywords (e.g., `["basketball", "soccer ball", "baseball", ...]` for
  ball). Up to 12 random photos per category sampled with seed=42 + cat
  initial.
- Abstract: WikiArt shard 0 (1132 rows), filtered by `style ∈ {0, 1, 4,
  5, 6, 25}` (heuristic abstract-related styles).
- All photos are square-padded with white and resized to 512×512.
- License recorded per photo in `photo_metadata.csv`.

Known curation caveats:
- COCO photos are scene-rich (a "ball" image often shows a baseball
  player, not an isolated ball). Acceptable for the M8c hypothesis test
  (does photo-realism shift PMR?), but limits clean per-object
  comparison with synthetic stim.
- WikiArt shard 0 has more figurative than abstract paintings; the
  abstract category is best read as "diverse painting styles" rather
  than "purely abstract".

## Headline numbers

`PMR(_nolabel)` baseline by (model × category):

| category | Qwen photo | LLaVA photo |
|----------|-----------:|------------:|
| ball     | 0.667      | 0.500       |
| car      | 0.500      | 0.000       |
| person   | 0.667      | 0.417       |
| bird     | 0.417      | 0.500       |
| abstract | 0.500      | 0.000       |

`PMR(_nolabel)` synthetic-textured baseline (from M8a circle + M8d):

| category | Qwen synth-textured | LLaVA synth-textured |
|----------|--------------------:|---------------------:|
| ball (circle) | 0.900 | 0.450 |
| car           | 0.975 | 0.375 |
| person        | 0.850 | 0.025 |
| bird          | 0.875 | 0.600 |

Synthetic−photo `Δ` (paired by category):

| category | Qwen Δ | LLaVA Δ |
|----------|-------:|--------:|
| ball     | −0.233 | +0.050 |
| car      | −0.475 | −0.375 |
| person   | −0.183 | +0.392 |
| bird     | −0.458 | −0.100 |

`PMR` by (category × label_role) — Qwen:

| category | physical | abstract | exotic |
|----------|---------:|---------:|-------:|
| ball     | 0.667 | 0.583 | 0.583 |
| car      | 0.667 | 0.833 | 0.500 |
| person   | 0.750 | 0.250 | 0.500 |
| bird     | 0.750 | 0.750 | 0.583 |
| abstract | 0.500 | 0.500 | 0.417 |

`PMR` by (category × label_role) — LLaVA:

| category | physical | abstract | exotic |
|----------|---------:|---------:|-------:|
| ball     | 0.667 | 0.500 | 0.667 |
| car      | 0.083 | 0.083 | 0.333 |
| person   | 0.333 | 0.583 | 0.417 |
| bird     | 0.833 | 0.167 | 0.500 |
| abstract | 0.000 | 0.083 | 0.167 |

H7 paired-difference `physical − abstract` on photos:

| category | Qwen | LLaVA |
|----------|-----:|------:|
| ball     | +0.083 | +0.167 |
| car      | −0.167 | 0.000  |
| person   | **+0.500** | **−0.250** |
| bird     | 0.000 | **+0.667** |
| abstract | 0.000 | −0.083 |

## Files

- `outputs/m8c_summary/` — per-model rollups + concatenated annotated
  parquet (`m8c_qwen_annotated.parquet`, `m8c_llava_annotated.parquet`).
- `outputs/m8c_summary/m8c_synthetic_baseline.csv` — synthetic baselines.
- `outputs/m8c_summary/m8c_synthetic_vs_photo.csv` — paired delta per
  (model × category).
- `docs/figures/m8c_{photo_grid,pmr_by_category,paired_synthetic_vs_photo}.png`.
- `notebooks/m8c_real_photos.ipynb` — cell-by-cell reproduction.
