# Design Spec — M8d (non-ball categories) + M8c (real photographs)

> **Status**: design approved 2026-04-25, awaiting implementation plan.
> **Owner**: namam3gy.
> **Predecessors**: M8a (non-circle synthetic shapes, complete 2026-04-25).
> **Related research docs**: `references/roadmap.md` §3 (M8c, M8d work plans), `references/project.md` §2 (sub-task structure), `docs/insights/m8a_non_circle_shapes.md` (M8a baseline patterns).

## 1. Scope and sequencing

Three phases, each producing an independent milestone deliverable:

```
[Phase 1] M8d  →  [Phase 2] M8c  →  [Phase 3 — optional] M8e super-milestone
   (~6-8h)         (~4-6h)            (~2-4h)
```

Phase 3 is decided after Phases 1 and 2 ship; it consolidates M8d's category coverage with M8c's photo coverage into a `(category × source_type)` paired analysis.

### 1.1 Hypothesis mapping

| Phase | Hypothesis | Test |
|---|---|---|
| M8d | H7 cross-category generalization | car/person/bird × `(physical, abstract, exotic)` triplet — does label alone shift physics regime in non-ball categories? |
| M8d | H1 ramp + H-encoder-saturation | per-category abstraction ramp (line→textured) PMR — does the M8a saturation pattern (Qwen ceiling-flat / LLaVA monotone) hold? |
| M8c | H-encoder-saturation external validity | does photo-realism saturate the encoder further? does paired (synthetic-textured-ball vs photo-ball) PMR delta close LLaVA's gap? |
| M8e | M8d × M8c paired | per-category synthetic-textured vs photo PMR delta; does the encoder process photographic detail differently from synthetic-textured detail? |

### 1.2 Success criteria

**M8d phase**:
- All three categories (car, person, bird) produce recognizable primitives at every abstraction level (line/filled/shaded/textured) — verified by spot-check on 10 stimuli per category × Qwen + LLaVA generating responses that name the category in the label-free condition.
- Pre-registered scoring: H1 ramp (PMR(textured) − PMR(line) ≥ 0.05) on each category; H7 paired-delta (PMR(physical) − PMR(abstract) ≥ 0.05) on each category. Cross-category PASS rate per model recorded the same way as M8a (`Qwen X/3, LLaVA Y/3`).
- Regime distribution per `(category, label)` cell visible in figures.
- 50-stimulus hand-annotation validates the keyword classifier with false-positive + false-negative rate < 15 %.

**M8c phase**:
- 60 photos curated (12 per category × 5 categories), license metadata recorded.
- Photo PMR(_nolabel) ≥ synthetic-textured PMR(_nolabel) for each model on every physical category — confirms direction of the H-encoder-saturation prediction.
- Paired (synthetic-textured vs photo) figures emitted per (model, category).

**M8e phase (if pursued)**:
- Cross-source paired-delta heatmap (model × category × source_type).
- Insight on whether photo-realism is a *new* saturation axis or merely scales the existing one.

### 1.3 Estimated cost (single GPU 1, ~1.0 it/s on H200)

| Phase | Stimuli | Inferences | Wall-clock |
|---|---|---|---|
| M8d | 480 | 480 × 4 (3 labels + 1 nolabel) × 2 models = 3840 | ~52-65 min |
| M8c | 60 | 60 × 4 × 2 models = 480 | ~8-10 min |
| M8e | 0 (analysis only) | 0 | ~30-60 min |

Code + curation effort estimates: M8d ~6-8 h, M8c ~4-6 h (mostly photo curation), M8e ~2-4 h.

## 2. M8d — non-ball physical-object categories

### 2.1 Categories and labels

Three categories selected for primitive-drawing tractability and clean physical-vs-abstract separation: `car`, `person`, `bird`. (Plant deferred — primitives harder to draw recognizably at low resolution; chosen for a possible M8d round 2.)

Label triplets stay parallel to M8a's `(physical, abstract, exotic)` structure for analysis-code reuse:

```python
# physical_mode/inference/prompts.py — additions to LABELS_BY_SHAPE
"car":    ("car",    "silhouette",  "figurine"),
"person": ("person", "stick figure", "statue"),
"bird":   ("bird",   "silhouette",  "duck"),
```

Role rationale:
- **physical** — canonical category label invoking default kinetic regime (drives / walks / flies).
- **abstract** — depiction-style label that suppresses physics reading (silhouette = 2D dark form; stick figure = symbolic person drawing). `silhouette` chosen over `rectangle` (for car) and `outline` (for bird) because silhouette is a depiction style that cleanly cuts the physics reading rather than imposing a forced geometric class. `stick figure` retained for person because it is the iconic abstract for that category specifically.
- **exotic** — physical label that shifts regime away from the category's default. `figurine` shifts car from "drives" to "static toy on display"; `statue` shifts person from "walks" to "static stone"; `duck` shifts bird from "flies" to "swims/waddles" (note: duck is a mixed-regime exotic; the cleanest H7 signal would come from a flightless bird like ostrich/penguin, but duck is preferred because 7B VLM recognition of canonical bird shape is more robust than ostrich/penguin recognition at low resolution).

Known weak points (logged for M8d round 2):
- `bird/duck`: regime shift is partial (duck flies + swims + walks). Effect size on H7 paired-delta may be smaller than circle/planet's clean orbital shift.
- `person/silhouette` was considered as a unification of abstract role across categories but rejected to preserve `stick figure`'s strong iconic abstraction for person.
- Person's category label (`person`) may produce more abstract responses than car/bird because "person" is itself somewhat generic (cf. M8a `polygon/polygon` weak label).

### 2.2 Stimulus factorial

```python
# configs/m8d_qwen.py (and three siblings)
factorial = FactorialSpec(
    shapes=("car", "person", "bird"),
    object_levels=("line", "filled", "shaded", "textured"),
    bg_levels=("blank", "ground"),
    cue_levels=("none", "both"),
    event_templates=("fall", "horizontal"),
    seeds_per_cell=5,
)
# Total: 3 × 4 × 2 × 2 × 2 × 5 = 480 stimuli
```

Two-event grid is intentional. Per-category natural events:

| Category | natural event | unnatural event |
|---|---|---|
| car | horizontal (drives) | fall (cliff/cliff-fall) |
| person | horizontal (walks) | fall (cliff/cliff-fall) |
| bird | horizontal (flies) | fall (gravity-fall — birds do not naturally fall) |

**Analysis convention**: H1 ramp (`PMR(textured) − PMR(line)`) is computed on the union (fall + horizontal) — this is the abstraction-axis signal, not an event-axis signal. H7 paired-delta (`PMR(physical) − PMR(abstract)`) is computed on the **horizontal subset** for each category — the natural-event subset where regime selection is cleanest. The fall subset is reported separately as a stress test.

Recorded explicitly in the M8d insight document so future readers know the split.

### 2.3 Primitive drawing

`primitives.py` extended with category-specific draw functions, one per (category, abstraction-level) pair = 12 new functions:

```
_draw_line_car        _draw_filled_car        _draw_shaded_car        _draw_textured_car
_draw_line_person     _draw_filled_person     _draw_shaded_person     _draw_textured_person
_draw_line_bird       _draw_filled_bird       _draw_shaded_bird       _draw_textured_bird
```

Drawing constraints (informal — verified at spot-check time):
- **Line level** must be category-recognizable. Examples:
  - `_draw_line_car`: rectangular body outline + two circular wheel outlines + a small windshield rectangle.
  - `_draw_line_person`: classic stick figure (head circle + torso line + 2 arm lines + 2 leg lines).
  - `_draw_line_bird`: oval body outline + small head circle + beak triangle + simple wing curve.
- **Filled level**: same shapes filled with solid black (silhouette).
- **Shaded level**: same shapes with PIL gradient fill suggesting 3D depth (top-lit, bottom-shadowed) — analogous to `_draw_shaded_sphere` for circle.
- **Textured level**: shaded base + category-specific texture markings:
  - car: body color + window glass (lighter blue) + wheel hub detail.
  - person: skin-tone face circle + simple clothing block (different color from skin).
  - bird: feather hatching pattern + eye dot + beak in different color.

`stimuli/scenes.py::draw_object` dispatch updated to route the new shapes through these primitives. Existing M8a primitives unchanged.

### 2.4 Regime classifier (Hybrid D)

Implementation in `metrics/scoring.py` — extending the M8a keyword pattern:

```python
# Pseudocode
CATEGORY_REGIME_KEYWORDS: dict[str, dict[str, set[str]]] = {
    "car":    {"kinetic": {"drives", "rolls", "moves", "speeds"}, "static": {"parked", "stops"}},
    "person": {"kinetic": {"walks", "runs", "moves", "steps"},   "static": {"stands", "stays"}},
    "bird":   {"kinetic": {"flies", "swims", "waddles", "soars"}, "static": {"perches", "sits"}},
}

def classify_regime(category: str, response_text: str) -> str:
    """Return one of {kinetic, static, abstract, ambiguous}."""
    ...
```

`abstract` keywords (cross-category): `{"abstract", "geometric", "drawing", "shape", "nothing happens", "depicts", "depicted"}`.

**Validation procedure**: after the M8d run completes, randomly sample 50 responses (stratified across model × category × label = 24 cells, ~2 responses per cell, with ties broken by seed). Hand-annotate each response with one of `{kinetic, static, abstract, ambiguous}`. Compare against the keyword classifier output. Report false-positive + false-negative rate. Threshold for paper-ready signal: combined error rate < 15 %. If above 15 %, refine keywords and re-validate.

`is_physical_response()` and `is_gravity_aligned_response()` from `metrics/scoring.py` are left **unchanged** to preserve backward compatibility for circle/square/triangle/hexagon/polygon analysis code (M8a). The new `classify_regime(category, response_text)` is added in parallel and produces the regime label per response. M8d analysis code calls `classify_regime` directly; M8a analysis code still calls `is_physical_response` / `is_gravity_aligned_response` unchanged.

### 2.5 Configs and scripts

Four configs mirroring M8a:

```
configs/m8d_qwen.py              # labeled, 3 labels × open
configs/m8d_qwen_label_free.py   # 1 _nolabel × open_no_label
configs/m8d_llava.py             # labeled, 3 labels × open
configs/m8d_llava_label_free.py  # 1 _nolabel × open_no_label
```

Each config matches the M8a sampling settings (T=0.7, top_p=0.95) so paired-delta numbers align with M8a.

Scripts (paralleling M8a):

```
scripts/m8d_run_all.sh        # runs all four configs sequentially on GPU 1
scripts/m8d_analyze.py        # roll-up + paired-delta tables
scripts/m8d_figures.py        # 6 standard figures (see §2.7)
scripts/m8d_spot_check.py     # primitive-recognizability sampling for §2.3
```

### 2.6 Inference pipeline

No changes needed to `scripts/02_run_inference.py` or `inference/run.py` — the existing multi-shape dispatch (`labels_for_row(row_shape)`) already handles per-shape label triplets via `LABELS_BY_SHAPE`. New shapes just register in that dict.

### 2.7 Artifacts

Following the M8a/insight + experiment + figures + notebook convention:

- `docs/insights/m8d_non_ball_categories.md` (+ `_ko.md`) — deep dive.
- `docs/experiments/m8d_non_ball_categories.md` (+ `_ko.md`) — raw numbers.
- `docs/figures/m8d_shape_grid.png` — primitive samples per category × abstraction level.
- `docs/figures/m8d_full_scene_samples.png` — full stimuli per category × cue × bg.
- `docs/figures/m8d_pmr_ramp.png` — H1 abstraction ramp per category × model.
- `docs/figures/m8d_pmr_by_role.png` — H7 paired-delta per category × model.
- `docs/figures/m8d_paired_delta.png` — paired (physical, abstract, exotic) per category × model.
- `docs/figures/m8d_regime_distribution.png` — regime class probability per (category, label, model).
- `notebooks/m8d_non_ball_categories.ipynb` — cell-by-cell reproduction.

## 3. M8c — real photographs

### 3.1 Categories and counts

Five categories × 12 photos = 60 photos:

```
ball, car, person, bird       — 12 photos each (M8d-matched physical categories)
abstract                      — 12 photos (drawings, diagrams, line art, blueprints)
```

Equal counts simplify paired (synthetic vs photo) statistical comparison. Photo recognizability and lighting variation are favored over precise count.

### 3.2 Source strategy (Hybrid D)

| Category | Source | Method |
|---|---|---|
| ball, car, person, bird | OpenImages V7 (Apache 2.0) | Programmatic via HF `datasets` — class-filtered + bbox-cropped to a square aspect, downsampled to 512×512 to match synthetic. |
| abstract | PEXELS / Unsplash (Pexels License) | Manual curation — search terms `diagram`, `blueprint`, `line art`, `wireframe`, `technical drawing`. |

Implementation: `scripts/m8c_curate_photos.py` — single driver that:
1. Pulls OpenImages bounding boxes for `Ball`, `Car`, `Person`, `Bird` classes.
2. Downloads + crops to 12 representative samples per class (sampling strategy: random with rejection on low resolution / cluttered backgrounds).
3. Reads a hand-curated abstract photo list from `inputs/m8c_curated/abstract_urls.txt` — manual entries with PEXELS / Unsplash URLs, downloaded via `urllib`.
4. Emits `inputs/m8c_photos_<ts>/manifest.parquet` + `images/*.png` + `photo_metadata.csv` (source, URL, license).

### 3.3 Manifest schema

`inputs/m8c_photos_<ts>/manifest.parquet`:

```
sample_id          (str)   "ball_photo_000" ... "abstract_photo_011"
image_path         (str)   relative path to images/
shape              (str)   ball / car / person / bird / abstract
source_type        (str)   "photo"  (vs "synthetic" in synthetic manifests)
object_level       (str)   "photo"  (single value)
bg_level           (str)   "natural"  (single value — we don't control photo bg)
cue_level          (str)   "none"    (no synthetic cues on photos)
event_template     (str)   ball→fall, car/person/bird→horizontal, abstract→fall
seed               (int)   index 0..N within (shape, source_type)
```

Defaults are placeholders to maintain `run.py` compatibility — they are not informative axes for photos. `event_template` is metadata-only (the existing inference pipeline never inserts `event_template` into the prompt; it only uses it for synthetic stimulus generation, which we skip for photos), so the placeholder choice does not affect inference behavior.

`prompts.LABELS_BY_SHAPE` extension for the new `abstract` category:
```python
"abstract": ("object", "drawing", "diagram"),
```

### 3.4 Configs

Four configs mirroring M8d:

```
configs/m8c_qwen.py              # labeled
configs/m8c_qwen_label_free.py   # label-free
configs/m8c_llava.py             # labeled
configs/m8c_llava_label_free.py  # label-free
```

### 3.5 Inference pipeline

No changes to `scripts/02_run_inference.py` — the schema handling already accepts the manifest columns. The `m8c_curate_photos.py` step replaces `01_generate_stimuli.py` for this milestone.

### 3.6 Artifacts

- `docs/insights/m8c_real_photos.md` (+ `_ko.md`) — deep dive.
- `docs/experiments/m8c_real_photos.md` (+ `_ko.md`) — raw numbers.
- `docs/figures/m8c_photo_grid.png` — sample photos per category.
- `docs/figures/m8c_pmr_by_category.png` — PMR per (category, model, prompt).
- `docs/figures/m8c_paired_synthetic_vs_photo.png` — paired delta per (category, model).
- `notebooks/m8c_real_photos.ipynb`.
- `inputs/m8c_photos_<ts>/photo_metadata.csv` — license attribution table.
- `scripts/m8c_{run_all.sh, curate_photos.py, analyze.py, figures.py}`.

## 4. M8e super-milestone (optional)

Decided after M8d + M8c ship. Goals:
- `(model, category, source_type)` cross-tabulation: per-model, per-category synthetic-textured vs photo paired delta.
- Does photo-realism scale the existing saturation axis or open a new one? (Quantitative test: if photo PMR(_nolabel) − textured PMR(_nolabel) is constant across models, photo is a scale shift; if it varies, photo is a separate axis.)
- Abstract photos vs synthetic abstract: does the encoder treat abstract photos (drawings, diagrams) like abstract synthetic? Cross-saturation test.

**Synthetic counterparts for paired comparison**:

| M8c category | Synthetic counterpart |
|---|---|
| ball | M8a `circle` × `object_level=textured` × `label=ball` |
| car | M8d `car` × `object_level=textured` × `label=car` |
| person | M8d `person` × `object_level=textured` × `label=person` |
| bird | M8d `bird` × `object_level=textured` × `label=bird` |
| abstract | No clean synthetic counterpart — analyzed separately as a saturation reference. (Loose comparison: M8a `circle` × `object_level=line` × `label=circle` provides the synthetic abstract baseline.) |

The `event_template` for the synthetic counterpart is fixed to `horizontal` for car/person/bird and `fall` for ball/abstract, matching the per-category natural event from §2.2 and §3.3.

Artifacts (if pursued):
- `docs/insights/m8e_synthetic_vs_photo.md` (+ `_ko.md`).
- `docs/figures/m8e_{cross_source_heatmap, paired_grid}.png`.
- `notebooks/m8e_synthetic_vs_photo.ipynb`.

## 5. Roadmap and changelog updates

After each phase ships, update:

`references/roadmap.md` (+ `_ko.md`):
- §2 milestone overview: status column → ✅, completion date.
- §3 detailed status: new section per milestone (key results, hypothesis-scorecard updates).
- §1.3 hypothesis scorecard: H7 cross-category status, H-encoder-saturation external-validity status.
- §6 change log: phase completion entry with commit hash.

## 6. Test plan

`tests/`:
- `test_m8d_primitives.py`:
  - Each new draw function returns a non-blank image of the configured size.
  - Determinism: two calls with the same seed produce byte-identical output.
  - Pixel-mean differs across abstraction levels (rough sanity that levels are visually distinct).
- `test_m8d_labels.py`:
  - `LABELS_BY_SHAPE["car"|"person"|"bird"]` exist, are 3-tuples, all-string.
  - `prompts.labels_for_shape("car")` returns the expected triplet.
- `test_m8c_manifest.py`:
  - Synthetic minimal photo manifest (3 rows) loads via `pd.read_parquet`.
  - `inference.run.run_inference` accepts the photo manifest schema (smoke test with `cfg.limit=1`).

## 7. Implementation order and dependencies

```
M8d:
    1. Add primitives + draw_object dispatch              (~1.5h)
    2. Add LABELS_BY_SHAPE entries + Shape literal        (~15min)
    3. Write test_m8d_primitives.py + test_m8d_labels.py  (~30min)
    4. Write configs/m8d_*.py (4 configs)                 (~30min)
    5. Write scripts/m8d_run_all.sh                       (~15min)
    6. Generate stimuli, smoke (limit=5) on each config   (~15min)
    7. Full M8d run on GPU 1                              (~52-65 min wall)
    8. Add regime classifier + scoring + 50-stim hand-annot (~1.5h)
    9. Write m8d_analyze.py + m8d_figures.py              (~1.5h)
    10. Write insight + experiment + notebook             (~1.5h)
    11. Update roadmap + commit                           (~30min)

M8c (depends on M8d primitives finalized + LABELS_BY_SHAPE update for `abstract`):
    1. Write m8c_curate_photos.py + abstract_urls.txt     (~2h)
    2. Run curation, verify 60 photos                     (~20min)
    3. Write configs/m8c_*.py                             (~20min)
    4. Write scripts/m8c_run_all.sh                       (~10min)
    5. Smoke (limit=2) + full run                         (~10min)
    6. Write m8c_analyze.py + m8c_figures.py              (~1h)
    7. Write insight + experiment + notebook              (~1.5h)
    8. Update roadmap + commit                            (~30min)

M8e (optional, after M8d + M8c):
    1. Write m8e cross-source analysis script             (~1h)
    2. Generate cross-source figures                      (~1h)
    3. Write insight + notebook                           (~1.5h)
    4. Update roadmap + commit                            (~30min)
```

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Bird/duck primitive not recognizable at line level | Spot-check after step 6 of M8d implementation; re-design line primitives if VLM fails to identify category in label-free condition. |
| Regime keyword classifier miscount | 50-stim hand-annotation gates publication-readiness; if false rate ≥ 15 %, refine keywords or fall back to LLM judge. |
| OpenImages download bandwidth / API issues | Fallback to PEXELS for physical categories — slower but no infrastructure dependency. |
| `figurine` / `statue` / `duck` not invoking regime shift | Recorded as a finding (parallel to M8a's `wedge`/`polygon` weak-label exposure); regime distribution figure visualizes the (negative) result. |
| `silhouette` LABEL invokes "depiction-style" reading rather than abstract reject | Both readings produce abstract physics-mode response; either way the H7 abstract pole is operational. |
| Two-event factorial doubles inference cost | Already budgeted (~52-65 min on GPU 1). |
| Photo license edge cases | All sources are commercial-OK by default (Apache 2.0 / Pexels License); per-photo metadata CSV records source for paper-time attribution. |
| Photo curation includes near-duplicates that inflate paired-delta significance | Hand-review at curation step; reject visually similar photos. |

## 9. Out of scope

- Activation captures on Qwen / LLaVA for M8d / M8c (deferred to a possible future M8d-r2 mechanism milestone).
- InternVL3 inference on M8d / M8c (saturated baseline; deferred to M9 generalization audit).
- Forced-choice prompts (FC excluded same as M8a; LLaVA "A" bias unresolved at logit level).
- Cross-encoder swap (separate priority §4.5 milestone).
- Plant / boat / other M8d round-2 categories.
