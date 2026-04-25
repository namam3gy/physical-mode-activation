# M8d — Non-ball physical-object categories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `car`, `person`, and `bird` as non-ball physical-object categories to the existing M8a stimulus pipeline so that H7 (label-selects-regime) and H1 ramp generalization can be tested cross-category.

**Architecture:** Extends `physical_mode/stimuli/primitives.py` with 12 new draw functions (3 categories × 4 abstraction levels), `inference/prompts.py::LABELS_BY_SHAPE` with 3 new entries, and `metrics/pmr.py` with a parallel `classify_regime()` keyword classifier. Two-event factorial (`fall + horizontal`) is run on Qwen2.5-VL-7B and LLaVA-1.5-7B in labeled + label-free arms. Reuses M8a's `02_run_inference.py` and analysis pipeline unchanged.

**Tech Stack:** Python 3.10+, PIL/Pillow for primitives, pandas for analysis, pytest for tests, transformers / Qwen2.5-VL / LLaVA-1.5 for inference, matplotlib for figures, `uv run python` for all Python execution.

**Spec reference:** `docs/superpowers/specs/2026-04-25-m8d-m8c-stimulus-diversification-design.md` §2 (M8d component).

**GPU:** All inference on `CUDA_VISIBLE_DEVICES=1`.

---

## File Structure

```
NEW FILES
├── configs/
│   ├── m8d_qwen.py                       # labeled, Qwen
│   ├── m8d_qwen_label_free.py            # label-free, Qwen
│   ├── m8d_llava.py                      # labeled, LLaVA
│   └── m8d_llava_label_free.py           # label-free, LLaVA
├── scripts/
│   ├── m8d_run_all.sh                    # sequential 4-config runner
│   ├── m8d_spot_check.py                 # primitive recognizability sampler
│   ├── m8d_analyze.py                    # roll-up + paired-delta tables
│   ├── m8d_figures.py                    # 6 figures
│   └── m8d_hand_annotate.py              # 50-stim sampler + CSV scaffold
├── tests/
│   ├── test_m8d_labels.py                # LABELS_BY_SHAPE additions
│   ├── test_m8d_primitives.py            # new draw functions determinism + recognizability
│   └── test_m8d_regime.py                # classify_regime unit tests
├── docs/insights/
│   ├── m8d_non_ball_categories.md        # English deep dive
│   └── m8d_non_ball_categories_ko.md     # Korean translation
├── docs/experiments/
│   ├── m8d_non_ball_categories.md        # English raw numbers
│   └── m8d_non_ball_categories_ko.md     # Korean translation
├── docs/figures/
│   ├── m8d_shape_grid.png
│   ├── m8d_full_scene_samples.png
│   ├── m8d_pmr_ramp.png
│   ├── m8d_pmr_by_role.png
│   ├── m8d_paired_delta.png
│   └── m8d_regime_distribution.png
└── notebooks/
    └── m8d_non_ball_categories.ipynb     # cell-by-cell reproduction

MODIFIED FILES
├── src/physical_mode/config.py            # extend Shape + Label literals
├── src/physical_mode/stimuli/primitives.py  # 12 new draw functions + dispatch
├── src/physical_mode/inference/prompts.py   # LABELS_BY_SHAPE 3 new entries
├── src/physical_mode/metrics/pmr.py         # classify_regime() + helpers
├── src/physical_mode/metrics/lexicons.py    # CATEGORY_REGIME_KEYWORDS
└── references/roadmap.md (+ _ko.md)         # M8d completion entry
```

**Responsibility split**:
- `stimuli/primitives.py` owns visual rendering (no metric awareness).
- `metrics/pmr.py` owns response scoring; new `classify_regime` lives next to existing `score_pmr` for reuse of `_words` / `_any_stem_hit` helpers.
- `metrics/lexicons.py` owns vocabulary; new `CATEGORY_REGIME_KEYWORDS` lives next to existing `PHYSICS_VERB_STEMS`.
- `inference/prompts.py` owns prompt + label dispatch; `LABELS_BY_SHAPE` is the single source of truth for `(physical, abstract, exotic)` triplets.
- `configs/m8d_*.py` owns experiment specs (4 files mirroring M8a).
- `scripts/m8d_*.py` owns analysis + figure generation drivers.

---

## Task 1: Extend `Shape` and `Label` type aliases

**Files:**
- Modify: `src/physical_mode/config.py`
- Modify: `src/physical_mode/stimuli/primitives.py:17` (also has a `Shape` literal — keep in sync)
- Test: `tests/test_m8d_labels.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_m8d_labels.py`:

```python
"""Sanity tests that M8d additions to type aliases and label tables exist."""

from __future__ import annotations

from physical_mode.config import FactorialSpec
from physical_mode.inference.prompts import LABELS_BY_SHAPE, labels_for_shape


def test_m8d_shapes_factorial_iter():
    """FactorialSpec accepts the new M8d shapes without error."""
    spec = FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line",),
        bg_levels=("blank",),
        cue_levels=("none",),
        event_templates=("fall",),
        seeds_per_cell=1,
    )
    rows = list(spec.iter())
    assert len(rows) == 3
    assert {r.shape for r in rows} == {"car", "person", "bird"}


def test_m8d_labels_by_shape_present():
    """All three new categories registered in LABELS_BY_SHAPE."""
    for category in ("car", "person", "bird"):
        assert category in LABELS_BY_SHAPE, f"{category!r} missing from LABELS_BY_SHAPE"
        triplet = LABELS_BY_SHAPE[category]
        assert isinstance(triplet, tuple), f"{category!r} entry not a tuple"
        assert len(triplet) == 3, f"{category!r} entry not a 3-tuple: {triplet}"
        assert all(isinstance(x, str) for x in triplet), f"{category!r} entry has non-str: {triplet}"


def test_m8d_label_triplet_values():
    """Spec-pinned label triplets per category."""
    assert LABELS_BY_SHAPE["car"]    == ("car",    "silhouette",  "figurine")
    assert LABELS_BY_SHAPE["person"] == ("person", "stick figure", "statue")
    assert LABELS_BY_SHAPE["bird"]   == ("bird",   "silhouette",  "duck")


def test_m8d_labels_for_shape():
    """labels_for_shape() returns the configured triplet."""
    assert labels_for_shape("car")    == ("car",    "silhouette",  "figurine")
    assert labels_for_shape("person") == ("person", "stick figure", "statue")
    assert labels_for_shape("bird")   == ("bird",   "silhouette",  "duck")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && CUDA_VISIBLE_DEVICES=1 uv run python -m pytest tests/test_m8d_labels.py -v`
Expected: FAIL — `KeyError: 'car'` (or similar) because `LABELS_BY_SHAPE` doesn't have new entries yet.

- [ ] **Step 3: Update `Shape` literal in `config.py`**

Edit `src/physical_mode/config.py` line 23:

```python
# Old
Shape = Literal["circle", "square", "triangle", "hexagon", "polygon"]

# New (add car/person/bird)
Shape = Literal["circle", "square", "triangle", "hexagon", "polygon", "car", "person", "bird"]
```

Edit `src/physical_mode/config.py` lines 24-39 — extend the `Label` literal:

```python
Label = Literal[
    # circle labels (pilot)
    "circle", "ball", "planet",
    # generic / no-label sentinels
    "shape", "object", "_nolabel",
    # square labels (M8a)
    "brick", "square", "tile",
    # triangle labels (M8a)
    "wedge", "triangle", "sign",
    # hexagon labels (M8a)
    "nut", "hexagon", "coin",
    # polygon labels (M8a)
    "rock", "polygon", "boulder",
    # car labels (M8d)
    "car", "silhouette", "figurine",
    # person labels (M8d)
    "person", "stick figure", "statue",
    # bird labels (M8d)
    "bird", "duck",
    # M8a label-role aliases (resolved at run.py to literal labels)
    "physical", "abstract", "exotic",
]
```

Note: `silhouette` is reused across car and bird — list it only once.

- [ ] **Step 4: Update `Shape` literal in `primitives.py`**

Edit `src/physical_mode/stimuli/primitives.py` line 17:

```python
# Old
Shape = Literal["circle", "square", "triangle", "hexagon", "polygon"]

# New
Shape = Literal["circle", "square", "triangle", "hexagon", "polygon", "car", "person", "bird"]
```

- [ ] **Step 5: Add `LABELS_BY_SHAPE` entries**

Edit `src/physical_mode/inference/prompts.py` lines 70-81 — append the new entries:

```python
LABELS_BY_SHAPE: dict[str, tuple[str, str, str]] = {
    "circle":   ("ball",   "circle",   "planet"),
    "square":   ("brick",  "square",   "tile"),
    "triangle": ("wedge",  "triangle", "sign"),
    "hexagon":  ("nut",    "hexagon",  "coin"),
    "polygon":  ("rock",   "polygon",  "boulder"),
    # M8d non-ball categories. abstract role is depiction-style ("silhouette",
    # "stick figure") rather than a forced geometric class because non-ball
    # categories don't have natural geometric-class names.
    "car":      ("car",    "silhouette",  "figurine"),
    "person":   ("person", "stick figure", "statue"),
    "bird":     ("bird",   "silhouette",  "duck"),
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && uv run python -m pytest tests/test_m8d_labels.py -v`
Expected: PASS — all 4 tests green.

- [ ] **Step 7: Run full pytest suite to confirm no regressions**

Run: `uv run python -m pytest -v`
Expected: All existing M8a tests still pass.

- [ ] **Step 8: Commit**

```bash
git add tests/test_m8d_labels.py src/physical_mode/config.py src/physical_mode/stimuli/primitives.py src/physical_mode/inference/prompts.py
git commit -m "$(cat <<'EOF'
feat(m8d): register car/person/bird shapes + LABELS_BY_SHAPE entries

(physical, abstract, exotic) triplets:
  car    -> (car,    silhouette,    figurine)
  person -> (person, stick figure,  statue)
  bird   -> (bird,   silhouette,    duck)

abstract role uses depiction-style labels (silhouette / stick figure)
rather than a forced geometric class because non-ball categories
don't have natural geometric-class names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add car primitives (4 abstraction levels)

**Files:**
- Modify: `src/physical_mode/stimuli/primitives.py` (append before `# Object primitives` end / near line 530)
- Test: `tests/test_m8d_primitives.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_m8d_primitives.py`:

```python
"""Tests for M8d (car / person / bird) primitive draw functions."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from physical_mode.stimuli.primitives import (
    blank_canvas,
    draw_object,
)

CANVAS = 512
RADIUS = 64


def _img_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"))


@pytest.mark.parametrize("shape", ["car", "person", "bird"])
@pytest.mark.parametrize("mode", ["line", "filled", "shaded", "textured"])
def test_m8d_primitive_returns_image_of_canvas_size(shape, mode):
    img = blank_canvas(CANVAS)
    out = draw_object(img, mode=mode, cx=CANVAS // 2, cy=CANVAS // 2, radius=RADIUS, seed=42, shape=shape)
    assert out.size == (CANVAS, CANVAS)


@pytest.mark.parametrize("shape", ["car", "person", "bird"])
@pytest.mark.parametrize("mode", ["line", "filled", "shaded", "textured"])
def test_m8d_primitive_writes_non_white_pixels(shape, mode):
    """The new primitives must render some non-white pixels (otherwise nothing was drawn)."""
    img = blank_canvas(CANVAS)
    out = draw_object(img, mode=mode, cx=CANVAS // 2, cy=CANVAS // 2, radius=RADIUS, seed=42, shape=shape)
    arr = _img_array(out)
    n_non_white = int(((arr < 250).any(axis=-1)).sum())
    assert n_non_white >= 200, f"{shape}/{mode} drew only {n_non_white} non-white px — primitive looks empty"


@pytest.mark.parametrize("shape", ["car", "person", "bird"])
@pytest.mark.parametrize("mode", ["line", "filled", "shaded", "textured"])
def test_m8d_primitive_deterministic(shape, mode):
    """Same seed → byte-identical output."""
    img1 = blank_canvas(CANVAS)
    img2 = blank_canvas(CANVAS)
    out1 = draw_object(img1, mode=mode, cx=256, cy=256, radius=RADIUS, seed=123, shape=shape)
    out2 = draw_object(img2, mode=mode, cx=256, cy=256, radius=RADIUS, seed=123, shape=shape)
    assert _img_array(out1).tobytes() == _img_array(out2).tobytes()


@pytest.mark.parametrize("shape", ["car", "person", "bird"])
def test_m8d_levels_are_visually_distinct(shape):
    """line/filled/shaded/textured should produce different pixel arrays for a given shape."""
    img = blank_canvas(CANVAS)
    arrs = {}
    for mode in ("line", "filled", "shaded", "textured"):
        out = draw_object(blank_canvas(CANVAS), mode=mode, cx=256, cy=256, radius=RADIUS, seed=7, shape=shape)
        arrs[mode] = _img_array(out)
    # Each pair should differ in at least 0.5% of pixels.
    keys = list(arrs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff = (arrs[keys[i]] != arrs[keys[j]]).any(axis=-1).mean()
            assert diff > 0.005, f"{shape}: {keys[i]} ≈ {keys[j]} (only {diff:.4f} pixels differ)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v -k car`
Expected: FAIL — `ValueError: unknown (shape, mode): (car, line)` from `draw_object`.

- [ ] **Step 3: Implement car primitives**

Append to `src/physical_mode/stimuli/primitives.py` before the `# Background helpers` block (search for `def draw_ground`). Insert after `_draw_block_stack`:

```python
# ---------------------------------------------------------------------------
# M8d car primitives. Body = wide rectangle, two circular wheels below.
# Compositional drawing — recognizable at every abstraction level.
# ---------------------------------------------------------------------------


def _car_geometry(cx: int, cy: int, r: int) -> dict:
    """Bounding-box geometry shared by all four car abstractions.

    Body is a horizontal rectangle ~2.0r wide, ~0.7r tall, centered around (cx, cy).
    Two wheels sit just below, ~0.45r radius each.
    """
    body_w = int(r * 2.0)
    body_h = int(r * 0.7)
    body_left = cx - body_w // 2
    body_right = cx + body_w // 2
    body_top = cy - body_h // 2
    body_bottom = cy + body_h // 2
    wheel_r = int(r * 0.35)
    wheel_y = body_bottom + wheel_r // 2
    wheel_lx = body_left + int(body_w * 0.22)
    wheel_rx = body_left + int(body_w * 0.78)
    # Windshield rectangle (smaller, top-left of body).
    win_w = int(body_w * 0.4)
    win_h = int(body_h * 0.55)
    win_left = body_left + int(body_w * 0.15)
    win_top = body_top + int(body_h * 0.1)
    return dict(
        body_box=(body_left, body_top, body_right, body_bottom),
        wheel_r=wheel_r,
        wheel_l=(wheel_lx, wheel_y),
        wheel_r_pos=(wheel_rx, wheel_y),
        windshield_box=(win_left, win_top, win_left + win_w, win_top + win_h),
    )


def _draw_line_car(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.rectangle(g["body_box"], outline=(0, 0, 0), width=3)
    d.rectangle(g["windshield_box"], outline=(0, 0, 0), width=2)
    wr = g["wheel_r"]
    for (wx, wy) in (g["wheel_l"], g["wheel_r_pos"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), outline=(0, 0, 0), width=3)
    return img


def _draw_filled_car(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.rectangle(g["body_box"], fill=(0, 0, 0))
    wr = g["wheel_r"]
    for (wx, wy) in (g["wheel_l"], g["wheel_r_pos"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), fill=(0, 0, 0))
    # Windshield in lighter color so silhouette is still recognizably a car.
    d.rectangle(g["windshield_box"], fill=(120, 120, 120))
    return img


def _draw_shaded_car(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Car with top-lit gradient (lighter top, darker bottom)."""
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    bx0, by0, bx1, by1 = g["body_box"]
    body_h = by1 - by0
    n_strips = 32
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        c = int(220 - 110 * t)  # 220 (top) → 110 (bottom)
        y0 = by0 + int(body_h * (i / n_strips))
        y1 = by0 + int(body_h * ((i + 1) / n_strips))
        d.rectangle((bx0, y0, bx1, y1), fill=(c, c, c + 20))
    d.rectangle(g["body_box"], outline=(40, 40, 40), width=2)
    # Wheels: dark circles with subtle gradient.
    wr = g["wheel_r"]
    for (wx, wy) in (g["wheel_l"], g["wheel_r_pos"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), fill=(40, 40, 40), outline=(0, 0, 0), width=2)
        d.ellipse((wx - wr // 2, wy - wr // 2, wx + wr // 2, wy + wr // 2), fill=(80, 80, 80))
    # Windshield: light blue glassy shade.
    d.rectangle(g["windshield_box"], fill=(180, 200, 220), outline=(60, 60, 60), width=1)
    return img


def _draw_textured_car(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish car with body color, glass detail, wheel hubs."""
    rng = random.Random(seed + 11000)
    # Body color: a saturated automotive hue.
    palette = [(180, 30, 30), (40, 80, 160), (30, 120, 60), (200, 160, 30)]
    body_color = palette[rng.randrange(len(palette))]
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.rectangle(g["body_box"], fill=body_color, outline=(20, 20, 20), width=2)
    # Top highlight strip.
    bx0, by0, bx1, by1 = g["body_box"]
    d.rectangle((bx0 + 4, by0 + 4, bx1 - 4, by0 + (by1 - by0) // 5), fill=(min(255, body_color[0] + 60), min(255, body_color[1] + 60), min(255, body_color[2] + 60)))
    # Wheels with hubs.
    wr = g["wheel_r"]
    for (wx, wy) in (g["wheel_l"], g["wheel_r_pos"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), fill=(20, 20, 20), outline=(0, 0, 0), width=2)
        # Hub (center disc).
        hr = wr // 2
        d.ellipse((wx - hr, wy - hr, wx + hr, wy + hr), fill=(180, 180, 180))
        # Hub center bolt.
        d.ellipse((wx - 4, wy - 4, wx + 4, wy + 4), fill=(60, 60, 60))
    # Windshield: glassy blue with a slight gradient.
    wx0, wy0, wx1, wy1 = g["windshield_box"]
    d.rectangle(g["windshield_box"], fill=(150, 190, 220), outline=(40, 40, 40), width=1)
    # Glare line.
    d.line((wx0 + 4, wy0 + 4, wx1 - 4, wy0 + (wy1 - wy0) // 3), fill=(230, 240, 250), width=2)
    return img
```

- [ ] **Step 4: Update `draw_object` dispatch**

Insert into `src/physical_mode/stimuli/primitives.py::draw_object` before the final `raise ValueError(...)` (around line 96):

```python
    if shape == "car":
        if mode == "line":
            return _draw_line_car(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_car(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_car(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_car(img, cx, cy, radius, seed)
```

- [ ] **Step 5: Run test to verify car-specific tests pass**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v -k car`
Expected: PASS — `test_m8d_primitive_returns_image_of_canvas_size`, `test_m8d_primitive_writes_non_white_pixels`, `test_m8d_primitive_deterministic`, `test_m8d_levels_are_visually_distinct` all green for car.

- [ ] **Step 6: Commit**

```bash
git add src/physical_mode/stimuli/primitives.py tests/test_m8d_primitives.py
git commit -m "$(cat <<'EOF'
feat(m8d): car primitives (line/filled/shaded/textured)

Body = horizontal rectangle ~2.0r wide, two circular wheels, small
windshield rectangle. Recognizable at every abstraction level.

Textured palette samples one of four automotive colors per seed;
shaded uses a top-lit gradient with subtle wheel hub detail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add person primitives (4 abstraction levels)

**Files:**
- Modify: `src/physical_mode/stimuli/primitives.py`
- Test: `tests/test_m8d_primitives.py` (already covers person via parametrize)

- [ ] **Step 1: Run person test (currently fails because person primitives don't exist)**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v -k person`
Expected: FAIL — `ValueError: unknown (shape, mode): (person, line)`.

- [ ] **Step 2: Implement person primitives**

Append to `src/physical_mode/stimuli/primitives.py` after the car block:

```python
# ---------------------------------------------------------------------------
# M8d person primitives. Stick-figure family: head circle + body line + arms + legs.
# Recognizable at every abstraction level.
# ---------------------------------------------------------------------------


def _person_geometry(cx: int, cy: int, r: int) -> dict:
    """Stick-figure geometry — head, torso, arm/leg endpoints."""
    head_r = int(r * 0.28)
    head_cy = cy - r + head_r
    torso_top = head_cy + head_r
    torso_bottom = cy + r // 2
    arm_y = torso_top + int(r * 0.28)
    leg_y = cy + r
    return dict(
        head=(cx, head_cy, head_r),
        torso=((cx, torso_top), (cx, torso_bottom)),
        left_arm=((cx, arm_y), (cx - int(r * 0.65), arm_y + int(r * 0.4))),
        right_arm=((cx, arm_y), (cx + int(r * 0.65), arm_y + int(r * 0.4))),
        left_leg=((cx, torso_bottom), (cx - int(r * 0.45), leg_y)),
        right_leg=((cx, torso_bottom), (cx + int(r * 0.45), leg_y)),
    )


def _draw_line_person(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _person_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    hx, hy, hr = g["head"]
    d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), outline=(0, 0, 0), width=3)
    for limb in ("torso", "left_arm", "right_arm", "left_leg", "right_leg"):
        p1, p2 = g[limb]
        d.line((p1, p2), fill=(0, 0, 0), width=3)
    return img


def _draw_filled_person(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Filled silhouette: thick black body + filled head."""
    g = _person_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    hx, hy, hr = g["head"]
    d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(0, 0, 0))
    for limb in ("torso", "left_arm", "right_arm", "left_leg", "right_leg"):
        p1, p2 = g[limb]
        d.line((p1, p2), fill=(0, 0, 0), width=10)
    return img


def _draw_shaded_person(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Person with subtle 3D shading: head with gradient, body as gradient column."""
    g = _person_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    hx, hy, hr = g["head"]
    # Head with simple top-light gradient.
    n_strips = 16
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        c = int(230 - 90 * t)
        y0 = hy - hr + int(2 * hr * (i / n_strips))
        y1 = hy - hr + int(2 * hr * ((i + 1) / n_strips))
        d.ellipse((hx - hr, y0, hx + hr, y1), fill=(c, c, c))
    d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), outline=(60, 60, 60), width=2)
    # Body strokes as filled rectangles with mid-grey + outline.
    body_color = (140, 140, 150)
    body_outline = (60, 60, 70)
    for limb in ("torso", "left_arm", "right_arm", "left_leg", "right_leg"):
        p1, p2 = g[limb]
        d.line((p1, p2), fill=body_color, width=14)
        d.line((p1, p2), fill=body_outline, width=2)
    return img


def _draw_textured_person(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Person with skin-tone face + clothing color block."""
    rng = random.Random(seed + 22000)
    skin_palette = [(232, 192, 158), (210, 165, 130), (170, 120, 90), (130, 90, 70)]
    skin = skin_palette[rng.randrange(len(skin_palette))]
    clothes_palette = [(60, 90, 160), (160, 60, 60), (60, 120, 60), (180, 130, 50)]
    clothes = clothes_palette[rng.randrange(len(clothes_palette))]
    g = _person_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    hx, hy, hr = g["head"]
    # Head: skin color + outline + simple eyes.
    d.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=skin, outline=(40, 30, 30), width=2)
    eye_r = max(2, hr // 6)
    eye_y = hy - hr // 8
    d.ellipse((hx - hr // 2 - eye_r, eye_y - eye_r, hx - hr // 2 + eye_r, eye_y + eye_r), fill=(20, 20, 20))
    d.ellipse((hx + hr // 2 - eye_r, eye_y - eye_r, hx + hr // 2 + eye_r, eye_y + eye_r), fill=(20, 20, 20))
    # Torso (clothing block) and limbs.
    for limb in ("torso", "left_arm", "right_arm"):
        p1, p2 = g[limb]
        d.line((p1, p2), fill=clothes, width=14)
    # Legs (a different darker tone).
    legs = (max(0, clothes[0] - 50), max(0, clothes[1] - 50), max(0, clothes[2] - 50))
    for limb in ("left_leg", "right_leg"):
        p1, p2 = g[limb]
        d.line((p1, p2), fill=legs, width=14)
    return img
```

- [ ] **Step 3: Update `draw_object` dispatch**

Insert into `draw_object` after the car block:

```python
    if shape == "person":
        if mode == "line":
            return _draw_line_person(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_person(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_person(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_person(img, cx, cy, radius, seed)
```

- [ ] **Step 4: Run person tests to verify they pass**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v -k person`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/physical_mode/stimuli/primitives.py
git commit -m "$(cat <<'EOF'
feat(m8d): person primitives (line/filled/shaded/textured)

Stick-figure family: head circle + body line + 2 arms + 2 legs.
Filled level uses thick limb strokes for silhouette readability.
Shaded uses gradient head + greyscale body; textured uses
skin-tone palette + clothing color blocks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add bird primitives (4 abstraction levels)

**Files:**
- Modify: `src/physical_mode/stimuli/primitives.py`
- Test: `tests/test_m8d_primitives.py` (already covers bird via parametrize)

- [ ] **Step 1: Run bird test to confirm it fails**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v -k bird`
Expected: FAIL — `ValueError: unknown (shape, mode): (bird, line)`.

- [ ] **Step 2: Implement bird primitives**

Append to `src/physical_mode/stimuli/primitives.py` after the person block:

```python
# ---------------------------------------------------------------------------
# M8d bird primitives. Oval body + small head + beak + wing curve.
# Recognizable at every abstraction level.
# ---------------------------------------------------------------------------


def _bird_geometry(cx: int, cy: int, r: int) -> dict:
    """Bird geometry — oval body, head circle to upper-right, beak triangle, wing arc."""
    body_w = int(r * 1.6)
    body_h = int(r * 0.95)
    body_box = (cx - body_w // 2, cy - body_h // 2, cx + body_w // 2, cy + body_h // 2)
    head_r = int(r * 0.32)
    head_cx = cx + int(body_w * 0.35)
    head_cy = cy - int(body_h * 0.35)
    head_box = (head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r)
    # Beak triangle pointing right.
    beak = [
        (head_cx + head_r, head_cy),
        (head_cx + head_r + int(r * 0.35), head_cy - int(r * 0.04)),
        (head_cx + head_r + int(r * 0.04), head_cy + int(r * 0.10)),
    ]
    # Wing arc (a chord) — three points across upper body.
    wing = [
        (cx - int(body_w * 0.30), cy - int(body_h * 0.05)),
        (cx, cy - int(body_h * 0.45)),
        (cx + int(body_w * 0.20), cy - int(body_h * 0.05)),
    ]
    eye_cx = head_cx + int(head_r * 0.25)
    eye_cy = head_cy - int(head_r * 0.10)
    return dict(
        body_box=body_box,
        head_box=head_box,
        head_center=(head_cx, head_cy, head_r),
        beak=beak,
        wing=wing,
        eye=(eye_cx, eye_cy),
    )


def _draw_line_bird(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _bird_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.ellipse(g["body_box"], outline=(0, 0, 0), width=3)
    d.ellipse(g["head_box"], outline=(0, 0, 0), width=3)
    d.polygon(g["beak"], outline=(0, 0, 0))
    d.line(g["wing"], fill=(0, 0, 0), width=3)
    return img


def _draw_filled_bird(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _bird_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.ellipse(g["body_box"], fill=(0, 0, 0))
    d.ellipse(g["head_box"], fill=(0, 0, 0))
    d.polygon(g["beak"], fill=(0, 0, 0))
    return img


def _draw_shaded_bird(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Bird with greyscale gradient body."""
    g = _bird_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    bx0, by0, bx1, by1 = g["body_box"]
    body_h = by1 - by0
    n_strips = 24
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        c = int(220 - 110 * t)
        y0 = by0 + int(body_h * (i / n_strips))
        y1 = by0 + int(body_h * ((i + 1) / n_strips))
        d.ellipse((bx0, y0, bx1, y1), fill=(c, c, c))
    d.ellipse(g["body_box"], outline=(60, 60, 60), width=2)
    # Head: filled grey + outline.
    d.ellipse(g["head_box"], fill=(160, 160, 160), outline=(40, 40, 40), width=2)
    # Beak.
    d.polygon(g["beak"], fill=(120, 100, 60), outline=(60, 50, 30))
    # Wing line.
    d.line(g["wing"], fill=(40, 40, 40), width=3)
    # Eye dot.
    ex, ey = g["eye"]
    d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(0, 0, 0))
    return img


def _draw_textured_bird(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish bird: body color + feather hatching + colored beak + eye."""
    rng = random.Random(seed + 33000)
    body_palette = [(120, 90, 60), (60, 80, 130), (180, 130, 60), (90, 120, 70)]
    body_color = body_palette[rng.randrange(len(body_palette))]
    g = _bird_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.ellipse(g["body_box"], fill=body_color, outline=(30, 30, 30), width=2)
    # Feather hatching: short curved strokes inside the body.
    bx0, by0, bx1, by1 = g["body_box"]
    body_w = bx1 - bx0
    body_h = by1 - by0
    for _ in range(28):
        sx = bx0 + rng.randint(8, body_w - 8)
        sy = by0 + rng.randint(8, body_h - 8)
        # Inside-ellipse check.
        if ((sx - cx) / (body_w / 2)) ** 2 + ((sy - cy) / (body_h / 2)) ** 2 > 0.85:
            continue
        ex = sx + rng.randint(-8, 8)
        ey = sy + rng.randint(2, 8)
        feather_color = (
            max(0, body_color[0] - 30),
            max(0, body_color[1] - 30),
            max(0, body_color[2] - 30),
        )
        d.line(((sx, sy), (ex, ey)), fill=feather_color, width=1)
    # Head with same base color but slightly lighter.
    head_color = (min(255, body_color[0] + 20), min(255, body_color[1] + 20), min(255, body_color[2] + 20))
    d.ellipse(g["head_box"], fill=head_color, outline=(30, 30, 30), width=2)
    # Beak (warm orange).
    d.polygon(g["beak"], fill=(220, 140, 40), outline=(140, 80, 20))
    # Eye.
    ex, ey = g["eye"]
    d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(0, 0, 0))
    return img
```

- [ ] **Step 3: Update `draw_object` dispatch**

Insert into `draw_object` after the person block:

```python
    if shape == "bird":
        if mode == "line":
            return _draw_line_bird(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_bird(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_bird(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_bird(img, cx, cy, radius, seed)
```

- [ ] **Step 4: Run bird tests + full M8d primitive suite**

Run: `uv run python -m pytest tests/test_m8d_primitives.py -v`
Expected: PASS — all 36 parametrized tests (3 shapes × 4 modes × 3 base tests, plus 3 levels-distinct tests = 39 cases) green.

- [ ] **Step 5: Run full pytest to ensure no regressions**

Run: `uv run python -m pytest -v`
Expected: All M8a + M8d tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/physical_mode/stimuli/primitives.py
git commit -m "$(cat <<'EOF'
feat(m8d): bird primitives (line/filled/shaded/textured)

Oval body + small head circle + beak triangle + wing arc.
Textured uses feather-hatching for body fill; shaded uses
greyscale gradient with colored beak. All four levels visually
distinct (≥0.5% pixel diff between any pair).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Visual spot-check primitives via VLM

**Files:**
- Create: `scripts/m8d_spot_check.py`
- Create: `docs/figures/m8d_shape_grid.png`

**Goal:** Confirm primitives are recognizable as the intended category by Qwen2.5-VL-7B in the label-free condition. If recognition fails (e.g., model says "sphere" for car-line), the line primitive needs revision.

- [ ] **Step 1: Write `scripts/m8d_spot_check.py`**

Create `scripts/m8d_spot_check.py`:

```python
"""M8d primitive spot-check.

Renders every (shape, mode) combination as a single PNG grid for visual
inspection. Optionally runs Qwen2.5-VL-7B label-free inference on each
to confirm the model identifies the intended category.

Run:
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/m8d_spot_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from physical_mode.stimuli.primitives import blank_canvas, draw_object


SHAPES = ("car", "person", "bird")
MODES = ("line", "filled", "shaded", "textured")
CANVAS = 512
RADIUS = 64


def render_grid(out_path: Path) -> None:
    """Render a 3x4 grid (shape rows, abstraction columns)."""
    cell = 256
    grid = Image.new("RGB", (cell * len(MODES), cell * len(SHAPES)), (255, 255, 255))
    for i, shape in enumerate(SHAPES):
        for j, mode in enumerate(MODES):
            img = blank_canvas(CANVAS)
            img = draw_object(img, mode=mode, cx=CANVAS // 2, cy=CANVAS // 2, radius=RADIUS, seed=42, shape=shape)
            small = img.resize((cell, cell), Image.LANCZOS)
            grid.paste(small, (j * cell, i * cell))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"wrote {out_path}")


def run_vlm_check(shapes: list[str]) -> None:
    """Optional: load Qwen2.5-VL-7B and ask label-free what each primitive depicts."""
    # Lazy import — heavy.
    from physical_mode.models.vlm_runner import InferenceArgs, PhysModeVLM
    from physical_mode.inference.prompts import render

    vlm = PhysModeVLM(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="bfloat16",
        device="cuda",
    )
    args = InferenceArgs(max_new_tokens=48, temperature=0.0, top_p=1.0)
    rp = render("open_no_label", "_nolabel")

    for shape in shapes:
        for mode in MODES:
            img = blank_canvas(CANVAS)
            img = draw_object(img, mode=mode, cx=CANVAS // 2, cy=CANVAS // 2, radius=RADIUS, seed=42, shape=shape)
            tmp = Path("/tmp/_m8d_spot.png")
            img.save(tmp)
            gen = vlm.generate(image=tmp, prompt=rp.user, args=args, system_prompt=rp.system)
            print(f"[{shape}/{mode}] {gen['raw_text'][:120]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-only", action="store_true", help="Only render the grid; skip VLM check")
    p.add_argument("--shapes", nargs="*", default=list(SHAPES))
    p.add_argument("--out", type=Path, default=Path("docs/figures/m8d_shape_grid.png"))
    args = p.parse_args()

    render_grid(args.out)

    if not args.grid_only:
        run_vlm_check(args.shapes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render the grid**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && uv run python scripts/m8d_spot_check.py --grid-only`
Expected: STDOUT `wrote docs/figures/m8d_shape_grid.png`. Open the PNG and visually confirm:
  - Row 1 (car): all four cells look like a car (rectangular body + 2 wheels).
  - Row 2 (person): all four cells look like a person (head + body + arms + legs).
  - Row 3 (bird): all four cells look like a bird (oval body + head + beak).

- [ ] **Step 3: Run VLM-based recognition check**

Run: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/m8d_spot_check.py`
Expected: STDOUT shows 12 lines like `[car/line] The image shows a simple line drawing of a car with two wheels.`

Acceptance: ≥10 of 12 cells produce a description that contains the canonical category word (`car`, `person`/`man`/`figure`, `bird`). Record outliers — if any line-level primitive is misread (e.g., bird/line mistaken for "fish"), it's a candidate for revision before proceeding.

- [ ] **Step 4: Commit grid figure**

```bash
git add scripts/m8d_spot_check.py docs/figures/m8d_shape_grid.png
git commit -m "$(cat <<'EOF'
feat(m8d): primitive spot-check script + shape grid figure

scripts/m8d_spot_check.py renders the 3x4 (shape x mode) grid and
optionally runs Qwen2.5-VL-7B label-free inference to confirm
category recognition. Shape-grid PNG saved to docs/figures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `classify_regime` keyword classifier

**Files:**
- Modify: `src/physical_mode/metrics/lexicons.py` (append `CATEGORY_REGIME_KEYWORDS`)
- Modify: `src/physical_mode/metrics/pmr.py` (add `classify_regime`)
- Test: `tests/test_m8d_regime.py` (Create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_m8d_regime.py`:

```python
"""Tests for M8d regime classifier (kinetic / static / abstract / ambiguous)."""

from __future__ import annotations

import pytest

from physical_mode.metrics.pmr import classify_regime


# (category, response_text, expected_regime)
KINETIC_CASES = [
    ("car",    "The car drives forward at high speed.",         "kinetic"),
    ("car",    "It rolls along the road.",                       "kinetic"),
    ("car",    "The race car speeds away.",                      "kinetic"),
    ("person", "The person walks forward.",                      "kinetic"),
    ("person", "The athlete runs across the field.",             "kinetic"),
    ("bird",   "The bird flies into the sky.",                   "kinetic"),
    ("bird",   "The duck swims across the pond.",                "kinetic"),
    ("bird",   "The duck waddles along the shore.",              "kinetic"),
]

STATIC_CASES = [
    ("car",    "The car is parked beside the curb.",             "static"),
    ("car",    "The figurine stays on display.",                 "static"),
    ("person", "The statue stands motionless in the square.",    "static"),
    ("person", "The person stays still and waits.",              "static"),
    ("bird",   "The bird perches on the branch.",                "static"),
]

ABSTRACT_CASES = [
    ("car",    "This is just a drawing of a rectangle.",         "abstract"),
    ("person", "This is an abstract stick figure — nothing happens.", "abstract"),
    ("bird",   "It is a silhouette; nothing moves.",             "abstract"),
]

AMBIGUOUS_CASES = [
    ("car",    "Hmm, I am not sure.",                            "ambiguous"),
    ("bird",   "It is unclear what will happen.",                "ambiguous"),
]


@pytest.mark.parametrize("category,text,expected", KINETIC_CASES)
def test_kinetic_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", STATIC_CASES)
def test_static_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", ABSTRACT_CASES)
def test_abstract_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


@pytest.mark.parametrize("category,text,expected", AMBIGUOUS_CASES)
def test_ambiguous_responses(category, text, expected):
    assert classify_regime(category, text) == expected, f"{category}: {text!r}"


def test_abstract_overrides_kinetic_keyword():
    """'It is a silhouette; the bird flies' → abstract wins because of explicit reject."""
    assert classify_regime("bird", "It is just a silhouette — nothing physical happens.") == "abstract"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_m8d_regime.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify_regime' from 'physical_mode.metrics.pmr'`.

- [ ] **Step 3: Add `CATEGORY_REGIME_KEYWORDS` to lexicons.py**

Append to `src/physical_mode/metrics/lexicons.py`:

```python
# ---------------------------------------------------------------------------
# M8d category-specific regime keywords.
# Used by metrics.pmr.classify_regime to assign one of {kinetic, static,
# abstract, ambiguous} to a model response.
# ---------------------------------------------------------------------------

CATEGORY_REGIME_KEYWORDS: dict[str, dict[str, frozenset[str]]] = {
    "car": {
        "kinetic": frozenset({"driv", "roll", "spe", "moves", "moving", "moved", "race", "accel"}),
        "static":  frozenset({"park", "stop", "stay", "stays", "stayed", "still", "stationary", "display"}),
    },
    "person": {
        "kinetic": frozenset({"walk", "run", "jog", "step", "stride", "moves", "moving", "moved"}),
        "static":  frozenset({"stand", "stays", "still", "stationary", "stand still", "motionless", "frozen"}),
    },
    "bird": {
        "kinetic": frozenset({"fly", "fli", "flew", "flown", "swim", "swam", "soar", "soaring", "waddl", "moves", "moving", "glid"}),
        "static":  frozenset({"perch", "sit", "stays", "still", "stationary", "rests", "rest"}),
    },
}
```

- [ ] **Step 4: Add `classify_regime` to pmr.py**

Append to `src/physical_mode/metrics/pmr.py`:

```python
from .lexicons import CATEGORY_REGIME_KEYWORDS  # noqa: E402  (extending existing imports)


def classify_regime(category: str, text: str) -> str:
    """Classify a free-form response into one of {kinetic, static, abstract, ambiguous}.

    Order of checks:
      1. abstract markers override everything (e.g., "this is just a silhouette,
         the bird flies" → abstract because the explicit reject takes precedence
         over the kinetic keyword).
      2. category-specific kinetic / static stems decided by `_any_stem_hit`.
      3. fallback: ambiguous.

    Categories without an entry in CATEGORY_REGIME_KEYWORDS produce a ValueError
    so M8a categories don't accidentally fall through.
    """
    if category not in CATEGORY_REGIME_KEYWORDS:
        raise ValueError(f"classify_regime called for unsupported category {category!r}")
    if not text:
        return "ambiguous"
    if _any_phrase_hit(text, ABSTRACT_MARKERS):
        return "abstract"
    words = _words(text)
    table = CATEGORY_REGIME_KEYWORDS[category]
    if _any_stem_hit(words, table["kinetic"]):
        return "kinetic"
    if _any_stem_hit(words, table["static"]):
        return "static"
    return "ambiguous"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_m8d_regime.py -v`
Expected: PASS — all parametrized cases green.

- [ ] **Step 6: Run full pytest to confirm no regressions**

Run: `uv run python -m pytest -v`
Expected: All tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/physical_mode/metrics/lexicons.py src/physical_mode/metrics/pmr.py tests/test_m8d_regime.py
git commit -m "$(cat <<'EOF'
feat(m8d): classify_regime keyword classifier (kinetic/static/abstract)

CATEGORY_REGIME_KEYWORDS holds per-category kinetic + static stems
for car, person, bird. classify_regime() checks abstract markers
first (so explicit reject overrides any kinetic verb that follows),
then category-specific stems.

Reuses the existing _any_phrase_hit / _any_stem_hit helpers from
the M8a scoring module — no changes to M8a interfaces.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Write 4 M8d configs

**Files:**
- Create: `configs/m8d_qwen.py`
- Create: `configs/m8d_qwen_label_free.py`
- Create: `configs/m8d_llava.py`
- Create: `configs/m8d_llava_label_free.py`

- [ ] **Step 1: Write `configs/m8d_qwen.py`**

```python
"""M8d — non-ball physical-object categories, Qwen2.5-VL-7B run.

External-validity round 2 (after M8a non-circle synthetic shapes):
does the H7 (label-selects-regime) and H1 (abstraction ramp) pattern
generalize to non-ball categories that have no clean geometric class?

Factorial:
    3 shapes × 4 obj × 2 bg × 2 cue × 2 events × 5 seeds = 480 stimuli
    × 3 label roles × 1 open prompt                      = 1440 inferences

Shapes:    car, person, bird
Obj:       line, filled, shaded, textured
Bg:        blank, ground
Cue:       none, both       (= cast_shadow + motion_arrow)
Events:    fall + horizontal — natural event per category is
           horizontal (drives/walks/flies); fall is stress test.
           H1 ramp is union; H7 paired-delta is horizontal subset.
Seeds:     5

Labels per shape (LABELS_BY_SHAPE):
    car    : (car,    silhouette,   figurine)
    person : (person, stick figure, statue)
    bird   : (bird,   silhouette,   duck)

cfg.labels lists role names; run.py resolves to literal labels via
LABELS_BY_SHAPE.

Sampling settings match M8a so paired-delta is comparable to the
M8a circle/square/triangle/hexagon/polygon baseline.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_qwen",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
```

- [ ] **Step 2: Write `configs/m8d_qwen_label_free.py`**

```python
"""M8d label-free pass on Qwen2.5-VL-7B.

Pairs with `configs/m8d_qwen.py` to give the (label, _nolabel)
paired-delta needed for H-encoder-saturation cross-category test.

Same factorial as the labeled config; uses `_nolabel` and
`open_no_label` prompt variant.
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_qwen_label_free",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
```

- [ ] **Step 3: Write `configs/m8d_llava.py`**

```python
"""M8d on LLaVA-1.5-7B, labeled.

Cross-model arm — pairs with M8a's LLaVA labeled run for the
(model, shape, paired-delta) heatmap (M9 input).
"""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_llava",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("physical", "abstract", "exotic"),
    prompt_variants=("open",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
```

- [ ] **Step 4: Write `configs/m8d_llava_label_free.py`**

```python
"""M8d label-free on LLaVA-1.5-7B."""

from __future__ import annotations

from physical_mode.config import EvalConfig, FactorialSpec

CONFIG = EvalConfig(
    run_name="m8d_llava_label_free",
    model_id="llava-hf/llava-1.5-7b-hf",
    torch_dtype="bfloat16",
    max_new_tokens=96,
    temperature=0.7,
    top_p=0.95,
    factorial=FactorialSpec(
        shapes=("car", "person", "bird"),
        object_levels=("line", "filled", "shaded", "textured"),
        bg_levels=("blank", "ground"),
        cue_levels=("none", "both"),
        event_templates=("fall", "horizontal"),
        seeds_per_cell=5,
    ),
    labels=("_nolabel",),
    prompt_variants=("open_no_label",),
    capture_lm_layers=None,
    capture_lm_attentions=False,
    random_seed=42,
)
```

- [ ] **Step 5: Verify configs load without error**

Run:
```bash
uv run python -c "
import importlib.util
for cfg in ['configs/m8d_qwen.py', 'configs/m8d_qwen_label_free.py', 'configs/m8d_llava.py', 'configs/m8d_llava_label_free.py']:
    spec = importlib.util.spec_from_file_location('m', cfg)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    print(f'{cfg}: shapes={m.CONFIG.factorial.shapes} labels={m.CONFIG.labels} prompts={m.CONFIG.prompt_variants}')
"
```
Expected: 4 lines confirming each config loads with the expected `shapes`, `labels`, `prompts` values.

- [ ] **Step 6: Commit**

```bash
git add configs/m8d_qwen.py configs/m8d_qwen_label_free.py configs/m8d_llava.py configs/m8d_llava_label_free.py
git commit -m "$(cat <<'EOF'
feat(m8d): four configs (Qwen + LLaVA, labeled + label-free)

Compact factorial — 3 shapes x 4 obj x 2 bg x 2 cue x 2 events x
5 seeds = 480 stimuli; 1440 labeled + 480 label-free inferences
per model. Sampling matches M8a (T=0.7, top_p=0.95).

Two-event grid (fall + horizontal) lets H1 ramp use the union and
H7 paired-delta use the horizontal subset (natural event for
car/person/bird).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Generate stimuli

**Files:**
- Use existing: `scripts/01_generate_stimuli.py`
- Output: `inputs/m8d_qwen_<ts>_<hash>/`

- [ ] **Step 1: Generate the M8d stimulus set**

The factorial is the same across all four configs — generate once and reuse.

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && CUDA_VISIBLE_DEVICES=1 uv run python scripts/01_generate_stimuli.py --config configs/m8d_qwen.py`
Expected: progress bar `Rendering 480 stimuli` followed by `Wrote 480 stimuli to inputs/m8d_qwen_<timestamp>_<hash>`. Wall-clock ~30 s.

- [ ] **Step 2: Inspect a sample of generated stimuli**

Run:
```bash
M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
echo "M8D_DIR=$M8D_DIR"
ls "$M8D_DIR/images/" | head -10
ls "$M8D_DIR/images/" | wc -l
```
Expected: STDOUT shows `480` and a sample of filenames like `car_line_blank_none_fall_000.png`, `bird_textured_ground_both_horizontal_004.png`.

- [ ] **Step 3: Visually inspect 3 cells (one per category)**

Use `Read` tool to view:
- `inputs/m8d_qwen_*/images/car_textured_ground_both_horizontal_000.png` (textured car with ground + cue)
- `inputs/m8d_qwen_*/images/person_filled_blank_none_horizontal_000.png` (filled person, blank, no cue)
- `inputs/m8d_qwen_*/images/bird_line_ground_both_horizontal_000.png` (line bird with ground + cue)

Expected: each category recognizable; cues (shadow + arrow) visible where present; no obvious rendering bug.

- [ ] **Step 4: No commit needed (inputs are gitignored)**

---

## Task 9: Smoke test all 4 M8d configs (limit=5)

**Files:**
- Use existing: `scripts/02_run_inference.py`

- [ ] **Step 1: Smoke Qwen labeled config**

Run:
```bash
cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation
M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
echo "Smoking against $M8D_DIR"
CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py --config configs/m8d_qwen.py --stimulus-dir "$M8D_DIR" --limit 5
```
Expected: `15` inferences (5 stimuli × 3 labels), wall-clock ~30 s after model load. Final `Wrote 15 predictions to outputs/m8d_qwen_*`. Spot-check `predictions.csv` — `raw_text` column contains plausible responses naming the category.

- [ ] **Step 2: Smoke Qwen label-free config**

Run: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py --config configs/m8d_qwen_label_free.py --stimulus-dir "$M8D_DIR" --limit 5`
Expected: 5 inferences, all with `label='_nolabel'` and `prompt_variant='open_no_label'`.

- [ ] **Step 3: Smoke LLaVA labeled config**

Run: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py --config configs/m8d_llava.py --stimulus-dir "$M8D_DIR" --limit 5`
Expected: 15 inferences, model loads as `llava-hf/llava-1.5-7b-hf` (~15 s extra on first load due to download if not cached).

- [ ] **Step 4: Smoke LLaVA label-free config**

Run: `CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py --config configs/m8d_llava_label_free.py --stimulus-dir "$M8D_DIR" --limit 5`
Expected: 5 inferences.

- [ ] **Step 5: Clean up smoke output dirs**

```bash
# Optional: remove the smoke outputs so they don't clutter outputs/
ls outputs/m8d_*_2026* | head
# After confirming they look like smoke (small N), remove or keep; they're gitignored either way.
```

---

## Task 10: Write `m8d_run_all.sh` and execute full M8d run

**Files:**
- Create: `scripts/m8d_run_all.sh`
- Output: 4 run dirs under `outputs/m8d_*_2026*/`

- [ ] **Step 1: Write the runner**

Create `scripts/m8d_run_all.sh`:

```bash
#!/bin/bash
# Run all four M8d inference configs in sequence on GPU 1.
# Each model is loaded fresh per config (small overhead, simpler memory profile).

set -euo pipefail

cd "$(dirname "$0")/.."

M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
echo "===== M8D_DIR: $M8D_DIR ====="

LOG=outputs/m8d_run_all.log
: > "$LOG"

run() {
    local cfg=$1
    echo "----- $(date -u +%H:%M:%S) starting $cfg -----" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py \
        --config "$cfg" --stimulus-dir "$M8D_DIR" 2>&1 | tee -a "$LOG"
    echo "----- $(date -u +%H:%M:%S) done with $cfg -----" | tee -a "$LOG"
}

run configs/m8d_qwen.py
run configs/m8d_qwen_label_free.py
run configs/m8d_llava.py
run configs/m8d_llava_label_free.py

echo "===== ALL DONE $(date -u +%H:%M:%S) =====" | tee -a "$LOG"
```

Make executable: `chmod +x scripts/m8d_run_all.sh`

- [ ] **Step 2: Sanity-check the script**

Run: `head -5 scripts/m8d_run_all.sh && ls -l scripts/m8d_run_all.sh`
Expected: confirms first 5 lines + executable bit set.

- [ ] **Step 3: Execute the full run**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && bash scripts/m8d_run_all.sh`
Expected wall-clock: ~52-65 minutes total (1440 + 480 + 1440 + 480 = 3840 inferences at ~1 it/s on H200 GPU 1).

While running, the log goes to both stdout and `outputs/m8d_run_all.log`. Each config produces a separate `outputs/m8d_<cfg>_<ts>_<hash>/` directory with `predictions.{jsonl,parquet,csv}` and `run_meta.json`.

If a config fails (OOM / hf-download / network), inspect the log and re-run only the failing config:
```bash
M8D_DIR=$(ls -td inputs/m8d_qwen_* | head -1)
CUDA_VISIBLE_DEVICES=1 uv run python scripts/02_run_inference.py --config configs/m8d_<failed>.py --stimulus-dir "$M8D_DIR"
```

- [ ] **Step 4: Verify outputs**

Run:
```bash
ls -td outputs/m8d_qwen_2026* outputs/m8d_qwen_label_free_2026* outputs/m8d_llava_2026* outputs/m8d_llava_label_free_2026* | head -8
for d in $(ls -td outputs/m8d_*_2026* | head -4); do
    echo -n "$d: "
    wc -l "$d/predictions.jsonl"
done
```
Expected: 4 dirs (one per config). Line counts:
- m8d_qwen: 1440 lines
- m8d_qwen_label_free: 480
- m8d_llava: 1440
- m8d_llava_label_free: 480

- [ ] **Step 5: Commit the runner script and log**

```bash
git add scripts/m8d_run_all.sh
# outputs/ is gitignored
git commit -m "$(cat <<'EOF'
feat(m8d): m8d_run_all.sh — sequential 4-config runner on GPU 1

Mirrors scripts/m8a_run_all.sh: picks the most recent m8d_qwen_*
input dir, runs each config in sequence with logging to
outputs/m8d_run_all.log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Write `m8d_analyze.py` and run analysis

**Files:**
- Create: `scripts/m8d_analyze.py`
- Output: `outputs/m8d_summary.{csv,parquet}` — concatenated predictions across the 4 configs with PMR/GAR/regime columns added.

- [ ] **Step 1: Write the analyzer**

Create `scripts/m8d_analyze.py`:

```python
"""M8d analysis driver.

Concatenates predictions from the four M8d run dirs, scores PMR / GAR /
regime, and emits per-(model, shape, label) rolled-up tables:
  - PMR by (shape, model, label)
  - GAR by (shape, model, label)        (only on `fall + ground` cells)
  - paired delta: PMR(label) - PMR(_nolabel)
  - regime distribution by (shape, model, label) on horizontal subset

Outputs to `outputs/m8d_summary/` (csv + parquet).

Usage:
    uv run python scripts/m8d_analyze.py
    uv run python scripts/m8d_analyze.py --run-dirs outputs/m8d_qwen_2026... outputs/m8d_qwen_label_free_2026... ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from physical_mode.metrics.pmr import (
    classify_regime,
    score_abstract_reject,
    score_gar,
    score_hold_still,
    score_pmr,
)


def _model_short(model_id: str) -> str:
    if "Qwen" in model_id:
        return "qwen"
    if "llava" in model_id.lower():
        return "llava"
    if "InternVL" in model_id or "internvl" in model_id.lower():
        return "internvl3"
    return model_id.split("/")[-1]


def _autodetect_run_dirs() -> list[Path]:
    """Pick the most-recent run dir per (run_name) pattern."""
    base = Path("outputs")
    expected = ("m8d_qwen", "m8d_qwen_label_free", "m8d_llava", "m8d_llava_label_free")
    out = []
    for run_name in expected:
        # Sort by mtime, take newest matching dir.
        candidates = sorted(
            (p for p in base.glob(f"{run_name}_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no run dir matching outputs/{run_name}_*")
        out.append(candidates[0])
    return out


def _load(run_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(run_dir / "predictions.parquet")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    df["model"] = _model_short(meta["model_id"])
    df["run_dir"] = str(run_dir)
    return df


def _score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pmr"] = out["raw_text"].map(score_pmr)
    out["hold_still"] = out["raw_text"].map(score_hold_still)
    out["abstract_reject"] = out["raw_text"].map(score_abstract_reject)
    out["gar"] = [
        score_gar(t, e, b)
        for t, e, b in zip(out["raw_text"], out["event_template"], out["bg_level"])
    ]
    out["regime"] = [
        classify_regime(s, t) if s in {"car", "person", "bird"} else None
        for s, t in zip(out["shape"], out["raw_text"])
    ]
    return out


def _paired_delta(df_labeled: pd.DataFrame, df_nolabel: pd.DataFrame) -> pd.DataFrame:
    """For each (model, shape, label, sample_id), compute PMR(label) − PMR(_nolabel)."""
    keys = ["model", "shape", "sample_id"]
    base = df_nolabel.set_index(keys)["pmr"].rename("pmr_nolabel")
    rows = []
    for label in df_labeled["label"].unique():
        sub = df_labeled[df_labeled["label"] == label].set_index(keys)
        merged = sub.join(base, how="inner")
        merged["delta"] = merged["pmr"] - merged["pmr_nolabel"]
        agg = merged.reset_index().groupby(["model", "shape"])["delta"].mean().reset_index()
        agg["label"] = label
        rows.append(agg)
    return pd.concat(rows).reset_index(drop=True)[["model", "shape", "label", "delta"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", nargs="*", default=None, help="If omitted, autodetect from outputs/")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/m8d_summary"))
    args = p.parse_args()

    run_dirs = [Path(s) for s in args.run_dirs] if args.run_dirs else _autodetect_run_dirs()
    print("Run dirs:")
    for d in run_dirs:
        print(f"  {d}")

    dfs = [_load(d) for d in run_dirs]
    df = pd.concat(dfs, ignore_index=True)
    df = _score(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "predictions_scored.parquet", index=False)
    df.to_csv(args.out_dir / "predictions_scored.csv", index=False)
    print(f"Wrote {len(df)} scored predictions to {args.out_dir}")

    # Summary tables.
    pmr_by_shape_model_label = (
        df.groupby(["model", "shape", "label", "object_level"], dropna=False)["pmr"]
        .mean()
        .reset_index()
    )
    pmr_by_shape_model_label.to_csv(args.out_dir / "pmr_by_shape_model_label.csv", index=False)
    print(f"Wrote PMR summary: {len(pmr_by_shape_model_label)} rows")

    # GAR (defined only on fall + ground/scene; NaN otherwise).
    gar_by_shape_model_label = (
        df.dropna(subset=["gar"])
        .groupby(["model", "shape", "label"], dropna=False)["gar"]
        .mean()
        .reset_index()
    )
    gar_by_shape_model_label.to_csv(args.out_dir / "gar_by_shape_model_label.csv", index=False)
    print(f"Wrote GAR summary: {len(gar_by_shape_model_label)} rows")

    # Paired delta vs label-free.
    df_labeled = df[df["prompt_variant"] == "open"].copy()
    df_nolabel = df[df["prompt_variant"] == "open_no_label"].copy()
    paired = _paired_delta(df_labeled, df_nolabel)
    paired.to_csv(args.out_dir / "paired_delta_vs_nolabel.csv", index=False)
    print(f"Wrote paired-delta: {len(paired)} rows")

    # Regime distribution on horizontal subset (where regime classifier is most informative).
    horiz = df[(df["event_template"] == "horizontal") & (df["regime"].notna())].copy()
    regime_dist = (
        horiz.groupby(["model", "shape", "label", "regime"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    regime_dist["share"] = regime_dist.groupby(["model", "shape", "label"])["n"].transform(
        lambda s: s / s.sum()
    )
    regime_dist.to_csv(args.out_dir / "regime_distribution_horizontal.csv", index=False)
    print(f"Wrote regime distribution (horizontal subset): {len(regime_dist)} rows")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run analysis on real outputs**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && uv run python scripts/m8d_analyze.py`
Expected:
- STDOUT: 4 run dirs printed, then "Wrote N scored predictions" with N = 3840.
- 5 files under `outputs/m8d_summary/`: `predictions_scored.{parquet,csv}`, `pmr_by_shape_model_label.csv`, `gar_by_shape_model_label.csv`, `paired_delta_vs_nolabel.csv`, `regime_distribution_horizontal.csv`.

- [ ] **Step 3: Spot-check the paired delta table**

Run:
```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/m8d_summary/paired_delta_vs_nolabel.csv')
print(df.pivot_table(index=['model','shape'], columns='label', values='delta'))
"
```
Expected: a small table with rows for each (model, shape) pair and columns for each label. The expected pattern (per design doc):
- LLaVA `physical` columns positive (~+0.2 to +0.5).
- Qwen close to zero or negative across labels (saturation-flat).

This is the headline result for H7 cross-category — record the table in step 7.

- [ ] **Step 4: Spot-check the regime distribution**

Run:
```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/m8d_summary/regime_distribution_horizontal.csv')
print(df.pivot_table(index=['model','shape','label'], columns='regime', values='share', fill_value=0))
"
```
Expected: per-(model, shape, label) regime shares. The H7 prediction:
- `physical` label → mostly `kinetic`.
- `abstract` label → some `abstract` share.
- `exotic` label → shifted regime mix (e.g., `figurine` → more `static`; `duck` → mixed `kinetic`).

If `exotic` regime distribution doesn't differ from `physical`, that's the "weak exotic label" finding flagged in the spec §2.1 — record it in the insight doc.

- [ ] **Step 5: Commit the analyzer**

```bash
git add scripts/m8d_analyze.py
git commit -m "$(cat <<'EOF'
feat(m8d): m8d_analyze.py — roll-up + paired-delta + regime tables

Concatenates the four M8d run dirs, scores PMR/GAR/regime, and
emits five summary CSVs to outputs/m8d_summary/. Auto-detects the
most-recent run dir per config when --run-dirs is omitted.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Hand-annotate 50 stimuli for regime classifier validation

**Files:**
- Create: `scripts/m8d_hand_annotate.py`
- Create: `outputs/m8d_summary/hand_annotation.csv`

**Goal:** Sample 50 responses (stratified by model × shape × label) and hand-label the regime, then compute false-pos/neg rates of the keyword classifier.

- [ ] **Step 1: Write the annotation sampler**

Create `scripts/m8d_hand_annotate.py`:

```python
"""Sample 50 M8d responses for hand-annotation of the regime classifier.

Outputs a CSV with `auto_regime` (from classify_regime), an empty
`hand_regime` column for the human to fill in, and the raw text +
metadata. After hand-annotation, run with --validate to compute the
false-pos / false-neg rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HAND_CSV = Path("outputs/m8d_summary/hand_annotation.csv")


def sample(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Stratified sample across (model, shape, label, prompt_variant=open)."""
    df = pd.read_parquet("outputs/m8d_summary/predictions_scored.parquet")
    df = df[df["prompt_variant"] == "open"].copy()
    # ~24 cells (2 models × 3 shapes × 4 labels including _nolabel-equivalent), 2 per cell.
    cells = df.groupby(["model", "shape", "label"], dropna=False)
    rng = pd.Series(range(len(df)), index=df.index)
    rows = []
    for _, group in cells:
        take = group.sample(n=min(2, len(group)), random_state=seed)
        rows.append(take)
    out = pd.concat(rows).head(n).reset_index(drop=True)
    out = out[["model", "shape", "label", "object_level", "bg_level", "cue_level", "event_template", "raw_text", "regime"]].copy()
    out = out.rename(columns={"regime": "auto_regime"})
    out["hand_regime"] = ""
    out["notes"] = ""
    return out


def validate() -> None:
    df = pd.read_csv(HAND_CSV)
    if df["hand_regime"].isna().any() or (df["hand_regime"].astype(str).str.strip() == "").any():
        n = (df["hand_regime"].astype(str).str.strip() == "").sum() + df["hand_regime"].isna().sum()
        raise ValueError(f"{n} rows are not hand-annotated yet")
    df["match"] = (df["auto_regime"].astype(str).str.strip() == df["hand_regime"].astype(str).str.strip())
    err = (~df["match"]).mean()
    print(f"Hand annotation N = {len(df)}")
    print(f"Disagreement rate (errors / N) = {err:.3f}  ({(~df['match']).sum()}/{len(df)})")
    confusion = pd.crosstab(df["hand_regime"], df["auto_regime"], margins=True)
    print("\nConfusion matrix (rows = hand, cols = auto):")
    print(confusion)
    if err >= 0.15:
        print("\n⚠  Disagreement >= 15 % — refine CATEGORY_REGIME_KEYWORDS and re-run.")
    else:
        print("\n✓ Disagreement < 15 % — classifier passes pre-registered threshold.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true", help="Compute false-pos/neg rate after annotation")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.validate:
        validate()
    else:
        df = sample(n=args.n, seed=args.seed)
        HAND_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(HAND_CSV, index=False)
        print(f"Wrote {len(df)} rows to {HAND_CSV}")
        print("Edit the `hand_regime` column with one of: kinetic, static, abstract, ambiguous")
        print(f"Then run: uv run python scripts/m8d_hand_annotate.py --validate")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sample 50 stimuli**

Run: `uv run python scripts/m8d_hand_annotate.py --n 50`
Expected: STDOUT `Wrote 50 rows to outputs/m8d_summary/hand_annotation.csv` + instruction to edit.

- [ ] **Step 3: Hand-annotate**

Open `outputs/m8d_summary/hand_annotation.csv` (e.g., in VS Code or LibreOffice). For each row, read `raw_text` and write one of `kinetic`, `static`, `abstract`, `ambiguous` in the `hand_regime` column. Use `notes` for ambiguous cases. Aim for ~30 minutes total.

Acceptance rules:
- `kinetic` if the response describes movement (drives, walks, flies, runs, swims, glides, etc.) for the depicted category.
- `static` if the response says the object stays/stands/perches/parks/sits.
- `abstract` if the response explicitly rejects physical interpretation ("just a drawing", "abstract shape").
- `ambiguous` for anything else.

- [ ] **Step 4: Validate**

Run: `uv run python scripts/m8d_hand_annotate.py --validate`
Expected: STDOUT prints disagreement rate + confusion matrix. If disagreement < 15 %, accept and proceed. If ≥ 15 %, refine `CATEGORY_REGIME_KEYWORDS` in `lexicons.py`, re-run `m8d_analyze.py` (regenerates `regime` column), and re-validate. Iterate until threshold met.

- [ ] **Step 5: Commit annotation script + filled CSV**

```bash
git add scripts/m8d_hand_annotate.py outputs/m8d_summary/hand_annotation.csv
git commit -m "$(cat <<'EOF'
feat(m8d): regime classifier hand-annotation (50 stim)

Stratified 50-stim sample across model x shape x label cells.
Each row hand-annotated with kinetic/static/abstract/ambiguous;
disagreement rate vs keyword classifier reported by --validate.

Pre-registered threshold: disagreement < 15 %.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Generate M8d figures

**Files:**
- Create: `scripts/m8d_figures.py`
- Output: 6 PNGs under `docs/figures/m8d_*.png`

- [ ] **Step 1: Write figure generator**

Create `scripts/m8d_figures.py`:

```python
"""Generate M8d figures from outputs/m8d_summary/ tables.

Outputs:
  - m8d_full_scene_samples.png — 3x4 grid of full stimulus scenes per (shape, mode)
  - m8d_pmr_ramp.png — H1 ramp per category x model
  - m8d_pmr_by_role.png — PMR(physical) vs PMR(abstract) vs PMR(exotic) per category x model
  - m8d_paired_delta.png — paired-delta per category x model x label
  - m8d_regime_distribution.png — regime share per (category, label, model) on horizontal subset

(m8d_shape_grid.png is generated separately by m8d_spot_check.py.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from physical_mode.stimuli.primitives import blank_canvas, draw_object
from physical_mode.stimuli.scenes import render_scene
from physical_mode.config import StimulusRow

FIG_DIR = Path("docs/figures")
SUMMARY_DIR = Path("outputs/m8d_summary")
SHAPES = ("car", "person", "bird")
MODES = ("line", "filled", "shaded", "textured")
MODELS = ("qwen", "llava")
ROLES = ("physical", "abstract", "exotic")


def fig_full_scene_samples() -> None:
    """3x4 grid of one full stimulus scene per (shape, mode), with ground + cue=both."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for i, shape in enumerate(SHAPES):
        for j, mode in enumerate(MODES):
            row = StimulusRow(
                sample_id=f"{shape}_{mode}_ground_both_horizontal_000",
                event_template="horizontal",
                object_level=mode,
                bg_level="ground",
                cue_level="both",
                seed=42,
                shape=shape,
            )
            img = render_scene(row, size=512)
            axes[i, j].imshow(np.asarray(img))
            axes[i, j].set_title(f"{shape} / {mode}")
            axes[i, j].axis("off")
    fig.suptitle("M8d full stimulus scenes (ground + cue=both, event=horizontal, seed=42)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "m8d_full_scene_samples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_pmr_ramp() -> None:
    """PMR vs object_level (line/filled/shaded/textured) per (model, shape).
    Averages over labels except `_nolabel`."""
    df = pd.read_csv(SUMMARY_DIR / "pmr_by_shape_model_label.csv")
    df = df[df["label"] != "_nolabel"].copy()
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(15, 4), sharey=True)
    for i, shape in enumerate(SHAPES):
        for model in MODELS:
            sub = df[(df["model"] == model) & (df["shape"] == shape)]
            agg = sub.groupby("object_level")["pmr"].mean().reindex(MODES).reset_index()
            axes[i].plot(MODES, agg["pmr"], marker="o", label=model)
        axes[i].set_title(f"{shape}")
        axes[i].set_xlabel("object_level")
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    axes[0].set_ylabel("PMR")
    fig.suptitle("M8d H1 ramp — PMR vs object abstraction (labels averaged over physical/abstract/exotic)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "m8d_pmr_ramp.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_pmr_by_role() -> None:
    """PMR per (model, shape, role) — horizontal subset only."""
    scored = pd.read_parquet(SUMMARY_DIR / "predictions_scored.parquet")
    horiz = scored[(scored["event_template"] == "horizontal") & (scored["prompt_variant"] == "open")].copy()
    # Map literal label back to role.
    from physical_mode.inference.prompts import LABELS_BY_SHAPE
    role_idx = {r: i for i, r in enumerate(ROLES)}
    def label_to_role(shape, label):
        if shape not in LABELS_BY_SHAPE:
            return None
        triplet = LABELS_BY_SHAPE[shape]
        for r, l in zip(ROLES, triplet):
            if l == label:
                return r
        return None
    horiz["role"] = [label_to_role(s, l) for s, l in zip(horiz["shape"], horiz["label"])]
    horiz = horiz[horiz["role"].notna()].copy()
    pmr = horiz.groupby(["model", "shape", "role"], dropna=False)["pmr"].mean().reset_index()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4), sharey=True)
    width = 0.25
    x = np.arange(len(SHAPES))
    for i, model in enumerate(MODELS):
        for j, role in enumerate(ROLES):
            sub = pmr[(pmr["model"] == model) & (pmr["role"] == role)]
            sub = sub.set_index("shape").reindex(SHAPES).reset_index()
            axes[i].bar(x + j * width, sub["pmr"], width, label=role)
        axes[i].set_xticks(x + width)
        axes[i].set_xticklabels(SHAPES)
        axes[i].set_title(f"{model} — horizontal subset")
        axes[i].set_ylim(0, 1.05)
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    axes[0].set_ylabel("PMR")
    fig.suptitle("M8d H7 — PMR by role (horizontal subset)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "m8d_pmr_by_role.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_paired_delta() -> None:
    """Paired delta = PMR(label) − PMR(_nolabel) per (model, shape, label)."""
    df = pd.read_csv(SUMMARY_DIR / "paired_delta_vs_nolabel.csv")
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4), sharey=True)
    width = 0.25
    x = np.arange(len(SHAPES))
    for i, model in enumerate(MODELS):
        sub = df[df["model"] == model]
        for j, label_role in enumerate(ROLES):
            # Recover the literal label for each shape's role.
            from physical_mode.inference.prompts import LABELS_BY_SHAPE
            literals = [LABELS_BY_SHAPE[s][ROLES.index(label_role)] for s in SHAPES]
            ys = []
            for shape, lit in zip(SHAPES, literals):
                row = sub[(sub["shape"] == shape) & (sub["label"] == lit)]
                ys.append(row["delta"].mean() if len(row) else float("nan"))
            axes[i].bar(x + j * width, ys, width, label=label_role)
        axes[i].axhline(0, color="black", linewidth=0.7)
        axes[i].set_xticks(x + width)
        axes[i].set_xticklabels(SHAPES)
        axes[i].set_title(model)
        axes[i].grid(alpha=0.3)
        axes[i].legend()
    axes[0].set_ylabel("PMR(label) − PMR(_nolabel)")
    fig.suptitle("M8d paired delta — H-encoder-saturation cross-category test", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "m8d_paired_delta.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_regime_distribution() -> None:
    """Regime share per (model, shape, role) on horizontal subset."""
    df = pd.read_csv(SUMMARY_DIR / "regime_distribution_horizontal.csv")
    from physical_mode.inference.prompts import LABELS_BY_SHAPE
    role_for = {(s, lit): r for s, triplet in LABELS_BY_SHAPE.items() for r, lit in zip(ROLES, triplet)}
    df["role"] = [role_for.get((s, l)) for s, l in zip(df["shape"], df["label"])]
    df = df[df["role"].notna()].copy()
    pivot = df.pivot_table(index=["model", "shape", "role"], columns="regime", values="share", fill_value=0.0).reset_index()
    fig, axes = plt.subplots(len(MODELS), len(SHAPES), figsize=(13, 7), sharey=True)
    if len(MODELS) == 1:
        axes = np.array([axes])
    regime_classes = ["kinetic", "static", "abstract", "ambiguous"]
    for i, model in enumerate(MODELS):
        for j, shape in enumerate(SHAPES):
            sub = pivot[(pivot["model"] == model) & (pivot["shape"] == shape)].set_index("role").reindex(ROLES).reset_index()
            bottom = np.zeros(len(ROLES))
            for cls in regime_classes:
                vals = sub[cls].values if cls in sub.columns else np.zeros(len(ROLES))
                axes[i, j].bar(range(len(ROLES)), vals, bottom=bottom, label=cls)
                bottom = bottom + vals
            axes[i, j].set_xticks(range(len(ROLES)))
            axes[i, j].set_xticklabels(ROLES, rotation=20)
            axes[i, j].set_title(f"{model} / {shape}")
            axes[i, j].set_ylim(0, 1.05)
            if j == 0:
                axes[i, j].set_ylabel("regime share")
    axes[0, -1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("M8d regime distribution — horizontal subset", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "m8d_regime_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_full_scene_samples()
    fig_pmr_ramp()
    fig_pmr_by_role()
    fig_paired_delta()
    fig_regime_distribution()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the figure generator**

Run: `uv run python scripts/m8d_figures.py`
Expected: STDOUT lists 5 `wrote docs/figures/m8d_*.png` lines (full_scene_samples, pmr_ramp, pmr_by_role, paired_delta, regime_distribution). Plus `m8d_shape_grid.png` already exists from Task 5.

- [ ] **Step 3: Visually inspect each figure**

Use `Read` tool to view each:
- `docs/figures/m8d_full_scene_samples.png` — full scenes at every (shape, mode) cell, ground + cue visible.
- `docs/figures/m8d_pmr_ramp.png` — line goes up-and-to-right for each (model, shape) for LLaVA; flat / saturated for Qwen (matches H-encoder-saturation prediction).
- `docs/figures/m8d_pmr_by_role.png` — `physical` and `exotic` ≥ `abstract` for LLaVA; Qwen all near saturation.
- `docs/figures/m8d_paired_delta.png` — `physical` and `exotic` bars positive for LLaVA, near-zero for Qwen.
- `docs/figures/m8d_regime_distribution.png` — `physical` mostly kinetic; `abstract` has visible abstract share; `exotic` shifted (figurine → static, statue → static, duck → kinetic but mixed).

- [ ] **Step 4: Commit figures**

```bash
git add scripts/m8d_figures.py docs/figures/m8d_full_scene_samples.png docs/figures/m8d_pmr_ramp.png docs/figures/m8d_pmr_by_role.png docs/figures/m8d_paired_delta.png docs/figures/m8d_regime_distribution.png
git commit -m "$(cat <<'EOF'
feat(m8d): figure generator + 5 figures

Five figures cover H1 ramp (pmr_ramp), H7 PMR-by-role and paired
delta, regime distribution on the horizontal subset, and full
stimulus scene samples. Generated from outputs/m8d_summary/ tables.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Write insight + experiment markdown

**Files:**
- Create: `docs/insights/m8d_non_ball_categories.md`
- Create: `docs/insights/m8d_non_ball_categories_ko.md`
- Create: `docs/experiments/m8d_non_ball_categories.md`
- Create: `docs/experiments/m8d_non_ball_categories_ko.md`

**Goal:** Match the M8a documentation pattern — `docs/insights/` is the deep-dive interpretation; `docs/experiments/` is raw numbers + provenance.

- [ ] **Step 1: Read the M8a templates**

Use `Read` tool to view `docs/insights/m8a_non_circle_shapes.md` and `docs/experiments/m8a_non_circle_shapes.md` to learn the structure and section ordering. Don't copy text — copy the *structure* and write fresh M8d content from the actual numbers in `outputs/m8d_summary/`.

- [ ] **Step 2: Write `docs/insights/m8d_non_ball_categories.md`**

Sections:
1. **Headline** — 1-2 sentences with the key finding (e.g., "Pre-registered scoring: Qwen X/3, LLaVA Y/3 — encoder-saturation prediction holds for non-ball categories").
2. **Setup** — pointer to spec + run dirs + factorial summary.
3. **Hypothesis-by-hypothesis results** — H1 ramp, H7 paired-delta, H7 regime distribution (with the literal numbers).
4. **Per-category notes** — car / person / bird specific findings (e.g., "duck mixed-regime exotic" caveat).
5. **Hypothesis-scorecard updates** — what changes in roadmap §1.3.
6. **Limitations and follow-ups** — weak label points, M8c parallels, encoder-swap test.
7. **Cross-link to figures + raw numbers**.

Length: 250-400 lines, similar to `docs/insights/m8a_non_circle_shapes.md`.

- [ ] **Step 3: Write the Korean translation**

Create `docs/insights/m8d_non_ball_categories_ko.md` — direct translation of the English insight doc. Keep technical terms in English mid-sentence (PMR, paired-delta, kinetic regime, etc.). Match the sentence-by-sentence structure so cross-lookup is straightforward.

- [ ] **Step 4: Write `docs/experiments/m8d_non_ball_categories.md`**

Sections:
1. **Run provenance** — 4 run dirs, model IDs, sampling settings, wall-clock.
2. **Stimulus inventory** — 480 stimuli with the factorial breakdown.
3. **Top-level numbers** — overall PMR per (model × shape × label × prompt_variant).
4. **Paired delta tables** (literal CSV from `outputs/m8d_summary/paired_delta_vs_nolabel.csv`).
5. **Regime distribution tables** (from `regime_distribution_horizontal.csv`).
6. **Hand-annotation validation** — disagreement rate from Task 12.
7. **Confidence intervals / cell counts** where appropriate.

Length: 200-300 lines, focused on numbers (not interpretation — that goes in insights).

- [ ] **Step 5: Write the Korean translation**

Create `docs/experiments/m8d_non_ball_categories_ko.md` — direct translation.

- [ ] **Step 6: Commit**

```bash
git add docs/insights/m8d_non_ball_categories.md docs/insights/m8d_non_ball_categories_ko.md docs/experiments/m8d_non_ball_categories.md docs/experiments/m8d_non_ball_categories_ko.md
git commit -m "$(cat <<'EOF'
docs(m8d): insight + experiment per-milestone deep dive

docs/insights/m8d_non_ball_categories(_ko).md — H1 / H7 / encoder-
saturation results for car/person/bird, hypothesis-scorecard
updates, limitations.

docs/experiments/m8d_non_ball_categories(_ko).md — raw numbers,
paired-delta and regime tables, hand-annotation validation
results.

Bilingual per project rule #6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Write reproduction notebook

**Files:**
- Create: `notebooks/m8d_non_ball_categories.ipynb`

**Goal:** Cell-by-cell reproduction of the M8d analysis from `outputs/m8d_summary/` — matches the existing `notebooks/m8a_non_circle_shapes.ipynb` and `notebooks/demo.ipynb` patterns.

- [ ] **Step 1: Use `m8a_non_circle_shapes.ipynb` as a template**

Run: `cp notebooks/m8a_non_circle_shapes.ipynb notebooks/m8d_non_ball_categories.ipynb`

- [ ] **Step 2: Adapt the notebook for M8d**

Open `notebooks/m8d_non_ball_categories.ipynb` (in Jupyter or VS Code), update each cell:
- Replace `m8a` → `m8d` in summary directory paths.
- Replace shapes from `(circle, square, triangle, hexagon, polygon)` to `(car, person, bird)`.
- Replace `outputs/m8a_summary/` references with `outputs/m8d_summary/`.
- Add a cell that loads `regime_distribution_horizontal.csv` and visualizes regime shares (this is M8d-specific; M8a didn't have a regime classifier).
- Add a markdown cell at the top with the same structure as `m8a` notebooks but referencing `docs/insights/m8d_non_ball_categories.md`.
- Add the hand-annotation validation result as a cell.

- [ ] **Step 3: Execute every cell**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/m8d_non_ball_categories.ipynb --output m8d_non_ball_categories.ipynb`
Expected: every cell executes without error, plots render, no `# TODO` markers left.

- [ ] **Step 4: Open + verify visually**

Run: `uv run jupyter lab notebooks/m8d_non_ball_categories.ipynb`
Expected: notebook opens, all charts display, all markdown narrative is consistent with the insight doc.

- [ ] **Step 5: Commit**

```bash
git add notebooks/m8d_non_ball_categories.ipynb
git commit -m "$(cat <<'EOF'
docs(m8d): cell-by-cell reproduction notebook

Adapts notebooks/m8a_non_circle_shapes.ipynb to M8d's three
categories (car/person/bird) with the regime distribution +
hand-annotation cells added. Per project rule #7 (notebook per
milestone).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Update roadmap

**Files:**
- Modify: `references/roadmap.md`
- Modify: `references/roadmap_ko.md`

- [ ] **Step 1: Update roadmap §2 milestone overview**

Edit `references/roadmap.md`:
- Find the M8c row (currently `▶ PRIORITY 2 (next)`) and confirm the M8d row above it.
- Mark the M8d row's status `▶` → `✅` and date → `2026-04-25` (or actual completion date).
- M8c stays `▶ PRIORITY 2 (next)`.

- [ ] **Step 2: Update roadmap §1.3 hypothesis scorecard**

For each hypothesis affected by M8d, update the "Status" column:
- **H1 ramp**: append "M8d cross-category result — Qwen X/3, LLaVA Y/3 (matches encoder-saturation prediction)".
- **H7 (label-selects-regime)**: append "M8d cross-category result — Qwen X/3, LLaVA Y/3 on horizontal subset; regime distribution per (label, model) emitted".
- **H7-GAR**: append the M8d horizontal-subset GAR numbers.
- **H-encoder-saturation**: append "M8d 3-shape × 2-model = 6 paired-delta points; saturation pattern Qwen near-zero / LLaVA positive replicates".

- [ ] **Step 3: Add roadmap §3 — M8d completion entry**

After the existing M8a section in §3, add an M8d section with the same structure:
- Run dirs (4)
- Headline numbers (PMR, paired-delta, regime)
- Notable per-category findings
- Hypothesis-scorecard impact
- Artifacts list (insight, experiment, figures, notebook)

- [ ] **Step 4: Add roadmap §6 change log entry**

Append:
```markdown
| 2026-04-25 | **M8d complete (non-ball categories)**: 3 shapes (car/person/bird) × Qwen + LLaVA, labeled + label-free arms, ~3840 inferences. Pre-registered strict scoring: Qwen X/3 / LLaVA Y/3. H7 cross-category result + paired-delta confirms H-encoder-saturation cross-category. Regime classifier hand-annotation disagreement Z%. M8c (real photographs) remains ▶ priority 2. | (this commit) |
```

- [ ] **Step 5: Mirror the changes to `references/roadmap_ko.md`**

Translate each updated section to Korean — match the structure sentence-by-sentence.

- [ ] **Step 6: Commit**

```bash
git add references/roadmap.md references/roadmap_ko.md
git commit -m "$(cat <<'EOF'
docs(roadmap): M8d (non-ball categories) complete

Update §2 milestone overview, §1.3 hypothesis scorecard (H1, H7,
H7-GAR, H-encoder-saturation), §3 detailed status (M8d section),
§6 change log. Bilingual update — English canonical, Korean mirror.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: Final integration test + memory update

**Files:**
- Run: full pytest suite
- Possibly modify: project memory if user requests

- [ ] **Step 1: Run the full pytest suite**

Run: `cd /mnt/ddn/prod-runs/thyun.park/src/physical_mode_activation && uv run python -m pytest -v`
Expected: all tests pass — M8a stimulus determinism + PMR scoring tests, plus M8d's `test_m8d_labels.py`, `test_m8d_primitives.py`, `test_m8d_regime.py`. Total ~50+ tests, all green.

- [ ] **Step 2: Run the M8d configs validate-only (no inference)**

Run:
```bash
uv run python -c "
import importlib.util
for cfg in ['configs/m8d_qwen.py', 'configs/m8d_qwen_label_free.py', 'configs/m8d_llava.py', 'configs/m8d_llava_label_free.py']:
    spec = importlib.util.spec_from_file_location('m', cfg)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print(f'{cfg}: total={m.CONFIG.factorial.total()}')
"
```
Expected: all four configs report `total=480`.

- [ ] **Step 3: Verify the figures all exist**

Run: `ls -l docs/figures/m8d_*.png`
Expected: 6 PNGs (`shape_grid`, `full_scene_samples`, `pmr_ramp`, `pmr_by_role`, `paired_delta`, `regime_distribution`).

- [ ] **Step 4: Verify all M8d artifacts present**

Run:
```bash
ls -l \
    docs/insights/m8d_non_ball_categories.md \
    docs/insights/m8d_non_ball_categories_ko.md \
    docs/experiments/m8d_non_ball_categories.md \
    docs/experiments/m8d_non_ball_categories_ko.md \
    notebooks/m8d_non_ball_categories.ipynb \
    references/roadmap.md
```
Expected: all 6 paths exist.

- [ ] **Step 5: Final summary commit (if anything missed earlier)**

Optional — if previous tasks left any uncommitted file (e.g., a `__pycache__` cleanup), this is the catch-all:

```bash
git status
# If anything tracked is uncommitted:
# git add ...
# git commit -m "chore(m8d): final cleanup"
```

- [ ] **Step 6: Tag the milestone (optional, matches pattern from prior milestones)**

If user wants explicit milestone marker:
```bash
git tag -a m8d -m "M8d (non-ball categories) complete — H7 cross-category test, paired-delta encoder-saturation replicated"
```

---

## Self-Review

After writing this plan, I checked it against the spec:

**Spec coverage check (§1.2 success criteria):**
- ✓ Primitives recognizable at every level: Tasks 2-5 (primitives + spot-check VLM verification).
- ✓ H1 ramp + H7 paired-delta + cross-category PASS scoring: Task 11 analyzer + Task 13 figures.
- ✓ Regime distribution per (category, label) cell: Task 11 analyzer (regime column) + Task 13 figure 5.
- ✓ 50-stim hand-annotation < 15 % error: Task 12.
- ✓ Bilingual artifacts: Task 14 + Task 16.
- ✓ Reproduction notebook: Task 15.
- ✓ Roadmap update: Task 16.

**Spec coverage check (§2.1-§2.7 M8d component):**
- ✓ Categories + label triplets: Task 1.
- ✓ Stimulus factorial 3 × 4 × 2 × 2 × 2 × 5: Task 7 configs (FactorialSpec) + Task 8 generation.
- ✓ Two-event grid with horizontal-subset H7 split: Task 11 analyzer + insight doc Task 14.
- ✓ Primitives 12 functions: Tasks 2-4.
- ✓ Regime classifier Hybrid (D): Task 6 (keyword) + Task 12 (hand-annotation).
- ✓ Configs + scripts: Tasks 7, 9, 10, 11, 12, 13.
- ✓ Inference pipeline unchanged: design verified — `run.py::labels_for_row` already routes through `LABELS_BY_SHAPE`, no edits needed.

**Placeholder scan:** no TBD/TODO markers in any task; every code block is concrete.

**Type consistency:**
- `classify_regime(category, text) -> str` matches between Task 6 implementation and Task 11 / Task 12 / Task 13 callers.
- `LABELS_BY_SHAPE` keys (`"car"`, `"person"`, `"bird"`) match across Task 1, Task 7 configs, Task 11 analyzer, Task 13 figures.
- `Shape` literal in `config.py` and `primitives.py` are kept in sync (Task 1).

**Risk register cross-check:**
- Bird/duck primitive recognizability: Task 5 spot-check explicitly catches this before paying for the full inference run.
- Regime classifier miscount: Task 12 gates on disagreement < 15 %.
- Photo license: out of scope for M8d (M8c milestone).
- Two-event factorial cost: budgeted ~52-65 min in Task 10.
