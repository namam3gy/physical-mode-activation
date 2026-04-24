"""Canonical stimulus renderings must be byte-for-byte reproducible."""

from __future__ import annotations

import hashlib
import io

import pytest

from physical_mode.config import StimulusRow
from physical_mode.stimuli.scenes import render_scene


def _png_hash(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


CANON_ROWS = [
    StimulusRow("line_blank_none_fall_000", "fall", "line", "blank", "none", 1000),
    StimulusRow("textured_ground_none_fall_000", "fall", "textured", "ground", "none", 1001),
    StimulusRow("shaded_ground_cast_shadow_fall_000", "fall", "shaded", "ground", "cast_shadow", 1002),
    StimulusRow("filled_blank_motion_arrow_fall_000", "fall", "filled", "blank", "motion_arrow", 1003),
    StimulusRow("textured_scene_both_fall_000", "fall", "textured", "scene", "both", 1004),
    # Legacy pilot values retained for backward compat:
    StimulusRow("shaded_ground_arrow_shadow_fall_000", "fall", "shaded", "ground", "arrow_shadow", 1005),
    StimulusRow("filled_blank_wind_horizontal_000", "horizontal", "filled", "blank", "wind", 1006),
]


@pytest.mark.parametrize("row", CANON_ROWS, ids=[r.sample_id for r in CANON_ROWS])
def test_rendering_is_deterministic(row):
    """Same row rendered twice must produce identical bytes."""
    a = render_scene(row)
    b = render_scene(row)
    assert _png_hash(a) == _png_hash(b), f"non-deterministic rendering for {row.sample_id}"
    assert a.size == (512, 512)


def test_different_object_levels_produce_different_pixels():
    base = StimulusRow("x", "fall", "line", "blank", "none", 42)
    line = render_scene(base)
    filled = render_scene(StimulusRow("x", "fall", "filled", "blank", "none", 42))
    shaded = render_scene(StimulusRow("x", "fall", "shaded", "blank", "none", 42))
    textured = render_scene(StimulusRow("x", "fall", "textured", "blank", "none", 42))
    hashes = {_png_hash(im) for im in (line, filled, shaded, textured)}
    assert len(hashes) == 4, "abstraction levels collapsed to the same image"


def test_cue_split_produces_distinct_images():
    """cast_shadow alone, motion_arrow alone, and both must differ from none and each other."""
    base = StimulusRow("x", "fall", "shaded", "ground", "none", 7)
    no_cue = render_scene(base)
    shadow = render_scene(StimulusRow("x", "fall", "shaded", "ground", "cast_shadow", 7))
    arrow = render_scene(StimulusRow("x", "fall", "shaded", "ground", "motion_arrow", 7))
    both = render_scene(StimulusRow("x", "fall", "shaded", "ground", "both", 7))
    hashes = {_png_hash(im) for im in (no_cue, shadow, arrow, both)}
    assert len(hashes) == 4, "cue_level split did not produce four distinct images"


def test_both_equals_legacy_arrow_shadow():
    """`both` is the full cue stack; `arrow_shadow` (legacy) should render identically."""
    new = render_scene(StimulusRow("x", "fall", "textured", "ground", "both", 99))
    legacy = render_scene(StimulusRow("x", "fall", "textured", "ground", "arrow_shadow", 99))
    assert _png_hash(new) == _png_hash(legacy), "both/arrow_shadow should render the same"
