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
    StimulusRow("shaded_ground_arrow_shadow_fall_000", "fall", "shaded", "ground", "arrow_shadow", 1002),
    StimulusRow("filled_blank_wind_horizontal_000", "horizontal", "filled", "blank", "wind", 1003),
    StimulusRow("textured_scene_arrow_shadow_fall_000", "fall", "textured", "scene", "arrow_shadow", 1004),
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


def test_cue_changes_image():
    base = StimulusRow("x", "fall", "shaded", "ground", "none", 7)
    no_cue = render_scene(base)
    with_arrow = render_scene(StimulusRow("x", "fall", "shaded", "ground", "arrow_shadow", 7))
    assert _png_hash(no_cue) != _png_hash(with_arrow)
