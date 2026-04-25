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
