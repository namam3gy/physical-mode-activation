"""Atomic stimulus drawers. Deterministic given an explicit seed.

Canvas convention: (size, size) RGB, white background (255, 255, 255).
All positions use PIL conventions: (0, 0) is top-left; y grows downward.
"""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

ObjectMode = Literal["line", "filled", "shaded", "textured", "block_stack"]


def blank_canvas(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), (255, 255, 255))


# ---------------------------------------------------------------------------
# Object primitives
# ---------------------------------------------------------------------------


def draw_object(
    img: Image.Image,
    mode: ObjectMode,
    cx: int,
    cy: int,
    radius: int,
    seed: int,
) -> Image.Image:
    if mode == "line":
        return _draw_line_circle(img, cx, cy, radius)
    if mode == "filled":
        return _draw_filled_circle(img, cx, cy, radius)
    if mode == "shaded":
        return _draw_shaded_sphere(img, cx, cy, radius)
    if mode == "textured":
        return _draw_textured_ball(img, cx, cy, radius, seed)
    if mode == "block_stack":
        return _draw_block_stack(img, cx, cy, radius, seed)
    raise ValueError(f"unknown object mode: {mode}")


def _draw_line_circle(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=3, fill=(255, 255, 255))
    return img


def _draw_filled_circle(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 0), width=2, fill=(150, 150, 150))
    return img


def _draw_shaded_sphere(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Light-from-above-left radial gradient — the core 3D cue per Ramachandran 1988."""
    arr = np.asarray(img, dtype=np.float32).copy()
    H, W, _ = arr.shape
    # Light source offset inside the disk, toward top-left.
    light_x = cx - int(0.4 * r)
    light_y = cy - int(0.4 * r)
    # Per-pixel distances (in a bounding box for speed).
    y0, y1 = max(0, cy - r - 2), min(H, cy + r + 2)
    x0, x1 = max(0, cx - r - 2), min(W, cx + r + 2)
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    dist_from_light = np.sqrt((xs - light_x) ** 2 + (ys - light_y) ** 2)
    # Normalize to 0..1 across the possible range (0..2r).
    t = np.clip(dist_from_light / (2.0 * r), 0.0, 1.0)
    # Brightness: near the light is ~235, far side is ~55.
    brightness = (235.0 - 180.0 * t**0.9).astype(np.float32)
    arr[y0:y1, x0:x1][inside] = np.stack([brightness[inside]] * 3, axis=-1)
    # Thin dark outline so the sphere reads against the background.
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(40, 40, 40), width=2)
    return out


def _draw_textured_ball(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Shaded sphere + seam line + discrete texture spots (soccer-ball-like cue)."""
    img = _draw_shaded_sphere(img, cx, cy, r)
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    # Seam: an arc near the vertical mid-line, simulating a 3D meridian.
    d.arc(
        (cx - int(r * 0.6), cy - r, cx + int(r * 0.6), cy + r),
        start=270,
        end=450,
        fill=(40, 40, 40),
        width=2,
    )
    # Scatter small dark spots on the front-lit hemisphere.
    n_spots = 7
    for _ in range(n_spots):
        # Sample inside circle via rejection.
        while True:
            dx = rng.uniform(-r * 0.8, r * 0.8)
            dy = rng.uniform(-r * 0.8, r * 0.8)
            if dx * dx + dy * dy <= (r * 0.7) ** 2:
                break
        px, py = int(cx + dx), int(cy + dy)
        spot_r = rng.randint(4, 8)
        d.ellipse(
            (px - spot_r, py - spot_r, px + spot_r, py + spot_r),
            fill=(70, 50, 35),  # leathery brown
        )
    return img


def _draw_block_stack(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Three stacked cubes — an unambiguous physical-object cue without a ball."""
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    block_w = int(r * 1.2)
    block_h = int(r * 0.9)
    base_bottom = cy + r
    palette = [(180, 100, 60), (60, 120, 160), (180, 170, 60)]
    for i in range(3):
        top = base_bottom - (i + 1) * block_h
        bot = base_bottom - i * block_h
        # Slight horizontal jitter so the stack reads as gravity-under-tension.
        jitter = rng.randint(-3, 3)
        left = cx - block_w // 2 + jitter
        right = cx + block_w // 2 + jitter
        d.rectangle((left, top, right, bot), fill=palette[i % 3], outline=(30, 30, 30), width=2)
    return img


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def draw_ground(img: Image.Image, y: int) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.line(((0, y), (img.width, y)), fill=(0, 0, 0), width=3)
    return img


def draw_scene(img: Image.Image, ground_y: int, seed: int) -> Image.Image:
    """Ground + horizon + a small obstacle on the ground."""
    img = draw_ground(img, ground_y)
    d = ImageDraw.Draw(img)
    # Horizon shading: light blue above ground, light tan below.
    arr = np.asarray(img, dtype=np.float32).copy()
    arr[:ground_y] = arr[:ground_y] * 0.85 + np.array([200, 220, 240], dtype=np.float32) * 0.15
    arr[ground_y + 3 :] = arr[ground_y + 3 :] * 0.85 + np.array([220, 200, 170], dtype=np.float32) * 0.15
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # Small obstacle (triangle "ramp") to the right.
    rng = random.Random(seed)
    obs_x = int(img.width * rng.uniform(0.75, 0.85))
    obs_top = ground_y - rng.randint(30, 55)
    d.polygon(
        [(obs_x - 40, ground_y), (obs_x + 40, ground_y), (obs_x + 40, obs_top)],
        fill=(130, 110, 90),
        outline=(40, 40, 40),
    )
    return img


# ---------------------------------------------------------------------------
# Context cues
# ---------------------------------------------------------------------------


def draw_wind_marks(img: Image.Image, side: Literal["left", "right"], cx: int, cy: int, seed: int) -> Image.Image:
    """Short curved streaks suggesting rightward (or leftward) airflow behind the object."""
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    direction = 1 if side == "right" else -1
    anchor_x = cx - direction * 90
    for i in range(5):
        y_off = rng.randint(-60, 60)
        ax = anchor_x + rng.randint(-10, 10)
        ay = cy + y_off
        length = rng.randint(28, 55)
        for k in range(3):
            offset = k * 6 - 6
            d.arc(
                (ax - length, ay + offset - 5, ax + 5, ay + offset + 5),
                start=340 if direction > 0 else 160,
                end=380 if direction > 0 else 200,
                fill=(120, 120, 120),
                width=2,
            )
    return img


def draw_trajectory_arrow(
    img: Image.Image, from_xy: tuple[int, int], to_xy: tuple[int, int]
) -> Image.Image:
    d = ImageDraw.Draw(img)
    x0, y0 = from_xy
    x1, y1 = to_xy
    d.line(((x0, y0), (x1, y1)), fill=(200, 30, 30), width=4)
    # Arrowhead.
    ang = math.atan2(y1 - y0, x1 - x0)
    head_len = 18
    for s in (+1, -1):
        hx = x1 - head_len * math.cos(ang + s * 0.45)
        hy = y1 - head_len * math.sin(ang + s * 0.45)
        d.line(((x1, y1), (int(hx), int(hy))), fill=(200, 30, 30), width=4)
    return img


def draw_cast_shadow(img: Image.Image, cx: int, cy: int, r: int, ground_y: int) -> Image.Image:
    """Elliptical cast shadow on the ground — anchors the object physically (Kersten et al. 1997)."""
    d = ImageDraw.Draw(img)
    shadow_w = int(r * 1.3)
    shadow_h = int(r * 0.35)
    sy = ground_y + 2
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.ellipse(
        (cx - shadow_w, sy - shadow_h, cx + shadow_w, sy + shadow_h),
        fill=(30, 30, 30, 110),
    )
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
