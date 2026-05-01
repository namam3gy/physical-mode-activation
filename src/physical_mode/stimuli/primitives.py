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
Shape = Literal[
    "circle", "square", "triangle", "hexagon", "polygon",
    "car", "person", "bird", "boat", "fish", "plant",
]


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
    shape: Shape = "circle",
) -> Image.Image:
    """Render an object of the given `shape` at (cx, cy) with the given abstraction `mode`.

    `block_stack` ignores `shape` (kept for backward compat with axis-A level 5).
    """
    if mode == "block_stack":
        return _draw_block_stack(img, cx, cy, radius, seed)

    if shape == "circle":
        if mode == "line":
            return _draw_line_circle(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_circle(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_sphere(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_ball(img, cx, cy, radius, seed)

    if shape == "square":
        if mode == "line":
            return _draw_line_polygon(img, _square_vertices(cx, cy, radius))
        if mode == "filled":
            return _draw_filled_polygon(img, _square_vertices(cx, cy, radius))
        if mode == "shaded":
            return _draw_shaded_cube(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_block(img, cx, cy, radius, seed)

    if shape == "triangle":
        if mode == "line":
            return _draw_line_polygon(img, _triangle_vertices(cx, cy, radius))
        if mode == "filled":
            return _draw_filled_polygon(img, _triangle_vertices(cx, cy, radius))
        if mode == "shaded":
            return _draw_shaded_wedge(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_stone(img, cx, cy, radius, seed, _triangle_vertices(cx, cy, radius))

    if shape == "hexagon":
        if mode == "line":
            return _draw_line_polygon(img, _hexagon_vertices(cx, cy, radius))
        if mode == "filled":
            return _draw_filled_polygon(img, _hexagon_vertices(cx, cy, radius))
        if mode == "shaded":
            return _draw_shaded_hex_prism(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_metal_nut(img, cx, cy, radius, seed)

    if shape == "polygon":
        verts = _irregular_polygon_vertices(cx, cy, radius, seed)
        if mode == "line":
            return _draw_line_polygon(img, verts)
        if mode == "filled":
            return _draw_filled_polygon(img, verts)
        if mode == "shaded":
            return _draw_shaded_polygon(img, cx, cy, radius, verts)
        if mode == "textured":
            return _draw_textured_stone(img, cx, cy, radius, seed, verts)

    if shape == "car":
        if mode == "line":
            return _draw_line_car(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_car(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_car(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_car(img, cx, cy, radius, seed)

    if shape == "person":
        if mode == "line":
            return _draw_line_person(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_person(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_person(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_person(img, cx, cy, radius, seed)

    if shape == "bird":
        if mode == "line":
            return _draw_line_bird(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_bird(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_bird(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_bird(img, cx, cy, radius, seed)

    if shape == "boat":
        if mode == "line":
            return _draw_line_boat(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_boat(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_boat(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_boat(img, cx, cy, radius, seed)

    if shape == "fish":
        if mode == "line":
            return _draw_line_fish(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_fish(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_fish(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_fish(img, cx, cy, radius, seed)

    if shape == "plant":
        if mode == "line":
            return _draw_line_plant(img, cx, cy, radius)
        if mode == "filled":
            return _draw_filled_plant(img, cx, cy, radius)
        if mode == "shaded":
            return _draw_shaded_plant(img, cx, cy, radius)
        if mode == "textured":
            return _draw_textured_plant(img, cx, cy, radius, seed)

    raise ValueError(f"unknown (shape, mode): ({shape}, {mode})")


# ---------------------------------------------------------------------------
# Shape vertex helpers (for polygonal shapes)
# ---------------------------------------------------------------------------


def _square_vertices(cx: int, cy: int, r: int) -> list[tuple[int, int]]:
    s = int(r * 0.95)  # half-side so visual area roughly matches a circle of the same r
    return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]


def _triangle_vertices(cx: int, cy: int, r: int) -> list[tuple[int, int]]:
    # Equilateral, apex up.
    h = int(r * 1.1)
    half = int(r * 1.05)
    return [(cx, cy - h), (cx + half, cy + int(h * 0.55)), (cx - half, cy + int(h * 0.55))]


def _hexagon_vertices(cx: int, cy: int, r: int) -> list[tuple[int, int]]:
    # Flat-top hexagon. Scale so the bounding box ~matches the circle's diameter.
    s = int(r * 1.05)
    pts = []
    for k in range(6):
        ang = math.radians(60 * k)  # 0, 60, 120, ...
        pts.append((cx + int(s * math.cos(ang)), cy + int(s * math.sin(ang))))
    return pts


def _irregular_polygon_vertices(cx: int, cy: int, r: int, seed: int) -> list[tuple[int, int]]:
    """Seeded irregular polygon — between 5 and 7 vertices, jittered radius and angle."""
    rng = random.Random(seed)
    n = rng.choice([5, 6, 7])
    base = rng.uniform(0, 2 * math.pi)
    pts: list[tuple[int, int]] = []
    for k in range(n):
        ang = base + 2 * math.pi * k / n + rng.uniform(-0.18, 0.18)
        rad = r * rng.uniform(0.78, 1.08)
        pts.append((cx + int(rad * math.cos(ang)), cy + int(rad * math.sin(ang))))
    return pts


# ---------------------------------------------------------------------------
# Generic polygonal mode drawers (shared across square / triangle / hexagon /
# polygon for `line` and `filled`).
# ---------------------------------------------------------------------------


def _draw_line_polygon(img: Image.Image, verts: list[tuple[int, int]]) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.polygon(verts, outline=(0, 0, 0), fill=(255, 255, 255))
    # Re-stroke for visible outline width — PIL's polygon outline=1px only.
    closed = verts + [verts[0]]
    for a, b in zip(closed, closed[1:]):
        d.line((a, b), fill=(0, 0, 0), width=3)
    return img


def _draw_filled_polygon(img: Image.Image, verts: list[tuple[int, int]]) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.polygon(verts, outline=(0, 0, 0), fill=(150, 150, 150))
    closed = verts + [verts[0]]
    for a, b in zip(closed, closed[1:]):
        d.line((a, b), fill=(0, 0, 0), width=2)
    return img


# ---------------------------------------------------------------------------
# Directional shading — used by all non-circle `shaded` modes.
# ---------------------------------------------------------------------------


# Light source direction (upper-left): unit vector pointing FROM the surface TO the light.
_LIGHT_DIR = (-1.0, -1.0)
_LIGHT_NORM = math.sqrt(_LIGHT_DIR[0] ** 2 + _LIGHT_DIR[1] ** 2)


def _face_brightness(face_normal_2d: tuple[float, float], base: int = 180, span: int = 70) -> int:
    """Lambert-ish brightness for a 2D face normal.

    `face_normal_2d` is the outward normal of the face (in image plane).
    Faces pointing toward `_LIGHT_DIR` are brightest; faces pointing away are darkest.
    """
    nx, ny = face_normal_2d
    n_norm = math.sqrt(nx * nx + ny * ny) or 1.0
    cos_t = (nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1]) / (n_norm * _LIGHT_NORM)
    # cos_t in [-1, 1]; map to [base - span, base + span/2].
    val = int(base + span * cos_t)
    return max(40, min(235, val))


# ---------------------------------------------------------------------------
# Square: shaded cube + textured wooden block
# ---------------------------------------------------------------------------


def _draw_shaded_cube(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Square front face + parallelogram top face + parallelogram right face (3D cube)."""
    s = int(r * 0.85)  # half-side, slightly smaller to leave room for the projected faces
    depth = int(r * 0.45)
    dx = int(depth * 0.85)  # x-shift of the back face
    dy = -int(depth * 0.55)  # y-shift of the back face (up = negative)

    front = [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]
    top = [
        (cx - s, cy - s),
        (cx - s + dx, cy - s + dy),
        (cx + s + dx, cy - s + dy),
        (cx + s, cy - s),
    ]
    right = [
        (cx + s, cy - s),
        (cx + s + dx, cy - s + dy),
        (cx + s + dx, cy + s + dy),
        (cx + s, cy + s),
    ]

    d = ImageDraw.Draw(img)
    front_b = _face_brightness((0.0, 0.0))  # front face — moderate (no normal toward light)
    top_b = _face_brightness((0.0, -1.0))   # top face — bright
    right_b = _face_brightness((1.0, 0.0))  # right face — dark
    d.polygon(front, fill=(front_b, front_b, front_b), outline=(40, 40, 40))
    d.polygon(top, fill=(top_b, top_b, top_b), outline=(40, 40, 40))
    d.polygon(right, fill=(right_b, right_b, right_b), outline=(40, 40, 40))
    return img


def _draw_textured_block(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Shaded cube + wood-grain lines on the front face."""
    img = _draw_shaded_cube(img, cx, cy, r)
    s = int(r * 0.85)
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    # Tint front face warm-brown.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.polygon(
        [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)],
        fill=(140, 90, 50, 90),
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    # Horizontal wood grain wavy lines on front face.
    for i in range(6):
        y = cy - s + int((2 * i + 1) * s / 6)
        amp = rng.randint(2, 4)
        period = rng.randint(28, 44)
        prev = (cx - s, y)
        for x in range(cx - s + 4, cx + s + 1, 4):
            yy = y + int(amp * math.sin((x - (cx - s)) / period * 2 * math.pi))
            d.line((prev, (x, yy)), fill=(80, 50, 25), width=1)
            prev = (x, yy)
    return img


# ---------------------------------------------------------------------------
# Triangle: shaded wedge
# ---------------------------------------------------------------------------


def _draw_shaded_wedge(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """3D triangular prism — front face + slanted top face suggesting depth."""
    apex_x, apex_y = cx, cy - int(r * 1.1)
    half = int(r * 1.05)
    base_y = cy + int(r * 0.6)
    front = [(apex_x, apex_y), (cx + half, base_y), (cx - half, base_y)]
    # Back-top offset to suggest a wedge (prism) extending back-up-left.
    dx = -int(r * 0.35)
    dy = -int(r * 0.20)
    back_apex = (apex_x + dx, apex_y + dy)
    back_left = (cx - half + dx, base_y + dy)
    back_right = (cx + half + dx, base_y + dy)

    d = ImageDraw.Draw(img)
    front_b = _face_brightness((0.0, 0.0))
    # Top-left slanted face (visible because light is from upper-left).
    top_left_b = _face_brightness((-0.7, -0.7))
    top_right_b = _face_brightness((0.7, -0.7))

    # Draw back face faintly (mostly hidden), then top faces, then front.
    d.polygon([back_apex, back_left, back_right], fill=(110, 110, 110), outline=(40, 40, 40))
    # Top-left face: front_apex → back_apex → back_left → front_left.
    d.polygon(
        [(apex_x, apex_y), back_apex, back_left, (cx - half, base_y)],
        fill=(top_left_b, top_left_b, top_left_b),
        outline=(40, 40, 40),
    )
    # Top-right face: front_apex → back_apex → back_right → front_right.
    d.polygon(
        [(apex_x, apex_y), back_apex, back_right, (cx + half, base_y)],
        fill=(top_right_b, top_right_b, top_right_b),
        outline=(40, 40, 40),
    )
    d.polygon(front, fill=(front_b, front_b, front_b), outline=(40, 40, 40))
    return img


# ---------------------------------------------------------------------------
# Hexagon: shaded hex prism + textured metal nut
# ---------------------------------------------------------------------------


def _draw_shaded_hex_prism(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Hex front face + slight extrusion to upper-left for a hex-prism look."""
    front = _hexagon_vertices(cx, cy, r)
    dx = -int(r * 0.30)
    dy = -int(r * 0.18)
    back = [(x + dx, y + dy) for (x, y) in front]
    d = ImageDraw.Draw(img)
    # Top-facing prism faces (the ones whose outward normal has a -y component).
    for i in range(6):
        v0 = front[i]
        v1 = front[(i + 1) % 6]
        b0 = back[i]
        b1 = back[(i + 1) % 6]
        # Outward normal of the side face (perpendicular to v1-v0, pointing outward).
        ex, ey = v1[0] - v0[0], v1[1] - v0[1]
        nx, ny = ey, -ex  # rotate -90 deg: outward in flat-top hexagon
        # Only draw side faces whose normal has any component toward the light
        # (i.e. cos > 0); this avoids drawing back-side polygons over the front.
        cos_n = (nx * _LIGHT_DIR[0] + ny * _LIGHT_DIR[1])
        if cos_n <= 0:
            continue
        b = _face_brightness((nx, ny))
        d.polygon([v0, v1, b1, b0], fill=(b, b, b), outline=(50, 50, 50))
    # Front face brightness (no normal toward light → moderate).
    fb = _face_brightness((0.0, 0.0))
    d.polygon(front, fill=(fb, fb, fb), outline=(30, 30, 30))
    # Restroke front edges.
    closed = front + [front[0]]
    for a, b in zip(closed, closed[1:]):
        d.line((a, b), fill=(30, 30, 30), width=2)
    return img


def _draw_textured_metal_nut(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Hex prism + metallic radial gradient + central bolt-hole circle."""
    img = _draw_shaded_hex_prism(img, cx, cy, r)
    # Metallic tint on front face: brighter near upper-left.
    arr = np.asarray(img, dtype=np.float32).copy()
    H, W, _ = arr.shape
    front = _hexagon_vertices(cx, cy, r)
    # Create a polygon mask.
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(front, fill=255)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    # Radial-from-light brightness: distance from upper-left light source.
    lx = cx - int(0.6 * r)
    ly = cy - int(0.6 * r)
    ys, xs = np.indices(arr.shape[:2])
    dist = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2)
    t = np.clip(dist / (2.5 * r), 0.0, 1.0)
    bright = (215.0 - 90.0 * t).astype(np.float32)
    # Cool metallic tint (slightly bluer than gray).
    metal = np.stack([bright * 0.95, bright * 0.97, bright * 1.02], axis=-1)
    metal = np.clip(metal, 30, 240)
    arr = arr * (1.0 - m[..., None]) + metal * m[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Central bolt-hole.
    d = ImageDraw.Draw(img)
    hr = int(r * 0.35)
    d.ellipse((cx - hr, cy - hr, cx + hr, cy + hr), fill=(60, 60, 65), outline=(30, 30, 30), width=2)
    # Inner highlight rim on the bolt-hole.
    d.arc((cx - hr, cy - hr, cx + hr, cy + hr), start=180, end=315, fill=(150, 150, 155), width=2)
    return img


# ---------------------------------------------------------------------------
# Polygon: faceted shaded + rocky textured
# ---------------------------------------------------------------------------


def _draw_shaded_polygon(
    img: Image.Image, cx: int, cy: int, r: int, verts: list[tuple[int, int]]
) -> Image.Image:
    """Faceted shading: split the polygon into triangles from the centroid and
    shade each triangle by its outward-edge normal direction. Approximates
    a faceted rock.
    """
    d = ImageDraw.Draw(img)
    n = len(verts)
    for i in range(n):
        v0 = verts[i]
        v1 = verts[(i + 1) % n]
        # Edge midpoint, outward direction relative to centroid.
        mx = (v0[0] + v1[0]) / 2.0
        my = (v0[1] + v1[1]) / 2.0
        ox, oy = mx - cx, my - cy
        b = _face_brightness((ox, oy))
        d.polygon([(cx, cy), v0, v1], fill=(b, b, b), outline=(60, 60, 60))
    # Outline.
    closed = verts + [verts[0]]
    for a, b in zip(closed, closed[1:]):
        d.line((a, b), fill=(30, 30, 30), width=2)
    return img


def _draw_textured_stone(
    img: Image.Image, cx: int, cy: int, r: int, seed: int, verts: list[tuple[int, int]]
) -> Image.Image:
    """Rocky texture — fill polygon with noisy gray + scatter small pits."""
    rng = random.Random(seed)
    H, W = img.height, img.width
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(verts, fill=255)
    m = np.asarray(mask, dtype=np.float32) / 255.0

    # Per-pixel noise modulated by directional shading.
    np_rng = np.random.default_rng(seed)
    noise = np_rng.normal(loc=0.0, scale=18.0, size=(H, W))
    # Directional brightness gradient (light from upper-left).
    ys, xs = np.indices((H, W))
    grad = -((xs - cx) / (1.5 * r)) - ((ys - cy) / (1.5 * r))  # higher = toward light
    grad = np.clip(grad, -1.2, 1.2)
    bright = 145.0 + 50.0 * grad + noise
    bright = np.clip(bright, 50, 230)
    rock = np.stack([bright, bright * 0.97, bright * 0.92], axis=-1)
    arr = np.asarray(img, dtype=np.float32).copy()
    arr = arr * (1.0 - m[..., None]) + rock * m[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Outline.
    d = ImageDraw.Draw(img)
    closed = verts + [verts[0]]
    for a, b in zip(closed, closed[1:]):
        d.line((a, b), fill=(40, 40, 40), width=2)
    # Scatter dark pits.
    for _ in range(12):
        # Sample a random vertex pair and lerp toward centroid.
        i = rng.randrange(len(verts))
        vx, vy = verts[i]
        t = rng.uniform(0.2, 0.85)
        px = int(cx + (vx - cx) * t + rng.randint(-4, 4))
        py = int(cy + (vy - cy) * t + rng.randint(-4, 4))
        pr = rng.randint(3, 6)
        d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(70, 60, 50))
    return img


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
        wheel_radius=wheel_r,
        wheel_left=(wheel_lx, wheel_y),
        wheel_right=(wheel_rx, wheel_y),
        windshield_box=(win_left, win_top, win_left + win_w, win_top + win_h),
    )


def _draw_line_car(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.rectangle(g["body_box"], outline=(0, 0, 0), width=3)
    d.rectangle(g["windshield_box"], outline=(0, 0, 0), width=2)
    wr = g["wheel_radius"]
    for (wx, wy) in (g["wheel_left"], g["wheel_right"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), outline=(0, 0, 0), width=3)
    return img


def _draw_filled_car(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _car_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.rectangle(g["body_box"], fill=(0, 0, 0))
    wr = g["wheel_radius"]
    for (wx, wy) in (g["wheel_left"], g["wheel_right"]):
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
    wr = g["wheel_radius"]
    for (wx, wy) in (g["wheel_left"], g["wheel_right"]):
        d.ellipse((wx - wr, wy - wr, wx + wr, wy + wr), fill=(40, 40, 40), outline=(0, 0, 0), width=2)
        d.ellipse((wx - wr // 2, wy - wr // 2, wx + wr // 2, wy + wr // 2), fill=(80, 80, 80))
    # Windshield: light blue glassy shade.
    d.rectangle(g["windshield_box"], fill=(180, 200, 220), outline=(60, 60, 60), width=1)
    return img


def _draw_textured_car(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish car with body color, glass detail, wheel hubs."""
    # Per-category RNG offset: decouples palette/jitter draws across
    # categories for the same input seed (car=11000, person=22000, bird=33000).
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
    wr = g["wheel_radius"]
    for (wx, wy) in (g["wheel_left"], g["wheel_right"]):
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
    # Per-category RNG offset: decouples palette/jitter draws across
    # categories for the same input seed (car=11000, person=22000, bird=33000).
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
    # Per-category RNG offset: decouples palette/jitter draws across
    # categories for the same input seed (car=11000, person=22000, bird=33000).
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
    feather_color = (
        max(0, body_color[0] - 30),
        max(0, body_color[1] - 30),
        max(0, body_color[2] - 30),
    )
    for _ in range(28):
        sx = bx0 + rng.randint(8, body_w - 8)
        sy = by0 + rng.randint(8, body_h - 8)
        # Inside-ellipse check.
        if ((sx - cx) / (body_w / 2)) ** 2 + ((sy - cy) / (body_h / 2)) ** 2 > 0.85:
            continue
        tx = sx + rng.randint(-8, 8)
        ty = sy + rng.randint(2, 8)
        d.line(((sx, sy), (tx, ty)), fill=feather_color, width=1)
    # Wing chord, drawn beneath the head/beak/eye (after the feather loop, before the head).
    d.line(g["wing"], fill=feather_color, width=3)
    # Head with same base color but slightly lighter.
    head_color = (min(255, body_color[0] + 20), min(255, body_color[1] + 20), min(255, body_color[2] + 20))
    d.ellipse(g["head_box"], fill=head_color, outline=(30, 30, 30), width=2)
    # Beak (warm orange).
    d.polygon(g["beak"], fill=(220, 140, 40), outline=(140, 80, 20))
    # Eye.
    ex, ey = g["eye"]
    d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(0, 0, 0))
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


# ===========================================================================
# C6-orig (2026-05-01) — boat / fish / plant primitives.
# Three new categories chosen to test cross-category H7 generalization beyond
# car / person / bird. Each shape has 4 abstraction levels: line, filled,
# shaded, textured. Geometry intentionally simple so models can recognize at
# every abstraction level.
# ===========================================================================


def _boat_geometry(cx: int, cy: int, r: int) -> dict:
    """Sailboat geometry — hull (trapezoid) + mast (vertical) + sail (triangle)."""
    hull = [
        (cx - int(r * 0.9), cy),
        (cx + int(r * 0.9), cy),
        (cx + int(r * 0.7), cy + int(r * 0.5)),
        (cx - int(r * 0.7), cy + int(r * 0.5)),
    ]
    mast_top = (cx, cy - int(r * 0.95))
    mast_base = (cx, cy)
    sail = [
        (cx, cy - int(r * 0.95)),
        (cx, cy - int(r * 0.1)),
        (cx + int(r * 0.65), cy - int(r * 0.45)),
    ]
    return dict(hull=hull, mast_top=mast_top, mast_base=mast_base, sail=sail)


def _draw_line_boat(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _boat_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.polygon(g["hull"], outline=(0, 0, 0))
    d.line([g["mast_base"], g["mast_top"]], fill=(0, 0, 0), width=3)
    d.polygon(g["sail"], outline=(0, 0, 0))
    return img


def _draw_filled_boat(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _boat_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.polygon(g["hull"], fill=(0, 0, 0))
    d.line([g["mast_base"], g["mast_top"]], fill=(0, 0, 0), width=3)
    d.polygon(g["sail"], fill=(0, 0, 0))
    return img


def _draw_shaded_boat(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Boat with hull gradient (lighter top, darker bottom) + glossy sail."""
    g = _boat_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    # Hull gradient — fake with 3 horizontal strips of decreasing brightness.
    hull_top_y = cy
    hull_bot_y = cy + int(r * 0.5)
    h_height = hull_bot_y - hull_top_y
    n_strips = 16
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        c = int(180 - 100 * t)
        y0 = hull_top_y + int(h_height * (i / n_strips))
        y1 = hull_top_y + int(h_height * ((i + 1) / n_strips))
        # Strip across the hull width, clipped by trapezoid edges (approx).
        edge_t = (y0 - hull_top_y) / max(1, h_height)
        left = cx - int(r * (0.9 - 0.2 * edge_t))
        right = cx + int(r * (0.9 - 0.2 * edge_t))
        d.rectangle((left, y0, right, y1), fill=(c, c, c + 15))
    d.polygon(g["hull"], outline=(40, 40, 40), width=2)
    # Mast and sail.
    d.line([g["mast_base"], g["mast_top"]], fill=(60, 40, 20), width=3)
    d.polygon(g["sail"], fill=(220, 220, 230), outline=(50, 50, 50), width=2)
    # Sail glare strip.
    d.line(
        (cx + 4, cy - int(r * 0.85), cx + int(r * 0.5), cy - int(r * 0.45)),
        fill=(245, 245, 250), width=2,
    )
    return img


def _draw_textured_boat(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish sailboat with painted hull + cloth sail + reflection."""
    rng = random.Random(seed + 44000)
    g = _boat_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    # Hull color: nautical palette (red/blue/green hulls).
    palette = [(160, 30, 30), (30, 60, 130), (30, 110, 70), (200, 150, 30)]
    hull_color = palette[rng.randrange(len(palette))]
    d.polygon(g["hull"], fill=hull_color, outline=(20, 20, 20), width=2)
    # Top hull highlight strip.
    d.line(
        (cx - int(r * 0.85), cy + 4, cx + int(r * 0.85), cy + 4),
        fill=tuple(min(255, c + 60) for c in hull_color), width=2,
    )
    # Mast: brown wood color.
    d.line([g["mast_base"], g["mast_top"]], fill=(110, 70, 30), width=4)
    # Sail: off-white cloth with slight gradient.
    d.polygon(g["sail"], fill=(240, 235, 220), outline=(40, 40, 40), width=1)
    # Sail seam.
    d.line(
        (cx + 4, cy - int(r * 0.85), cx + int(r * 0.55), cy - int(r * 0.4)),
        fill=(190, 180, 160), width=1,
    )
    # Small flag at mast top.
    d.polygon(
        [(cx, cy - int(r * 0.95)), (cx + int(r * 0.2), cy - int(r * 0.85)), (cx, cy - int(r * 0.75))],
        fill=(200, 30, 30),
    )
    return img


def _fish_geometry(cx: int, cy: int, r: int) -> dict:
    """Fish geometry — body ellipse (head left, tail right) + tail triangle + eye."""
    body_box = (cx - int(r * 0.8), cy - int(r * 0.45), cx + int(r * 0.55), cy + int(r * 0.45))
    tail = [
        (cx + int(r * 0.55), cy),
        (cx + int(r * 0.95), cy - int(r * 0.55)),
        (cx + int(r * 0.95), cy + int(r * 0.55)),
    ]
    eye = (cx - int(r * 0.5), cy - int(r * 0.12))
    return dict(body_box=body_box, tail=tail, eye=eye)


def _draw_line_fish(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _fish_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.ellipse(g["body_box"], outline=(0, 0, 0), width=3)
    d.polygon(g["tail"], outline=(0, 0, 0))
    ex, ey = g["eye"]
    d.ellipse((ex - 4, ey - 4, ex + 4, ey + 4), outline=(0, 0, 0), width=2)
    return img


def _draw_filled_fish(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _fish_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.ellipse(g["body_box"], fill=(0, 0, 0))
    d.polygon(g["tail"], fill=(0, 0, 0))
    # Eye in white so it's visible.
    ex, ey = g["eye"]
    d.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill=(255, 255, 255))
    d.ellipse((ex - 2, ey - 2, ex + 2, ey + 2), fill=(0, 0, 0))
    return img


def _draw_shaded_fish(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Fish with horizontal gradient (lighter belly, darker back) + tail shading."""
    g = _fish_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    bx0, by0, bx1, by1 = g["body_box"]
    body_h = by1 - by0
    # Vertical gradient strips on the body (lighter near belly = bottom).
    n_strips = 24
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        c = int(80 + 130 * t)  # back darker top; belly lighter bottom
        y0 = by0 + int(body_h * (i / n_strips))
        y1 = by0 + int(body_h * ((i + 1) / n_strips))
        od.ellipse((bx0, y0 - 2, bx1, y1 + 2), fill=(c, c, c + 20, 255))
    img2 = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d2 = ImageDraw.Draw(img2)
    d2.ellipse(g["body_box"], outline=(40, 40, 40), width=2)
    d2.polygon(g["tail"], fill=(60, 60, 70), outline=(20, 20, 20), width=2)
    ex, ey = g["eye"]
    d2.ellipse((ex - 5, ey - 5, ex + 5, ey + 5), fill=(240, 240, 240), outline=(40, 40, 40), width=1)
    d2.ellipse((ex - 2, ey - 2, ex + 2, ey + 2), fill=(20, 20, 20))
    return img2


def _draw_textured_fish(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish fish with body color + scale dots + fin highlight."""
    rng = random.Random(seed + 55000)
    g = _fish_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    palette = [(180, 90, 50), (90, 150, 180), (150, 130, 60), (130, 60, 130)]
    body_color = palette[rng.randrange(len(palette))]
    d.ellipse(g["body_box"], fill=body_color, outline=(20, 20, 20), width=2)
    # Top highlight strip (back).
    bx0, by0, bx1, by1 = g["body_box"]
    body_h = by1 - by0
    d.ellipse(
        (bx0 + 4, by0 + 4, bx1 - 4, by0 + body_h // 3),
        fill=tuple(max(0, c - 40) for c in body_color),
    )
    # Belly highlight strip (lighter).
    d.ellipse(
        (bx0 + 8, by1 - body_h // 3, bx1 - 8, by1 - 4),
        fill=tuple(min(255, c + 50) for c in body_color),
    )
    # Tail.
    tail_color = tuple(max(0, c - 30) for c in body_color)
    d.polygon(g["tail"], fill=tail_color, outline=(20, 20, 20), width=2)
    # Scale dots.
    for _ in range(8):
        sx = rng.randint(bx0 + 10, bx1 - 30)
        sy = rng.randint(by0 + 6, by1 - 6)
        d.ellipse((sx - 2, sy - 2, sx + 2, sy + 2), fill=(255, 255, 255, 200))
    # Eye.
    ex, ey = g["eye"]
    d.ellipse((ex - 6, ey - 6, ex + 6, ey + 6), fill=(255, 255, 255), outline=(20, 20, 20), width=1)
    d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=(20, 20, 20))
    return img


def _plant_geometry(cx: int, cy: int, r: int) -> dict:
    """Potted-plant geometry — pot (trapezoid) + stem (vertical) + 3 leaves (ellipses)."""
    pot = [
        (cx - int(r * 0.45), cy + int(r * 0.55)),
        (cx + int(r * 0.45), cy + int(r * 0.55)),
        (cx + int(r * 0.35), cy + int(r * 0.95)),
        (cx - int(r * 0.35), cy + int(r * 0.95)),
    ]
    stem_top = (cx, cy - int(r * 0.85))
    stem_base = (cx, cy + int(r * 0.55))
    leaf_left = (cx - int(r * 0.6), cy - int(r * 0.05), cx - int(r * 0.05), cy + int(r * 0.30))
    leaf_right = (cx + int(r * 0.05), cy - int(r * 0.40), cx + int(r * 0.6), cy - int(r * 0.05))
    leaf_top = (cx - int(r * 0.30), cy - int(r * 0.95), cx + int(r * 0.30), cy - int(r * 0.55))
    return dict(
        pot=pot, stem_top=stem_top, stem_base=stem_base,
        leaf_left=leaf_left, leaf_right=leaf_right, leaf_top=leaf_top,
    )


def _draw_line_plant(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _plant_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.polygon(g["pot"], outline=(0, 0, 0))
    d.line([g["stem_base"], g["stem_top"]], fill=(0, 0, 0), width=3)
    d.ellipse(g["leaf_left"], outline=(0, 0, 0), width=2)
    d.ellipse(g["leaf_right"], outline=(0, 0, 0), width=2)
    d.ellipse(g["leaf_top"], outline=(0, 0, 0), width=2)
    return img


def _draw_filled_plant(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    g = _plant_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    d.polygon(g["pot"], fill=(0, 0, 0))
    d.line([g["stem_base"], g["stem_top"]], fill=(0, 0, 0), width=3)
    d.ellipse(g["leaf_left"], fill=(0, 0, 0))
    d.ellipse(g["leaf_right"], fill=(0, 0, 0))
    d.ellipse(g["leaf_top"], fill=(0, 0, 0))
    return img


def _draw_shaded_plant(img: Image.Image, cx: int, cy: int, r: int) -> Image.Image:
    """Plant with terracotta pot (shaded) + green leaves (graded)."""
    g = _plant_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    # Pot — terracotta-shaded.
    pot_pts = g["pot"]
    pot_y0 = pot_pts[0][1]
    pot_y1 = pot_pts[2][1]
    pot_h = pot_y1 - pot_y0
    n_strips = 12
    for i in range(n_strips):
        t = i / max(1, n_strips - 1)
        red = int(190 - 70 * t)
        grn = int(110 - 50 * t)
        blu = int(80 - 40 * t)
        y0 = pot_y0 + int(pot_h * (i / n_strips))
        y1 = pot_y0 + int(pot_h * ((i + 1) / n_strips))
        edge_t = (y0 - pot_y0) / max(1, pot_h)
        left = cx - int(r * (0.45 - 0.10 * edge_t))
        right = cx + int(r * (0.45 - 0.10 * edge_t))
        d.rectangle((left, y0, right, y1), fill=(red, grn, blu))
    d.polygon(g["pot"], outline=(60, 30, 20), width=2)
    # Stem.
    d.line([g["stem_base"], g["stem_top"]], fill=(60, 100, 50), width=4)
    # Leaves — graded green ellipses.
    for box in (g["leaf_left"], g["leaf_right"], g["leaf_top"]):
        d.ellipse(box, fill=(60, 150, 70), outline=(20, 80, 30), width=2)
        # Highlight stripe within leaf.
        x0, y0, x1, y1 = box
        d.line(
            (x0 + 4, (y0 + y1) // 2, x1 - 4, (y0 + y1) // 2),
            fill=(120, 200, 120), width=2,
        )
    return img


def _draw_textured_plant(img: Image.Image, cx: int, cy: int, r: int, seed: int) -> Image.Image:
    """Photorealistic-ish potted plant with detailed leaves + pot rim + soil."""
    rng = random.Random(seed + 66000)
    g = _plant_geometry(cx, cy, r)
    d = ImageDraw.Draw(img)
    # Pot — terracotta with rim.
    pot_palette = [(180, 100, 70), (160, 90, 60), (200, 130, 80)]
    pot_color = pot_palette[rng.randrange(len(pot_palette))]
    d.polygon(g["pot"], fill=pot_color, outline=(60, 30, 20), width=2)
    # Pot rim (top edge highlight).
    pot_pts = g["pot"]
    d.line(
        (pot_pts[0][0], pot_pts[0][1], pot_pts[1][0], pot_pts[1][1]),
        fill=(min(255, pot_color[0] + 40), min(255, pot_color[1] + 40), min(255, pot_color[2] + 40)),
        width=3,
    )
    # Soil at top of pot.
    soil_y = pot_pts[0][1] + 3
    d.rectangle(
        (pot_pts[0][0] + 4, pot_pts[0][1] + 1, pot_pts[1][0] - 4, soil_y + 6),
        fill=(70, 50, 35),
    )
    # Stem — green with darker edge.
    d.line([g["stem_base"], g["stem_top"]], fill=(60, 110, 50), width=4)
    # Leaves with vein detail.
    leaf_palette = [(70, 160, 80), (90, 140, 60), (60, 150, 100)]
    for box in (g["leaf_left"], g["leaf_right"], g["leaf_top"]):
        leaf_color = leaf_palette[rng.randrange(len(leaf_palette))]
        d.ellipse(box, fill=leaf_color, outline=(20, 70, 30), width=2)
        x0, y0, x1, y1 = box
        # Central vein.
        d.line(((x0 + x1) // 2, y0 + 4, (x0 + x1) // 2, y1 - 4), fill=(20, 70, 30), width=1)
        # Side veins (3 small diagonals).
        cy_leaf = (y0 + y1) // 2
        for j in range(-1, 2):
            yj = cy_leaf + j * (y1 - y0) // 6
            d.line(((x0 + x1) // 2 - 6, yj, x0 + 8, yj - 3), fill=(20, 70, 30), width=1)
            d.line(((x0 + x1) // 2 + 6, yj, x1 - 8, yj - 3), fill=(20, 70, 30), width=1)
    # Small flower / bud on top leaf.
    tx = (g["leaf_top"][0] + g["leaf_top"][2]) // 2
    ty = g["leaf_top"][1] + 4
    d.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(230, 80, 100))
    return img
