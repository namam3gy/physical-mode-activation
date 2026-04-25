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
Shape = Literal["circle", "square", "triangle", "hexagon", "polygon", "car", "person", "bird"]


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
