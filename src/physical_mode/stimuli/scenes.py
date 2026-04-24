"""Compose a full stimulus image from (object × background × cue × event)."""

from __future__ import annotations

from PIL import Image

from ..config import StimulusRow
from . import primitives as P


# Canonical layout constants for a 512x512 canvas.
CANVAS = 512
OBJ_RADIUS = 64  # 128px diameter
GROUND_FRACTION = 0.78  # ground y = 0.78 * canvas = 399


def _object_center_for_event(event: str, canvas: int, ground_y: int) -> tuple[int, int]:
    """Return where the object should sit for each event template."""
    mid_x = canvas // 2
    if event == "fall":
        # Object high above ground, centered horizontally.
        return (mid_x, int(canvas * 0.25))
    if event == "horizontal":
        # Mid-height, slightly left of center so wind/arrow has room.
        return (int(canvas * 0.35), int(canvas * 0.45))
    if event == "hover":
        return (mid_x, int(canvas * 0.4))
    if event == "wall_bounce":
        # Near-right side, mid-height.
        return (int(canvas * 0.7), int(canvas * 0.45))
    if event == "roll_slope":
        # Near top of a notional ramp.
        return (int(canvas * 0.25), int(canvas * 0.6))
    return (mid_x, int(canvas * 0.4))


def render_scene(row: StimulusRow, size: int = CANVAS) -> Image.Image:
    """Render a single stimulus image for the given StimulusRow."""
    img = P.blank_canvas(size)
    ground_y = int(size * GROUND_FRACTION)

    # 1. Background.
    if row.bg_level == "blank":
        pass
    elif row.bg_level == "ground":
        img = P.draw_ground(img, ground_y)
    elif row.bg_level == "scene":
        img = P.draw_scene(img, ground_y, seed=row.seed)

    # 2. Object position based on event template.
    cx, cy = _object_center_for_event(row.event_template, size, ground_y)

    # 3. Context cue rendered *behind* the object where appropriate, *in front* otherwise.
    if row.cue_level == "wind":
        # Wind marks behind the direction of motion — for horizontal, marks trail to the left.
        img = P.draw_wind_marks(img, side="right", cx=cx, cy=cy, seed=row.seed)
    elif row.cue_level == "arrow_shadow":
        # Cast shadow on ground (if a ground exists) + trajectory arrow.
        if row.bg_level in ("ground", "scene"):
            img = P.draw_cast_shadow(img, cx, cy, OBJ_RADIUS, ground_y)
        # Arrow: for fall event, down-pointing; for horizontal, rightward.
        if row.event_template == "fall":
            img = P.draw_trajectory_arrow(img, (cx, cy + OBJ_RADIUS + 10), (cx, cy + OBJ_RADIUS + 90))
        elif row.event_template == "horizontal":
            img = P.draw_trajectory_arrow(img, (cx + OBJ_RADIUS + 10, cy), (cx + OBJ_RADIUS + 110, cy))
        elif row.event_template == "wall_bounce":
            img = P.draw_trajectory_arrow(img, (cx - OBJ_RADIUS - 10, cy), (cx - OBJ_RADIUS - 110, cy - 30))
        elif row.event_template == "roll_slope":
            img = P.draw_trajectory_arrow(img, (cx + OBJ_RADIUS + 10, cy + 10), (cx + OBJ_RADIUS + 110, cy + 70))
        elif row.event_template == "hover":
            img = P.draw_trajectory_arrow(img, (cx, cy - OBJ_RADIUS - 10), (cx, cy - OBJ_RADIUS - 70))

    # 4. Object last, so it sits on top of cues where they overlap.
    img = P.draw_object(
        img,
        mode=row.object_level,
        cx=cx,
        cy=cy,
        radius=OBJ_RADIUS,
        seed=row.seed,
    )

    return img
