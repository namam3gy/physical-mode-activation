"""Compose a full stimulus image from (object × background × cue × event)."""

from __future__ import annotations

from PIL import Image

from ..config import StimulusRow
from . import primitives as P


# Canonical layout constants for a 512x512 canvas.
CANVAS = 512
OBJ_RADIUS = 64  # 128px diameter
GROUND_FRACTION = 0.78  # ground y = 0.78 * canvas = 399


_GROUND_BOUND_SHAPES: frozenset[str] = frozenset({"car", "person"})


def _object_center_for_event(
    event: str,
    canvas: int,
    ground_y: int,
    shape: str = "circle",
    radius: int = 64,
) -> tuple[int, int]:
    """Return where the object should sit for each event template.

    For `horizontal` events, ground-bound categories (`car`, `person`) sit
    *on* the ground line so that the natural-motion reading (drives /
    walks) is geometrically consistent with the cast shadow. Birds and
    abstract shapes keep the legacy mid-height placement (a bird flying
    horizontally is naturally airborne).
    """
    mid_x = canvas // 2
    if event == "fall":
        # Object high above ground, centered horizontally.
        return (mid_x, int(canvas * 0.25))
    if event == "horizontal":
        x = int(canvas * 0.35)
        if shape in _GROUND_BOUND_SHAPES:
            # Place the object so its base touches the ground line.
            return (x, ground_y - radius)
        return (x, int(canvas * 0.45))
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

    # 2. Object position based on event template (and shape for ground-bound
    #    categories whose horizontal-motion natural event is ground-level).
    cx, cy = _object_center_for_event(
        row.event_template,
        size,
        ground_y,
        shape=getattr(row, "shape", "circle"),
        radius=OBJ_RADIUS,
    )

    # 3. Context cue rendered *behind* the object where appropriate, *in front* otherwise.
    shadow_cues = {"cast_shadow", "both", "arrow_shadow"}
    arrow_cues = {"motion_arrow", "both", "arrow_shadow"}

    if row.cue_level == "wind":
        # Legacy pilot cue — invisible to Qwen2.5-VL; kept for reproducibility.
        img = P.draw_wind_marks(img, side="right", cx=cx, cy=cy, seed=row.seed)

    if row.cue_level in shadow_cues:
        # Cast shadow anchors the object to the (implicit or explicit) ground plane.
        # Draw regardless of bg_level so "shadow alone" cells are measurable.
        img = P.draw_cast_shadow(img, cx, cy, OBJ_RADIUS, ground_y)

    if row.cue_level in arrow_cues:
        # Arrow direction by event template.
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
        shape=getattr(row, "shape", "circle"),
    )

    return img
