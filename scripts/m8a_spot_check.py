"""Render a 5x4 spot-check grid for M8a non-circle shapes.

Rows = shape (circle, square, triangle, hexagon, polygon).
Cols = abstraction mode (line, filled, shaded, textured).

Saves to docs/figures/m8a_shape_grid.png so we can visually inspect the
new primitives before generating the full M8a stimulus set.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from physical_mode.stimuli import primitives as P


SHAPES = ["circle", "square", "triangle", "hexagon", "polygon"]
MODES = ["line", "filled", "shaded", "textured"]
CELL = 256
CANVAS_R = 80
SEED = 1000
PAD = 30  # padding for row/col labels


def render_cell(shape: str, mode: str, seed: int) -> Image.Image:
    img = P.blank_canvas(CELL)
    img = P.draw_object(img, mode=mode, cx=CELL // 2, cy=CELL // 2, radius=CANVAS_R, seed=seed, shape=shape)
    return img


def main(out_path: Path) -> None:
    grid_w = PAD + len(MODES) * CELL
    grid_h = PAD + len(SHAPES) * CELL
    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    d = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    # Column labels.
    for ci, mode in enumerate(MODES):
        d.text((PAD + ci * CELL + 8, 6), mode, font=font, fill=(20, 20, 20))
    # Row labels (rotated would be nicer; horizontal vertical-text is fine).
    for ri, shape in enumerate(SHAPES):
        d.text((4, PAD + ri * CELL + 8), shape, font=font, fill=(20, 20, 20))

    for ri, shape in enumerate(SHAPES):
        for ci, mode in enumerate(MODES):
            cell = render_cell(shape, mode, SEED + ri * len(MODES) + ci)
            grid.paste(cell, (PAD + ci * CELL, PAD + ri * CELL))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, format="PNG")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    # Re-import after path mutation.
    from physical_mode.stimuli import primitives as P  # noqa: F401, F811

    target = project_root / "docs" / "figures" / "m8a_shape_grid.png"
    main(target)
