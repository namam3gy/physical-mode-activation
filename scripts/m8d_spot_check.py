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
