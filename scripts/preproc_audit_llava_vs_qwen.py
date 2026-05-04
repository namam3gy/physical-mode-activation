"""Preprocessing fidelity audit — LLaVA-1.5 (CLIP) vs Qwen2.5-VL.

Discriminates "real LLaVA encoder weakness" from "preprocessing-induced
weakness". Runs the same M2 stim through both processors, denormalizes
`pixel_values` back to viewable PNGs, and saves a side-by-side panel.

Cues to eyeball: ground line, cast_shadow, motion_arrow, object outline.
If LLaVA's 336×336 path silently loses cues that Qwen's native path
preserves, the encoder-bottleneck framing has a confound. If both paths
preserve cues comparably, the framing is solid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPR_STIM = [
    "shaded_ground_cast_shadow_fall_000.png",  # M5b NULL on LLaVA
    "line_blank_none_fall_000.png",              # M5a flip target (Qwen 10/10, LLaVA 0/10)
    "filled_blank_both_fall_000.png",            # Strong-cue baseline (Qwen M5b break cell)
    "textured_ground_motion_arrow_fall_000.png", # Strongest cue stack
]


def denormalize_clip(pix: torch.Tensor) -> Image.Image:
    """LLaVA-1.5 uses CLIP image processor: mean=(0.481,0.458,0.408), std=(0.269,0.261,0.276)."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    arr = (pix.detach().float().cpu() * std + mean).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def denormalize_qwen(pix_flat: torch.Tensor, grid_thw: torch.Tensor) -> Image.Image:
    """Qwen2.5-VL flattens patches into (n_patches, c*tph*ph*pw).

    grid_thw = (T, H_patches, W_patches) where each spatial patch is 14×14
    and 2×2 patches are merged into one token in spatial_merge.
    Pixel grid layout: H_pixels = H_patches * 14, W_pixels = W_patches * 14.

    Mean/std are CLIP defaults (Qwen reuses them).
    """
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    T, Hp, Wp = grid_thw.tolist()
    patch = 14
    arr = pix_flat.detach().float().cpu().numpy()
    # Qwen layout: (T, Hp//2, Wp//2, merge_size_h=2, merge_size_w=2, c, temporal_patch_size=2, ph, pw)
    # flattened to (n, c*tps*ph*pw) where n = T*Hp*Wp.
    # We'll just reshape spatial-only assuming T=1, tps=2.
    c = 3
    tps = 2
    expected = T * Hp * Wp
    assert arr.shape[0] == expected, f"got {arr.shape[0]} patches, expected {expected}"
    # Reshape: (T, Hp//2, Wp//2, 2, 2, c, tps, ph, pw)
    Hpm = Hp // 2
    Wpm = Wp // 2
    arr = arr.reshape(T, Hpm, Wpm, 2, 2, c, tps, patch, patch)
    # Take first temporal frame
    arr = arr[0, :, :, :, :, :, 0, :, :]  # (Hpm, Wpm, 2, 2, c, ph, pw)
    # Move c to last and flatten patch grid → (Hp*ph, Wp*pw, c)
    arr = arr.transpose(0, 2, 5, 1, 3, 6, 4)  # (Hpm, 2, ph, Wpm, 2, pw, c)
    H_pix = Hpm * 2 * patch
    W_pix = Wpm * 2 * patch
    arr = arr.reshape(H_pix, W_pix, c)
    arr = arr * std + mean
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_panel(images: list[tuple[str, Image.Image]], out_path: Path) -> None:
    """Save a horizontal panel with title strip per image."""
    pad = 8
    title_h = 28
    target_h = max(im.height for _, im in images)
    # Resize all to common height for visual comparison
    resized = []
    for label, im in images:
        scale = target_h / im.height
        new_w = int(round(im.width * scale))
        resized.append((label, im.resize((new_w, target_h), Image.LANCZOS)))
    total_w = sum(im.width for _, im in resized) + pad * (len(resized) + 1)
    total_h = target_h + title_h + pad * 2
    panel = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    x = pad
    for label, im in resized:
        panel.paste(im, (x, title_h + pad))
        draw.text((x + 4, 6), f"{label} ({im.width}×{im.height})", fill=(240, 240, 240), font=font)
        x += im.width + pad
    panel.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-dir", type=Path,
                    default=Path("inputs/mvp_full_20260424-093926_e9d79da3/images"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/preproc_audit_llava_vs_qwen"))
    ap.add_argument("--llava-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--qwen-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    print(f"Loading LLaVA processor: {args.llava_id}")
    llava_proc = AutoProcessor.from_pretrained(args.llava_id)
    print(f"Loading Qwen processor: {args.qwen_id}")
    qwen_proc = AutoProcessor.from_pretrained(args.qwen_id)

    # Print LLaVA image-processor config (the suspected confound)
    llip = llava_proc.image_processor
    print("\nLLaVA image_processor config:")
    print(f"  size                = {getattr(llip, 'size', '?')}")
    print(f"  crop_size           = {getattr(llip, 'crop_size', '?')}")
    print(f"  do_resize           = {getattr(llip, 'do_resize', '?')}")
    print(f"  do_center_crop      = {getattr(llip, 'do_center_crop', '?')}")
    print(f"  do_normalize        = {getattr(llip, 'do_normalize', '?')}")
    print(f"  resample            = {getattr(llip, 'resample', '?')}")
    print(f"  image_mean          = {getattr(llip, 'image_mean', '?')}")
    print(f"  image_std           = {getattr(llip, 'image_std', '?')}")

    qip = qwen_proc.image_processor
    print("\nQwen image_processor config:")
    print(f"  min_pixels          = {getattr(qip, 'min_pixels', '?')}")
    print(f"  max_pixels          = {getattr(qip, 'max_pixels', '?')}")
    print(f"  patch_size          = {getattr(qip, 'patch_size', '?')}")
    print(f"  merge_size          = {getattr(qip, 'merge_size', '?')}")
    print(f"  temporal_patch_size = {getattr(qip, 'temporal_patch_size', '?')}")
    print(f"  do_resize           = {getattr(qip, 'do_resize', '?')}")

    for stim_name in REPR_STIM:
        stim_path = args.stim_dir / stim_name
        if not stim_path.exists():
            print(f"[skip] {stim_path} not found")
            continue
        orig = Image.open(stim_path).convert("RGB")
        print(f"\n=== {stim_name} (orig {orig.size}) ===")

        # LLaVA pipeline
        llava_inputs = llip(images=[orig], return_tensors="pt")
        pix_l = llava_inputs["pixel_values"][0]  # (3, H, W)
        print(f"  LLaVA pixel_values shape = {tuple(pix_l.shape)}")
        llava_img = denormalize_clip(pix_l)

        # Qwen pipeline
        qwen_inputs = qip(images=[orig], return_tensors="pt")
        pix_q_flat = qwen_inputs["pixel_values"]
        grid = qwen_inputs["image_grid_thw"][0]  # (T, Hp, Wp)
        print(f"  Qwen pixel_values shape  = {tuple(pix_q_flat.shape)}, grid_thw = {grid.tolist()}")
        try:
            qwen_img = denormalize_qwen(pix_q_flat, grid)
            qwen_label = "Qwen native"
        except Exception as e:
            print(f"  Qwen denorm failed: {e}; using orig as placeholder")
            qwen_img = orig.copy()
            qwen_label = "Qwen (denorm failed)"

        # Save individual + panel
        stem = stim_path.stem
        llava_img.save(args.out_dir / f"{stem}_llava.png")
        qwen_img.save(args.out_dir / f"{stem}_qwen.png")
        make_panel(
            [
                ("orig 512×512", orig),
                ("LLaVA CLIP", llava_img),
                (qwen_label, qwen_img),
            ],
            args.out_dir / f"{stem}_panel.png",
        )
        print(f"  → wrote {stem}_panel.png")

    print(f"\nAll outputs in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
