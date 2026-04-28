"""§4.6 cross-model — Idefics2-8B variant of pixel-space gradient ascent.

Idefics2 uses sub-image tiling similar to LLaVA-Next AnyRes:
  - `pixel_values` shape `(1, num_tiles, 3, H, W)` — typically 5 tiles for
    a 512×512 input, each at 512×512.
  - `pixel_attention_mask` shape `(1, num_tiles, H, W)` flags valid pixels
    (1 = valid, 0 = padding). For our square 512×512 stim, the mask is
    typically all-1 within each tile.

The mask must be passed to the model at every forward but is *not* a leaf
tensor — gradients flow only through `pixel_values`. Eps-clipping is per-
element (per tile, per channel, per spatial position).

Reuses `forward_get_layer_hidden` from `counterfactual.py` (model-agnostic
visual-token mask via image_token_id resolution).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .counterfactual import (
    forward_get_layer_hidden,
    _resolve_image_token_id,  # noqa: F401  (re-export)
)


def pixel_values_from_pil_idefics2(
    pil_image: Image.Image,
    processor,
    prompt: str = "What will happen next?",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pixel_values shape `(1, T, 3, H, W)`, pixel_attention_mask `(1, T, H, W)`)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    pv = raw["pixel_values"].detach().to(torch.float32)
    mask = raw["pixel_attention_mask"].detach()
    return pv, mask


def reconstruct_pil_idefics2(
    pixel_values: torch.Tensor,
    processor,
    tile_idx: int = 0,
) -> Image.Image:
    """De-normalize `pixel_values[0, tile_idx]` into a PIL.

    Idefics2's image_processor uses standard CLIP-style mean/std. tile_idx=0
    is the first sub-image; for our square 512×512 stim multiple tiles are
    similar (the processor splits even square inputs into a 4-tile grid +
    base).
    """
    image_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)

    pv = pixel_values.detach().cpu().to(torch.float32)
    if pv.dim() != 5 or pv.shape[0] != 1:
        raise ValueError(f"expected pixel_values shape (1, T, 3, H, W), got {tuple(pv.shape)}")
    img = pv[0, tile_idx]
    if img.shape[0] != 3:
        raise ValueError(f"expected 3-channel tile, got {tuple(img.shape)}")

    img_unnorm = (img * image_std + image_mean).clamp(0, 1)
    img_uint8 = (img_unnorm * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(img_uint8, "RGB")


def prepare_inputs_for_grad_idefics2(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = "What will happen next?",
) -> tuple[dict[str, Any], torch.Tensor]:
    """Idefics2 variant: pixel_values shape `(1, T, 3, 512, 512)` + mask."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    raw = {k: v.to(model.device) for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)
    return inputs, pv_leaf


def gradient_ascent_idefics2(
    model,
    processor,
    pil_image: Image.Image,
    v_unit: torch.Tensor | np.ndarray,
    layer: int = 10,
    n_steps: int = 200,
    lr: float = 1e-2,
    eps: float | None = 0.1,
    mode: str = "bounded",
    prompt: str = "What will happen to the circle in the next moment? Answer in one short sentence.",
    log_every: int = 10,
) -> dict[str, Any]:
    """Pixel-space gradient ascent for Idefics2.

    pixel_attention_mask is forwarded along with pixel_values but not
    optimized (it's a binary mask). Eps-clip is per-element on
    pixel_values only.
    """
    if isinstance(v_unit, np.ndarray):
        v_unit = torch.from_numpy(v_unit)
    v_unit = v_unit.to(device=model.device, dtype=torch.float32)

    inputs, pv_leaf = prepare_inputs_for_grad_idefics2(model, processor, pil_image, prompt)
    pv_initial = pv_leaf.detach().clone()

    optimizer = torch.optim.Adam([pv_leaf], lr=lr)
    trajectory: list[tuple[int, float]] = []
    baseline_proj: float | None = None

    for step in range(n_steps):
        optimizer.zero_grad()
        inputs["pixel_values"] = pv_leaf.to(model.dtype)

        h_visual = forward_get_layer_hidden(model, inputs, layer)
        h_mean = h_visual.mean(dim=0).to(torch.float32)
        projection = (h_mean * v_unit).sum()
        loss = -projection
        loss.backward()
        optimizer.step()

        if mode == "bounded" and eps is not None:
            with torch.no_grad():
                delta = pv_leaf.data - pv_initial
                delta.clamp_(-eps, eps)
                pv_leaf.data.copy_(pv_initial + delta)

        proj_val = float(projection.detach().item())
        if baseline_proj is None:
            baseline_proj = proj_val
        if step % log_every == 0 or step == n_steps - 1:
            trajectory.append((step, proj_val))

    return {
        "pixel_values_initial": pv_initial.detach().cpu(),
        "pixel_values_final": pv_leaf.detach().cpu(),
        "projection_trajectory": trajectory,
        "final_projection": trajectory[-1][1],
        "baseline_projection": baseline_proj,
    }
