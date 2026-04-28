"""§4.6 cross-model — LLaVA-Next-Mistral-7B variant of pixel-space gradient ascent.

Differences from `counterfactual.py` (Qwen2.5-VL) and `counterfactual_llava.py`
(LLaVA-1.5):
  - LLaVA-Next uses the AnyRes preprocessing scheme: the processor outputs
    `pixel_values` of shape `(1, num_tiles, 3, 336, 336)` where `num_tiles ∈
    {1, 2, 5}` depending on the resolution selected from
    `image_grid_pinpoints`. For our 512×512 M2 stim the processor selects
    the 672×672 grid → 4 sub-tiles + 1 base = 5 tiles.
  - `image_sizes` `(1, 2)` is also returned and required at forward time.
  - We optimize the full 5-D `pixel_values` tensor; per-element eps-clip.
  - Reconstruction returns the *first tile* (the base / scaled-down view)
    de-normalized for human visualization. The full multi-tile structure
    cannot be losslessly reassembled into a single PIL without resizing
    artifacts — for our use case (visualizing the counterfactual stim),
    showing the base tile is sufficient and reflects the gradient-altered
    input the model saw.
  - `forward_get_layer_hidden` from `counterfactual.py` is model-agnostic
    (visual-token mask via `_resolve_image_token_id`) and works for
    LLaVA-Next too.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from .counterfactual import (
    forward_get_layer_hidden,
    _resolve_image_token_id,  # noqa: F401  (re-export for callers)
)


def pixel_values_from_pil_llava_next(
    pil_image: Image.Image,
    processor,
    prompt: str = "What will happen next?",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pixel_values shape `(1, T, 3, H, W)`, image_sizes shape `(1, 2)`)."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    pv = raw["pixel_values"].detach().to(torch.float32)
    image_sizes = raw["image_sizes"].detach()
    return pv, image_sizes


def reconstruct_pil_llava_next(
    pixel_values: torch.Tensor,
    processor,
    tile_idx: int = 0,
) -> Image.Image:
    """De-normalize `pixel_values[0, tile_idx]` and return a PIL.

    LLaVA-Next AnyRes outputs multiple tiles; tile 0 is conventionally the
    first sub-tile of the original image's split (or the base, depending on
    image_grid_pinpoints). For visualization purposes any single tile shows
    what the gradient ascent altered.
    """
    image_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)

    pv = pixel_values.detach().cpu().to(torch.float32)
    if pv.dim() != 5 or pv.shape[0] != 1:
        raise ValueError(f"expected pixel_values shape (1, T, 3, H, W), got {tuple(pv.shape)}")
    img = pv[0, tile_idx]  # (3, H, W)
    if img.shape[0] != 3:
        raise ValueError(f"expected 3-channel tile, got {tuple(img.shape)}")

    img_unnorm = (img * image_std + image_mean).clamp(0, 1)
    img_uint8 = (img_unnorm * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(img_uint8, "RGB")


def prepare_inputs_for_grad_llava_next(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = "What will happen next?",
) -> tuple[dict[str, Any], torch.Tensor]:
    """LLaVA-Next variant: pixel_values shape `(1, T, 3, 336, 336)`."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    raw = {k: v.to(model.device) for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)
    return inputs, pv_leaf


def gradient_ascent_llava_next(
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
    """Pixel-space gradient ascent for LLaVA-Next.

    Mirrors `counterfactual.gradient_ascent` but uses LLaVA-Next's 5-D AnyRes
    `pixel_values` layout. eps-clip is per-element (per tile per channel per
    spatial position) — symmetric with LLaVA-1.5's per-pixel-channel clip.
    """
    if isinstance(v_unit, np.ndarray):
        v_unit = torch.from_numpy(v_unit)
    v_unit = v_unit.to(device=model.device, dtype=torch.float32)

    inputs, pv_leaf = prepare_inputs_for_grad_llava_next(model, processor, pil_image, prompt)
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
