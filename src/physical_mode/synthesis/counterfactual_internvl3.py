"""§4.6 cross-model — InternVL3-8B-hf variant of pixel-space gradient ascent.

The HF `OpenGVLab/InternVL3-8B-hf` `InternVLProcessor` produces a single-tile
`pixel_values` shape `(1, 3, 448, 448)` for our 480×480 M2 stim — simpler
than LLaVA-Next AnyRes / Idefics2 sub-image tiling. Reconstruction is
straight de-normalization with InternViT's image_mean / image_std.

Reuses `forward_get_layer_hidden` from `counterfactual.py` (model-agnostic
via image_token_id resolution).
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


def pixel_values_from_pil_internvl3(
    pil_image: Image.Image,
    processor,
    prompt: str = "What will happen next?",
) -> torch.Tensor:
    """Return float32 pixel_values shape `(1, 3, 448, 448)`."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    pv = raw["pixel_values"].detach().to(torch.float32)
    return pv


def reconstruct_pil_internvl3(
    pixel_values: torch.Tensor,
    processor,
) -> Image.Image:
    """De-normalize `(1, 3, H, W)` pixel_values into a PIL."""
    image_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)

    pv = pixel_values.detach().cpu().to(torch.float32)
    if pv.dim() == 4:
        pv = pv[0]
    if pv.dim() != 3 or pv.shape[0] != 3:
        raise ValueError(f"expected (3, H, W) after batch removal, got {tuple(pv.shape)}")

    img_unnorm = (pv * image_std + image_mean).clamp(0, 1)
    img_uint8 = (img_unnorm * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(img_uint8, "RGB")


def prepare_inputs_for_grad_internvl3(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = "What will happen next?",
) -> tuple[dict[str, Any], torch.Tensor]:
    """InternVL3 variant: pixel_values shape `(1, 3, 448, 448)`."""
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    raw = {k: v.to(model.device) for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)
    return inputs, pv_leaf


def gradient_ascent_internvl3(
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
    """Pixel-space gradient ascent for InternVL3."""
    if isinstance(v_unit, np.ndarray):
        v_unit = torch.from_numpy(v_unit)
    v_unit = v_unit.to(device=model.device, dtype=torch.float32)

    inputs, pv_leaf = prepare_inputs_for_grad_internvl3(model, processor, pil_image, prompt)
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
