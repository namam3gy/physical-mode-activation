"""§4.6 — VTI-reverse counterfactual stimulus generation utilities.

Pixel-space gradient ascent on Qwen2.5-VL's post-processor `pixel_values`
tensor (the patch-flattened, normalized representation), maximizing the
projection of the LM L10 hidden state at visual tokens onto the M5a
`v_L10` steering direction.

The processor's PIL-side preprocessing (resize, patchify, normalize) is
not differentiable, so we snapshot `pixel_values` once, then optimize a
leaf tensor of the same shape `(T_patches, 1176)` (Qwen2.5-VL flattens
each merged patch as 1176 = temporal_patch_size * channels * patch *
patch = 2 * 3 * 14 * 14).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


def pixel_values_from_pil(
    pil_image: Image.Image,
    processor,
    prompt: str = "What will happen next?",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the processor once and extract the pixel_values + image_grid_thw.

    Returns (pixel_values, image_grid_thw):
      - pixel_values: float32 (T_patches, 1176) for Qwen2.5-VL.
      - image_grid_thw: int (1, 3) with [t, h_pre_merge, w_pre_merge].
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    pv = raw["pixel_values"].detach().to(torch.float32)
    grid_thw = raw["image_grid_thw"].detach()
    return pv, grid_thw


def reconstruct_pil(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    processor,
) -> Image.Image:
    """Inverse of `pixel_values_from_pil`: un-patch + de-normalize back to a
    viewable RGB PIL image.

    Qwen2VLImageProcessor's forward step (see transformers source) does:

        view(grid_t, tp, c, h_m, merge, patch, w_m, merge, patch)
        permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)  # excluding batch dim
        reshape(grid_t * h_m * w_m * merge * merge, c * tp * patch * patch)
        # = (T, 1176) where T = grid_t * grid_h * grid_w = grid_t * h_m * w_m * merge^2

    Where h_m = grid_h // merge, w_m = grid_w // merge. We invert this.

    image_grid_thw is `(1, 3)` with `[grid_t, grid_h, grid_w]` (in unmerged
    14-px patch units). For Qwen2.5-VL on a 504×504 single image,
    grid_t=1, grid_h=grid_w=36, merge=2 → T=1296, h_m=w_m=18.
    """
    image_mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)

    pv = pixel_values.detach().cpu().to(torch.float32)
    if pv.dim() == 3 and pv.shape[0] == 1:
        pv = pv[0]
    if pv.dim() != 2:
        raise ValueError(f"expected 2D pixel_values, got shape {tuple(pv.shape)}")

    grid_t, grid_h, grid_w = [int(x) for x in image_grid_thw[0].tolist()]
    patch = int(processor.image_processor.patch_size)
    temporal_patch = int(processor.image_processor.temporal_patch_size)
    merge = int(processor.image_processor.merge_size)
    channels = 3
    expected_T = grid_t * grid_h * grid_w
    if pv.shape[0] != expected_T:
        raise ValueError(
            f"pixel_values T={pv.shape[0]} but expected {expected_T} = "
            f"grid_t {grid_t} * grid_h {grid_h} * grid_w {grid_w}"
        )
    h_m = grid_h // merge
    w_m = grid_w // merge

    # Inverse of the processor's reshape+permute:
    #   forward: view(grid_t, tp, c, h_m, mh, ph, w_m, mw, pw)
    #            permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    #              → (grid_t, h_m, w_m, mh, mw, c, tp, ph, pw)
    #            reshape(T, 1176)
    #
    # Reverse: reshape(T, 1176) → (grid_t, h_m, w_m, mh, mw, c, tp, ph, pw)
    #          permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    #            → (grid_t, tp, c, h_m, mh, ph, w_m, mw, pw)
    #          reshape(grid_t, tp, c, h_m * mh * ph, w_m * mw * pw)
    #            = (grid_t, tp, c, grid_h * patch, grid_w * patch)
    pv_unflat = pv.reshape(
        grid_t, h_m, w_m, merge, merge, channels, temporal_patch, patch, patch
    )
    pv_perm = pv_unflat.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()
    pv_image = pv_perm.reshape(
        grid_t, temporal_patch, channels, h_m * merge * patch, w_m * merge * patch
    )

    # Single image: grid_t=1, take temporal slot 0 (frames duplicate).
    img = pv_image[0, 0]  # (C, H, W)

    img_unnorm = (img * image_std + image_mean).clamp(0, 1)
    img_uint8 = (img_unnorm * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(img_uint8, "RGB")


def prepare_inputs_for_grad(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = "What will happen next?",
) -> tuple[dict[str, Any], torch.Tensor]:
    """Build model inputs with a leaf pixel_values tensor for gradient ascent.

    Returns (inputs, pv_leaf). `inputs["pixel_values"]` is the bf16-cast of
    the float32 leaf — gradients flow back through the cast to `pv_leaf`.
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    raw = {k: v.to(model.device) for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)
    return inputs, pv_leaf


def _resolve_image_token_id(model) -> int:
    """Locate the image-token id for visual-token masking."""
    cfg = model.config
    for attr in ("image_token_id", "image_token_index"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise RuntimeError("could not resolve image_token_id from model.config")


def forward_get_layer_hidden(
    model,
    inputs: dict[str, Any],
    layer: int,
) -> torch.Tensor:
    """Forward pass with hidden_states; returns LM hidden after `layer` at
    visual-token positions only.

    Returns: Tensor `(n_visual_tokens, hidden_dim)`, differentiable back to
    `inputs["pixel_values"]` if that traces to a leaf tensor.
    """
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer + 1][0]  # +1 to skip embedding; batch=1
    image_token_id = _resolve_image_token_id(model)
    mask = (inputs["input_ids"][0] == image_token_id)
    return h[mask]


def gradient_ascent(
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
    """Pixel-space gradient ascent on `<mean(h_layer[visual]), v_unit>`.

    Args:
        model, processor: Qwen2.5-VL instance + processor.
        pil_image: starting baseline PIL.Image (RGB).
        v_unit: unit-norm steering direction in LM hidden space `(hidden_dim,)`.
        layer: LM layer index for projection target (10 for `v_L10`).
        n_steps: optimization steps.
        lr: Adam learning rate.
        eps: L_inf bound on `pv_leaf − pv_initial`. None or `mode='unconstrained'` disables clipping.
        mode: "bounded" (clip after each step) or "unconstrained".
        prompt: text prompt fed alongside the image.
        log_every: log projection every N steps.

    Returns dict:
        - 'pixel_values_initial': float32 tensor on CPU
        - 'pixel_values_final':   float32 tensor on CPU
        - 'projection_trajectory': list[(step:int, projection:float)]
        - 'final_projection': float
        - 'baseline_projection': float (step 0)
    """
    if isinstance(v_unit, np.ndarray):
        v_unit = torch.from_numpy(v_unit)
    v_unit = v_unit.to(device=model.device, dtype=torch.float32)

    inputs, pv_leaf = prepare_inputs_for_grad(model, processor, pil_image, prompt)
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
