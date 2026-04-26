# §4.6 — VTI-Reverse Counterfactual Stimulus Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Synthesize an "adversarial physics-mode stimulus" by gradient-ascent on the M5a `v_L10` direction — a perturbation of a baseline abstract circle that humans still call abstract but Qwen2.5-VL reads as physics-mode (high PMR).

**Architecture:** Pixel-space gradient ascent on Qwen2.5-VL's *post-processor* `pixel_values` tensor (shape `(1, T_patches, 1176)`), bypassing the non-differentiable PIL preprocessing. Loss = `−⟨mean(h_L10[visual_tokens]), v_L10_unit⟩`. Bounded `‖δ‖_∞ ≤ ε` sweep (ε ∈ {0.05, 0.1, 0.2}) + unconstrained ablation. Random-direction control (n=3) confirms `v_L10`-specificity. PIL-roundtrip + fresh PMR inference is the causal sanity check.

**Tech Stack:** PyTorch (autograd, Adam), `transformers` AutoModelForImageTextToText / AutoProcessor for Qwen2.5-VL-7B-Instruct, NumPy, Matplotlib, PIL. New module `src/physical_mode/synthesis/counterfactual.py`. All Python via `uv run python`.

---

## Reference: spec + key paths

- **Spec**: `docs/superpowers/specs/2026-04-26-sec4_6-counterfactual-stim-design.md` (approved 2026-04-26).
- **`v_L10`**: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz` (key `v_unit_10`, shape `(3584,)`, float32, unit-norm).
- **Baseline stim** (5 seeds): `inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_{000..004}.png`.
- **Inference reference**: `src/physical_mode/models/vlm_runner.py::PhysModeVLM` (uses `@torch.inference_mode()`; we cannot reuse `generate()` for backward — write a parallel forward path with grad enabled).
- **Existing M5a steering hook pattern**: `scripts/06_vti_steering.py::make_hook` (forward hook injecting `α·v` into LM layer output; we use `v_L10` as the loss target rather than the hook input).

---

## Phase 1: Differentiability smoke (gate)

If this phase fails (gradient does not reach `pixel_values`), all of Approach A is blocked and we switch to Approach C (activation-space optimization). The smoke is a single self-contained script.

### Task 1: Read v_L10 and sanity-check shape

**Files:**
- Read: `outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz`

- [ ] **Step 1: Inspect the npz contents**

Run:
```bash
uv run python -c "
import numpy as np
data = np.load('outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz')
print('keys:', sorted(data.keys()))
v = data['v_unit_10']
print('shape:', v.shape, 'dtype:', v.dtype, 'norm:', np.linalg.norm(v))
"
```

Expected output:
```
keys: ['mean_neg_10', ..., 'v_unit_25', ...]
shape: (3584,) dtype: float32 norm: 1.0
```

If shape is not `(3584,)` or norm is not `1.0`, the steering vector is corrupted; abort and re-extract.

### Task 2: Write the differentiability smoke script

**Files:**
- Create: `scripts/sec4_6_differentiability_smoke.py`

- [ ] **Step 1: Create the smoke script**

```python
"""§4.6 iter-1 gate: verify gradient flows from pixel_values through Qwen2.5-VL
vision tower + projector + LM 0..10 to the loss `−⟨mean(h_L10[visual]), v_L10⟩`.

If this passes, Approach A (pixel-space gradient ascent) is feasible. If it
fails (e.g., grad is None, or NaN, or the vision tower is not differentiable),
fall back to Approach C (activation-space optimization).

Usage:
    uv run python scripts/sec4_6_differentiability_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
STEERING_NPZ = PROJECT_ROOT / "outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz"
BASELINE_IMG = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_000.png"
PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."
LAYER = 10


def main() -> None:
    print("Loading processor + model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    v_unit = torch.from_numpy(np.load(STEERING_NPZ)["v_unit_10"]).to("cuda:0", torch.float32)
    print(f"v_L10 shape={tuple(v_unit.shape)} norm={v_unit.norm().item():.6f}")

    pil = Image.open(BASELINE_IMG).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil], text=[text], return_tensors="pt")
    raw = {k: v.to("cuda:0") for k, v in raw.items()}

    pv_leaf = raw["pixel_values"].detach().to(torch.float32).clone().requires_grad_(True)
    inputs = dict(raw)
    inputs["pixel_values"] = pv_leaf.to(model.dtype)

    print(f"pixel_values shape={tuple(pv_leaf.shape)} dtype={pv_leaf.dtype} requires_grad={pv_leaf.requires_grad}")

    with torch.enable_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    # +1 to skip embedding layer
    h = out.hidden_states[LAYER + 1][0]  # (seq_len, hidden_dim)
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_index", None)
    print(f"image_token_id={image_token_id}")
    mask = (inputs["input_ids"][0] == image_token_id)
    n_visual = int(mask.sum().item())
    print(f"n_visual_tokens={n_visual}")
    h_visual = h[mask]  # (n_visual, hidden_dim)

    h_mean = h_visual.mean(dim=0).to(torch.float32)
    projection = (h_mean * v_unit).sum()
    print(f"baseline projection={projection.item():.6f}")

    loss = -projection
    loss.backward()

    grad = pv_leaf.grad
    if grad is None:
        print("FAIL: pixel_values.grad is None — backward did not reach the leaf.")
        raise SystemExit(1)
    print(f"pixel_values.grad shape={tuple(grad.shape)} max_abs={grad.abs().max().item():.4e} "
          f"has_nan={torch.isnan(grad).any().item()}")
    if torch.isnan(grad).any():
        print("FAIL: gradient contains NaN.")
        raise SystemExit(1)
    if grad.abs().max().item() < 1e-12:
        print("FAIL: gradient is zero — graph is detached somewhere.")
        raise SystemExit(1)

    print("PASS: gradient reaches pixel_values; magnitudes finite and non-trivial.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke**

Run: `uv run python scripts/sec4_6_differentiability_smoke.py`

Expected output (~3 min including model load):
```
Loading processor + model...
v_L10 shape=(3584,) norm=1.000000
pixel_values shape=(1, ..., 1176) dtype=torch.float32 requires_grad=True
image_token_id=...
n_visual_tokens=...
baseline projection=...
pixel_values.grad shape=(1, ..., 1176) max_abs=... has_nan=False
PASS: gradient reaches pixel_values; magnitudes finite and non-trivial.
```

If any line says `FAIL`, the autograd graph is broken. Diagnose:
- `grad is None`: check `inputs["pixel_values"]` is the bf16-cast of the leaf, not a fresh tensor.
- `has_nan=True`: bf16 numerical issue; force float32 forward (cast inside model — slow but diagnostic).
- `max_abs < 1e-12`: a `torch.no_grad`/`@torch.inference_mode()` decorator is hiding inside the model. Patch with `torch.set_grad_enabled(True)` in the call.

If any of those diagnostics still fail, escalate to the user with the script output. We do not switch to Approach C without explicit confirmation that Approach A is unrecoverable.

- [ ] **Step 3: Commit the smoke script**

```bash
git add scripts/sec4_6_differentiability_smoke.py
git commit -m "feat(§4.6 iter-1): differentiability smoke gate

Confirms gradient reaches pixel_values through Qwen2.5-VL vision tower
+ projector + LM 0..10. Required gate before Approach A optimizer loop.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: Core counterfactual module (`synthesis/counterfactual.py`)

TDD where unit-testable. Helpers that need the real model (e.g., end-to-end gradient ascent) get smoke tests in `scripts/`, not `tests/` — keeping the project convention of "no model in pytest."

### Task 3: Create the synthesis package

**Files:**
- Create: `src/physical_mode/synthesis/__init__.py`

- [ ] **Step 1: Create the empty package marker**

```python
"""Synthesis utilities (§4.6) — counterfactual stimulus generation."""
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "import physical_mode.synthesis; print('ok')"`
Expected: `ok`

### Task 4: Test for `reconstruct_pil` (round-trip)

**Files:**
- Create: `tests/test_counterfactual.py`

- [ ] **Step 1: Write the round-trip test (no model required, uses real processor)**

```python
"""Unit tests for src/physical_mode/synthesis/counterfactual.py.

These tests use the real Qwen2.5-VL processor (CPU-loadable, ~50 MB) but
NOT the model weights. Runtime ~5s for processor load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

from physical_mode.synthesis.counterfactual import (
    pixel_values_from_pil,
    reconstruct_pil,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_IMG = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_000.png"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def test_pixel_values_from_pil_returns_correct_shape(processor):
    pil = Image.open(BASELINE_IMG).convert("RGB")
    pv, grid_thw = pixel_values_from_pil(pil, processor)
    assert pv.dim() == 3, f"expected 3D (1, T, P), got {pv.shape}"
    assert pv.shape[0] == 1
    assert pv.shape[2] == 1176, f"expected last dim 1176 for Qwen2.5-VL, got {pv.shape[2]}"
    assert grid_thw.shape == (1, 3), f"expected (1, 3), got {grid_thw.shape}"


def test_reconstruct_pil_round_trip(processor):
    """Round-trip: PIL → pixel_values → reconstruct_pil should be ≤ 1 pixel
    different (uint8 quantization is the only loss)."""
    pil_orig = Image.open(BASELINE_IMG).convert("RGB")
    pv, grid_thw = pixel_values_from_pil(pil_orig, processor)
    pil_recon = reconstruct_pil(pv, grid_thw, processor)
    arr_orig = np.array(pil_orig.resize(pil_recon.size))
    arr_recon = np.array(pil_recon)
    assert arr_orig.shape == arr_recon.shape, f"shape mismatch {arr_orig.shape} vs {arr_recon.shape}"
    max_diff = int(np.abs(arr_orig.astype(int) - arr_recon.astype(int)).max())
    assert max_diff <= 2, f"reconstruction max abs diff {max_diff} > 2 (uint8 round-trip should be ≤ 1)"
```

- [ ] **Step 2: Run test to verify it fails (functions not yet implemented)**

Run: `uv run python -m pytest tests/test_counterfactual.py -q`
Expected: FAIL with `ImportError: cannot import name 'pixel_values_from_pil'`.

### Task 5: Implement `pixel_values_from_pil`

**Files:**
- Create: `src/physical_mode/synthesis/counterfactual.py`

- [ ] **Step 1: Add the helper**

```python
"""§4.6 — VTI-reverse counterfactual stimulus generation utilities.

Pixel-space gradient ascent on Qwen2.5-VL's post-processor `pixel_values`
tensor (the patch-flattened, normalized representation), maximizing the
projection of the LM L10 hidden state at visual tokens onto the M5a
`v_L10` steering direction.

The processor's PIL-side preprocessing (resize, patchify, normalize) is
not differentiable, so we snapshot `pixel_values` once, then optimize a
leaf tensor of the same shape.
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

    Returns (pixel_values_float32, image_grid_thw). pixel_values has shape
    (1, T_patches, 1176) for Qwen2.5-VL; image_grid_thw is (1, 3) with
    [t, h, w] grid sizes used for un-patching.
    """
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    raw = processor(images=[pil_image], text=[text], return_tensors="pt")
    pv = raw["pixel_values"].detach().to(torch.float32)
    grid_thw = raw["image_grid_thw"].detach()
    return pv, grid_thw
```

- [ ] **Step 2: Run the first test (still fails — reconstruct_pil missing)**

Run: `uv run python -m pytest tests/test_counterfactual.py::test_pixel_values_from_pil_returns_correct_shape -q`
Expected: PASS.

### Task 6: Implement `reconstruct_pil`

**Files:**
- Modify: `src/physical_mode/synthesis/counterfactual.py`

- [ ] **Step 1: Add the un-patch + de-normalize helper**

Append to `counterfactual.py`:

```python
def reconstruct_pil(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    processor,
) -> Image.Image:
    """Inverse of `pixel_values_from_pil`: un-patch + de-normalize back to a
    viewable RGB PIL image.

    Qwen2.5-VL flattens patches as `(temporal=2, channels=3, h=14, w=14)` →
    1176 dims per merged patch. For single images, both temporal slots are
    duplicates; we average them (or take the first) for reconstruction.

    image_grid_thw is `(1, 3)` with `[t, h_patches, w_patches]`. Note that h
    and w are in *pre-merge* units; the spatial merge is 2×2, so the
    pixel_values' T_patches = (h * w) // 4 grouped by 2×2 merger.
    """
    # Pixel value normalization stats (SigLIP / Qwen2.5-VL: image_mean = [0.5,
    # 0.5, 0.5], image_std = [0.5, 0.5, 0.5] → values in roughly [-1, 1]).
    image_mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1)

    pv = pixel_values.detach().cpu().to(torch.float32)
    t, h_pre, w_pre = [int(x) for x in image_grid_thw[0].tolist()]
    merge = int(processor.image_processor.merge_size)
    patch = int(processor.image_processor.patch_size)
    temporal_patch = int(processor.image_processor.temporal_patch_size)
    channels = 3

    # pv: (1, T, 1176) where T = (h_pre / merge) * (w_pre / merge)
    T = pv.shape[1]
    expected_T = (h_pre // merge) * (w_pre // merge)
    if T != expected_T:
        raise ValueError(f"pixel_values T={T} but expected {expected_T} for grid h={h_pre}, w={w_pre}, merge={merge}")

    # Reshape: (1, T, 1176) → (1, T, temporal, channels, merge*patch, merge*patch)?
    # Actually the flatten order is: temporal × channels × (merge_h × patch_h) × (merge_w × patch_w).
    # 1176 = 2 * 3 * 14 * 14 = 2 * 3 * 196 — so each merged patch is 1×1 in the
    # output grid (merge already applied). Let's reshape:
    pv_4d = pv.reshape(1, T, temporal_patch, channels, patch, patch)
    # Take temporal slot 0 (frames are duplicates for single images).
    pv_3d = pv_4d[:, :, 0]  # (1, T, 3, patch, patch)

    # Place T merged-patches into a (1, 3, h_pre/merge * patch, w_pre/merge * patch) grid.
    # Note: this is the post-merge dimension, so each "merged patch" is one
    # `patch × patch` tile. (Spatial merge is already collapsed in the pixel_values.)
    h_grid = h_pre // merge
    w_grid = w_pre // merge
    H = h_grid * patch
    W = w_grid * patch
    img = pv_3d.reshape(1, h_grid, w_grid, channels, patch, patch)
    img = img.permute(0, 3, 1, 4, 2, 5).contiguous()  # (1, C, h_grid, patch, w_grid, patch)
    img = img.reshape(1, channels, H, W)

    # Un-normalize.
    img_unnorm = img * image_std + image_mean
    img_unnorm = img_unnorm.clamp(0, 1)
    img_uint8 = (img_unnorm * 255).to(torch.uint8)[0].permute(1, 2, 0).numpy()
    return Image.fromarray(img_uint8, "RGB")
```

- [ ] **Step 2: Run both tests**

Run: `uv run python -m pytest tests/test_counterfactual.py -q`
Expected: PASS (2 tests).

If the round-trip test fails with `max_diff > 2`, the patch reshape order in `reconstruct_pil` doesn't match the processor's forward order. Inspect `transformers/models/qwen2_5_vl/image_processing_qwen2_5_vl.py::_preprocess` to confirm the flatten order and adjust `permute` accordingly.

- [ ] **Step 3: Commit Phase 2 helpers (so far)**

```bash
git add src/physical_mode/synthesis/__init__.py src/physical_mode/synthesis/counterfactual.py tests/test_counterfactual.py
git commit -m "feat(§4.6): pixel_values <-> PIL helpers + round-trip test

Implements pixel_values_from_pil and reconstruct_pil for Qwen2.5-VL.
Round-trip test verifies max abs uint8 diff <= 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 7: Implement `prepare_inputs_for_grad`

**Files:**
- Modify: `src/physical_mode/synthesis/counterfactual.py`

- [ ] **Step 1: Add the input-prep helper**

Append:

```python
def prepare_inputs_for_grad(
    model,
    processor,
    pil_image: Image.Image,
    prompt: str = "What will happen next?",
) -> tuple[dict[str, Any], torch.Tensor]:
    """Build the model inputs dict for a backward-enabled forward pass.

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
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from physical_mode.synthesis.counterfactual import prepare_inputs_for_grad; print('ok')"`
Expected: `ok`.

### Task 8: Implement `forward_get_layer_hidden`

**Files:**
- Modify: `src/physical_mode/synthesis/counterfactual.py`

- [ ] **Step 1: Add the forward helper (no `@torch.inference_mode()`)**

Append:

```python
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

    Returns: Tensor `(n_visual_tokens, hidden_dim)` in the model's dtype,
    differentiable back to `inputs["pixel_values"]` if that traces to a
    leaf tensor.
    """
    out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer + 1][0]  # +1 to skip embedding
    image_token_id = _resolve_image_token_id(model)
    mask = (inputs["input_ids"][0] == image_token_id)
    return h[mask]
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from physical_mode.synthesis.counterfactual import forward_get_layer_hidden; print('ok')"`
Expected: `ok`.

### Task 9: Implement `gradient_ascent`

**Files:**
- Modify: `src/physical_mode/synthesis/counterfactual.py`

- [ ] **Step 1: Add the main loop**

Append:

```python
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
    prompt: str = "What will happen next?",
    log_every: int = 10,
) -> dict[str, Any]:
    """Pixel-space gradient ascent on `<mean(h_layer[visual]), v_unit>`.

    Args:
        model, processor: Qwen2.5-VL instance + processor.
        pil_image: starting baseline PIL.Image (RGB).
        v_unit: unit-norm steering direction in LM hidden space (shape
                `(hidden_dim,)`). May be ndarray or Tensor.
        layer: LM layer index for projection target (10 for `v_L10`).
        n_steps: optimization steps.
        lr: Adam learning rate.
        eps: L_inf bound on `pv_leaf - pv_initial`. None or `mode='unconstrained'` disables clipping.
        mode: "bounded" (clip after each step) or "unconstrained".
        prompt: text prompt fed alongside the image.
        log_every: log projection every N steps.

    Returns dict:
        - 'pixel_values_initial': float32 (1, T, 1176)
        - 'pixel_values_final':   float32 (1, T, 1176)
        - 'projection_trajectory': list[(step:int, projection:float)]
        - 'final_projection': float
        - 'baseline_projection': float (step 0, pre-optimization)
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

        # Re-cast leaf to model dtype for forward (autograd tracks the cast).
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
```

- [ ] **Step 2: Smoke-test the loop end-to-end (10 steps, no clip)**

Add to `scripts/sec4_6_differentiability_smoke.py` a `--with-loop` flag, OR write a quick standalone:

Run:
```bash
uv run python -c "
import numpy as np, torch
from PIL import Image
from pathlib import Path
from transformers import AutoModelForImageTextToText, AutoProcessor
from physical_mode.synthesis.counterfactual import gradient_ascent

ROOT = Path('.')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
model = AutoModelForImageTextToText.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16, device_map='cuda:0')
model.eval()
v = np.load('outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz')['v_unit_10']
pil = Image.open('inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_000.png').convert('RGB')
out = gradient_ascent(model, processor, pil, v, layer=10, n_steps=10, lr=1e-2, eps=0.1, mode='bounded', log_every=1)
print('trajectory:', out['projection_trajectory'])
print('baseline=', out['baseline_projection'], 'final=', out['final_projection'])
assert out['final_projection'] > out['baseline_projection'], 'projection did not increase'
print('PASS: projection increased over 10 bounded steps.')
"
```

Expected: trajectory is monotonically (roughly) increasing; final > baseline; PASS line printed.

If projection decreases or fluctuates wildly: lr too high; try `lr=1e-3`. If still failing, the clip step may be undoing the optimizer step — switch the order so clip happens BEFORE re-binding `inputs["pixel_values"]` next step (already correct in code, but verify).

- [ ] **Step 3: Commit Phase 2 complete**

```bash
git add src/physical_mode/synthesis/counterfactual.py
git commit -m "feat(§4.6): gradient_ascent loop + prepare_inputs_for_grad

Adds the main pixel-space gradient ascent driver. Smoke-tested with 10
bounded steps on M8a abstract baseline — projection monotonically
increases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3: Driver script (`sec4_6_counterfactual_stim.py`)

End-to-end runner: 5 seeds × {3 ε bounded + 1 unconstrained + 3 random-direction control} × 200 steps. Saves all artifacts to `outputs/sec4_6_counterfactual_<ts>/`.

### Task 10: Skeleton driver with arg parsing

**Files:**
- Create: `scripts/sec4_6_counterfactual_stim.py`

- [ ] **Step 1: Write the argparse + main skeleton**

```python
"""§4.6 — VTI-reverse counterfactual stimulus generation driver.

Runs gradient_ascent on each baseline stim × each (mode, eps) configuration.
Saves: per-seed reconstructed PNGs, per-config delta tensors, projection
trajectories, and a config manifest. Inference re-evaluation (PMR pre/post)
is delegated to scripts/sec4_6_summarize.py.

Usage:
    uv run python scripts/sec4_6_counterfactual_stim.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.synthesis.counterfactual import gradient_ascent, reconstruct_pil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "inputs/mvp_full_20260424-093926_e9d79da3/images"
DEFAULT_STEERING_NPZ = PROJECT_ROOT / "outputs/mvp_full_20260424-094103_8ae1fa3d/probing_steering/steering_vectors.npz"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--baseline-pattern", default="line_blank_none_fall_*.png")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--steering-npz", type=Path, default=DEFAULT_STEERING_NPZ)
    p.add_argument("--steering-key", default="v_unit_10")
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--eps-list", default="0.05,0.1,0.2",
                   help="comma-separated L_inf bounds for bounded mode")
    p.add_argument("--include-unconstrained", action="store_true", default=True)
    p.add_argument("--n-random-controls", type=int, default=3,
                   help="number of random-direction controls (eps=0.1 fixed)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="if None, autogenerates outputs/sec4_6_counterfactual_<ts>")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for random-direction controls")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = PROJECT_ROOT / f"outputs/sec4_6_counterfactual_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {args.model_id} on {args.device}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    v_unit = np.load(args.steering_npz)[args.steering_key]
    print(f"Steering: {args.steering_key} shape={v_unit.shape} norm={np.linalg.norm(v_unit):.6f}")

    # Random-direction controls
    rng = np.random.default_rng(args.seed)
    random_dirs = []
    for i in range(args.n_random_controls):
        r = rng.standard_normal(v_unit.shape).astype(np.float32)
        r /= np.linalg.norm(r) + 1e-8
        random_dirs.append((f"v_random_{i}", r))

    # Baseline stim list
    baselines = sorted(args.baseline_dir.glob(args.baseline_pattern))[: args.n_seeds]
    print(f"Baselines ({len(baselines)}): {[b.name for b in baselines]}")

    eps_list = [float(x) for x in args.eps_list.split(",")]
    configs: list[dict] = []
    for eps in eps_list:
        configs.append({"name": f"bounded_eps{eps}", "mode": "bounded", "eps": eps, "v": ("v_unit_10", v_unit)})
    if args.include_unconstrained:
        configs.append({"name": "unconstrained", "mode": "unconstrained", "eps": None, "v": ("v_unit_10", v_unit)})
    for r_name, r_vec in random_dirs:
        configs.append({"name": f"control_{r_name}", "mode": "bounded", "eps": 0.1, "v": (r_name, r_vec)})

    manifest_rows = []
    for stim_path in baselines:
        sid = stim_path.stem
        pil = Image.open(stim_path).convert("RGB")
        for cfg in configs:
            v_name, v_arr = cfg["v"]
            print(f"  [{sid}] {cfg['name']} (v={v_name}, mode={cfg['mode']}, eps={cfg['eps']})")
            out = gradient_ascent(
                model, processor, pil, v_arr,
                layer=args.layer, n_steps=args.n_steps, lr=args.lr,
                eps=cfg["eps"], mode=cfg["mode"], prompt=args.prompt,
                log_every=10,
            )
            # Save artifacts
            cfg_dir = args.output_dir / cfg["name"] / sid
            cfg_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "pixel_values_initial": out["pixel_values_initial"],
                "pixel_values_final": out["pixel_values_final"],
            }, cfg_dir / "pixel_values.pt")
            traj_arr = np.array(out["projection_trajectory"], dtype=np.float64)
            np.save(cfg_dir / "trajectory.npy", traj_arr)
            # Reconstruct image
            from physical_mode.synthesis.counterfactual import pixel_values_from_pil
            _, grid_thw = pixel_values_from_pil(pil, processor, args.prompt)
            recon = reconstruct_pil(out["pixel_values_final"], grid_thw, processor)
            recon.save(cfg_dir / "synthesized.png")
            pil.save(cfg_dir / "baseline.png")

            manifest_rows.append({
                "sample_id": sid,
                "config_name": cfg["name"],
                "v_name": v_name,
                "mode": cfg["mode"],
                "eps": cfg["eps"],
                "layer": args.layer,
                "n_steps": args.n_steps,
                "baseline_projection": out["baseline_projection"],
                "final_projection": out["final_projection"],
                "synthesized_path": str((cfg_dir / "synthesized.png").relative_to(PROJECT_ROOT)),
                "trajectory_path": str((cfg_dir / "trajectory.npy").relative_to(PROJECT_ROOT)),
            })

    manifest = {"args": vars(args), "rows": manifest_rows}
    # vars(args) contains Path objects — convert to str
    manifest["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in manifest["args"].items()}
    with open(args.output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nDone. Manifest at {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check arg parsing**

Run: `uv run python scripts/sec4_6_counterfactual_stim.py --help`
Expected: argparse usage printed with all flags.

### Task 11: Smoke-run the driver (1 seed, 10 steps, 1 config)

- [ ] **Step 1: Run reduced-scope smoke**

Run:
```bash
uv run python scripts/sec4_6_counterfactual_stim.py \
    --n-seeds 1 \
    --n-steps 10 \
    --eps-list 0.1 \
    --n-random-controls 0 \
    --output-dir outputs/sec4_6_smoke_test
```

Expected (~3-5 min including model load): one stim × one config (`bounded_eps0.1`) → `outputs/sec4_6_smoke_test/bounded_eps0.1/line_blank_none_fall_000/{baseline,synthesized}.png + pixel_values.pt + trajectory.npy`. `manifest.json` has 1 row.

- [ ] **Step 2: Spot-check trajectory and reconstructed image**

Run:
```bash
uv run python -c "
import numpy as np
t = np.load('outputs/sec4_6_smoke_test/bounded_eps0.1/line_blank_none_fall_000/trajectory.npy')
print('trajectory:', t)
assert t[-1, 1] > t[0, 1], f'projection did not increase ({t[0,1]} -> {t[-1,1]})'
print('PASS: trajectory increased.')
"
ls outputs/sec4_6_smoke_test/bounded_eps0.1/line_blank_none_fall_000/
```

Expected: `PASS: trajectory increased.` printed; both PNGs visible.

- [ ] **Step 3: Commit Phase 3**

```bash
git add scripts/sec4_6_counterfactual_stim.py
git commit -m "feat(§4.6): driver script for full sweep

End-to-end driver: 5 seeds × {3 eps bounded + unconstrained + 3 random
controls} × 200 steps. Saves baseline.png / synthesized.png / pixel_values.pt
/ trajectory.npy per (config, sid) + manifest.json.

Smoke-tested with 1 seed × 10 steps × bounded eps=0.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 12: Full driver run (5 seeds × full configs)

- [ ] **Step 1: Run the full sweep (~30-40 min on H200)**

Run in background (the run is long):
```bash
nohup uv run python scripts/sec4_6_counterfactual_stim.py > outputs/sec4_6_full_run.log 2>&1 &
```

Expected configs per seed: 3 bounded (eps 0.05, 0.1, 0.2) + 1 unconstrained + 3 random controls = 7. Total runs: 5 × 7 = 35. At ~30s per run on H200 → ~17 min plus model load.

- [ ] **Step 2: Wait + verify**

When notified of completion, check:
```bash
tail -5 outputs/sec4_6_full_run.log
ls outputs/sec4_6_counterfactual_*/manifest.json
uv run python -c "
import json
with open(sorted(__import__('pathlib').Path('outputs').glob('sec4_6_counterfactual_*/manifest.json'))[-1]) as f:
    m = json.load(f)
print('rows:', len(m['rows']))
print('configs:', sorted({r['config_name'] for r in m['rows']}))
print('seeds:', sorted({r['sample_id'] for r in m['rows']}))
"
```

Expected: 35 rows; 7 configs; 5 seeds.

---

## Phase 4: PMR re-inference + summarize + figures

The synthesized images need a fresh PMR inference (independent of the gradient-ascent path) to confirm "the projection increase translates to actual model behavior change."

### Task 13: PMR re-inference on synthesized images

**Files:**
- Modify: `scripts/sec4_6_summarize.py` (create)

- [ ] **Step 1: Write the re-inference + aggregation script**

```python
"""§4.6 — re-infer PMR on baseline + synthesized stim, aggregate, plot.

Loads the manifest produced by sec4_6_counterfactual_stim.py, runs a
fresh inference on each baseline.png and synthesized.png using the
Qwen2.5-VL inference pipeline (independent of the gradient-ascent path),
scores PMR, and writes a results CSV + figure.

Usage:
    uv run python scripts/sec4_6_summarize.py --run-dir outputs/sec4_6_counterfactual_<ts>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physical_mode.metrics.pmr import score_pmr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT = "What will happen to the circle in the next moment? Answer in one short sentence."


def _generate(model, processor, pil: Image.Image, prompt: str = PROMPT) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(images=[pil], text=[text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return raw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    with open(args.run_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(f"Loading {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()

    rows = []
    for r in manifest["rows"]:
        sid = r["sample_id"]
        cfg = r["config_name"]
        synth_path = PROJECT_ROOT / r["synthesized_path"]
        baseline_path = synth_path.parent / "baseline.png"

        baseline_resp = _generate(model, processor, Image.open(baseline_path).convert("RGB"))
        synth_resp = _generate(model, processor, Image.open(synth_path).convert("RGB"))
        baseline_pmr = score_pmr(baseline_resp)
        synth_pmr = score_pmr(synth_resp)

        rows.append({
            **r,
            "baseline_response": baseline_resp,
            "synthesized_response": synth_resp,
            "baseline_pmr": baseline_pmr,
            "synthesized_pmr": synth_pmr,
            "delta_pmr": synth_pmr - baseline_pmr,
        })
        print(f"  [{sid}] {cfg}: baseline_pmr={baseline_pmr} synth_pmr={synth_pmr} Δ={synth_pmr - baseline_pmr:+d}")

    df = pd.DataFrame(rows)
    df_csv = args.run_dir / "results.csv"
    df.to_csv(df_csv, index=False)
    print(f"\nWrote {df_csv}")

    # Aggregate per config
    agg = df.groupby("config_name").agg(
        n=("sample_id", "count"),
        baseline_pmr_mean=("baseline_pmr", "mean"),
        synth_pmr_mean=("synthesized_pmr", "mean"),
        delta_mean=("delta_pmr", "mean"),
        n_flipped=("delta_pmr", lambda s: int((s > 0).sum())),
    ).reset_index()
    print("\n=== Aggregated PMR per config ===")
    print(agg.round(3).to_string(index=False))
    agg.to_csv(args.run_dir / "results_aggregated.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on the full output**

Run:
```bash
uv run python scripts/sec4_6_summarize.py \
    --run-dir outputs/sec4_6_counterfactual_<ts>
```

Replace `<ts>` with the timestamp from Task 12.

Expected (~5 min): 35 inferences (5 seeds × 7 configs); per-row print; `results.csv` + `results_aggregated.csv` written.

### Task 14: 4-panel canonical figure

**Files:**
- Modify: `scripts/sec4_6_summarize.py`

- [ ] **Step 1: Add 4-panel figure generator**

Append to `scripts/sec4_6_summarize.py` before `if __name__`:

```python
def _plot_canonical_panels(args, manifest_rows: list[dict], canonical_sid: str) -> None:
    """4-panel figure: baseline / eps=0.05 / eps=0.1 / unconstrained for one seed."""
    panels = [
        ("baseline", "baseline_eps_0"),
        ("bounded_eps0.05", "ε = 0.05"),
        ("bounded_eps0.1", "ε = 0.1"),
        ("unconstrained", "unconstrained"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (cfg, title) in zip(axes, panels):
        if cfg == "baseline":
            row = next(r for r in manifest_rows if r["sample_id"] == canonical_sid)
            img_path = (Path(row["synthesized_path"])).parent / "baseline.png"
            img = Image.open(PROJECT_ROOT / img_path)
        else:
            row = next((r for r in manifest_rows
                       if r["sample_id"] == canonical_sid and r["config_name"] == cfg), None)
            if row is None:
                ax.set_visible(False); continue
            img = Image.open(PROJECT_ROOT / row["synthesized_path"])
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"§4.6 — VTI-reverse counterfactual (seed: {canonical_sid})", fontsize=14)
    out = PROJECT_ROOT / "docs/figures/sec4_6_counterfactual_stim_panels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_trajectories(args, manifest_rows: list[dict]) -> None:
    """Per-config mean trajectory ± std across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cfgs = sorted({r["config_name"] for r in manifest_rows})
    for cfg in cfgs:
        seed_rows = [r for r in manifest_rows if r["config_name"] == cfg]
        trajs = []
        for r in seed_rows:
            t = np.load(PROJECT_ROOT / r["trajectory_path"])
            trajs.append(t[:, 1])
        if not trajs:
            continue
        max_len = max(len(x) for x in trajs)
        # Pad with last value for unequal lengths.
        padded = np.array([np.pad(x, (0, max_len - len(x)), mode="edge") for x in trajs])
        steps = np.arange(max_len)
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        ls = "--" if cfg.startswith("control_") else "-"
        ax.plot(steps, mean, ls, label=cfg, linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Optimization step"); ax.set_ylabel("Projection on v_L10")
    ax.set_title("§4.6 — projection trajectory per config (mean ± std over 5 seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    out = PROJECT_ROOT / "docs/figures/sec4_6_counterfactual_stim_trajectory.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
```

Insert the calls in `main()` after writing the CSV:

```python
    canonical = manifest["rows"][0]["sample_id"]  # first seed in stim list
    _plot_canonical_panels(args, manifest["rows"], canonical)
    _plot_trajectories(args, manifest["rows"])
```

Note the imports at top of file: `from pathlib import Path` and `import matplotlib.pyplot as plt` (already present). Make sure both are there.

- [ ] **Step 2: Re-run the summarize**

Run:
```bash
uv run python scripts/sec4_6_summarize.py --run-dir outputs/sec4_6_counterfactual_<ts>
```

Expected: 4-panel + trajectory figures written to `docs/figures/`.

- [ ] **Step 3: Commit Phase 4**

```bash
git add scripts/sec4_6_summarize.py docs/figures/sec4_6_counterfactual_stim_panels.png docs/figures/sec4_6_counterfactual_stim_trajectory.png
git commit -m "feat(§4.6): summarize + figures (PMR re-inference + 4-panel viz)

Re-infers PMR on synthesized images via the standard inference path,
aggregates per (mode, eps), generates the 4-panel canonical-seed figure
and the cross-seed projection trajectory.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5: Insight doc + notebook + roadmap

### Task 15: Insight doc EN

**Files:**
- Create: `docs/insights/sec4_6_counterfactual_stim.md`

- [ ] **Step 1: Write the insight doc using the actual numbers**

Read the numbers from `outputs/sec4_6_counterfactual_<ts>/results_aggregated.csv` and write `docs/insights/sec4_6_counterfactual_stim.md` with sections:
- Frontmatter (`section: §4.6`, `date: 2026-04-26`, `status: complete`).
- Question (the 1-sentence framing from the spec).
- Method (Approach A pixel-values gradient ascent).
- Result table: per-config PMR pre/post + Δ + n_flipped (from `results_aggregated.csv`).
- Headlines (3-5 bullets covering: did the bounded eps=0.1 sweep flip ≥3/5? did random control stay flat? what does the unconstrained image look like?).
- Implication for hypotheses (H-shortcut: does the synthesized stim look abstract to humans yet flip the model?).
- Limitations.
- Reproducer block.
- Artifacts list.

Use `docs/insights/sec4_3_korean_vs_english.md` as a structure template.

- [ ] **Step 2: Skim for accuracy**

Compare the numbers in the doc to `results_aggregated.csv`. They must match.

### Task 16: Insight doc KO mirror

**Files:**
- Create: `docs/insights/sec4_6_counterfactual_stim_ko.md`

- [ ] **Step 1: Write the Korean translation**

Translate the EN doc section by section. Project rule: English work, Korean for terminal-visible / user-facing markdown sibling. Mid-sentence English technical terms are fine.

### Task 17: Roadmap update

**Files:**
- Modify: `references/roadmap.md`
- Modify: `references/roadmap_ko.md`

- [ ] **Step 1: Update §4.6 in milestone table to ✅**

Find the row in `references/roadmap.md` containing:

```
| **4.6** | **Counterfactual stimulus generation via SAE / VTI reverse** | ... | ▶ **PRIORITY 5 (next)** | — |
```

Replace status column with `✅` and date column with `2026-04-26`. Mirror in `roadmap_ko.md`.

- [ ] **Step 2: Update §3 detailed-status section for §4.6**

Replace `### 4.6 Counterfactual stimulus generation via SAE / VTI reverse — work plan ▶ priority 5 (promoted)` and its work-plan body with a one-paragraph summary of the result + a pointer to the insight doc:

```markdown
### 4.6 Counterfactual stimulus generation via VTI reverse ✅ (2026-04-26)

Pixel-space gradient ascent on Qwen2.5-VL post-processor `pixel_values`,
maximizing projection of LM L10 hidden state onto the M5a `v_L10`
direction. 5 seeds × {bounded ε ∈ {0.05, 0.1, 0.2}, unconstrained,
random-direction control n=3}. **Headlines from `results_aggregated.csv`**:
[insert headline numbers from the actual results].

Doc: `docs/insights/sec4_6_counterfactual_stim.md` (+ ko).
Figures: `docs/figures/sec4_6_counterfactual_stim_{panels,trajectory}.png`.
```

Mirror in KO.

### Task 18: Reproduction notebook

**Files:**
- Create: `notebooks/sec4_6_counterfactual_stim.ipynb`

- [ ] **Step 1: Build a notebook that reproduces the canonical seed**

Use `notebooks/m5_vti_steering.ipynb` as the template. Cells:
1. Imports + paths.
2. Load processor + model (with `# %% [warning: 15 GB download on first run]`).
3. Load `v_L10`.
4. Load 1 baseline image + show.
5. Run `gradient_ascent` for 200 steps × bounded ε=0.1.
6. Plot trajectory.
7. Show baseline / synthesized side-by-side.
8. Generate response on baseline + synthesized.
9. Print PMR delta.

Output the notebook as a `.ipynb` file. Run all cells before committing (notebook must be pre-executed per project convention from CLAUDE.md).

- [ ] **Step 2: Pre-execute and save**

Run:
```bash
uv run jupyter nbconvert --to notebook --execute notebooks/sec4_6_counterfactual_stim.ipynb --output sec4_6_counterfactual_stim.ipynb
```

Expected: notebook runs without errors; `.ipynb` updated in place with outputs.

### Task 19: Final commit + advisor sanity check

- [ ] **Step 1: Commit Phase 5**

```bash
git add docs/insights/sec4_6_counterfactual_stim.md docs/insights/sec4_6_counterfactual_stim_ko.md notebooks/sec4_6_counterfactual_stim.ipynb references/roadmap.md references/roadmap_ko.md
git commit -m "feat(§4.6): close — insight doc + notebook + roadmap

§4.6 (VTI-reverse counterfactual stim generation) marked complete.
[paste headline result line here]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 2: Advisor sanity check**

Call `advisor()` to validate:
- Headlines in the insight doc are consistent with `results_aggregated.csv`.
- The "shortcut interpretation" claim is appropriately scoped (single model, single direction, single starting baseline).
- Random-direction control numbers genuinely separate from `v_L10` numbers (otherwise the claim is unsupported).
- Limitations section flags every known caveat.

If advisor flags anything load-bearing, address before pushing.

- [ ] **Step 3: Update session summary**

Append a "Late-session addition #5: §4.6 closed" section to
`docs/insights/session_2026-04-26_summary.md` (+ ko) summarizing the
result and the artifacts. Commit.

---

## Self-Review

**1. Spec coverage:**
- Goal (adversarial physics-mode stim via gradient ascent on `v_L10`) → Phase 2 + Phase 3 cover it.
- Approach A (pixel-space `pixel_values`) → Phase 2 Tasks 5-9.
- Bounded + unconstrained → Phase 3 Task 10 configs list.
- Random-direction control n=3 → Phase 3 Task 10 configs list.
- Causal sanity (PIL roundtrip + fresh PMR) → Phase 4 Task 13.
- Differentiability check → Phase 1 Task 2 (gate, mandatory before Phase 2).
- 4-panel figure → Phase 4 Task 14.
- Trajectory figure → Phase 4 Task 14.
- Insight doc EN + KO → Phase 5 Tasks 15-16.
- Notebook → Phase 5 Task 18.
- Roadmap update → Phase 5 Task 17.

All 11 spec sections covered. No gaps.

**2. Placeholder scan:** None (every code step has full code; commit messages are full).

**3. Type consistency:** Function signatures match across tasks. `pixel_values_from_pil(pil, processor)` returns `(Tensor, Tensor)`; `reconstruct_pil(pixel_values, image_grid_thw, processor)` accepts that grid_thw. `gradient_ascent` returns dict with documented keys consumed by `sec4_6_summarize.py`. The `manifest.json` schema written in Task 10 matches the schema read in Task 13 + 14.

---

## Time budget

| Phase | Tasks | Estimated | Notes |
|---|---|---|---|
| 1 — Differentiability smoke | 2 | 1 hr | Includes model load + first-time download check |
| 2 — Core counterfactual module | 7 | 3 hrs | TDD on round-trip; smoke on gradient ascent |
| 3 — Driver script | 3 | 2 hrs | Includes 30-40 min inference |
| 4 — Re-infer + figures | 2 | 2 hrs | Includes 5 min PMR re-inference |
| 5 — Doc + notebook + roadmap | 5 | 1.5 hrs | EN + KO mirror + notebook re-execute |
| **Total** | **19** | **9.5-10 hrs** | Across 4-5 sessions |

The spec estimated 10-11 hrs; this plan is at the lower end of that band because Phase 2 TDD compresses time vs the spec's "differentiability smoke as iter 1" framing (we keep the smoke but it's smaller scope here).
