"""Unit tests for src/physical_mode/synthesis/counterfactual.py.

Uses the real Qwen2.5-VL processor (CPU-loadable, ~50 MB cached) but NOT
the model weights. Runtime ~5-10s for processor load.
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
BASELINE_IMG = (
    PROJECT_ROOT
    / "inputs/mvp_full_20260424-093926_e9d79da3/images/line_blank_none_fall_000.png"
)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_ID)


def test_pixel_values_from_pil_returns_correct_shape(processor):
    pil = Image.open(BASELINE_IMG).convert("RGB")
    pv, grid_thw = pixel_values_from_pil(pil, processor)
    # Qwen2.5-VL flattens to (T_total, 1176)
    assert pv.dim() == 2, f"expected 2D, got shape {tuple(pv.shape)}"
    assert pv.shape[1] == 1176, f"expected last dim 1176, got {pv.shape[1]}"
    assert grid_thw.shape == (1, 3), f"expected (1, 3), got {tuple(grid_thw.shape)}"
    t, h_pre, w_pre = [int(x) for x in grid_thw[0].tolist()]
    assert pv.shape[0] == h_pre * w_pre, (
        f"T_total {pv.shape[0]} should equal h_pre*w_pre = {h_pre*w_pre}"
    )


def test_reconstruct_pil_round_trip(processor):
    """Round-trip: PIL → pixel_values → reconstruct_pil should preserve the
    image up to processor resize + uint8 quantization. Threshold is loose
    (≤ 32) because the processor resizes arbitrary inputs to a 14-multiple
    grid before patchification — pure round-trip without resize is the
    relevant test (same input dims as output dims here)."""
    pil_orig = Image.open(BASELINE_IMG).convert("RGB")
    pv, grid_thw = pixel_values_from_pil(pil_orig, processor)
    pil_recon = reconstruct_pil(pv, grid_thw, processor)
    # Resize the original to match reconstructed size (handles any processor resize)
    pil_orig_matched = pil_orig.resize(pil_recon.size, Image.BILINEAR)
    arr_orig = np.array(pil_orig_matched)
    arr_recon = np.array(pil_recon)
    assert arr_orig.shape == arr_recon.shape, (
        f"shape mismatch {arr_orig.shape} vs {arr_recon.shape}"
    )
    max_diff = int(np.abs(arr_orig.astype(int) - arr_recon.astype(int)).max())
    # Bilinear resize + processor's resize alg may differ; allow up to 32/255 ≈ 12% per channel.
    assert max_diff <= 32, (
        f"reconstruction max abs diff {max_diff} > 32 (round-trip + resize)"
    )


def test_reconstruct_pil_preserves_perturbation(processor):
    """Reconstruct(pv + δ) ≠ Reconstruct(pv) when δ ≠ 0 — basic sanity."""
    pil = Image.open(BASELINE_IMG).convert("RGB")
    pv, grid_thw = pixel_values_from_pil(pil, processor)
    import torch
    pv_perturbed = pv + 0.1  # small uniform perturbation
    pil_a = reconstruct_pil(pv, grid_thw, processor)
    pil_b = reconstruct_pil(pv_perturbed, grid_thw, processor)
    arr_a = np.array(pil_a).astype(int)
    arr_b = np.array(pil_b).astype(int)
    diff = int(np.abs(arr_a - arr_b).max())
    assert diff > 0, "perturbed reconstruction is identical to original — δ was lost"
