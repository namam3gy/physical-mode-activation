"""Identify SAE features that selectively activate on physics-mode stim."""

from __future__ import annotations

import torch

from physical_mode.sae.train import SAE


@torch.no_grad()
def rank_physics_features(
    sae: SAE,
    activations_phys: torch.Tensor,
    activations_abs: torch.Tensor,
    top_k: int = 20,
) -> dict:
    """Rank SAE features by activation gap between physics-mode and abstract-mode stim.

    Args:
        sae: trained SAE.
        activations_phys: (N_phys, d_in) — activations on physics-mode stim
            (per-token; flatten before passing).
        activations_abs: (N_abs, d_in) — activations on abstract-mode stim.
        top_k: number of top features to return.

    Returns:
        dict with `mean_phys`, `mean_abs`, `delta`, `top_k_indices`,
        `top_k_deltas`.
    """
    device = next(sae.parameters()).device
    z_phys = sae.encode(activations_phys.to(device).float())  # (N_phys, F)
    z_abs = sae.encode(activations_abs.to(device).float())  # (N_abs, F)

    mean_phys = z_phys.mean(dim=0)
    mean_abs = z_abs.mean(dim=0)
    delta = mean_phys - mean_abs

    top_idx = torch.argsort(delta, descending=True)[:top_k]
    return {
        "mean_phys": mean_phys.cpu(),
        "mean_abs": mean_abs.cpu(),
        "delta": delta.cpu(),
        "top_k_indices": top_idx.cpu(),
        "top_k_deltas": delta[top_idx].cpu(),
    }
