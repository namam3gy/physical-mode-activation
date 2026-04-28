"""Identify SAE features that selectively activate on physics-mode stim."""

from __future__ import annotations

import torch

from physical_mode.sae.train import SAE


@torch.no_grad()
def _encode_chunked(sae: SAE, activations: torch.Tensor, chunk_size: int = 50_000) -> torch.Tensor:
    """Encode in chunks to avoid OOM on large activation sets.

    Idefics2 5-tile × ~390 phys stim ≈ 2.5M tokens — encoding all at once
    requires ~46 GB output (4608 features × float32) which OOMs even on H200
    when external workloads are also resident. Chunked encoding moves chunks
    to CPU before concatenating, leaving only working memory on GPU.
    """
    device = next(sae.parameters()).device
    out_chunks = []
    for i in range(0, activations.shape[0], chunk_size):
        chunk = activations[i:i + chunk_size].to(device).float()
        z = sae.encode(chunk).cpu()
        out_chunks.append(z)
        del chunk
    return torch.cat(out_chunks, dim=0)


@torch.no_grad()
def rank_physics_features(
    sae: SAE,
    activations_phys: torch.Tensor,
    activations_abs: torch.Tensor,
    top_k: int = 20,
) -> dict:
    """Rank SAE features by activation gap between physics-mode and abstract-mode stim.

    Returns both raw mean delta and Cohen's d (mean delta divided by pooled
    std). Cohen's d filters high-baseline outliers whose raw delta is large in
    absolute terms but small relative to the per-feature variance.

    Args:
        sae: trained SAE.
        activations_phys: (N_phys, d_in) — activations on physics-mode stim
            (per-token; flatten before passing).
        activations_abs: (N_abs, d_in) — activations on abstract-mode stim.
        top_k: number of top features to return.

    Returns:
        dict with `mean_phys`, `mean_abs`, `std_phys`, `std_abs`, `pooled_std`,
        `delta`, `cohens_d`, plus `top_k_indices_delta`, `top_k_deltas`,
        `top_k_indices_cohen`, `top_k_cohens_d`.
    """
    z_phys = _encode_chunked(sae, activations_phys)  # CPU (N_phys, F)
    z_abs = _encode_chunked(sae, activations_abs)  # CPU (N_abs, F)

    mean_phys = z_phys.mean(dim=0)
    mean_abs = z_abs.mean(dim=0)
    delta = mean_phys - mean_abs

    n_phys = z_phys.shape[0]
    n_abs = z_abs.shape[0]
    std_phys = z_phys.std(dim=0, unbiased=True)
    std_abs = z_abs.std(dim=0, unbiased=True)
    # Pooled SD with Welch-style fallback to avoid div-by-zero on dead features.
    pooled_var = (
        (n_phys - 1) * std_phys.pow(2) + (n_abs - 1) * std_abs.pow(2)
    ) / max(n_phys + n_abs - 2, 1)
    pooled_std = pooled_var.clamp_min(1e-8).sqrt()
    cohens_d = delta / pooled_std

    top_idx_delta = torch.argsort(delta, descending=True)[:top_k]
    top_idx_cohen = torch.argsort(cohens_d, descending=True)[:top_k]
    return {
        "mean_phys": mean_phys,
        "mean_abs": mean_abs,
        "std_phys": std_phys,
        "std_abs": std_abs,
        "pooled_std": pooled_std,
        "delta": delta,
        "cohens_d": cohens_d,
        "top_k_indices_delta": top_idx_delta,
        "top_k_deltas": delta[top_idx_delta],
        "top_k_indices_cohen": top_idx_cohen,
        "top_k_cohens_d": cohens_d[top_idx_cohen],
        # Back-compat alias for existing callers.
        "top_k_indices": top_idx_delta,
    }
