"""MLPResampler — Idefics2 perceiver-resampler counterfactual.

Replaces ``Idefics2PerceiverResampler`` with a 1-layer cross-attention pool +
2-layer GELU MLP that projects variable-length context to the same fixed
``(B, n_latents, hidden)`` budget the LM expects.

Strict pure-MLP per-token (LLaVA-style, no token compression) would change the
LM's expected visual-token count from 64 to thousands, which is a different
intervention. We hold the token budget fixed and only swap the *aggregation
operator*, so any downstream §4.6 / M5b shift is attributable to the operator
swap rather than to the visual-token count.

For Pillar B M-PSwap (`references/submission_plan.md` §2 / `references/paper_gaps.md` G3):
the controlled comparison is encoder + LM held fixed; only this connector
sub-module changes. The LM gets LoRA adapters (rank-32, alpha-64 on
q/v/k/o_proj) so it can learn to consume the new module's output statistics
without touching base weights.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPPoolResampler(nn.Module):
    """Cross-attention-pool + MLP replacement for Idefics2's perceiver-resampler.

    Forward signature matches ``Idefics2PerceiverResampler.forward``:
        forward(context, attention_mask, **kwargs) -> Tensor of shape (B, n_latents, hidden_size)

    Args:
        hidden_size: post-modality_projection embedding dim (4096 for Idefics2).
        n_latents: output token budget (64 for Idefics2 — held fixed for fair compare).
        n_heads: attention heads in the pool (8 by default — full MHA, not GQA).
        intermediate_size: MLP hidden dim (4 × hidden_size by default).
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        n_latents: int = 64,
        n_heads: int = 8,
        intermediate_size: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_latents = n_latents
        self.n_heads = n_heads
        intermediate_size = intermediate_size or 4 * hidden_size

        self.queries = nn.Parameter(torch.randn(n_latents, hidden_size) * 0.02)
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # context: (B, S, H), attention_mask: (B, S) where 1 = real token, 0 = pad.
        # FP32 master-weights regime: incoming context may be bf16 (the upstream
        # SigLIP encoder + modality_projection run in bf16), but this module's
        # parameters are fp32. Cast on entry, cast back on exit. This avoids the
        # ~1500-step bf16 round-off accumulation that caused two NaN collapses
        # in our earlier runs (Run 1 step 1475, Run 2 step 1999).
        in_dtype = context.dtype
        b, _s, h = context.shape
        if h != self.hidden_size:
            raise ValueError(f"context hidden {h} != module hidden {self.hidden_size}")

        context = context.float()
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        # attention_mask is sometimes redundant (Idefics2 do_image_splitting=False
        # → all keys valid) but we honor it when present.
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        pooled, _ = self.attn_pool(
            query=q,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm1(q + pooled)
        out = self.norm2(pooled + self.mlp(pooled))
        return out.to(in_dtype)


def swap_perceiver_to_mlp_pool(
    model: nn.Module,
    n_heads: int = 8,
    seed: int | None = 42,
) -> MLPPoolResampler:
    """Replace ``model.model.connector.perceiver_resampler`` with a fresh ``MLPPoolResampler``.

    Returns the new module so callers can register it with PEFT
    ``modules_to_save`` (full-finetune) or inspect it.

    The replacement preserves dtype/device of the original perceiver.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "connector"):
        raise AttributeError("model.model.connector not found — is this Idefics2?")
    connector = model.model.connector
    pr = connector.perceiver_resampler
    hidden_size = pr.hidden_size
    n_latents = pr.n_latents
    # Inherit device from the original perceiver, but force fp32 master weights:
    # this module is the only sub-module with full-finetune (not LoRA) so PEFT's
    # auto-promote to fp32 doesn't apply to it. Keeping it at the base model's
    # bf16 caused ~1500-step round-off accumulation → NaN in two prior runs.
    p = next(pr.parameters())
    device = p.device

    if seed is not None:
        torch.manual_seed(seed)

    new_module = MLPPoolResampler(
        hidden_size=hidden_size,
        n_latents=n_latents,
        n_heads=n_heads,
    ).to(dtype=torch.float32, device=device)

    connector.perceiver_resampler = new_module
    return new_module


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
