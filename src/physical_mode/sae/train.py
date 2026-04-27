"""Sparse autoencoder training (Anthropic-style L1-penalty SAE).

Reference: Bricken et al. 2023 ("Towards monosemanticity"); Pach et al. 2025
adaptation for VLM vision-encoder activations.

The SAE has tied-weight encoder/decoder (decoder is encoder^T) and a learned
input bias for centering. Training optimizes
    L = ||x - x_hat||² + λ * ||z||₁
with z = ReLU(W (x - b_pre) + b_enc), x_hat = W^T z + b_pre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, d_in: int, d_features: int):
        super().__init__()
        self.d_in = d_in
        self.d_features = d_features
        self.W = nn.Parameter(torch.empty(d_features, d_in))
        nn.init.kaiming_uniform_(self.W, a=5**0.5)
        # Normalize decoder columns (=encoder rows transposed) to unit norm.
        with torch.no_grad():
            self.W.div_(self.W.norm(dim=1, keepdim=True) + 1e-8)
        self.b_enc = nn.Parameter(torch.zeros(d_features))
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        # Input z-score normalization (set by train_sae or set_input_stats).
        self.register_buffer("input_mean", torch.zeros(d_in))
        self.register_buffer("input_std", torch.ones(d_in))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / self.input_std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in) — RAW activation. Normalization happens here.
        x_n = self.normalize_input(x)
        return torch.relu((x_n - self.b_pre) @ self.W.T + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Returns RAW-scale output (un-normalize after linear decode).
        x_hat_n = z @ self.W + self.b_pre
        return x_hat_n * self.input_std + self.input_mean

    def decode_normalized(self, z: torch.Tensor) -> torch.Tensor:
        # For training: stays in normalized space.
        return z @ self.W + self.b_pre

    def forward(self, x_normalized: torch.Tensor):
        # Training-time forward: assumes input is already normalized.
        z = torch.relu((x_normalized - self.b_pre) @ self.W.T + self.b_enc)
        x_hat = z @ self.W + self.b_pre
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W.norm(dim=1, keepdim=True) + 1e-8
        self.W.div_(norms)

    @torch.no_grad()
    def feature_contribution(self, x_raw: torch.Tensor, feature_idx: torch.Tensor) -> torch.Tensor:
        """Return the raw-scale contribution of `feature_idx` features to the SAE
        reconstruction. Used by intervention hooks to subtract specific features.

        x_raw: (..., d_in) raw activations.
        feature_idx: 1-D long tensor of feature indices.
        Returns: (..., d_in) — same shape as x_raw, contribution of those
        features (in raw scale, accounting for input_std).
        """
        z = self.encode(x_raw)  # (..., F)
        target_z = z[..., feature_idx]  # (..., k)
        target_W = self.W[feature_idx]  # (k, d_in) decoder rows in normalized space
        contribution_n = target_z @ target_W  # (..., d_in) normalized scale
        return contribution_n * self.input_std  # un-normalize


@dataclass
class TrainConfig:
    n_steps: int = 5000
    batch_size: int = 4096
    lr: float = 1e-3
    l1_lambda: float = 1.0
    log_every: int = 200
    device: str = "cuda:0"


def train_sae(activations: torch.Tensor, d_features: int,
              cfg: TrainConfig | None = None) -> tuple[SAE, dict]:
    """Train an SAE on flattened activations of shape (n_samples, d_in).

    Inputs are z-score normalized (per-dim mean/std from a sample of the
    data) so the L1 penalty has a stable scale across layers. The
    normalization stats are saved as buffers on the SAE so the intervention
    hook can apply the same transform.

    Returns the trained SAE plus a dict of training metrics.
    """
    cfg = cfg or TrainConfig()
    n, d_in = activations.shape
    sae = SAE(d_in=d_in, d_features=d_features).to(cfg.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    activations = activations.to(cfg.device)

    # Compute per-dim normalization stats from up to 100k samples.
    n_stat = min(100_000, n)
    sample_idx = torch.randperm(n, device=cfg.device)[:n_stat]
    sample = activations[sample_idx].float()
    mean = sample.mean(dim=0)
    std = sample.std(dim=0).clamp_min(1e-6)
    sae.input_mean.data.copy_(mean)
    sae.input_std.data.copy_(std)

    def normalize(x):
        return (x - sae.input_mean) / sae.input_std

    metrics = {"recon_loss": [], "l1_loss": [], "total_loss": [],
               "frac_alive": [], "frac_active_per_token": []}

    for step in range(cfg.n_steps):
        idx = torch.randint(0, n, (cfg.batch_size,), device=cfg.device)
        x = normalize(activations[idx].float())
        x_hat, z = sae(x)
        recon_loss = ((x - x_hat) ** 2).mean()
        l1_loss = z.abs().mean()
        loss = recon_loss + cfg.l1_lambda * l1_loss
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad_along = (sae.W.grad * sae.W).sum(dim=1, keepdim=True)
            sae.W.grad.sub_(grad_along * sae.W)
        optimizer.step()
        sae.normalize_decoder()

        if step % cfg.log_every == 0 or step == cfg.n_steps - 1:
            with torch.no_grad():
                frac_alive = (z.sum(dim=0) > 0).float().mean().item()
                frac_active = (z > 0).float().mean().item()
            metrics["recon_loss"].append(recon_loss.item())
            metrics["l1_loss"].append(l1_loss.item())
            metrics["total_loss"].append(loss.item())
            metrics["frac_alive"].append(frac_alive)
            metrics["frac_active_per_token"].append(frac_active)
            print(f"  step={step}/{cfg.n_steps} recon={recon_loss.item():.4f} "
                  f"l1={l1_loss.item():.4f} alive={frac_alive:.2%} "
                  f"active={frac_active:.2%}")

    return sae, metrics


def save_sae(sae: SAE, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": sae.state_dict(),
        "d_in": sae.d_in,
        "d_features": sae.d_features,
    }, path)


def load_sae(path: str | Path, device: str = "cpu") -> SAE:
    blob = torch.load(path, map_location=device)
    sae = SAE(d_in=blob["d_in"], d_features=blob["d_features"]).to(device)
    sae.load_state_dict(blob["state_dict"])
    return sae
