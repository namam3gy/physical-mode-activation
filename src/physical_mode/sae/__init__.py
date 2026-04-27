"""Sparse autoencoder utilities for physical-mode feature discovery."""

from physical_mode.sae.train import SAE, train_sae
from physical_mode.sae.feature_id import rank_physics_features

__all__ = ["SAE", "train_sae", "rank_physics_features"]
