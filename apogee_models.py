"""Shared PyTorch models and raw/projected apogee contracts.

Importing this module requires PyTorch.  The NumPy/pandas pipeline remains
usable without it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn as nn

G = 9.81


def _state_dict_from_physics_wrapper(state: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Strip the legacy PhysicsInformedGRU ``gru.`` prefix."""
    cleaned = OrderedDict()
    for key, value in state.items():
        cleaned[key[4:] if key.startswith("gru.") else key] = value
    return cleaned


class GRUApogeeModel(nn.Module):
    """GRU that returns raw r, without a hidden lower-bound projection."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128,
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        gru_out, hidden = self.gru(x, hidden)
        r_raw = 1.0 + 0.5 * self.fc(gru_out[:, -1, :]).squeeze(-1)
        return r_raw, hidden

    def load_legacy_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(_state_dict_from_physics_wrapper(state))


class SnapshotApogeeModel(nn.Module):
    """MLP baseline using only the endpoint feature vector."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        r_raw = 1.0 + 0.5 * self.net(x).squeeze(-1)
        return r_raw


def validate_physics_inputs(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(h).all() or not torch.isfinite(v).all():
        raise ValueError("h/v must be finite")
    denom = h + v**2 / (2 * G)
    if not torch.isfinite(denom).all() or not (denom > 0).all():
        raise ValueError("h + v^2/(2g) must be finite and positive")
    return denom


def raw_apogee(h: torch.Tensor, v: torch.Tensor, r_raw: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(r_raw).all():
        raise ValueError("r_raw must be finite")
    return validate_physics_inputs(h, v) * r_raw


def project_apogee(h: torch.Tensor, v: torch.Tensor, r_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projected height, projected ratio, and below-current mask."""
    denom = validate_physics_inputs(h, v)
    h_raw = denom * r_raw
    floor = h / denom
    below = h_raw < h
    projected_ratio = torch.maximum(r_raw, floor)
    return denom * projected_ratio, projected_ratio, below


def raw_apogee_loss(
    r_pred: torch.Tensor,
    r_target: torch.Tensor,
    h_pred: torch.Tensor,
    h_target: torch.Tensor,
    *,
    height_weight: float = 0.1,
) -> torch.Tensor:
    """MSE(r) + height_weight * MSE(H/100), with no lower-bound projection."""
    if not torch.isfinite(r_pred).all() or not torch.isfinite(h_pred).all():
        raise ValueError("model outputs must be finite")
    r_loss = torch.mean((r_pred - r_target) ** 2)
    h_loss = torch.mean((h_pred / 100.0 - h_target / 100.0) ** 2)
    return r_loss + height_weight * h_loss


def score_state_dict(state: Mapping[str, torch.Tensor]) -> OrderedDict:
    """Copy a state dict to CPU without sharing storage (for best restore)."""
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items())
