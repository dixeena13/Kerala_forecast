from __future__ import annotations

import torch
from torch import nn


class PrecipitationRegressor(nn.Module):
    """Small baseline model for local precipitation postprocessing."""

    def __init__(self, n_features: int = 5) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

