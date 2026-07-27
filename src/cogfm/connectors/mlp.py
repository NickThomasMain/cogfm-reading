"""Two-layer MLP connector with a non-linearity in between."""

from __future__ import annotations

import torch
from torch import nn

from cogfm.connectors.base import Connector
from cogfm.registry import CONNECTORS


@CONNECTORS.register("mlp")
class MLPConnector(Connector):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
