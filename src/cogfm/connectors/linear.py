"""Simplest connector: a single linear projection."""

from __future__ import annotations

import torch
from torch import nn

from cogfm.connectors.base import Connector
from cogfm.registry import CONNECTORS


@CONNECTORS.register("linear")
class LinearConnector(Connector):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
