"""Base class for connectors.

The connector is the trainable bridge: it maps a (frozen) modality embedding into
the anchor (LLM) space, where the contrastive loss compares it to the text
embedding. Input and output dims are wired at build time from the encoder's
embed_dim and the anchor's dim.

Most connectors receive one vector per trial. A connector that pools a sequence
itself receives (B, T, in_dim) instead and needs to know which steps are real,
so it sets ``wants_mask`` and is handed the batch mask alongside the input.
"""

from __future__ import annotations

import torch
from torch import nn


class Connector(nn.Module):
    """Maps a modality embedding (B, in_dim) to the anchor space (B, out_dim)."""

    # Whether the binding model should pass the batch mask to forward().
    wants_mask: bool = False

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
