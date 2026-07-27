"""Base class for the anchor.

The anchor turns text into the target vectors that every modality is bound to.
In the real system this is a frozen LLM (Qwen3) producing word/sub-word
embeddings that are pooled per sample. The contract: a list of B strings in, a
(B, dim) tensor out.
"""

from __future__ import annotations

import torch
from torch import nn


class AnchorEncoder(nn.Module):
    """Maps a batch of texts to anchor embeddings of shape (B, dim)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError
