"""Base class for modality encoders.

In the real system these wrap frozen pretrained models (ScanEZ for ET, LaBraM
for EEG, ...). The contract is simple: take a padded modality batch and return a
fixed-size embedding per sample.
"""

from __future__ import annotations

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    """Maps a raw modality batch to a fixed-size embedding of shape (B, embed_dim)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode a batch.

        Args:
            x: padded sequence of shape (B, T, F).
            mask: optional (B, T) tensor, 1 = real element, 0 = padding.

        Returns:
            Embeddings of shape (B, embed_dim).
        """
        raise NotImplementedError
