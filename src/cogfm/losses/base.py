"""Base class for binding losses.

A binding loss compares a batch of modality embeddings with the matching batch
of anchor (text) embeddings and returns:
  - a scalar loss to train on, and
  - a (B, B) similarity matrix (row = modality sample, col = anchor sample),
    reused downstream to compute retrieval@k.

By convention the i-th modality embedding and the i-th anchor embedding are the
true (positive) pair; all off-diagonal pairs are negatives.
"""

from __future__ import annotations

import torch
from torch import nn


class BindingLoss(nn.Module):
    def forward(
        self, modality: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: modality (B, D), anchor (B, D). Returns: (loss, logits (B, B))."""
        raise NotImplementedError
