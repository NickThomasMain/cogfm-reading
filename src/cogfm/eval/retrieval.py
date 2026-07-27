"""Retrieval@k from a (B, B) similarity matrix.

Row i holds the similarity of modality sample i to every text in the batch; by
convention the correct match for row i is column i (the diagonal). Retrieval@k
is the fraction of rows whose correct column is among that row's top-k highest
similarities. Chance level is k / B.
"""

from __future__ import annotations

import torch


def retrieval_at_k(logits: torch.Tensor, k: int = 1) -> float:
    """Fraction of rows whose diagonal entry is within the top-k of that row.

    Args:
        logits: (B, B) similarity matrix (higher = more similar).
        k: how many top candidates count as a hit.

    Returns:
        A float in [0, 1].
    """
    n = logits.size(0)
    k = min(k, n)
    topk_cols = logits.topk(k, dim=1).indices  # (B, k): best columns per row
    correct = torch.arange(n, device=logits.device).unsqueeze(1)  # (B, 1): the diagonal
    hits = (topk_cols == correct).any(dim=1)  # (B,): was the correct col in top-k?
    return hits.float().mean().item()
