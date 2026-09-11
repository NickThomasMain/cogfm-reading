"""A connector that learns how to weight a sequence before projecting it.

Where the modality arrives as a sequence -- one vector per one-second patch of
EEG, say -- something has to reduce it to a single vector before the contrastive
loss can compare it to a sentence. Doing that with a mean fixes the answer in
advance: every second counts the same. Over a five-second trial that erases
anything that sits in one of them, and language-related components in reading
EEG are exactly that kind of thing, a few hundred milliseconds wide.

This connector leaves the decision to training. A single learned query scores
every step, the scores become weights, and the weighted sum goes through the
same two-layer projection the plain MLP connector uses. The mean remains
reachable -- uniform scores produce it -- so this cannot do worse than the mean
for any reason other than optimisation.

Padded steps are removed before the scores are normalised, not after: masking
afterwards would leave the padding influencing the denominator, and a trial's
representation would then depend on which batch it landed in.

A (B, D) input is accepted and treated as one step, which makes the same
connector usable for a condition whose encoder returns a single vector -- the
floor condition of a result row, for instance -- so that a row does not have to
vary the connector alongside the encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from cogfm.connectors.base import Connector
from cogfm.registry import CONNECTORS

# Added to a fully padded row's denominator; such a row cannot occur through the
# batching, which always leaves at least one real step, but a weight vector that
# sums to zero would produce NaN rather than an error.
EPSILON = 1e-6


@CONNECTORS.register("attention_pool")
class AttentionPoolConnector(Connector):
    """Pools a sequence with learned weights, then projects it.

    Args:
        in_dim: width of one step.
        out_dim: width of the anchor space.
        hidden_dim: width of the projection's hidden layer.
        dropout: applied inside the projection, as in the MLP connector.
        score_dim: width of the space the query and the steps are scored in.
    """

    wants_mask = True

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        score_dim: int = 128,
    ) -> None:
        super().__init__(in_dim, out_dim)
        self.key = nn.Linear(in_dim, score_dim)
        self.query = nn.Parameter(torch.zeros(score_dim))
        nn.init.normal_(self.query, std=score_dim ** -0.5)
        self.scale = score_dim ** -0.5
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
            mask = None
        if x.ndim != 3:
            raise ValueError(f"expected (batch, steps, {self.in_dim}), got {tuple(x.shape)}")
        if x.shape[2] != self.in_dim:
            raise ValueError(f"in_dim {self.in_dim} does not match the input width {x.shape[2]}")

        scores = (self.key(x) @ self.query) * self.scale  # (B, T)
        if mask is not None:
            if mask.shape != x.shape[:2]:
                raise ValueError(
                    f"mask {tuple(mask.shape)} does not match the batch {tuple(x.shape[:2])}"
                )
            scores = scores.masked_fill(mask <= 0, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights)  # a fully padded row would be all -inf
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp(
            min=EPSILON
        )
        return self.net(pooled)
