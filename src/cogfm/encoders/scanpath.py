"""Placeholder eye-tracking scanpath encoder.

Random-initialised, NOT a real model — its only job is to give the pipeline a
correctly shaped modality embedding. It can be replaced by a real, pretrained,
frozen scanpath encoder (e.g. ScanEZ) behind the same interface.

It projects each fixation's (x, y, duration) to embed_dim and mean-pools over the
(unpadded) fixations. Since an affine map commutes with an average, the result
equals a projection of the mean fixation: the whole scanpath collapses to its
mean x, mean y and mean duration, and a shuffled scanpath encodes identically.
That makes it useful as a smoke-test component and unsuitable as a stand-in for
a sequence model; see ScanEZEncoder for one that reads the order.
"""

from __future__ import annotations

import torch
from torch import nn

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS


@ENCODERS.register("scanpath")
class ScanpathEncoder(ModalityEncoder):
    def __init__(self, embed_dim: int = 128, n_features: int = 3) -> None:
        super().__init__(embed_dim)
        self.proj = nn.Linear(n_features, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)  # (B, T, D)
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)  # (B, T, 1)
            h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)
        return h  # (B, D)
