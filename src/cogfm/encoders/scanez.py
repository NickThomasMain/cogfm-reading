"""A transformer over the fixation sequence.

Follows the shape ScanEZ describes: the three gaze features of each fixation
are projected into a latent space, sinusoidal encodings mark the position, and
self-attention layers relate the fixations to one another. Pooling over the
real positions turns the sequence into one vector.

The same class serves two conditions. Left at its random initialisation and
frozen, it shows what the architecture contributes on its own, before any
pretraining; loaded from a checkpoint, it is the pretrained encoder. Keeping
both in one class means the two conditions differ in their weights and in
nothing else.

Order matters here, which is the point. A projection followed by a mean can be
rewritten as a projection of the mean, so such an encoder sees only the average
fixation and cannot tell a scanpath from a shuffled copy of itself. Attention
between positions removes that, and the positional encoding makes the order
itself visible.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import nn

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS

log = logging.getLogger(__name__)


def sinusoidal_encoding(length: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """Fixed position codes of shape (length, width)."""
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    step = torch.arange(0, width, 2, device=device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * step / width)
    codes = torch.zeros(length, width, device=device, dtype=torch.float32)
    codes[:, 0::2] = torch.sin(position * frequency)
    codes[:, 1::2] = torch.cos(position * frequency[: codes[:, 1::2].shape[1]])
    return codes.to(dtype) if dtype is not None else codes


@ENCODERS.register("scanez")
class ScanEZEncoder(ModalityEncoder):
    """Encodes a scanpath into one vector via self-attention.

    Args:
        embed_dim: width of the latent space and of the output vector.
        n_features: features per fixation, three for (x, y, duration).
        n_layers: self-attention blocks.
        n_heads: attention heads per block.
        feedforward_dim: width of the block's inner layer.
        dropout: only active while training; the frozen conditions see none.
        max_length: longest sequence the position codes cover.
        checkpoint: weights to load; random initialisation when None.

    Raises:
        ValueError: when the width does not divide into the heads, or when a
            named checkpoint does not exist.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_features: int = 3,
        n_layers: int = 4,
        n_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        max_length: int = 512,
        checkpoint: str | None = None,
    ) -> None:
        if embed_dim % n_heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by n_heads {n_heads}")
        super().__init__(embed_dim)

        self.input_projection = nn.Linear(n_features, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.register_buffer(
            "positions", sinusoidal_encoding(max_length, embed_dim), persistent=False
        )
        self.max_length = max_length

        if checkpoint:
            path = Path(checkpoint)
            if not path.is_file():
                raise ValueError(f"checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu")
            self.load_state_dict(state.get("model", state))
            log.info("scanez weights loaded from %s", path)
        else:
            log.info("scanez left at its random initialisation")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        length = x.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence of {length} exceeds max_length {self.max_length}")

        h = self.input_projection(x) + self.positions[:length].unsqueeze(0).to(x.dtype)

        padding = None if mask is None else (mask == 0)
        h = self.blocks(h, src_key_padding_mask=padding)

        if mask is None:
            return h.mean(dim=1)
        weights = mask.unsqueeze(-1).to(h.dtype)
        return (h * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
