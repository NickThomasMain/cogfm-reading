"""An encoder for vectors that were computed once and stored.

Frozen encoders produce the same vector for a trial every time they are asked.
Where that vector is expensive -- LaBraM over a full trial -- it is computed
once by a separate script and read back from disk. This encoder is what puts
those stored vectors behind the encoder interface, so the condition runs through
the same connector, loss, split and pool as every other one.

It holds no parameters and learns nothing; the training signal reaches the
connector alone. That is the same arrangement as a frozen encoder in the loop,
with the forward pass already done.

The stored vector arrives as a length-one sequence, (B, 1, D), because batching
and evaluation expect a sequence with a mask. There is nothing to pool over and
nothing to mask, so the time axis is simply dropped.

With ``sequence=True`` the stored trial is a real sequence -- one vector per
one-second patch -- and the time axis is kept instead. Pooling then belongs to
the connector, which is the point: a mean fixed here would decide in advance
that every second of a trial counts the same.
"""

from __future__ import annotations

import torch

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS


@ENCODERS.register("precomputed")
class PrecomputedEncoder(ModalityEncoder):
    """Passes a stored embedding through unchanged.

    Args:
        embed_dim: width of the stored vectors. Checked against every batch, so
            a config pointing at an embedding file of a different width fails
            immediately instead of training on a silently reshaped tensor.
        sequence: keep the time axis and pass the sequence on, for stored files
            that hold one vector per patch. The connector then does the pooling
            and must accept a sequence. False expects one step per trial and
            rejects anything longer, which catches a config paired with the
            wrong embedding file instead of averaging it away unnoticed.
    """

    def __init__(self, embed_dim: int, sequence: bool = False) -> None:
        super().__init__(embed_dim)
        self.sequence = bool(sequence)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, steps, {self.embed_dim}), got {tuple(x.shape)}")
        if x.shape[2] != self.embed_dim:
            raise ValueError(
                f"embed_dim {self.embed_dim} does not match the stored width {x.shape[2]}"
            )
        if self.sequence:
            return x
        if x.shape[1] != 1:
            raise ValueError(
                f"a stored mean is one step long, got {x.shape[1]}. Either the file holds "
                "patch sequences, in which case set sequence=True, or the dataset is "
                "serving raw signal rather than embeddings."
            )
        return x[:, 0, :]
