"""The binding model: the two towers that end in the same anchor space.

Bundles the modality tower (encoder -> connector) and the anchor into one
module, exposing two methods that both return (B, D_anchor) embeddings:

    encode_modality(scanpath, mask) : (B, T, F) -> (B, D_anchor)
    encode_text(texts)             : list[str] -> (B, D_anchor)

The binding loss is applied outside this model (in the training loop), so it
stays swappable.
"""

from __future__ import annotations

import torch
from torch import nn

from cogfm.anchor.base import AnchorEncoder
from cogfm.connectors.base import Connector
from cogfm.encoders.base import ModalityEncoder


class BindingModel(nn.Module):
    def __init__(
        self,
        encoder: ModalityEncoder,
        connector: Connector,
        anchor: AnchorEncoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.connector = connector
        self.anchor = anchor

    def encode_modality(
        self, scanpath: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.encoder(scanpath, mask)  # (B, D_enc)
        return self.connector(features)  # (B, D_anchor)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.anchor(texts)  # (B, D_anchor)
