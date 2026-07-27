"""Tests for BindingModel: both towers end in the same anchor space."""

import torch

import cogfm.anchor  # noqa: F401  (register)
import cogfm.connectors  # noqa: F401  (register)
import cogfm.encoders  # noqa: F401  (register)
from cogfm.binding.model import BindingModel
from cogfm.registry import ANCHORS, CONNECTORS, ENCODERS


def _build(embed_dim: int = 16, anchor_dim: int = 16) -> BindingModel:
    encoder = ENCODERS.build("scanpath", embed_dim=embed_dim)
    connector = CONNECTORS.build("mlp", in_dim=embed_dim, out_dim=anchor_dim)
    anchor = ANCHORS.build("standin", dim=anchor_dim, vocab_size=500)
    return BindingModel(encoder, connector, anchor)


def test_encode_modality_shape():
    model = _build()
    out = model.encode_modality(torch.randn(4, 10, 3))
    assert out.shape == (4, 16)


def test_encode_text_shape():
    model = _build()
    out = model.encode_text(["lorem ipsum", "dolor sit", "amet", "a b c"])
    assert out.shape == (4, 16)


def test_both_towers_share_output_dim():
    model = _build(embed_dim=8, anchor_dim=16)
    m = model.encode_modality(torch.randn(3, 5, 3))
    t = model.encode_text(["a", "b c", "d"])
    assert m.shape[1] == t.shape[1] == 16
