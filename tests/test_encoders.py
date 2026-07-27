"""Tests for the encoder registry and the placeholder scanpath encoder."""

import pytest
import torch

import cogfm.encoders  # noqa: F401  (import triggers registration)
from cogfm.registry import ENCODERS


def test_scanpath_registered():
    assert "scanpath" in ENCODERS.available()


def test_scanpath_output_shape():
    enc = ENCODERS.build("scanpath", embed_dim=32)
    x = torch.randn(4, 7, 3)  # (B, T, F)
    out = enc(x)
    assert out.shape == (4, 32)


def test_scanpath_respects_mask():
    enc = ENCODERS.build("scanpath", embed_dim=16)
    x = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = enc(x, mask=mask)
    assert out.shape == (2, 16)


def test_unknown_encoder_raises():
    with pytest.raises(KeyError):
        ENCODERS.build("does-not-exist")
