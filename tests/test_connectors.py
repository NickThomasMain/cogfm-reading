"""Tests for the connector registry and the linear / MLP connectors."""

import pytest
import torch

import cogfm.connectors  # noqa: F401  (import triggers registration)
from cogfm.registry import CONNECTORS


def test_connectors_registered():
    names = CONNECTORS.available()
    assert "linear" in names
    assert "mlp" in names


@pytest.mark.parametrize("name", ["linear", "mlp"])
def test_connector_output_shape(name):
    conn = CONNECTORS.build(name, in_dim=32, out_dim=64)
    x = torch.randn(8, 32)  # (B, in_dim)
    out = conn(x)
    assert out.shape == (8, 64)


def test_unknown_connector_raises():
    with pytest.raises(KeyError):
        CONNECTORS.build("does-not-exist", in_dim=1, out_dim=1)
