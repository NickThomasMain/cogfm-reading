"""Tests for the anchor registry and the stand-in anchor."""

import torch

import cogfm.anchor  # noqa: F401  (import triggers registration)
from cogfm.registry import ANCHORS


def test_anchor_registered():
    assert "standin" in ANCHORS.available()


def test_anchor_output_shape():
    anc = ANCHORS.build("standin", dim=48, vocab_size=1000)
    out = anc(["lorem ipsum dolor", "sit amet"])
    assert out.shape == (2, 48)


def test_same_text_same_vector():
    anc = ANCHORS.build("standin", dim=32, vocab_size=1000)
    a = anc(["lorem ipsum"])
    b = anc(["lorem ipsum"])
    assert torch.equal(a, b)


def test_handles_empty_text():
    anc = ANCHORS.build("standin", dim=16, vocab_size=1000)
    out = anc(["", "word"])
    assert out.shape == (2, 16)
