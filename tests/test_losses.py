"""Tests for the loss registry and the InfoNCE loss."""

import torch

import cogfm.losses  # noqa: F401  (import triggers registration)
from cogfm.registry import LOSSES


def test_infonce_registered():
    assert "infonce" in LOSSES.available()


def test_infonce_returns_scalar_and_matrix():
    loss_fn = LOSSES.build("infonce", temperature=0.07)
    B, D = 6, 16
    loss, logits = loss_fn(torch.randn(B, D), torch.randn(B, D))
    assert loss.ndim == 0  # scalar
    assert logits.shape == (B, B)
    assert loss.item() > 0


def test_infonce_lower_when_aligned():
    loss_fn = LOSSES.build("infonce", temperature=0.07)
    B, D = 8, 32
    a = torch.randn(B, D)
    loss_aligned, _ = loss_fn(a.clone(), a.clone())  # perfect match
    loss_random, _ = loss_fn(torch.randn(B, D), a)  # unrelated
    assert loss_aligned < loss_random


def test_infonce_is_differentiable():
    loss_fn = LOSSES.build("infonce")
    m = torch.randn(4, 16, requires_grad=True)
    loss, _ = loss_fn(m, torch.randn(4, 16))
    loss.backward()
    assert m.grad is not None
