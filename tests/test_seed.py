"""Tests for the seeding helper: same seed -> same draws, different seed -> different."""

import numpy as np

from cogfm.seed import set_seed


def test_same_seed_reproduces_numpy_draws():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seed_gives_different_draws():
    set_seed(1)
    a = np.random.rand(5)
    set_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_torch_reproducible_if_available():
    try:
        import torch
    except ImportError:
        import pytest

        pytest.skip("torch not installed")
    set_seed(7)
    a = torch.rand(4)
    set_seed(7)
    b = torch.rand(4)
    assert torch.equal(a, b)
