"""Tests for retrieval_at_k."""

import torch

from cogfm.eval.retrieval import retrieval_at_k


def test_perfect_retrieval_is_one():
    logits = torch.eye(4) * 10.0  # diagonal is clearly the largest per row
    assert retrieval_at_k(logits, k=1) == 1.0


def test_worst_case_at_1_is_zero():
    logits = torch.ones(3, 3)
    logits.fill_diagonal_(-1.0)  # correct match is the smallest -> never top-1
    assert retrieval_at_k(logits, k=1) == 0.0


def test_larger_k_never_smaller():
    torch.manual_seed(0)
    logits = torch.randn(10, 10)
    assert retrieval_at_k(logits, k=5) >= retrieval_at_k(logits, k=1)


def test_k_equal_batch_is_one():
    logits = torch.randn(6, 6)
    assert retrieval_at_k(logits, k=6) == 1.0
