"""Tests for retrieval metrics: ranks, ties, chance levels, hand-computed cases."""

import numpy as np
import pytest

from cogfm.eval.metrics import (
    chance_mean_reciprocal_rank,
    chance_recall_at_k,
    format_summary,
    mean_reciprocal_rank,
    percentile_rank,
    recall_at_k,
    summarize_ranks,
    target_ranks,
)


def test_perfect_scores_give_rank_one():
    scores = np.array([[9.0, 1.0, 0.0], [0.0, 9.0, 1.0]])
    assert target_ranks(scores, [0, 1]).tolist() == [1.0, 1.0]


def test_worst_scores_give_the_last_rank():
    scores = np.array([[0.0, 1.0, 9.0], [9.0, 1.0, 0.0]])
    assert target_ranks(scores, [0, 2]).tolist() == [3.0, 3.0]


def test_rank_counts_how_many_score_higher():
    scores = np.array([[5.0, 9.0, 1.0, 7.0]])
    # target scores 5.0; 9.0 and 7.0 beat it, so it comes third
    assert target_ranks(scores, [0]).tolist() == [3.0]


def test_all_scores_equal_puts_the_target_in_the_middle():
    """A model with no preference must land at chance, not at either extreme."""
    scores = np.zeros((4, 25))
    ranks = target_ranks(scores, [0, 7, 24, 13])
    assert ranks.tolist() == [13.0] * 4
    assert percentile_rank(ranks, 25) == pytest.approx(0.5)


def test_partial_ties_take_the_centre_of_the_tied_block():
    scores = np.array([[9.0, 5.0, 5.0, 5.0]])
    # one candidate is better; the target shares second to fourth place
    assert target_ranks(scores, [1]).tolist() == [3.0]


def test_recall_counts_ranks_within_k():
    ranks = np.array([1.0, 2.0, 6.0, 25.0])
    assert recall_at_k(ranks, 1) == pytest.approx(0.25)
    assert recall_at_k(ranks, 5) == pytest.approx(0.5)
    assert recall_at_k(ranks, 10) == pytest.approx(0.75)


def test_mrr_matches_a_hand_computed_case():
    ranks = np.array([1.0, 2.0, 4.0])
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_percentile_spans_one_to_zero():
    assert percentile_rank(np.array([1.0]), 25) == pytest.approx(1.0)
    assert percentile_rank(np.array([25.0]), 25) == pytest.approx(0.0)


def test_chance_levels_follow_the_pool_size():
    assert chance_recall_at_k(25, 1) == pytest.approx(0.04)
    assert chance_recall_at_k(25, 5) == pytest.approx(0.20)
    assert chance_recall_at_k(25, 10) == pytest.approx(0.40)
    assert chance_mean_reciprocal_rank(2) == pytest.approx(0.75)


def test_random_scores_land_near_chance():
    """A large sample of noise must reproduce the analytic chance levels."""
    rng = np.random.default_rng(0)
    pool_size = 25
    scores = rng.normal(size=(20000, pool_size))
    ranks = target_ranks(scores, rng.integers(0, pool_size, 20000))
    assert recall_at_k(ranks, 1) == pytest.approx(chance_recall_at_k(pool_size, 1), abs=0.01)
    assert recall_at_k(ranks, 5) == pytest.approx(chance_recall_at_k(pool_size, 5), abs=0.02)
    assert mean_reciprocal_rank(ranks) == pytest.approx(
        chance_mean_reciprocal_rank(pool_size), abs=0.02
    )
    assert percentile_rank(ranks, pool_size) == pytest.approx(0.5, abs=0.02)


def test_summary_carries_a_chance_value_for_every_metric():
    summary = summarize_ranks(np.array([1.0, 3.0, 20.0]), pool_size=25)
    for key in ("recall@1", "recall@5", "recall@10", "mrr", "percentile", "mean_rank"):
        assert key in summary
        assert f"chance_{key}" in summary


def test_summary_skips_k_larger_than_the_pool():
    summary = summarize_ranks(np.array([1.0, 2.0]), pool_size=5)
    assert "recall@1" in summary
    assert "recall@5" in summary
    assert "recall@10" not in summary


def test_format_shows_value_and_chance():
    text = format_summary(summarize_ranks(np.array([1.0, 2.0, 3.0]), pool_size=25))
    assert "recall@1" in text
    assert "Zufall" in text


def test_empty_input_is_not_an_error():
    ranks = target_ranks(np.zeros((0, 25)), np.array([], dtype=int))
    assert len(ranks) == 0
    assert np.isnan(recall_at_k(ranks, 1))
    assert np.isnan(mean_reciprocal_rank(ranks))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="target positions"):
        target_ranks(np.zeros((3, 5)), [0, 1])


def test_target_outside_the_pool_is_rejected():
    with pytest.raises(ValueError, match="outside the pool"):
        target_ranks(np.zeros((2, 5)), [0, 7])


def test_one_dimensional_scores_are_rejected():
    with pytest.raises(ValueError, match="2-dimensional"):
        target_ranks(np.zeros(5), [0])
