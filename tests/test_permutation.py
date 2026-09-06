"""Tests for the permutation null: shuffling, p-values, and the two extremes."""

import numpy as np
import pytest

from cogfm.eval.metrics import mean_reciprocal_rank, recall_at_k
from cogfm.eval.permutation import (
    gather_scores,
    permutation_null,
    pool_columns,
)
from cogfm.eval.pools import build_pools


def r_at_1(ranks):
    return recall_at_k(ranks, 1)


def signal_setup(n_queries=120, n_sentences=40, pool_size=8, strength=1.0, seed=0):
    """A similarity matrix whose diagonal block is raised by ``strength``.

    With strength 0 the signal carries no information; raising it makes every
    query prefer its own sentence.
    """
    rng = np.random.default_rng(seed)
    similarity = rng.normal(size=(n_queries, n_sentences))
    targets_sentence = rng.integers(0, n_sentences, n_queries)
    similarity[np.arange(n_queries), targets_sentence] += strength

    columns = np.empty((n_queries, pool_size), dtype=int)
    positions = np.empty(n_queries, dtype=int)
    for i in range(n_queries):
        others = np.setdiff1d(np.arange(n_sentences), targets_sentence[i])
        drawn = rng.choice(others, pool_size - 1, replace=False)
        members = np.concatenate([[targets_sentence[i]], drawn])
        rng.shuffle(members)
        columns[i] = members
        positions[i] = int(np.flatnonzero(members == targets_sentence[i])[0])
    return similarity, columns, positions


def test_strong_signal_beats_the_null():
    similarity, columns, positions = signal_setup(strength=5.0)
    result = permutation_null(similarity, columns, positions, r_at_1, n_permutations=200)
    assert result.observed > result.null_mean
    assert result.p_value <= 1 / 201 + 1e-9
    assert result.z_score > 3


def test_absent_signal_lands_inside_the_null():
    similarity, columns, positions = signal_setup(strength=0.0)
    result = permutation_null(similarity, columns, positions, r_at_1, n_permutations=200)
    assert result.p_value > 0.05
    assert abs(result.z_score) < 3


def test_p_value_never_reaches_zero():
    similarity, columns, positions = signal_setup(strength=20.0)
    result = permutation_null(similarity, columns, positions, r_at_1, n_permutations=50)
    assert result.p_value == pytest.approx(1 / 51)
    assert result.p_value > 0


def test_null_mean_sits_near_the_analytic_chance_level():
    similarity, columns, positions = signal_setup(strength=0.0, pool_size=10)
    result = permutation_null(similarity, columns, positions, r_at_1, n_permutations=300)
    assert result.null_mean == pytest.approx(0.1, abs=0.03)


def test_same_seed_reproduces_the_null():
    similarity, columns, positions = signal_setup(strength=1.0)
    first = permutation_null(similarity, columns, positions, r_at_1, n_permutations=50, seed=3)
    second = permutation_null(similarity, columns, positions, r_at_1, n_permutations=50, seed=3)
    assert first.p_value == second.p_value
    assert first.null_mean == second.null_mean


def test_observed_value_does_not_depend_on_the_seed():
    similarity, columns, positions = signal_setup(strength=1.0)
    a = permutation_null(similarity, columns, positions, r_at_1, n_permutations=20, seed=0)
    b = permutation_null(similarity, columns, positions, r_at_1, n_permutations=20, seed=99)
    assert a.observed == b.observed


def test_it_works_for_any_statistic():
    similarity, columns, positions = signal_setup(strength=4.0)
    result = permutation_null(
        similarity, columns, positions, mean_reciprocal_rank, name="mrr", n_permutations=100
    )
    assert result.statistic == "mrr"
    assert result.observed > result.null_mean


def test_quantiles_bracket_the_null_mean():
    similarity, columns, positions = signal_setup(strength=0.0)
    result = permutation_null(similarity, columns, positions, r_at_1, n_permutations=300)
    assert result.quantiles["p02.5"] <= result.null_mean <= result.quantiles["p97.5"]


def test_gather_picks_the_configured_rows():
    similarity = np.arange(12, dtype=float).reshape(3, 4)
    columns = np.array([[0, 1], [2, 3], [0, 3]])
    assert gather_scores(similarity, columns).tolist() == [[0, 1], [6, 7], [8, 11]]
    assert gather_scores(similarity, columns, rows=[2, 2, 2]).tolist() == [[8, 9], [10, 11], [8, 11]]


def test_pool_columns_translates_sentence_ids():
    sentences = [{"id": i, "n_words": 20} for i in range(30)]
    pools, _ = build_pools(sentences, np.arange(30), pool_size=6, seed=0)
    order = np.arange(30)
    columns, positions = pool_columns(pools, order)
    assert columns.shape == (len(pools), 6)
    for pool, row, position in zip(pools, columns, positions):
        assert order[row].tolist() == pool.candidates.tolist()
        assert order[row[position]] == pool.sentence_id


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="pools for"):
        permutation_null(np.zeros((5, 10)), np.zeros((3, 4), dtype=int), np.zeros(3, dtype=int), r_at_1)


def test_zero_permutations_is_rejected():
    similarity, columns, positions = signal_setup()
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_null(similarity, columns, positions, r_at_1, n_permutations=0)
