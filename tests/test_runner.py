"""Tests for evaluate_similarity: pooling, dropped queries, and the two extremes."""

import numpy as np
import pytest

from cogfm.eval.runner import FoldEvaluation, evaluate_similarity


def sentences(lengths: list[int]) -> list[dict]:
    return [{"id": i, "n_words": n, "task": "SR"} for i, n in enumerate(lengths)]


def setup(n_sentences=60, n_per_sentence=3, strength=0.0, seed=0):
    """A similarity matrix with several queries per sentence, all same length."""
    rng = np.random.default_rng(seed)
    records = sentences([20] * n_sentences)
    order = np.arange(n_sentences)
    query_sentences = np.repeat(order, n_per_sentence)
    similarity = rng.normal(size=(len(query_sentences), n_sentences))
    similarity[np.arange(len(query_sentences)), query_sentences] += strength
    return similarity, order, query_sentences, records


def test_no_signal_stays_at_chance():
    similarity, order, queries, records = setup(strength=0.0)
    result = evaluate_similarity(
        similarity, order, queries, records, pool_size=25, n_permutations=100
    )
    assert result.metrics["recall@1"] == pytest.approx(0.04, abs=0.06)
    assert result.permutation["recall@1"].p_value > 0.05


def test_strong_signal_beats_the_null():
    similarity, order, queries, records = setup(strength=6.0)
    result = evaluate_similarity(
        similarity, order, queries, records, pool_size=25, n_permutations=100
    )
    assert result.metrics["recall@1"] > 0.9
    assert result.permutation["recall@1"].p_value <= 1 / 101 + 1e-9


def test_every_query_is_kept_when_all_sentences_have_pools():
    similarity, order, queries, records = setup()
    result = evaluate_similarity(similarity, order, queries, records, pool_size=25, n_permutations=0)
    assert result.n_queries == len(queries)
    assert result.n_dropped_queries == 0
    assert result.n_unservable_sentences == 0


def test_queries_on_unservable_sentences_are_dropped_and_counted():
    """One sentence is far longer than the rest, so it gets no pool."""
    records = sentences([20] * 40 + [95])
    order = np.arange(41)
    queries = np.repeat(order, 2)
    rng = np.random.default_rng(0)
    similarity = rng.normal(size=(len(queries), 41))

    result = evaluate_similarity(
        similarity, order, queries, records, pool_size=25, tolerance=3, n_permutations=0
    )
    assert result.n_unservable_sentences == 1
    assert result.n_dropped_queries == 2
    assert result.n_queries == len(queries) - 2


def test_several_subjects_share_one_pool():
    """Queries on the same sentence must face the same alternatives."""
    similarity, order, queries, records = setup(n_sentences=40, n_per_sentence=4)
    result = evaluate_similarity(similarity, order, queries, records, pool_size=10, n_permutations=0)
    assert result.n_pools == 40
    assert result.n_queries == 160


def test_pool_size_reaches_the_metrics_and_the_chance_level():
    similarity, order, queries, records = setup()
    result = evaluate_similarity(similarity, order, queries, records, pool_size=10, n_permutations=0)
    assert result.metrics["pool_size"] == 10
    assert result.metrics["chance_recall@1"] == pytest.approx(0.1)


def test_permutations_can_be_skipped():
    similarity, order, queries, records = setup()
    result = evaluate_similarity(similarity, order, queries, records, n_permutations=0)
    assert result.permutation == {}


def test_the_condition_label_is_carried_through():
    similarity, order, queries, records = setup()
    result = evaluate_similarity(
        similarity, order, queries, records, condition="trivial", fold=2, n_permutations=0
    )
    assert isinstance(result, FoldEvaluation)
    assert result.condition == "trivial"
    assert result.fold == 2
    assert "trivial" in str(result)


def test_same_seeds_reproduce_the_record():
    similarity, order, queries, records = setup(strength=1.0)
    a = evaluate_similarity(similarity, order, queries, records, n_permutations=30)
    b = evaluate_similarity(similarity, order, queries, records, n_permutations=30)
    assert a.metrics == b.metrics
    assert a.permutation["mrr"].p_value == b.permutation["mrr"].p_value


def test_a_column_mismatch_is_rejected():
    similarity, order, queries, records = setup()
    with pytest.raises(ValueError, match="similarity columns"):
        evaluate_similarity(similarity[:, :10], order, queries, records)


def test_a_query_count_mismatch_is_rejected():
    similarity, order, queries, records = setup()
    with pytest.raises(ValueError, match="query sentences"):
        evaluate_similarity(similarity, order, queries[:5], records)


def test_an_impossible_pool_is_reported_clearly():
    records = sentences([10, 30, 50])
    with pytest.raises(ValueError, match="no query has a pool"):
        evaluate_similarity(
            np.zeros((3, 3)), np.arange(3), np.arange(3), records, pool_size=25, tolerance=1
        )
