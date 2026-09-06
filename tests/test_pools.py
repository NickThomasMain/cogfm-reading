"""Tests for candidate pools: size, length matching, determinism, coverage."""

import numpy as np
import pytest

from cogfm.eval.pools import build_pools, describe_pools, pools_by_sentence


def sentences(lengths: list[int]) -> list[dict]:
    """Sentence records with the requested word counts and ids 0..n-1."""
    return [{"id": i, "n_words": n, "text": "w " * n, "task": "SR"} for i, n in enumerate(lengths)]


def uniform(n: int, n_words: int = 20) -> list[dict]:
    """A set of sentences that all share one length."""
    return sentences([n_words] * n)


def test_pool_has_the_requested_size():
    records = uniform(50)
    pools, _ = build_pools(records, np.arange(50), pool_size=25)
    assert pools
    assert all(len(pool) == 25 for pool in pools)


def test_target_appears_exactly_once_at_its_position():
    records = uniform(50)
    pools, _ = build_pools(records, np.arange(50), pool_size=10)
    for pool in pools:
        assert int(np.count_nonzero(pool.candidates == pool.sentence_id)) == 1
        assert int(pool.candidates[pool.target_position]) == pool.sentence_id


def test_candidates_are_distinct():
    records = uniform(40)
    pools, _ = build_pools(records, np.arange(40), pool_size=12)
    for pool in pools:
        assert len(set(pool.candidates.tolist())) == len(pool)


def test_distractors_respect_the_length_tolerance():
    records = sentences(list(range(5, 65)))
    pools, _ = build_pools(records, np.arange(60), pool_size=5, tolerance=3)
    words = {s["id"]: s["n_words"] for s in records}
    for pool in pools:
        target = words[pool.sentence_id]
        for candidate in pool.candidates:
            assert abs(words[int(candidate)] - target) <= 3


def test_candidates_come_only_from_the_eligible_set():
    records = uniform(60)
    eligible = np.arange(0, 60, 2)
    pools, _ = build_pools(records, eligible, pool_size=8)
    allowed = set(eligible.tolist())
    for pool in pools:
        assert set(pool.candidates.tolist()) <= allowed


def test_chance_matches_the_pool_size():
    pools, _ = build_pools(uniform(40), np.arange(40), pool_size=25)
    assert pools[0].chance == pytest.approx(0.04)


def test_same_seed_gives_the_same_pools():
    records = uniform(40)
    first, _ = build_pools(records, np.arange(40), pool_size=10, seed=7)
    second, _ = build_pools(records, np.arange(40), pool_size=10, seed=7)
    assert all(np.array_equal(a.candidates, b.candidates) for a, b in zip(first, second))


def test_a_different_seed_changes_the_pools():
    records = uniform(40)
    first, _ = build_pools(records, np.arange(40), pool_size=10, seed=0)
    second, _ = build_pools(records, np.arange(40), pool_size=10, seed=1)
    assert not all(np.array_equal(a.candidates, b.candidates) for a, b in zip(first, second))


def test_sentences_without_enough_neighbours_are_reported():
    """One sentence sits far from the rest and cannot fill a pool."""
    records = sentences([20] * 30 + [90])
    pools, unservable = build_pools(records, np.arange(31), pool_size=10, tolerance=3)
    assert unservable.tolist() == [30]
    assert all(pool.sentence_id != 30 for pool in pools)


def test_nothing_is_served_when_the_set_is_too_small():
    records = uniform(5)
    pools, unservable = build_pools(records, np.arange(5), pool_size=25)
    assert pools == []
    assert len(unservable) == 5


def test_pool_size_below_two_is_rejected():
    with pytest.raises(ValueError, match="pool_size"):
        build_pools(uniform(10), np.arange(10), pool_size=1)


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        build_pools(uniform(10), np.arange(10), tolerance=-1)


def test_unknown_sentence_id_is_rejected():
    with pytest.raises(ValueError, match="no sentence record"):
        build_pools(uniform(10), np.array([0, 1, 99]))


def test_lookup_covers_every_pool():
    records = uniform(40)
    pools, _ = build_pools(records, np.arange(40), pool_size=8)
    index = pools_by_sentence(pools)
    assert len(index) == len(pools)
    assert all(index[pool.sentence_id] is pool for pool in pools)


def test_description_reports_coverage():
    records = sentences([20] * 30 + [90])
    pools, unservable = build_pools(records, np.arange(31), pool_size=10, tolerance=3)
    text = describe_pools(pools, unservable, records, tolerance=3)
    assert "30 von 31" in text
    assert "Nicht bedient" in text
