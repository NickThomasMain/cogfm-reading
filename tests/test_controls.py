"""Tests for the two extra controls: within-subject shuffling and decoy pools."""

import numpy as np
import pytest

from cogfm.eval.metrics import recall_at_k
from cogfm.eval.permutation import (
    grouped_permutation,
    permutation_null,
    singleton_share,
)
from cogfm.eval.pools import build_decoy_pools, surface_features
from cogfm.eval.runner import evaluate_similarity


def r_at_1(ranks):
    return recall_at_k(ranks, 1)


def sentences(specs: list[tuple[int, int]]) -> list[dict]:
    """Records built from (word count, word length) pairs."""
    return [
        {"id": i, "n_words": n, "text": " ".join(["x" * length] * n), "task": "SR"}
        for i, (n, length) in enumerate(specs)
    ]


# --- within-subject shuffling -------------------------------------------------


def test_grouped_permutation_keeps_members_inside_their_group():
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    order = grouped_permutation(groups, np.random.default_rng(0))
    assert sorted(order.tolist()) == list(range(6))
    assert set(order[:3].tolist()) == {0, 1, 2}
    assert set(order[3:].tolist()) == {3, 4, 5}


def test_grouped_permutation_does_shuffle():
    groups = np.array(["A"] * 40)
    order = grouped_permutation(groups, np.random.default_rng(0))
    assert not np.array_equal(order, np.arange(40))


def test_singleton_share_counts_queries_that_cannot_move():
    assert singleton_share(np.array(["A", "A", "B"])) == pytest.approx(1 / 3)
    assert singleton_share(np.array(["A", "A"])) == 0.0
    assert singleton_share(np.array([])) == 0.0


def subject_effect_setup(n_subjects=4, per_subject=10, repeats=6, pool_size=8, seed=0):
    """Every subject reads their own block of sentences and leaves a mark on it.

    The similarity boost is tied to the block a subject read, not to the text.
    A query therefore scores well on its own target for a reason that has
    nothing to do with meaning, which is exactly the confound the grouped null
    is meant to expose.
    """
    rng = np.random.default_rng(seed)
    n_sentences = n_subjects * per_subject
    subjects = np.repeat(np.arange(n_subjects), per_subject * repeats)
    targets = np.concatenate(
        [np.tile(np.arange(s * per_subject, (s + 1) * per_subject), repeats)
         for s in range(n_subjects)]
    )

    similarity = rng.normal(scale=0.1, size=(len(subjects), n_sentences))
    for s in range(n_subjects):
        block = slice(s * per_subject, (s + 1) * per_subject)
        similarity[np.ix_(subjects == s, np.arange(n_sentences)[block])] += 3.0

    columns = np.empty((len(subjects), pool_size), dtype=int)
    positions = np.empty(len(subjects), dtype=int)
    for i, target in enumerate(targets):
        others = np.setdiff1d(np.arange(n_sentences), target)
        members = np.concatenate([[target], rng.choice(others, pool_size - 1, replace=False)])
        rng.shuffle(members)
        columns[i] = members
        positions[i] = int(np.flatnonzero(members == target)[0])
    return similarity, columns, positions, subjects


def test_a_subject_effect_inflates_the_grouped_null_but_not_the_free_one():
    """The reason the control exists, made explicit.

    Under free shuffling a query lands on some other subject's pool, where its
    own mark is worthless, so the null stays near chance and the confound reads
    as a result. Shuffling inside one subject keeps the mark useful, the null
    rises to meet the observation, and the confound is exposed.
    """
    similarity, columns, positions, subjects = subject_effect_setup()

    free = permutation_null(similarity, columns, positions, r_at_1, n_permutations=200, seed=0)
    grouped = permutation_null(
        similarity, columns, positions, r_at_1, n_permutations=200, seed=0, groups=subjects
    )

    assert grouped.null_mean > free.null_mean + 0.1
    assert free.p_value < 0.05
    assert grouped.p_value > 0.05


def test_a_genuine_text_signal_survives_both_nulls():
    """The counterpart: a signal tied to the text, not the reader, holds up."""
    rng = np.random.default_rng(1)
    n_subjects, per_subject, repeats, pool_size = 4, 10, 6, 8
    n_sentences = n_subjects * per_subject
    subjects = np.repeat(np.arange(n_subjects), per_subject * repeats)
    targets = np.concatenate(
        [np.tile(np.arange(s * per_subject, (s + 1) * per_subject), repeats)
         for s in range(n_subjects)]
    )
    similarity = rng.normal(scale=0.1, size=(len(subjects), n_sentences))
    similarity[np.arange(len(targets)), targets] += 3.0

    columns = np.empty((len(subjects), pool_size), dtype=int)
    positions = np.empty(len(subjects), dtype=int)
    for i, target in enumerate(targets):
        others = np.setdiff1d(np.arange(n_sentences), target)
        members = np.concatenate([[target], rng.choice(others, pool_size - 1, replace=False)])
        rng.shuffle(members)
        columns[i] = members
        positions[i] = int(np.flatnonzero(members == target)[0])

    grouped = permutation_null(
        similarity, columns, positions, r_at_1, n_permutations=200, seed=0, groups=subjects
    )
    assert grouped.observed > 0.9
    assert grouped.p_value <= 1 / 201 + 1e-9


def test_group_labels_must_match_the_queries():
    with pytest.raises(ValueError, match="group labels"):
        permutation_null(
            np.zeros((5, 10)),
            np.zeros((5, 4), dtype=int),
            np.zeros(5, dtype=int),
            r_at_1,
            n_permutations=5,
            groups=np.array(["A", "B"]),
        )


# --- decoy pools --------------------------------------------------------------


def test_a_decoy_pool_holds_exactly_two_candidates():
    records = sentences([(10 + i % 5, 4 + i % 3) for i in range(20)])
    pools = build_decoy_pools(records, np.arange(20))
    assert len(pools) == 20
    assert all(len(pool) == 2 for pool in pools)
    assert all(pool.chance == 0.5 for pool in pools)


def test_the_target_sits_at_its_recorded_position():
    records = sentences([(10 + i % 5, 4) for i in range(20)])
    for pool in build_decoy_pools(records, np.arange(20)):
        assert int(pool.candidates[pool.target_position]) == pool.sentence_id
        assert int(np.count_nonzero(pool.candidates == pool.sentence_id)) == 1


def test_the_target_does_not_always_take_the_first_slot():
    records = sentences([(10 + i % 7, 4 + i % 4) for i in range(60)])
    positions = {pool.target_position for pool in build_decoy_pools(records, np.arange(60))}
    assert positions == {0, 1}


def test_the_decoy_is_the_closest_match_not_an_arbitrary_sentence():
    """One sentence has a near twin and a set of clearly different others."""
    records = sentences([(10, 4), (10, 4), (40, 9), (41, 9), (42, 8)])
    pools = build_decoy_pools(records, np.arange(5))
    partner = {p.sentence_id: [int(c) for c in p.candidates if int(c) != p.sentence_id][0]
               for p in pools}
    assert partner[0] == 1
    assert partner[1] == 0
    assert partner[2] in (3, 4)


def test_surface_features_report_count_and_word_length():
    records = sentences([(3, 5), (10, 2)])
    features = surface_features(records, np.arange(2))
    assert features[0].tolist() == [3.0, 5.0]
    assert features[1].tolist() == [10.0, 2.0]


def test_extra_feature_columns_are_accepted():
    records = sentences([(10 + i % 5, 4) for i in range(20)])
    base = surface_features(records, np.arange(20))
    extended = np.hstack([base, np.arange(20).reshape(-1, 1)])
    assert len(build_decoy_pools(records, np.arange(20), features=extended)) == 20


def test_mismatched_feature_rows_are_rejected():
    records = sentences([(10, 4)] * 5)
    with pytest.raises(ValueError, match="feature rows"):
        build_decoy_pools(records, np.arange(5), features=np.zeros((3, 2)))


def test_a_single_sentence_cannot_form_a_pair():
    with pytest.raises(ValueError, match="at least 2"):
        build_decoy_pools(sentences([(10, 4)]), np.arange(1))


# --- both together in the runner ---------------------------------------------


def test_the_runner_accepts_ready_made_pools_and_reports_chance_at_one_half():
    records = sentences([(10 + i % 5, 4 + i % 3) for i in range(30)])
    order = np.arange(30)
    queries = np.repeat(order, 3)
    similarity = np.random.default_rng(0).normal(size=(len(queries), 30))

    pools = build_decoy_pools(records, order)
    result = evaluate_similarity(
        similarity, order, queries, records, pools=pools, n_permutations=50
    )
    assert result.metrics["pool_size"] == 2
    assert result.metrics["chance_recall@1"] == pytest.approx(0.5)
    assert result.n_pools == 30


def test_the_runner_passes_subjects_through_to_the_null():
    records = sentences([(20, 4)] * 40)
    order = np.arange(40)
    queries = np.repeat(order, 3)
    subjects = np.tile(np.array(["A", "B", "C"]), 40)
    similarity = np.random.default_rng(0).normal(size=(len(queries), 40))

    result = evaluate_similarity(
        similarity, order, queries, records, pool_size=10, n_permutations=50,
        query_subjects=subjects,
    )
    assert result.permutation["recall@1"].n_permutations == 50
