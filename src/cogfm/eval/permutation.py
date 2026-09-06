"""Empirical null distributions for retrieval metrics.

The analytic chance level of a pool, one over its size, assumes independent
queries, equally likely candidates and unbiased pools. None of that holds
exactly here: a sentence is read by many subjects, so queries come in
correlated groups; pools are drawn length-matched from a finite set, so some
sentences serve as distractors more often than others; and tied scores move
ranks in ways a formula does not describe.

Shuffling which signal is scored against which pool produces the same metric
under the same design but without any relationship between signal and text.
Repeating that gives the distribution chance actually takes, and the share of
shuffles reaching the observed value is the p-value.

Only the row index is permuted. Pools, their length matching and the position
of each target stay exactly as they were, so nothing about the design changes
between the observed run and the null.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from cogfm.eval.metrics import target_ranks

DEFAULT_PERMUTATIONS = 1000


@dataclass(frozen=True)
class PermutationResult:
    """An observed statistic held against the distribution chance produces."""

    statistic: str
    observed: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    quantiles: dict[str, float] = field(default_factory=dict)

    @property
    def z_score(self) -> float:
        """Distance from the null mean in null standard deviations."""
        if self.null_std == 0:
            return float("nan")
        return (self.observed - self.null_mean) / self.null_std

    def __str__(self) -> str:
        return (
            f"{self.statistic:14s} {self.observed:7.4f}   "
            f"Null {self.null_mean:7.4f} +/- {self.null_std:.4f}   "
            f"z {self.z_score:+6.2f}   p {self.p_value:.4f}"
        )


def pool_columns(pools, sentence_order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate candidate pools into column indices and target positions.

    Args:
        pools: CandidatePool objects, one per query, in query order.
        sentence_order: sentence ids in the order their anchor vectors are
            stored, so a candidate id can be turned into a column.

    Returns:
        An (n_queries, pool_size) index array and the target position per query.
    """
    column_of = {int(sentence_id): i for i, sentence_id in enumerate(sentence_order)}
    columns = np.array(
        [[column_of[int(candidate)] for candidate in pool.candidates] for pool in pools],
        dtype=np.int64,
    )
    targets = np.array([pool.target_position for pool in pools], dtype=np.int64)
    return columns, targets


def gather_scores(
    similarity: np.ndarray, columns: np.ndarray, rows: np.ndarray | None = None
) -> np.ndarray:
    """Cut each query's pool out of the full similarity matrix.

    Args:
        similarity: (n_queries, n_sentences) similarity of every signal to every
            anchor.
        columns: (n_queries, pool_size) candidate columns per query.
        rows: which signal to score against each pool; the identity by default.
            Passing a permutation here is what produces a null sample.
    """
    if rows is None:
        rows = np.arange(len(columns))
    return similarity[np.asarray(rows)[:, None], columns]


def permutation_null(
    similarity: np.ndarray,
    columns: np.ndarray,
    target_positions: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    name: str = "statistic",
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = 0,
) -> PermutationResult:
    """Compare an observed statistic against shuffled signal-to-pool assignments.

    Args:
        similarity: (n_queries, n_sentences) similarities.
        columns: (n_queries, pool_size) candidate columns per query.
        target_positions: column of the correct candidate within each pool.
        statistic: turns an array of ranks into one number, higher being better.
        name: label carried into the result.
        n_permutations: shuffles to draw; below a few hundred the p-value is
            too coarse to support a claim.
        seed: controls the shuffles.

    Returns:
        The observed value, the null it is held against, and a p-value computed
        as (1 + shuffles at least as extreme) / (1 + shuffles), which keeps the
        estimate above zero rather than reporting an impossible certainty.

    Raises:
        ValueError: on mismatched shapes or a permutation count below one.
    """
    similarity = np.asarray(similarity, dtype=float)
    columns = np.asarray(columns)
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2-dimensional, got {similarity.shape}")
    if len(columns) != len(similarity):
        raise ValueError(f"{len(columns)} pools for {len(similarity)} queries")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")

    observed = float(statistic(target_ranks(gather_scores(similarity, columns), target_positions)))

    rng = np.random.default_rng(seed)
    n_queries = len(columns)
    draws = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(n_queries)
        ranks = target_ranks(gather_scores(similarity, columns, shuffled), target_positions)
        draws[i] = statistic(ranks)

    at_least_as_extreme = int(np.count_nonzero(draws >= observed))
    return PermutationResult(
        statistic=name,
        observed=observed,
        null_mean=float(draws.mean()),
        null_std=float(draws.std(ddof=1)) if n_permutations > 1 else 0.0,
        p_value=(1 + at_least_as_extreme) / (1 + n_permutations),
        n_permutations=n_permutations,
        quantiles={
            "p02.5": float(np.percentile(draws, 2.5)),
            "p50": float(np.percentile(draws, 50)),
            "p97.5": float(np.percentile(draws, 97.5)),
        },
    )
