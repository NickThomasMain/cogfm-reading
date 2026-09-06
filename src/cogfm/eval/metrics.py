"""Rank-based retrieval metrics over candidate pools.

Everything here derives from one quantity: the rank the correct candidate
receives when a pool is sorted by similarity. Rank 1 means the target came
first. Metrics are then different summaries of the same rank distribution, and
each is reported next to the value chance alone would produce, since a number
like 0.31 means nothing without knowing whether chance is 0.04 or 0.40.

Scores arrive as an array rather than as model output, so this module needs no
tensors and can be checked against hand-computed cases.

Ties are resolved by averaging the positions they span. A model that returns
the same score for every candidate then lands exactly at chance instead of
being flattered or punished by the sort order, which matters because degenerate
outputs are a realistic failure mode rather than a hypothetical one.
"""

from __future__ import annotations

import numpy as np

DEFAULT_KS = (1, 5, 10)


def target_ranks(scores: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    """Rank of the correct candidate in each pool, counting from one.

    Args:
        scores: (n_queries, pool_size) similarities, higher meaning more similar.
        target_positions: column holding the correct candidate, per query.

    Returns:
        Float ranks in [1, pool_size]. Tied candidates share the mean of the
        positions they occupy, so a rank can fall between two integers.

    Raises:
        ValueError: on a non-2D score array, a length mismatch, or a target
            position outside the pool.
    """
    scores = np.asarray(scores, dtype=float)
    target_positions = np.asarray(target_positions)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-dimensional, got shape {scores.shape}")
    if len(target_positions) != len(scores):
        raise ValueError(
            f"{len(target_positions)} target positions for {len(scores)} queries"
        )
    if len(scores) and (target_positions.min() < 0 or target_positions.max() >= scores.shape[1]):
        raise ValueError("a target position lies outside the pool")

    rows = np.arange(len(scores))
    target_scores = scores[rows, target_positions][:, None]
    higher = np.count_nonzero(scores > target_scores, axis=1)
    equal = np.count_nonzero(scores == target_scores, axis=1)
    # The target sits somewhere inside its block of ties; take the block's centre.
    return higher + (equal + 1) / 2.0


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Share of queries whose target lands in the top k."""
    ranks = np.asarray(ranks, dtype=float)
    if not len(ranks):
        return float("nan")
    return float(np.count_nonzero(ranks <= k) / len(ranks))


def mean_reciprocal_rank(ranks: np.ndarray) -> float:
    """Mean of one over the rank, so early hits dominate the average."""
    ranks = np.asarray(ranks, dtype=float)
    if not len(ranks):
        return float("nan")
    return float(np.mean(1.0 / ranks))


def percentile_rank(ranks: np.ndarray, pool_size: int) -> float:
    """Mean position in the pool on a 0-to-1 scale, higher being better.

    A target ranked first scores 1, one ranked last scores 0. Unlike recall this
    does not depend on the pool size, which makes it the metric to use when
    comparing runs whose pools differ.
    """
    ranks = np.asarray(ranks, dtype=float)
    if not len(ranks) or pool_size < 2:
        return float("nan")
    return float(np.mean((pool_size - ranks) / (pool_size - 1)))


def chance_recall_at_k(pool_size: int, k: int) -> float:
    """Recall a model with no information reaches."""
    return min(k, pool_size) / pool_size


def chance_mean_reciprocal_rank(pool_size: int) -> float:
    """MRR a model with no information reaches, averaged over all ranks."""
    return float(np.mean(1.0 / np.arange(1, pool_size + 1)))


def summarize_ranks(
    ranks: np.ndarray, pool_size: int, ks: tuple[int, ...] = DEFAULT_KS
) -> dict[str, float]:
    """Collect every metric plus its chance level into one flat record."""
    ranks = np.asarray(ranks, dtype=float)
    summary: dict[str, float] = {"n_queries": float(len(ranks)), "pool_size": float(pool_size)}
    for k in ks:
        if k > pool_size:
            continue
        summary[f"recall@{k}"] = recall_at_k(ranks, k)
        summary[f"chance_recall@{k}"] = chance_recall_at_k(pool_size, k)
    summary["mrr"] = mean_reciprocal_rank(ranks)
    summary["chance_mrr"] = chance_mean_reciprocal_rank(pool_size)
    summary["percentile"] = percentile_rank(ranks, pool_size)
    summary["chance_percentile"] = 0.5
    summary["mean_rank"] = float(np.mean(ranks)) if len(ranks) else float("nan")
    summary["chance_mean_rank"] = (pool_size + 1) / 2.0
    return summary


def format_summary(summary: dict[str, float]) -> str:
    """Render a summary as aligned lines of value against chance."""
    lines = [f"Anfragen {int(summary['n_queries'])}, Poolgröße {int(summary['pool_size'])}"]
    pairs = [
        (key, f"chance_{key}")
        for key in summary
        if not key.startswith("chance_") and key not in ("n_queries", "pool_size")
    ]
    for key, chance_key in pairs:
        if chance_key not in summary:
            continue
        value, chance = summary[key], summary[chance_key]
        lines.append(f"  {key:14s} {value:7.3f}   Zufall {chance:7.3f}   Faktor {value / chance:5.2f}")
    return "\n".join(lines)
