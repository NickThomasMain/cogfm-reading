"""Aggregate fold evaluations into the result row.

A single fold rests on three test subjects, and with twelve subjects in the
corpus one unusual reader moves the number visibly. Every value is therefore
reported as a mean across folds with its spread beside it, and a difference
smaller than that spread is not a difference.

P-values are kept per fold rather than pooled. Combining them would need
assumptions about independence that four overlapping folds do not satisfy, so
the summary states how many folds cleared the threshold instead of inventing a
single number for all of them.

What the row is read for is the distance between conditions, not the absolute
values. Chance is what the permutation produced, and the encoder has to beat
the trivial features, not merely chance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cogfm.eval.runner import FoldEvaluation

SIGNIFICANCE = 0.01
ROW_METRICS = ("recall@1", "mrr", "percentile")


@dataclass(frozen=True)
class MetricSummary:
    """One metric for one condition, across folds."""

    name: str
    mean: float
    std: float
    null_mean: float
    per_fold: tuple[float, ...]
    p_values: tuple[float, ...]

    @property
    def n_significant(self) -> int:
        return int(np.count_nonzero(np.asarray(self.p_values) < SIGNIFICANCE))

    @property
    def lift(self) -> float:
        """How many times chance the mean reaches."""
        if self.null_mean == 0:
            return float("nan")
        return self.mean / self.null_mean


@dataclass(frozen=True)
class ConditionSummary:
    """Every metric of one condition, across folds."""

    condition: str
    n_folds: int
    n_queries: int
    metrics: dict[str, MetricSummary]


def aggregate(evaluations: list[FoldEvaluation], metrics: tuple[str, ...] = ROW_METRICS) -> ConditionSummary:
    """Collapse one condition's fold records into a single summary.

    Args:
        evaluations: records of the same condition, one per fold.
        metrics: which metrics to carry into the summary.

    Returns:
        Mean and spread per metric, the null it was held against, and the
        per-fold values so a single odd fold stays visible.

    Raises:
        ValueError: on an empty list or records from more than one condition.
    """
    if not evaluations:
        raise ValueError("no evaluations to aggregate")
    conditions = {record.condition for record in evaluations}
    if len(conditions) != 1:
        raise ValueError(f"records mix conditions: {sorted(conditions)}")

    summaries: dict[str, MetricSummary] = {}
    for name in metrics:
        values = np.array([record.metrics[name] for record in evaluations if name in record.metrics])
        if not len(values):
            continue
        nulls = [
            record.permutation[name].null_mean
            for record in evaluations
            if name in record.permutation
        ]
        p_values = [
            record.permutation[name].p_value for record in evaluations if name in record.permutation
        ]
        summaries[name] = MetricSummary(
            name=name,
            mean=float(values.mean()),
            std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            null_mean=float(np.mean(nulls)) if nulls else float("nan"),
            per_fold=tuple(float(v) for v in values),
            p_values=tuple(float(p) for p in p_values),
        )

    return ConditionSummary(
        condition=evaluations[0].condition,
        n_folds=len(evaluations),
        n_queries=sum(record.n_queries for record in evaluations),
        metrics=summaries,
    )


def format_result_row(
    summaries: list[ConditionSummary], metrics: tuple[str, ...] = ROW_METRICS
) -> str:
    """Render the conditions as one table, one line per condition."""
    if not summaries:
        return "no conditions to report"

    present = [name for name in metrics if any(name in s.metrics for s in summaries)]
    header = f"{'Bedingung':16s} {'Folds':>5s} {'Anfragen':>9s}"
    for name in present:
        header += f"   {name:>22s}   {'p<0.01':>7s}"
    lines = [header, "-" * len(header)]

    for summary in summaries:
        line = f"{summary.condition:16s} {summary.n_folds:5d} {summary.n_queries:9d}"
        for name in present:
            metric = summary.metrics.get(name)
            if metric is None:
                line += f"   {'-':>22s}   {'-':>7s}"
                continue
            body = f"{metric.mean:.3f} +/- {metric.std:.3f}"
            line += f"   {body:>22s}   {metric.n_significant}/{summary.n_folds:>5d}"
        lines.append(line)

    lines.append("")
    for name in present:
        nulls = [s.metrics[name].null_mean for s in summaries if name in s.metrics]
        if nulls:
            lines.append(f"  Permutationsnull {name:12s} {np.mean(nulls):.4f}")
    return "\n".join(lines)


def format_gaps(summaries: list[ConditionSummary], metric: str = "recall@1") -> str:
    """State the distance between neighbouring conditions, which carries the claim."""
    ordered = [s for s in summaries if metric in s.metrics]
    if len(ordered) < 2:
        return "at least two conditions are needed for a comparison"

    lines = [f"Abstände in {metric}:"]
    for earlier, later in zip(ordered, ordered[1:]):
        before, after = earlier.metrics[metric], later.metrics[metric]
        gap = after.mean - before.mean
        spread = float(np.hypot(before.std, after.std))
        verdict = "innerhalb der Streuung" if abs(gap) <= spread else "größer als die Streuung"
        lines.append(
            f"  {earlier.condition:14s} -> {later.condition:14s} "
            f"{gap:+.3f}   (Streuung {spread:.3f}, {verdict})"
        )
    return "\n".join(lines)
