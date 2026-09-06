"""Tests for aggregation across folds and the rendered result row."""

import numpy as np
import pytest

from cogfm.eval.permutation import PermutationResult
from cogfm.eval.report import aggregate, format_gaps, format_result_row
from cogfm.eval.runner import FoldEvaluation


def record(condition: str, fold: int, recall: float, null: float = 0.04, p: float = 0.001):
    return FoldEvaluation(
        condition=condition,
        fold=fold,
        metrics={"recall@1": recall, "mrr": recall + 0.1, "percentile": 0.5 + recall},
        permutation={
            name: PermutationResult(
                statistic=name,
                observed=value,
                null_mean=null,
                null_std=0.01,
                p_value=p,
                n_permutations=1000,
            )
            for name, value in (
                ("recall@1", recall),
                ("mrr", recall + 0.1),
                ("percentile", 0.5 + recall),
            )
        },
        n_queries=400,
    )


def test_mean_and_spread_come_from_the_folds():
    summary = aggregate([record("trivial", i, r) for i, r in enumerate([0.10, 0.12, 0.14, 0.16])])
    metric = summary.metrics["recall@1"]
    assert metric.mean == pytest.approx(0.13)
    assert metric.std == pytest.approx(np.std([0.10, 0.12, 0.14, 0.16], ddof=1))
    assert metric.per_fold == (0.10, 0.12, 0.14, 0.16)


def test_queries_are_summed_over_folds():
    summary = aggregate([record("trivial", i, 0.1) for i in range(4)])
    assert summary.n_folds == 4
    assert summary.n_queries == 1600


def test_significant_folds_are_counted_not_pooled():
    records = [record("x", 0, 0.2, p=0.001), record("x", 1, 0.2, p=0.001), record("x", 2, 0.2, p=0.4)]
    assert aggregate(records).metrics["recall@1"].n_significant == 2


def test_lift_relates_the_mean_to_the_null():
    summary = aggregate([record("trivial", 0, 0.12, null=0.04)])
    assert summary.metrics["recall@1"].lift == pytest.approx(3.0)


def test_mixed_conditions_are_rejected():
    with pytest.raises(ValueError, match="mix conditions"):
        aggregate([record("a", 0, 0.1), record("b", 1, 0.1)])


def test_an_empty_list_is_rejected():
    with pytest.raises(ValueError, match="no evaluations"):
        aggregate([])


def test_a_single_fold_reports_zero_spread():
    summary = aggregate([record("trivial", 0, 0.11)])
    assert summary.metrics["recall@1"].std == 0.0


def test_row_lists_every_condition():
    a = aggregate([record("trivial", i, 0.10) for i in range(4)])
    b = aggregate([record("scanez", i, 0.24) for i in range(4)])
    text = format_result_row([a, b])
    assert "trivial" in text
    assert "scanez" in text
    assert "Permutationsnull" in text


def test_gaps_flag_a_difference_inside_the_spread():
    """Two conditions that differ less than they scatter must not read as a result."""
    a = aggregate([record("trivial", i, r) for i, r in enumerate([0.10, 0.20, 0.05, 0.25])])
    b = aggregate([record("scanez", i, r) for i, r in enumerate([0.12, 0.22, 0.07, 0.27])])
    assert "innerhalb der Streuung" in format_gaps([a, b])


def test_gaps_flag_a_difference_beyond_the_spread():
    a = aggregate([record("trivial", i, r) for i, r in enumerate([0.10, 0.11, 0.10, 0.11])])
    b = aggregate([record("scanez", i, r) for i, r in enumerate([0.30, 0.31, 0.30, 0.31])])
    assert "größer als die Streuung" in format_gaps([a, b])


def test_gaps_need_two_conditions():
    a = aggregate([record("trivial", 0, 0.1)])
    assert "at least two" in format_gaps([a])
