"""Turn a trained model and one fold into a complete evaluation record.

Two halves, deliberately separated. ``embed_fold`` runs the model and produces
a similarity matrix; ``evaluate_similarity`` takes that matrix and produces the
numbers. Only the first half needs a model, so the arithmetic that decides what
a result means can be checked against hand-built matrices.

A pool belongs to a sentence, and several subjects read the same sentence, so
every trial on that sentence is scored against the same alternatives. Trials
whose sentence has no pool are dropped from the evaluation and counted, since
silently shrinking the query set would make coverage invisible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from cogfm.eval.metrics import (
    DEFAULT_KS,
    mean_reciprocal_rank,
    percentile_rank,
    recall_at_k,
    summarize_ranks,
    target_ranks,
)
from cogfm.eval.permutation import (
    DEFAULT_PERMUTATIONS,
    PermutationResult,
    gather_scores,
    permutation_null,
)
from cogfm.eval.pools import CandidatePool, build_pools, pools_by_sentence


@dataclass(frozen=True)
class FoldEvaluation:
    """Everything one condition produced on one fold."""

    condition: str
    fold: int
    metrics: dict[str, float]
    permutation: dict[str, PermutationResult] = field(default_factory=dict)
    n_queries: int = 0
    n_dropped_queries: int = 0
    n_pools: int = 0
    n_unservable_sentences: int = 0

    def __str__(self) -> str:
        head = (
            f"[{self.condition}] Fold {self.fold}: {self.n_queries} Anfragen "
            f"({self.n_dropped_queries} ohne Pool), {self.n_pools} Pools"
        )
        body = [f"  {result}" for result in self.permutation.values()]
        return "\n".join([head, *body])


def default_statistics(pool_size: int) -> dict[str, Callable[[np.ndarray], float]]:
    """The statistics a permutation null is drawn for."""
    return {
        "recall@1": lambda ranks: recall_at_k(ranks, 1),
        "mrr": mean_reciprocal_rank,
        "percentile": lambda ranks: percentile_rank(ranks, pool_size),
    }


def evaluate_similarity(
    similarity: np.ndarray,
    sentence_order: np.ndarray,
    query_sentence_ids: np.ndarray,
    sentences: list[dict],
    condition: str = "model",
    fold: int = 0,
    pool_size: int = 25,
    tolerance: int = 3,
    pool_seed: int = 0,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    permutation_seed: int = 0,
    ks: tuple[int, ...] = DEFAULT_KS,
    statistics: dict[str, Callable[[np.ndarray], float]] | None = None,
    pools: list[CandidatePool] | None = None,
    query_subjects: np.ndarray | None = None,
) -> FoldEvaluation:
    """Score every query against its sentence's pool and summarise the result.

    Args:
        similarity: (n_queries, n_sentences) similarity of each signal to each
            anchor, columns following ``sentence_order``.
        sentence_order: sentence ids in the order the columns hold them.
        query_sentence_ids: the sentence each query was recorded on.
        sentences: sentence records carrying ``id`` and ``n_words``.
        condition: label for the encoder or baseline being evaluated.
        fold: which fold this record belongs to.
        pool_size: candidates per pool.
        tolerance: word-count tolerance for distractors.
        pool_seed: controls which distractors are drawn.
        n_permutations: shuffles per statistic; zero skips the null entirely.
        permutation_seed: controls the shuffles.
        ks: recall cutoffs to report.
        statistics: statistics to draw a null for; the defaults if None.
        pools: ready-made pools to score against, which overrides ``pool_size``
            and ``tolerance``. Use it for the two-alternative decoy test.
        query_subjects: who produced each query. When given, the null shuffles
            only within a subject's own trials, which rules out a signal that
            comes from the reader rather than the text.

    Returns:
        The metrics, the nulls they are held against, and how many queries and
        sentences took part.

    Raises:
        ValueError: on shape mismatches, or when no query survives pooling.
    """
    similarity = np.asarray(similarity, dtype=float)
    sentence_order = np.asarray(sentence_order)
    query_sentence_ids = np.asarray(query_sentence_ids)
    if similarity.ndim != 2:
        raise ValueError(f"similarity must be 2-dimensional, got {similarity.shape}")
    if similarity.shape[1] != len(sentence_order):
        raise ValueError(
            f"{similarity.shape[1]} similarity columns for {len(sentence_order)} sentences"
        )
    if len(query_sentence_ids) != len(similarity):
        raise ValueError(
            f"{len(query_sentence_ids)} query sentences for {len(similarity)} queries"
        )

    if pools is None:
        pools, unservable = build_pools(
            sentences, sentence_order, pool_size=pool_size, tolerance=tolerance, seed=pool_seed
        )
    else:
        unservable = np.array([], dtype=np.int64)
        pool_size = len(pools[0]) if pools else pool_size
    by_sentence = pools_by_sentence(pools)

    kept = np.flatnonzero([int(s) in by_sentence for s in query_sentence_ids])
    if not len(kept):
        raise ValueError("no query has a pool; the eligible set is too small or too spread out")

    column_of = {int(sentence_id): i for i, sentence_id in enumerate(sentence_order)}
    columns = np.array(
        [
            [column_of[int(c)] for c in by_sentence[int(query_sentence_ids[i])].candidates]
            for i in kept
        ],
        dtype=np.int64,
    )
    targets = np.array(
        [by_sentence[int(query_sentence_ids[i])].target_position for i in kept], dtype=np.int64
    )

    kept_similarity = similarity[kept]
    ranks = target_ranks(gather_scores(kept_similarity, columns), targets)
    metrics = summarize_ranks(ranks, pool_size=pool_size, ks=ks)

    groups = None if query_subjects is None else np.asarray(query_subjects)[kept]

    nulls: dict[str, PermutationResult] = {}
    if n_permutations > 0:
        chosen = statistics if statistics is not None else default_statistics(pool_size)
        for name, statistic in chosen.items():
            nulls[name] = permutation_null(
                kept_similarity,
                columns,
                targets,
                statistic,
                name=name,
                n_permutations=n_permutations,
                seed=permutation_seed,
                groups=groups,
            )

    return FoldEvaluation(
        condition=condition,
        fold=fold,
        metrics=metrics,
        permutation=nulls,
        n_queries=len(kept),
        n_dropped_queries=len(similarity) - len(kept),
        n_pools=len(pools),
        n_unservable_sentences=len(unservable),
    )


def embed_fold(model, dataset, trial_indices: Sequence[int], sentence_ids: Sequence[int],
               batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over one fold and return a cosine similarity matrix.

    Encoder and anchor are frozen, so this is a single pass with no gradients.
    Both sides are length-normalised, which makes the dot product a cosine and
    matches how the contrastive loss compares them during training.

    Args:
        model: a BindingModel exposing ``encode_modality`` and ``encode_text``.
        dataset: the dataset the indices refer to.
        trial_indices: trials to evaluate, normally a fold's test split.
        sentence_ids: sentences whose anchors form the columns.
        batch_size: trials encoded at once.

    Returns:
        The (n_trials, n_sentences) similarity matrix, the sentence ids in
        column order, and the sentence each trial was recorded on.
    """
    import torch

    from cogfm.data.batching import batch_reading_samples

    model.eval()
    trial_indices = list(trial_indices)
    sentence_ids = np.asarray(sentence_ids)

    modality_chunks = []
    query_sentences = []
    with torch.no_grad():
        for start in range(0, len(trial_indices), batch_size):
            samples = [dataset[i] for i in trial_indices[start : start + batch_size]]
            batch = batch_reading_samples(samples)
            modality_chunks.append(model.encode_modality(batch["scanpath"], batch["mask"]))
            query_sentences.extend(batch["sentence_id"])

        modality = torch.cat(modality_chunks, dim=0)
        text_by_id = {int(s["sentence_id"]): s["text"] for s in (dataset[i] for i in trial_indices)}
        anchors = model.encode_text([text_by_id[int(i)] for i in sentence_ids])

        modality = torch.nn.functional.normalize(modality, dim=-1)
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
        similarity = (modality @ anchors.t()).cpu().numpy()

    return similarity, sentence_ids, np.asarray(query_sentences)
