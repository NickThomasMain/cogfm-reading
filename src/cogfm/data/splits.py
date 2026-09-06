"""Subject- and item-disjoint cross-validation folds.

A corpus in which every subject reads every sentence leaks in two directions.
Splitting by subject alone leaves the same sentence in training and test, read
by someone else, so a model can memorise the sentence rather than learn how
gaze relates to language. Splitting by sentence alone lets a model recognise a
person's reading style. Both axes therefore have to be separated at once.

Folds rotate the two axes together: in fold i the test set is the intersection
of subject group i and sentence group i, and training uses the intersection of
all other subject groups with all other sentence groups. Trials that overlap on
exactly one axis belong to neither and are dropped, which costs roughly a third
of the data and is the price of a result that cannot be explained by leakage.

Sentence groups are stratified by task and by length. Length is the strongest
surface cue a scanpath carries, and length-matched candidate pools during
evaluation need a comparable spread of lengths in every fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One cross-validation fold, as index arrays into the trial list."""

    index: int
    train: np.ndarray
    test: np.ndarray
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_sentences: np.ndarray
    test_sentences: np.ndarray

    @property
    def n_dropped(self) -> int:
        """Trials that overlap the test set on exactly one axis."""
        return self._n_total - len(self.train) - len(self.test)

    _n_total: int = 0


def _round_robin(order: np.ndarray, n_groups: int) -> np.ndarray:
    """Deal the items of ``order`` to groups one after another."""
    group = np.empty(len(order), dtype=np.int64)
    for position, item in enumerate(order):
        group[item] = position % n_groups
    return group


def _subject_groups(subjects: np.ndarray, n_groups: int, rng: np.random.Generator) -> dict:
    """Split subjects into equally sized groups, order randomised by the seed."""
    unique = np.array(sorted(set(subjects.tolist())))
    order = rng.permutation(len(unique))
    group = _round_robin(order, n_groups)
    return {str(name): int(g) for name, g in zip(unique, group, strict=True)}


def _sentence_groups(sentences: list[dict], n_groups: int, rng: np.random.Generator) -> dict:
    """Split sentences into groups with matched task mix and length spread.

    Within a task, sentences are ordered by word count and dealt out in turn, so
    every group receives a comparable share of short, typical and long items.
    Ties are broken by a seeded jitter rather than by sentence id, which would
    otherwise tie group membership to presentation order.
    """
    assignment: dict[int, int] = {}
    for task in sorted({s["task"] for s in sentences}):
        subset = [s for s in sentences if s["task"] == task]
        lengths = np.array([s["n_words"] for s in subset], dtype=float)
        lengths = lengths + rng.random(len(lengths)) * 0.5
        order = np.argsort(lengths)
        group = _round_robin(order, n_groups)
        for record, g in zip(subset, group, strict=True):
            assignment[int(record["id"])] = int(g)
    return assignment


def make_folds(
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    sentences: list[dict],
    n_folds: int = 4,
    seed: int = 0,
) -> list[Fold]:
    """Build subject- and item-disjoint folds over a list of trials.

    Args:
        subject_ids: subject of each trial.
        sentence_ids: sentence id of each trial.
        sentences: sentence records carrying ``id``, ``task`` and ``n_words``.
        n_folds: number of folds; subjects and sentences are split this many ways.
        seed: controls subject order and the tie-break among equal lengths.

    Returns:
        One Fold per split, each holding index arrays into the trial list.

    Raises:
        ValueError: if a fold would share a subject or a sentence between train
            and test, or if a fold would come out empty.
    """
    subject_ids = np.asarray(subject_ids)
    sentence_ids = np.asarray(sentence_ids)
    if len(subject_ids) != len(sentence_ids):
        raise ValueError("subject_ids and sentence_ids must have the same length")

    n_subjects = len(set(subject_ids.tolist()))
    if n_folds > n_subjects:
        raise ValueError(f"{n_folds} folds requested but only {n_subjects} subjects available")

    rng = np.random.default_rng(seed)
    subject_group = _subject_groups(subject_ids, n_folds, rng)
    sentence_group = _sentence_groups(sentences, n_folds, rng)

    trial_subject_group = np.array([subject_group[str(s)] for s in subject_ids])
    trial_sentence_group = np.array([sentence_group[int(s)] for s in sentence_ids])

    folds: list[Fold] = []
    for i in range(n_folds):
        subject_in = trial_subject_group == i
        sentence_in = trial_sentence_group == i
        test = np.flatnonzero(subject_in & sentence_in)
        train = np.flatnonzero(~subject_in & ~sentence_in)

        if len(test) == 0 or len(train) == 0:
            raise ValueError(f"fold {i} is empty: {len(train)} train, {len(test)} test")

        shared_subjects = set(subject_ids[train].tolist()) & set(subject_ids[test].tolist())
        if shared_subjects:
            raise ValueError(f"fold {i} shares subjects between train and test: {shared_subjects}")
        shared_sentences = set(sentence_ids[train].tolist()) & set(sentence_ids[test].tolist())
        if shared_sentences:
            raise ValueError(f"fold {i} shares {len(shared_sentences)} sentences between splits")

        folds.append(
            Fold(
                index=i,
                train=train,
                test=test,
                train_subjects=tuple(sorted(set(subject_ids[train].tolist()))),
                test_subjects=tuple(sorted(set(subject_ids[test].tolist()))),
                train_sentences=np.unique(sentence_ids[train]),
                test_sentences=np.unique(sentence_ids[test]),
                _n_total=len(subject_ids),
            )
        )

    _check_coverage(folds, sentence_ids)
    return folds


def _check_coverage(folds: list[Fold], sentence_ids: np.ndarray) -> None:
    """No subject and no sentence may be tested twice.

    Full coverage is not required. A sentence is only testable in its own fold
    if at least one subject of that fold has a usable recording for it, and the
    grid has holes wherever a session was lost or a trial fell below the
    fixation threshold. Missing coverage is reported by ``untested`` rather than
    treated as an error; being tested twice would be a real defect.
    """
    tested_subjects: list[str] = []
    tested_sentences: list[int] = []
    for fold in folds:
        tested_subjects.extend(fold.test_subjects)
        tested_sentences.extend(fold.test_sentences.tolist())

    if len(tested_subjects) != len(set(tested_subjects)):
        raise ValueError("a subject appears in the test set of more than one fold")
    if len(tested_sentences) != len(set(tested_sentences)):
        raise ValueError("a sentence appears in the test set of more than one fold")


def untested(folds: list[Fold], sentence_ids: np.ndarray) -> np.ndarray:
    """Sentences that no fold gets to test, sorted by id.

    A sentence drops out when none of the subjects in its fold produced a usable
    recording of it. Short sentences are the usual cause, since they rarely
    clear the minimum fixation count.
    """
    tested: set[int] = set()
    for fold in folds:
        tested |= set(fold.test_sentences.tolist())
    return np.array(sorted(set(sentence_ids.tolist()) - tested), dtype=np.int64)


def describe(folds: list[Fold], sentences: list[dict]) -> str:
    """Render a per-fold summary of sizes, subjects and length spread."""
    words = {int(s["id"]): int(s["n_words"]) for s in sentences}
    lines = [
        f"{'Fold':>4} {'Train':>7} {'Test':>6} {'verworfen':>10} "
        f"{'Testprobanden':>14} {'Testsätze':>10} {'Länge Median':>13} {'Länge p10-p90':>14}"
    ]
    for fold in folds:
        lengths = np.array([words[int(s)] for s in fold.test_sentences])
        lines.append(
            f"{fold.index:>4} {len(fold.train):>7} {len(fold.test):>6} {fold.n_dropped:>10} "
            f"{len(fold.test_subjects):>14} {len(fold.test_sentences):>10} "
            f"{np.median(lengths):>13.0f} "
            f"{str(int(np.percentile(lengths, 10))) + '-' + str(int(np.percentile(lengths, 90))):>14}"
        )
    return "\n".join(lines)
