"""Length-matched candidate pools for retrieval.

Retrieval asks whether a modality signal finds the text it was recorded on. The
answer depends on what it is asked to choose between. Comparing against every
held-out sentence makes the score depend on how many there happen to be, and a
score from a set of eighty is not comparable to one from a set of nine hundred.
A pool of fixed size removes that dependency and fixes the chance level at one
over the pool size.

The second reason for building pools rather than using the whole split is
length. A scanpath encodes how long a text is almost perfectly: more words means
more fixations. A model that has learned nothing about language can still rule
out candidates whose length does not fit, and would score far above chance for
that reason alone. Drawing every candidate from a narrow band of word counts
takes that shortcut away, so what remains has to come from somewhere else.

A pool belongs to a sentence, not to a trial. Every subject who read the same
sentence is therefore scored against the same alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_POOL_SIZE = 25
DEFAULT_TOLERANCE = 3


@dataclass(frozen=True)
class CandidatePool:
    """The alternatives one sentence is scored against.

    Attributes:
        sentence_id: the sentence a signal has to find.
        candidates: sentence ids to rank, including the target exactly once.
        target_position: where the target sits in ``candidates``.
    """

    sentence_id: int
    candidates: np.ndarray
    target_position: int

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def chance(self) -> float:
        """Probability of ranking the target first without any information."""
        return 1.0 / len(self.candidates)


def build_pools(
    sentences: list[dict],
    eligible: np.ndarray,
    pool_size: int = DEFAULT_POOL_SIZE,
    tolerance: int = DEFAULT_TOLERANCE,
    seed: int = 0,
) -> tuple[list[CandidatePool], np.ndarray]:
    """Draw one length-matched pool per eligible sentence.

    Args:
        sentences: sentence records carrying ``id`` and ``n_words``.
        eligible: sentence ids a pool may draw from, normally the test
            sentences of one fold. Training sentences are kept out because the
            connector was pushed away from them during training, which would
            make them easier to reject than an unseen alternative.
        pool_size: candidates per pool, the target included.
        tolerance: how many words a distractor may differ from the target.
        seed: controls which distractors are drawn.

    Returns:
        The pools that could be filled, and the ids of the sentences that could
        not. A sentence is unservable when too few others fall inside its length
        band; this happens at the ends of the length distribution and is
        reported rather than quietly worked around.

    Raises:
        ValueError: for a pool size below two or a negative tolerance.
    """
    if pool_size < 2:
        raise ValueError(f"pool_size must be at least 2, got {pool_size}")
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")

    words = {int(s["id"]): int(s["n_words"]) for s in sentences}
    eligible = np.asarray(eligible)
    missing = [int(i) for i in eligible if int(i) not in words]
    if missing:
        raise ValueError(f"no sentence record for ids {missing[:5]}")

    lengths = np.array([words[int(i)] for i in eligible])
    rng = np.random.default_rng(seed)

    pools: list[CandidatePool] = []
    unservable: list[int] = []
    for position, sentence_id in enumerate(eligible):
        within = np.flatnonzero(
            (lengths >= lengths[position] - tolerance) & (lengths <= lengths[position] + tolerance)
        )
        within = within[within != position]
        if len(within) < pool_size - 1:
            unservable.append(int(sentence_id))
            continue

        drawn = rng.choice(within, pool_size - 1, replace=False)
        members = np.concatenate([[position], drawn])
        # Shuffling keeps the target off a fixed slot, so a scorer that always
        # returns the same index lands at chance instead of looking perfect.
        rng.shuffle(members)
        pools.append(
            CandidatePool(
                sentence_id=int(sentence_id),
                candidates=eligible[members],
                target_position=int(np.flatnonzero(members == position)[0]),
            )
        )

    return pools, np.array(sorted(unservable), dtype=np.int64)


def pools_by_sentence(pools: list[CandidatePool]) -> dict[int, CandidatePool]:
    """Index pools by the sentence they belong to."""
    return {pool.sentence_id: pool for pool in pools}


def describe_pools(
    pools: list[CandidatePool],
    unservable: np.ndarray,
    sentences: list[dict],
    tolerance: int = DEFAULT_TOLERANCE,
) -> str:
    """Summarise coverage and the length range the pools actually span."""
    words = {int(s["id"]): int(s["n_words"]) for s in sentences}
    total = len(pools) + len(unservable)
    if total == 0:
        return "no eligible sentences"

    served = np.array([words[p.sentence_id] for p in pools]) if pools else np.array([])
    lines = [
        f"Pools gebaut      : {len(pools)} von {total} ({100 * len(pools) / total:.0f} %)",
        f"Poolgröße         : {len(pools[0]) if pools else 0}, Zufallsniveau "
        f"{pools[0].chance:.3f}" if pools else "Poolgröße         : -",
        f"Längentoleranz    : +/-{tolerance} Wörter",
    ]
    if len(served):
        lines.append(
            f"Bedient, Länge    : {served.min()} bis {served.max()} Wörter, "
            f"Median {np.median(served):.0f}"
        )
    if len(unservable):
        skipped = np.array([words[int(i)] for i in unservable])
        lines.append(
            f"Nicht bedient     : {len(unservable)}, Länge {skipped.min()} bis "
            f"{skipped.max()} Wörter, Median {np.median(skipped):.0f}"
        )
    return "\n".join(lines)


def surface_features(sentences: list[dict], eligible: np.ndarray) -> np.ndarray:
    """Properties a reader can exploit without understanding the text.

    Word count and mean word length in characters. Length drives how many
    fixations a scanpath contains; word length stands in for how common the
    words are, since frequent words are short ones. Both are visible in the
    gaze signal and say nothing about meaning.

    Returns an (n_eligible, 2) array in the order of ``eligible``. Callers that
    have proper frequency estimates can append them as further columns and pass
    the result to ``build_decoy_pools``.
    """
    text_of = {int(s["id"]): str(s["text"]) for s in sentences}
    rows = []
    for sentence_id in eligible:
        words = text_of[int(sentence_id)].split()
        mean_length = float(np.mean([len(w) for w in words])) if words else 0.0
        rows.append([float(len(words)), mean_length])
    return np.array(rows, dtype=float)


def build_decoy_pools(
    sentences: list[dict],
    eligible: np.ndarray,
    features: np.ndarray | None = None,
    seed: int = 0,
) -> list[CandidatePool]:
    """Pair every sentence with the one it is hardest to tell apart from.

    The two-alternative test. Where a pool of twenty-five can only control word
    count, a pool of two can be matched on several surface properties at once,
    because one close partner exists for almost any sentence while twenty-four
    do not. Chance is one half, so anything reliably above it distinguishes two
    texts that agree on everything a scanpath makes obvious.

    The partner is the nearest neighbour in standardised feature space, so no
    single feature dominates through its units. A sentence may serve as the
    decoy for several others.

    Args:
        sentences: sentence records carrying ``id``, ``text`` and ``n_words``.
        eligible: sentence ids to draw from, normally one fold's test set.
        features: (n_eligible, k) properties to match on; word count and mean
            word length when None.
        seed: decides which of the two slots holds the target.

    Returns:
        One two-candidate pool per eligible sentence.

    Raises:
        ValueError: with fewer than two eligible sentences, or when the feature
            rows do not line up with them.
    """
    eligible = np.asarray(eligible)
    if len(eligible) < 2:
        raise ValueError(f"a two-alternative pool needs at least 2 sentences, got {len(eligible)}")

    if features is None:
        features = surface_features(sentences, eligible)
    features = np.asarray(features, dtype=float)
    if len(features) != len(eligible):
        raise ValueError(f"{len(features)} feature rows for {len(eligible)} sentences")

    spread = features.std(axis=0)
    spread[spread == 0] = 1.0
    scaled = (features - features.mean(axis=0)) / spread

    distance = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    partners = distance.argmin(axis=1)

    rng = np.random.default_rng(seed)
    pools = []
    for position, sentence_id in enumerate(eligible):
        target_first = bool(rng.integers(0, 2))
        partner_id = eligible[partners[position]]
        candidates = (
            np.array([sentence_id, partner_id])
            if target_first
            else np.array([partner_id, sentence_id])
        )
        pools.append(
            CandidatePool(
                sentence_id=int(sentence_id),
                candidates=candidates,
                target_position=0 if target_first else 1,
            )
        )
    return pools
