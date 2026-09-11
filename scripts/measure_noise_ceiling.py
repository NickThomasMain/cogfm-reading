"""Measure how much sentence-specific signal the EEG contains at all.

Every result so far compares a trained model against chance and lands on chance.
That answers two questions at once and does not say which was asked: either the
model cannot find the signal, or the signal is not there. This script addresses
the second half without training a binding model.

Twelve people read the same sentence. Split them into two halves, average each
half's trials per sentence, and ask whether one half's average finds its own
sentence among the other half's. No encoder, no anchor, no connector.

The pools, the rank arithmetic and the metrics are imported from the evaluation
package rather than reimplemented, so the number is expressed in the same units
as a result row and the two can be read side by side.

Two readouts, because the answer depends on which is used.

``cosine``
    Plain cosine over the stored dimensions. Weights all of them alike, so a
    representation whose signal sits in few dimensions among many noisy ones is
    measured far below what it carries.
``ridge``
    A cross-validated linear map from one half to the other. It may scale the
    informative dimensions up and the noisy ones down, which is what a trained
    model would do. Sentences are split into folds and the map never sees the
    sentences it is scored on.

Neither is an absolute bound, and the script does not claim one. Two biases work
against each other: the estimator is handed several averaged trials per query
where a real model gets one, which is generous; and a readout may still be
weaker than what a non-linear model could do, which is pessimistic. What the
numbers support is a statement about how much is linearly accessible after
averaging, which is a reference point rather than a ceiling.

Reader identity is the strongest component in this space and is not what the
question is about, so each reader's own mean is subtracted first. That is
legitimate here in a way it is not during training: no binding model is fitted
and no held-out text label is touched.

A sanity check comes free. Run it on the probe embeddings, where every reader of
a sentence holds an identical vector by construction, and the result has to come
out near one. Anything less means this script is wrong, not the data.

Run, from the repo root:

    uv run python scripts/measure_noise_ceiling.py
    uv run python scripts/measure_noise_ceiling.py --embeddings eeg_bands_TRT_words.npz
    uv run python scripts/measure_noise_ceiling.py --readout cosine --tolerance 0
    uv run python scripts/measure_noise_ceiling.py \\
        --embeddings eeg_embeddings_probe_noise0.0.npz   # must come out near 1.0

The spectral split, which asks whether these features carry the reader rather
than the sentence, and then re-runs the ladder without the part that does:

    uv run python scripts/measure_noise_ceiling.py --diagnose-subject
    uv run python scripts/measure_noise_ceiling.py --spectral residual \\
        --target anchor --trials single --subjects disjoint \\
        --pools decoy --control-position
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

from cogfm.data.adapters.zuco_eeg_embeddings import ZuCoEEGEmbeddingDataset
from cogfm.data.splits import make_folds
from cogfm.data.shared_space import (
    project_fold,
    ridge_map,
    shared_space,
)
from cogfm.eval.metrics import summarize_ranks, target_ranks
from cogfm.eval.permutation import gather_scores
from cogfm.eval.pools import build_decoy_pools, build_pools, surface_features

# A sentence needs this many readers to be split at all: at least one per half.
# Two per half is the useful minimum, since a single trial per half measures the
# noise of one recording rather than what the readers share.
MIN_READERS = 4

TRANSFORMS = ("none", "log1p")

# Ridge strengths the inner validation chooses from. The range has to be wide
# because 840 dimensions meet a few hundred sentences and the right strength
# cannot be guessed in advance. It reaches below one because a first run chose
# the smallest value on offer almost every time, which means the grid rather
# than the data was setting the answer.
ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)

# Centre frequency of each band ZuCo reports, in Hz, in the order the extraction
# writes them. That extraction concatenates one full channel block per band, so
# a vector of 840 is read as (8 bands, 105 channels) and not the other way
# round. Getting this the wrong way would still run and still produce numbers.
BAND_CENTRES = np.array(
    [5.0, 7.25, 9.25, 11.75, 15.75, 24.25, 35.25, 44.75], dtype=np.float64
)
CHANNELS = 105
BAND_FEATURES = len(BAND_CENTRES) * CHANNELS


def split_spectrum(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Separate each channel's band powers into a 1/f part and what sits on it.

    A power spectrum is two things laid on top of each other: an aperiodic
    background falling off roughly as 1/f, described by an offset and a slope,
    and the oscillations that sit on it as peaks. Power in fixed bands measures
    their sum and cannot tell them apart, which is the objection Donoghue et al.
    (2020) raise against canonically defined bands.

    That matters here for one specific reason. The aperiodic part is a reader
    fingerprint: Demuru and Fraschini (2020) identify people from it and report
    that it does so better than the canonical bands do. If that holds on this
    corpus, then a large share of what these 840 numbers encode is who is
    wearing the net, and the subject-disjoint rung has been failing on a
    property of the features rather than on the absence of sentence content.

    Eight band centres is a coarse basis for a fit that specparam would do on a
    full spectrum, and ZuCo ships the collapsed bands rather than the spectra.
    So this is the cheap version: one straight line through eight points in log
    power over log frequency, per channel and per trial. Its two coefficients
    are returned as the aperiodic estimate, the residuals around it as the
    periodic remainder. It buys a test, not a measurement, and a result either
    way says what the expensive version would be worth.

    Returns the aperiodic parameters (2 per channel), the residuals (as many as
    went in), and how many values had to be lifted to stay loggable.
    """
    if vectors.shape[1] != BAND_FEATURES:
        raise ValueError(
            f"erwartet {BAND_FEATURES} Bandmerkmale, bekommen {vectors.shape[1]}"
        )
    rows = len(vectors)
    grid = vectors.reshape(rows, len(BAND_CENTRES), CHANNELS)

    # Band power from a Hilbert envelope is non-negative by construction, so a
    # value at or below zero is a gap in the source rather than a small number.
    # Counting them is the honest way to report it: a silent clip would make a
    # broken channel look like a flat spectrum.
    floor = 1e-12
    lifted = int((grid <= floor).sum())
    power = np.log10(np.clip(grid, floor, None))

    axis = np.log10(BAND_CENTRES)
    axis = (axis - axis.mean())[:, None]
    slope = (power * axis).sum(axis=1) / float((axis**2).sum())
    offset = power.mean(axis=1)
    fitted = offset[:, None, :] + slope[:, None, :] * axis
    residual = (power - fitted).reshape(rows, -1)
    aperiodic = np.concatenate([offset, slope], axis=1)
    return aperiodic, residual, lifted


def trial_vectors(dataset: ZuCoEEGEmbeddingDataset) -> np.ndarray:
    """One vector per trial, averaging a word sequence where the file holds one.

    A mean file already stores exactly this; a sequence file is collapsed here,
    because the question is what the readers share about the sentence and not
    how that is distributed over the sentence.
    """
    return np.stack([dataset.sequence(i).mean(axis=0) for i in range(len(dataset))])


def centre_per_reader(vectors: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Subtract each reader's own mean vector."""
    centred = vectors.copy()
    for name in set(subjects.tolist()):
        rows = np.flatnonzero(subjects == name)
        centred[rows] -= centred[rows].mean(axis=0, keepdims=True)
    return centred


def _ledoit_wolf(block: np.ndarray) -> tuple[np.ndarray, float]:
    """Covariance of one reader's trials, shrunk towards a scaled identity.

    With 840 features and a few hundred trials per reader the sample covariance
    is rank-deficient, and whitening by it would amplify directions that are
    pure estimation noise. Ledoit and Wolf's analytical shrinkage picks the
    weight between the sample estimate and a scaled identity that minimises the
    expected squared error, so no hyperparameter has to be guessed. The chosen
    weight is returned, and a value near one means the reader's own covariance
    carried almost nothing and the alignment barely acted.
    """
    n, p = block.shape
    cov = (block.T @ block) / n
    mu = float(np.trace(cov)) / p
    identity = np.eye(p)
    delta = float(np.linalg.norm(cov - mu * identity, "fro") ** 2) / p
    squared = (block**2).sum(axis=1)
    beta_bar = float((squared**2).sum() - n * np.linalg.norm(cov, "fro") ** 2) / (n**2 * p)
    weight = min(max(beta_bar, 0.0), delta) / delta if delta > 0 else 1.0
    return (1.0 - weight) * cov + weight * mu * identity, weight


def scale_per_reader(vectors: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Give every feature the same spread within each reader.

    The cautious half of alignment. It removes the scale differences between
    people without touching the correlation structure, so unlike a full
    whitening it cannot amplify directions that are only estimation noise. Where
    the full version destroys signal, this one bounds how much of that loss was
    the transform rather than the question.
    """
    scaled = vectors.copy()
    for name in sorted(set(subjects.tolist())):
        rows = np.flatnonzero(subjects == name)
        block = scaled[rows]
        block = block - block.mean(axis=0, keepdims=True)
        scaled[rows] = block / block.std(axis=0, keepdims=True).clip(min=1e-12)
    return scaled


def whiten_per_reader(
    vectors: np.ndarray, subjects: np.ndarray, shrinkage: float | None = None
) -> tuple[np.ndarray, list[float]]:
    """Give every reader the same second-order structure.

    The measurements say the mapping from band power to language is
    reader-specific, and subtracting each reader's mean does not fix it: the
    additive part was already removed and the subject-disjoint number still sat
    at chance. What differs is not an offset but the shape of each reader's
    space, because the net sits differently on every head and the same cortical
    source therefore reaches the scalp as a different pattern.

    Whitening by a reader's own covariance removes that shape. It is the
    band-power analogue of Euclidean alignment, which is normally applied to
    covariance matrices of the raw signal rather than to feature vectors, so it
    is the published idea adapted rather than the published method.

    Its limit is worth stating: this equalises the distribution of each reader
    but does not establish which direction means what. Where the discriminative
    directions themselves differ between people, a shared space has to be
    estimated instead.

    ``shrinkage`` fixes the weight towards the scaled identity; None asks
    Ledoit and Wolf's analytical value. That analytical value is the wrong tool
    here and a fixed sweep is the right one, for a reason worth recording: their
    formula minimises the error of the covariance estimate, while whitening
    needs its inverse, whose error is dominated by the smallest eigenvalues.
    With more features than trials per reader those eigenvalues are estimation
    noise, and a weight chosen for the forward problem leaves them small enough
    that inverting blows them up. Measured on this corpus, the analytical value
    came out at 0.014 and cost the subject-sharing control 0.15.
    """
    aligned = vectors.copy()
    weights = []
    for name in sorted(set(subjects.tolist())):
        rows = np.flatnonzero(subjects == name)
        block = aligned[rows]
        block = block - block.mean(axis=0, keepdims=True)
        if shrinkage is None:
            cov, weight = _ledoit_wolf(block)
        else:
            sample = (block.T @ block) / len(block)
            mu = float(np.trace(sample)) / sample.shape[0]
            weight = float(shrinkage)
            cov = (1.0 - weight) * sample + weight * mu * np.eye(sample.shape[0])
        weights.append(weight)
        values, directions = np.linalg.eigh(cov)
        values = np.clip(values, 1e-12, None)
        aligned[rows] = block @ ((directions * values**-0.5) @ directions.T)
    return aligned, weights


def readers_by_sentence(sentence_ids: np.ndarray) -> dict[int, list[int]]:
    """Trial indices per sentence, one list per sentence id."""
    groups: dict[int, list[int]] = {}
    for index, sentence in enumerate(sentence_ids.tolist()):
        groups.setdefault(int(sentence), []).append(index)
    return groups


def split_halves(
    vectors: np.ndarray,
    groups: dict[int, list[int]],
    rng: np.random.Generator,
    min_readers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average each sentence over two disjoint halves of its readers.

    Returns the two arrays and the sentence ids they follow. Sentences with too
    few readers are left out.
    """
    kept, first, second = [], [], []
    for sentence in sorted(groups):
        trials = np.array(groups[sentence])
        if len(trials) < min_readers:
            continue
        order = rng.permutation(len(trials))
        cut = len(trials) // 2
        first.append(vectors[trials[order[:cut]]].mean(axis=0))
        second.append(vectors[trials[order[cut : 2 * cut]]].mean(axis=0))
        kept.append(sentence)
    return np.stack(first), np.stack(second), np.array(kept, dtype=np.int64)


def sentence_anchors(sentences: list[dict], order: np.ndarray, name: str) -> dict[int, np.ndarray]:
    """The frozen language embedding of each sentence, keyed by sentence id.

    Predicting the other half's EEG shows that a sentence produces a
    reproducible pattern. It does not show that the pattern has anything to do
    with language: reading time, rhythm and difficulty are reproducible and
    sentence-specific too. Swapping the target for the anchor asks the question
    that matters, and it is the one point at which the whole line can still
    turn out to be about something other than text.
    """
    import torch
    from omegaconf import OmegaConf

    import cogfm.anchor  # noqa: F401  (registers the anchors)
    from cogfm.registry import ANCHORS

    config = OmegaConf.load(CONFIG_DIR / "anchor" / f"{name}.yaml")
    params = {k: v for k, v in config.items() if k not in ("name", "dim")}
    anchor = ANCHORS.build(config.name, dim=config.dim, **params)
    anchor.requires_grad_(False)

    text_by_id = {int(record["id"]): record["text"] for record in sentences}
    wanted = [int(s) for s in order]
    with torch.no_grad():
        embedded = anchor([text_by_id[i] for i in wanted]).cpu().numpy().astype(np.float64)
    return {sentence: embedded[row] for row, sentence in enumerate(wanted)}


def presentation_position(
    subjects: np.ndarray, tasks: np.ndarray, sentence_ids: np.ndarray
) -> dict[int, float]:
    """Where in the reading session each sentence sat, from 0 to 1.

    The corpus presents its sentences in the same order to every reader: the
    position of a sentence correlates at r = 0.9999 between any two of them.
    Sentence identity and time on task are therefore almost the same variable,
    and anything drifting over a session is sentence-specific and shared across
    readers at once. Electrode impedance rises as the sponges dry, alpha power
    grows with drowsiness, and muscle tension moves the gamma bands; six of the
    eight bands are exposed to at least one of those.

    Positions are read off the row order, which follows the source file, and
    normalised per reader because artefact rejection removes different trials
    for different people. Averaging over readers gives one number per sentence.
    """
    collected: dict[int, list[float]] = {}
    for subject in sorted(set(subjects.tolist())):
        for task in sorted(set(tasks.tolist())):
            rows = np.flatnonzero((subjects == subject) & (tasks == task))
            if len(rows) < 2:
                continue
            for rank, row in enumerate(rows):
                collected.setdefault(int(sentence_ids[row]), []).append(rank / (len(rows) - 1))
    return {sentence: float(np.mean(v)) for sentence, v in collected.items()}


def _decoy_features(
    sentences: list[dict], eligible: np.ndarray, positions: dict[int, float] | None
) -> np.ndarray | None:
    """Properties a decoy is matched on: word count, word length, and position.

    None keeps the default pair, which controls what a reader can see without
    understanding the text but not when the sentence was read.
    ``build_decoy_pools`` standardises the columns, so the added one carries the
    same weight as the two it joins.
    """
    if positions is None:
        return None
    base = surface_features(sentences, eligible)
    column = np.array([[positions.get(int(s), 0.5)] for s in eligible], dtype=float)
    return np.concatenate([base, column], axis=1)


def _cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Similarity of every row of one array to every row of the other."""
    first = first / np.linalg.norm(first, axis=1, keepdims=True).clip(min=1e-12)
    second = second / np.linalg.norm(second, axis=1, keepdims=True).clip(min=1e-12)
    return first @ second.T


def _diagonal_lead(first: np.ndarray, second: np.ndarray) -> float:
    """How far a sentence's own pair sits above the other pairs.

    Used to choose the ridge strength, where a single scalar is wanted and the
    pool machinery would be needless overhead.
    """
    similarity = _cosine(first, second)
    if len(similarity) < 2:
        return float("-inf")
    off = (similarity.sum() - np.trace(similarity)) / (similarity.size - len(similarity))
    return float(np.mean(np.diag(similarity)) - off)


def _lead(
    similarity: np.ndarray, query_sentences: np.ndarray, column_sentences: np.ndarray
) -> float:
    """How far each query sits above the average candidate, in cosine.

    Used to choose the ridge strength, where one scalar is wanted and the pool
    machinery would be needless overhead. Works whether the queries are the
    columns themselves or many trials of them.
    """
    index = {int(s): i for i, s in enumerate(column_sentences)}
    own = np.array(
        [similarity[row, index[int(s)]] for row, s in enumerate(query_sentences)]
    )
    return float(np.mean(own - similarity.mean(axis=1)))


def _pool_ranks(
    similarity: np.ndarray,
    sentence_order: np.ndarray,
    sentences: list[dict],
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    decoy: bool = False,
    positions: dict[int, float] | None = None,
    query_sentences: np.ndarray | None = None,
) -> np.ndarray:
    """Rank of the correct sentence among the candidates it is held against.

    Rows of ``similarity`` and entries of ``sentence_order`` follow each other,
    and the columns hold the same sentences, so a pool's candidates translate
    directly into columns.

    Two ways to choose those candidates. The length-matched pool holds
    ``pool_size`` sentences within a word-count band, which needs that many
    eligible sentences to exist; where the eligible set is small, as it is
    inside a cross-validation fold, a narrow band leaves pools unfillable. The
    decoy pool holds the single nearest neighbour in standardised surface
    space, matched on word count and mean word length at once, so it needs only
    one partner and controls more than word count alone.
    """
    if decoy:
        pools = build_decoy_pools(
            sentences,
            sentence_order,
            features=_decoy_features(sentences, sentence_order, positions),
            seed=pool_seed,
        )
    else:
        pools, _ = build_pools(
            sentences, sentence_order, pool_size=pool_size, tolerance=tolerance,
            seed=pool_seed,
        )
    by_sentence = {pool.sentence_id: pool for pool in pools}
    # One row per query. Where the queries are single trials there are many of
    # them per sentence, and ``sentence_order`` still names the columns.
    queries = sentence_order if query_sentences is None else query_sentences
    rows = np.flatnonzero([int(s) in by_sentence for s in queries])
    if not len(rows):
        return np.array([], dtype=float)

    column_of = {int(s): i for i, s in enumerate(sentence_order)}
    columns = np.array(
        [
            [column_of[int(c)] for c in by_sentence[int(queries[i])].candidates]
            for i in rows
        ],
        dtype=np.int64,
    )
    targets = np.array(
        [by_sentence[int(queries[i])].target_position for i in rows], dtype=np.int64
    )
    return target_ranks(gather_scores(similarity[rows], columns), targets)


def cosine_split(
    vectors: np.ndarray,
    groups: dict[int, list[int]],
    sentences: list[dict],
    rng: np.random.Generator,
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    min_readers: int,
    decoy: bool = False,
    positions: dict[int, float] | None = None,
) -> dict[str, float]:
    """Score one reader division with an unlearned cosine readout.

    Each half is centred over its own sentences first. Without that the shared
    channel and band profile dominates every pair alike and every similarity
    lands near one, which measures the montage rather than the sentence.
    """
    first, second, order = split_halves(vectors, groups, rng, min_readers)
    first = first - first.mean(axis=0, keepdims=True)
    second = second - second.mean(axis=0, keepdims=True)
    similarity = _cosine(first, second)

    ranks = _pool_ranks(
        similarity, order, sentences, pool_size, tolerance, pool_seed, decoy, positions
    )
    if not len(ranks):
        raise SystemExit(
            f"kein Satz konnte einen Pool fuellen: {len(order)} Saetze, Poolgroesse "
            f"{pool_size}, Toleranz +/-{tolerance}. Toleranz erhoehen, Pool verkleinern "
            f"oder --pools decoy nutzen."
        )
    effective = 2 if decoy else pool_size
    summary = summarize_ranks(ranks, pool_size=effective, ks=(1, 5))

    diagonal = float(np.mean(np.diag(similarity)))
    off = float((similarity.sum() - np.trace(similarity)) / (similarity.size - len(similarity)))
    summary["diagonal"] = diagonal
    summary["off_diagonal"] = off
    summary["separation"] = diagonal - off
    summary["n_sentences"] = float(len(order))
    return summary


def identify_readers(
    vectors: np.ndarray,
    subjects: np.ndarray,
    sentence_ids: np.ndarray,
    folds: int,
    seed: int,
) -> dict:
    """How much of a feature set says who was reading rather than what was read.

    A linear readout is fitted from the features to reader identity and scored
    on sentences it never saw. Sentences are held out rather than trials, so the
    readout cannot win by recognising a sentence and recalling who read it.

    Reader centering is deliberately not applied here: it subtracts exactly the
    quantity being measured. The number is therefore how separable the readers
    are before any alignment, which is what makes it comparable across feature
    sets rather than a statement about the pipeline that follows.

    A caveat that belongs before the run and not after it. The readers differ in
    when they sat in the scanner, and session drift is part of reader identity
    rather than a confound for this particular question, so nothing is matched
    on position. The number answers whether the reader is recoverable, not why.
    """
    names = sorted(set(subjects.tolist()))
    lookup = {name: position for position, name in enumerate(names)}
    truth = np.array([lookup[name] for name in subjects.tolist()])
    onehot = np.zeros((len(truth), len(names)))
    onehot[np.arange(len(truth)), truth] = 1.0

    sentences = np.array(sorted(set(sentence_ids.tolist())), dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    rows_of = np.array([int(s) for s in sentence_ids.tolist()])

    hits = 0
    total = 0
    picked: list[float] = []
    for fold in range(folds):
        held = set(sentences[order[fold::folds]].tolist())
        mask = np.array([value in held for value in rows_of.tolist()])
        test = np.flatnonzero(mask)
        train = np.flatnonzero(~mask)
        if len(test) == 0 or len(train) == 0:
            continue

        # Standardise on the training rows only, and give every set the same
        # treatment, so one strength grid means the same thing for 840 raw
        # bands as for 210 line coefficients and the comparison is about the
        # features rather than about their units.
        centre = vectors[train].mean(axis=0, keepdims=True)
        spread = vectors[train].std(axis=0, keepdims=True).clip(min=1e-12)
        fit_x = np.hstack([(vectors[train] - centre) / spread, np.ones((len(train), 1))])
        use_x = np.hstack([(vectors[test] - centre) / spread, np.ones((len(test), 1))])
        fit_y = onehot[train]
        fit_truth = truth[train]

        inner = rng.permutation(len(train))
        cut = max(1, len(inner) // 5)
        hold, keep = inner[:cut], inner[cut:]

        # The gram matrix does not depend on the strength, so it is built once
        # and the grid only touches its diagonal. Rebuilding it per strength
        # would multiply the cost of this diagnostic by the size of the grid for
        # no change in the answer.
        gram = fit_x[keep].T @ fit_x[keep]
        cross = fit_x[keep].T @ fit_y[keep]
        best_alpha = None
        best_score = -1.0
        for alpha in ALPHAS:
            shrunk = gram.copy()
            shrunk.flat[:: shrunk.shape[0] + 1] += alpha
            try:
                weights = np.linalg.solve(shrunk, cross)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(weights).all():
                continue
            score = float(
                (np.argmax(fit_x[hold] @ weights, axis=1) == fit_truth[hold]).mean()
            )
            if score > best_score:
                best_alpha, best_score = alpha, score
        if best_alpha is None:
            continue
        picked.append(best_alpha)

        weights = ridge_map(fit_x, fit_y, best_alpha)
        if weights is None:
            continue
        hits += int((np.argmax(use_x @ weights, axis=1) == truth[test]).sum())
        total += len(test)

    return {
        "accuracy": hits / total if total else float("nan"),
        "chance": 1.0 / len(names),
        "readers": len(names),
        "queries": total,
        "alpha": float(np.median(picked)) if picked else float("nan"),
    }


def learned_split(
    vectors: np.ndarray,
    groups: dict[int, list[int]],
    sentences: list[dict],
    rng: np.random.Generator,
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    min_readers: int,
    n_folds: int,
    decoy: bool = False,
    positions: dict[int, float] | None = None,
    anchors: dict[int, np.ndarray] | None = None,
) -> dict[str, float]:
    """Score one reader division through a cross-validated linear readout.

    Sentences are split into folds. The map is fitted on the training sentences
    of a fold and applied to its held-out ones, so no sentence contributes to
    the map that is later scored against it. Centring statistics come from the
    fitting sentences alone for the same reason.

    The ridge strength is chosen on a validation part of the training sentences
    rather than on the test fold, and the chosen values are reported: a strength
    sitting at the edge of the grid means the grid was too narrow.
    """
    first, second, order = split_halves(vectors, groups, rng, min_readers)
    # With an anchor target the second half is not the thing being predicted, so
    # only the first is used as input. Everything else stays as it was, which is
    # what makes the two numbers differ in the target and in nothing else.
    if anchors is not None:
        second = np.stack([anchors[int(s)] for s in order])
    fold_of = rng.permutation(len(order)) % n_folds

    collected, chosen = [], []
    for fold in range(n_folds):
        test = np.flatnonzero(fold_of == fold)
        # Shuffled before the inner split. flatnonzero returns sorted positions
        # and the sentences are ordered by id, which runs one reading task after
        # the other, so a prefix would validate on one task and fit on the other.
        train = rng.permutation(np.flatnonzero(fold_of != fold))
        inner = train[: max(2, len(train) // 5)]
        fit = train[len(inner) :]

        centre_a = first[fit].mean(axis=0, keepdims=True)
        centre_b = second[fit].mean(axis=0, keepdims=True)
        a_fit, b_fit = first[fit] - centre_a, second[fit] - centre_b

        best, best_alpha, best_weights = -np.inf, None, None
        for alpha in ALPHAS:
            weights = ridge_map(a_fit, b_fit, alpha)
            if weights is None:
                continue
            score = _diagonal_lead(
                (first[inner] - centre_a) @ weights, second[inner] - centre_b
            )
            if score > best:
                best, best_alpha, best_weights = score, alpha, weights
        if best_weights is None:
            raise SystemExit("keine Ridge-Staerke des Rasters liess sich loesen")
        chosen.append(best_alpha)

        similarity = _cosine(
            (first[test] - centre_a) @ best_weights, second[test] - centre_b
        )
        collected.append(
            _pool_ranks(
                similarity, order[test], sentences, pool_size, tolerance, pool_seed,
                decoy, positions,
            )
        )

    ranks = np.concatenate(collected)
    if not len(ranks):
        raise SystemExit(
            f"kein Satz konnte einen Pool fuellen: je Fold nur rund {len(order) // n_folds} "
            f"Kandidaten bei Poolgroesse {pool_size} und Toleranz +/-{tolerance}. "
            f"--folds senken, Pool verkleinern oder --pools decoy nutzen."
        )
    effective = 2 if decoy else pool_size
    summary = summarize_ranks(ranks, pool_size=effective, ks=(1, 5))
    summary["alpha_median"] = float(np.median(chosen))
    summary["alpha_at_low"] = float(np.mean([a == ALPHAS[0] for a in chosen]))
    summary["alpha_at_high"] = float(np.mean([a == ALPHAS[-1] for a in chosen]))
    summary["n_sentences"] = float(len(order))
    return summary


def single_trial_split(
    vectors: np.ndarray,
    sentence_ids: np.ndarray,
    sentences: list[dict],
    eligible: np.ndarray,
    anchors: dict[int, np.ndarray],
    rng: np.random.Generator,
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    n_folds: int,
    decoy: bool,
    positions: dict[int, float] | None,
) -> dict[str, float]:
    """Score every trial on its own, with no averaging over readers.

    The averaged measurement hands the estimator several recordings per query.
    A real run has one. Everything else is kept, so the distance between the two
    numbers is what the averaging was worth.

    Folds run over sentences and every trial inherits its sentence's fold, so a
    sentence never appears on both sides even though many readers produced it.
    Dropping the averaging costs signal per row and gains rows: roughly eleven
    times as many, which is why the direction of the net effect is worth
    measuring rather than assuming.
    """
    position_of = {int(s): i for i, s in enumerate(eligible)}
    rows = np.flatnonzero([int(s) in position_of for s in sentence_ids])
    signals = vectors[rows]
    row_sentence = sentence_ids[rows]

    fold_of_sentence = rng.permutation(len(eligible)) % n_folds
    fold_of_row = np.array([fold_of_sentence[position_of[int(s)]] for s in row_sentence])
    target = np.stack([anchors[int(s)] for s in row_sentence])

    collected, chosen = [], []
    for fold in range(n_folds):
        test = np.flatnonzero(fold_of_row == fold)
        train = rng.permutation(np.flatnonzero(fold_of_row != fold))
        inner = train[: max(2, len(train) // 5)]
        fit = train[len(inner) :]

        centre_x = signals[fit].mean(axis=0, keepdims=True)
        centre_y = target[fit].mean(axis=0, keepdims=True)
        x_fit, y_fit = signals[fit] - centre_x, target[fit] - centre_y

        inner_sentences = np.array(sorted(set(row_sentence[inner].tolist())), dtype=np.int64)
        inner_columns = np.stack([anchors[int(s)] for s in inner_sentences]) - centre_y

        best, best_alpha, best_weights = -np.inf, None, None
        for alpha in ALPHAS:
            weights = ridge_map(x_fit, y_fit, alpha)
            if weights is None:
                continue
            score = _lead(
                _cosine((signals[inner] - centre_x) @ weights, inner_columns),
                row_sentence[inner],
                inner_sentences,
            )
            if score > best:
                best, best_alpha, best_weights = score, alpha, weights
        if best_weights is None:
            raise SystemExit("keine Ridge-Staerke des Rasters liess sich loesen")
        chosen.append(best_alpha)

        test_sentences = np.array(sorted(set(row_sentence[test].tolist())), dtype=np.int64)
        test_columns = np.stack([anchors[int(s)] for s in test_sentences]) - centre_y
        similarity = _cosine((signals[test] - centre_x) @ best_weights, test_columns)
        collected.append(
            _pool_ranks(
                similarity, test_sentences, sentences, pool_size, tolerance, pool_seed,
                decoy, positions, row_sentence[test],
            )
        )

    ranks = np.concatenate(collected)
    if not len(ranks):
        raise SystemExit("kein Satz konnte einen Pool fuellen")
    effective = 2 if decoy else pool_size
    summary = summarize_ranks(ranks, pool_size=effective, ks=(1, 5))
    summary["alpha_median"] = float(np.median(chosen))
    summary["alpha_at_low"] = float(np.mean([a == ALPHAS[0] for a in chosen]))
    summary["alpha_at_high"] = float(np.mean([a == ALPHAS[-1] for a in chosen]))
    summary["n_sentences"] = float(len(eligible))
    summary["n_queries"] = float(len(ranks))
    return summary


def position_features(
    sentence_ids: np.ndarray, positions: dict[int, float], powers: int = 6
) -> np.ndarray:
    """Every row replaced by smooth functions of when its sentence was read.

    This is the control that decides whether the drift objection has any force
    left. Electrode impedance, alpha with fatigue and muscle in the gamma bands
    all change slowly over a session, and because ZuCo shows every reader the
    same sentence order, drift is both shared across readers and locked to
    sentence identity. A method that fuses readers by what they agree on will
    find the clock if the clock is the loudest agreement.

    The decoy is matched on presentation position for exactly that reason. The
    question this answers is whether that matching is tight enough: a decoder
    handed nothing but the clock, and a generous polynomial basis to read it
    with, is the strongest drift-only decoder that exists. If it lands on
    chance, then no amount of drift information can beat these pools, and the
    objection is closed for every other run using them. If it lands above,
    every number measured on these pools is suspect, including the good ones.

    A negative result here is therefore worth more than a positive one
    anywhere else, and it costs one run.
    """
    raw = np.array([positions[int(s)] for s in sentence_ids.tolist()])
    centred = 2.0 * raw - 1.0
    return np.stack([centred ** power for power in range(1, powers + 1)], axis=1)


def _fit_and_score(
    signals: np.ndarray,
    target: np.ndarray,
    row_sentence: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    anchors: dict[int, np.ndarray],
    sentences: list[dict],
    rng: np.random.Generator,
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    decoy: bool,
    positions: dict[int, float] | None,
) -> tuple[np.ndarray, float]:
    """One ridge fit on given rows, scored on the rest. Returns ranks and alpha."""
    train = rng.permutation(train)
    inner = train[: max(2, len(train) // 5)]
    fit = train[len(inner) :]

    centre_x = signals[fit].mean(axis=0, keepdims=True)
    centre_y = target[fit].mean(axis=0, keepdims=True)
    x_fit, y_fit = signals[fit] - centre_x, target[fit] - centre_y

    inner_sentences = np.array(sorted(set(row_sentence[inner].tolist())), dtype=np.int64)
    inner_columns = np.stack([anchors[int(s)] for s in inner_sentences]) - centre_y

    best, best_alpha, best_weights = -np.inf, None, None
    for alpha in ALPHAS:
        weights = ridge_map(x_fit, y_fit, alpha)
        if weights is None:
            continue
        score = _lead(
            _cosine((signals[inner] - centre_x) @ weights, inner_columns),
            row_sentence[inner],
            inner_sentences,
        )
        if score > best:
            best, best_alpha, best_weights = score, alpha, weights
    if best_weights is None:
        raise SystemExit("keine Ridge-Staerke des Rasters liess sich loesen")

    test_sentences = np.array(sorted(set(row_sentence[test].tolist())), dtype=np.int64)
    test_columns = np.stack([anchors[int(s)] for s in test_sentences]) - centre_y
    similarity = _cosine((signals[test] - centre_x) @ best_weights, test_columns)
    ranks = _pool_ranks(
        similarity, test_sentences, sentences, pool_size, tolerance, pool_seed,
        decoy, positions, row_sentence[test],
    )
    return ranks, best_alpha


def subject_split_draws(
    vectors: np.ndarray,
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    sentences: list[dict],
    eligible: np.ndarray,
    anchors: dict[int, np.ndarray],
    rng: np.random.Generator,
    pool_size: int,
    tolerance: int,
    pool_seed: int,
    n_folds: int,
    decoy: bool,
    positions: dict[int, float] | None,
    shared: dict | None = None,
    positions_for_null: dict[int, float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Hold out readers as well as sentences, and a size-matched control beside it.

    ``make_folds`` from the evaluation package builds the split, so the protocol
    is the one the result row uses rather than a second implementation of it:
    test is the intersection of one reader group and one sentence group, and
    training is the complement on both axes. Trials overlapping on exactly one
    axis belong to neither and are dropped, which costs roughly two thirds of
    the data.

    That cost is the reason for the second summary. A drop against the
    subject-sharing measurement would otherwise have two possible causes at
    once: unseen readers, and a training set a third of the size. The control
    keeps the same test trials and the same training sentences but draws its
    training rows from every reader, subsampled to the size the disjoint split
    left. Whatever separates the two is the reader axis alone.
    """
    position_of = {int(s): i for i, s in enumerate(eligible)}
    rows = np.flatnonzero([int(s) in position_of for s in sentence_ids])
    signals, row_sentence = vectors[rows], sentence_ids[rows]
    row_subject = subject_ids[rows]
    target = np.stack([anchors[int(s)] for s in row_sentence])

    folds = make_folds(
        row_subject, row_sentence, sentences, n_folds=n_folds,
        seed=int(rng.integers(1 << 30)),
    )

    disjoint, matched, alphas, sizes = [], [], [], []
    for fold in folds:
        train, test = fold.train, fold.test
        sizes.append(len(train))

        # The shared space is fitted here and not once for the whole corpus,
        # because it may not see this fold's readers or this fold's sentences.
        # Fitting it outside the loop would be the same leak the whole ladder
        # exists to avoid, and it would not announce itself in the numbers.
        fold_signals = signals
        if shared is not None:
            fold_signals = project_fold(
                signals, row_sentence, row_subject, train, test, anchors,
                shared["mode"], shared["components"], shared["view_pcs"],
                shared["ridge"], shared["iterations"], shared["stimulus"],
                shared.get("null", "none"),
                positions_for_null if positions_for_null is not None else positions,
                int(rng.integers(1 << 30)),
            )
            if fold_signals is None:
                continue

        ranks, alpha = _fit_and_score(
            fold_signals, target, row_sentence, train, test, anchors, sentences, rng,
            pool_size, tolerance, pool_seed, decoy, positions,
        )
        disjoint.append(ranks)
        alphas.append(alpha)

        # Same training sentences, every reader, cut to the same number of rows.
        training_sentences = set(fold.train_sentences.tolist())
        pool = np.flatnonzero([int(s) in training_sentences for s in row_sentence])
        pool = np.setdiff1d(pool, test, assume_unique=False)
        drawn = rng.choice(pool, size=min(len(train), len(pool)), replace=False)
        control, _ = _fit_and_score(
            fold_signals, target, row_sentence, drawn, test, anchors, sentences, rng,
            pool_size, tolerance, pool_seed, decoy, positions,
        )
        matched.append(control)

    effective = 2 if decoy else pool_size

    def wrap(collected: list[np.ndarray]) -> dict[str, float]:
        ranks = np.concatenate(collected)
        if not len(ranks):
            raise SystemExit("kein Satz konnte einen Pool fuellen")
        summary = summarize_ranks(ranks, pool_size=effective, ks=(1, 5))
        summary["alpha_median"] = float(np.median(alphas))
        summary["alpha_at_low"] = float(np.mean([a == ALPHAS[0] for a in alphas]))
        summary["alpha_at_high"] = float(np.mean([a == ALPHAS[-1] for a in alphas]))
        summary["n_sentences"] = float(len(eligible))
        summary["n_queries"] = float(len(ranks))
        summary["n_train"] = float(np.mean(sizes))
        return summary

    return wrap(disjoint), wrap(matched)


def _report(name: str, draws: list[dict], splits: int, pool_size: int) -> None:
    def spread(key: str) -> tuple[float, float]:
        values = np.array([d[key] for d in draws], dtype=float)
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    chance = 1.0 / pool_size
    print(f"  [{name}] {draws[0]['n_sentences']:.0f} Saetze, {splits} Leseraufteilungen")
    wanted = [("recall@1", chance), ("recall@5", 5 * chance), ("percentile", 0.5)]
    for key, level in [pair for pair in wanted if pair[0] in draws[0]]:
        mean, sd = spread(key)
        print(f"    {key:12s} {mean:6.3f} +/- {sd:.3f}   Zufall {level:.3f}   "
              f"Faktor {mean / level:5.2f}")
    if "separation" in draws[0]:
        mean, sd = spread("separation")
        d_mean, _ = spread("diagonal")
        o_mean, _ = spread("off_diagonal")
        print(f"    Kosinus gleicher Satz {d_mean:+.4f}, anderer Satz {o_mean:+.4f}, "
              f"Abstand {mean:+.4f} +/- {sd:.4f}")
    if "alpha_median" in draws[0]:
        alpha, _ = spread("alpha_median")
        low, _ = spread("alpha_at_low")
        high, _ = spread("alpha_at_high")
        if low > 0.05:
            verdict = f"Raster nach unten erweitern (unterster Wert {ALPHAS[0]:g})"
        elif high > 0.05:
            verdict = f"Raster nach oben erweitern (oberster Wert {ALPHAS[-1]:g})"
        else:
            verdict = "Raster ausreichend"
        print(f"    Ridge-Staerke Median {alpha:g}, unterer Rand {100 * low:.0f} %, "
              f"oberer Rand {100 * high:.0f} %  ->  {verdict}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/zuco/processed")
    parser.add_argument("--embeddings", default="eeg_bands_TRT_sentence.npz")
    parser.add_argument("--readout", default="both", choices=("cosine", "ridge", "both"))
    parser.add_argument("--target", default="eeg", choices=("eeg", "anchor"),
                        help="eeg: predict the other half's signal; anchor: predict the "
                             "frozen language embedding of the sentence")
    parser.add_argument("--anchor", default="qwen3", help="config name under configs/anchor")
    parser.add_argument("--subjects", default="shared", choices=("shared", "disjoint"),
                        help="disjoint: hold out readers as well as sentences, using the "
                             "same make_folds protocol as the result row; prints a "
                             "size-matched control beside it")
    parser.add_argument("--trials", default="averaged", choices=("averaged", "single"),
                        help="averaged: readers pooled per sentence; single: one trial per "
                             "query, which is what a real run has. single needs "
                             "--target anchor")
    parser.add_argument("--splits", type=int, default=50,
                        help="random reader divisions for the cosine readout")
    parser.add_argument("--ridge-splits", type=int, default=10,
                        help="random reader divisions for the ridge readout, which is slower")
    parser.add_argument("--folds", type=int, default=4,
                        help="sentence folds the ridge readout is cross-validated over")
    parser.add_argument("--min-readers", type=int, default=MIN_READERS)
    parser.add_argument("--pool-size", type=int, default=25)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--control-position", action="store_true",
                        help="match the decoy on when the sentence was read as well; "
                             "only meaningful with --pools decoy")
    parser.add_argument("--pools", default="lengths", choices=("lengths", "decoy"),
                        help="lengths: pool-size candidates inside a word-count band; "
                             "decoy: the single nearest neighbour on word count and mean "
                             "word length, chance 0.500")
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-reader-centering", action="store_true")
    parser.add_argument("--align", default="none", choices=("none", "diag", "whiten"),
                        help="diag: give every feature the same spread per reader; "
                             "whiten: give every reader the same covariance. Both use only "
                             "that reader's own trials and no labels")
    parser.add_argument("--shrinkage", default="0.1,0.3,0.5,0.9",
                        help="comma-separated weights towards a scaled identity for "
                             "--align whiten, or 'lw' for the analytical value. One result "
                             "block per value, so the choice is measured and not assumed")
    parser.add_argument("--spectral", default="none",
                        choices=("none", "residual", "aperiodic"),
                        help="split each channel's eight band powers into a 1/f line and "
                             "the residuals around it, before centering and before any "
                             "alignment. residual: keep the periodic remainder (840); "
                             "aperiodic: keep offset and slope only (210). Needs the band "
                             "features, not an arbitrary embedding")
    parser.add_argument("--shared", default="none",
                        choices=("none", "pca", "gcca", "all"),
                        help="estimate a space shared across readers from which sentence is "
                             "which, refitted inside every fold. gcca: MCCA at "
                             "--shared-stimulus 0, stimulus-informed GCCA above it. pca: the "
                             "control that uses no correspondence, same dimensions. all: "
                             "both, which is the only way the gcca number can be read. "
                             "Needs --subjects disjoint")
    parser.add_argument("--shared-components", default="5,10,25,50",
                        help="comma-separated widths of the shared space. The grid has to "
                             "reach low: on synthetic data at a signal-to-noise ratio of "
                             "0.5 a width of 50 scored below predicting the mean while 10 "
                             "recovered the truth, which is de Cheveigne's warning about "
                             "shared noise showing up as a number")
    parser.add_argument("--shared-stimulus", default="0,1,5",
                        help="comma-separated weights of the stimulus view, 0 being plain "
                             "MCCA. The anchor counts as one voice among eleven readers at "
                             "weight 1, so steering it needs more than that")
    parser.add_argument("--shared-null", default="none", choices=("none", "shift"),
                        help="shift: rotate every reader's sentences by a different amount "
                             "along the reading order before the shared space is fitted, so "
                             "the recordings and their temporal shape survive but no two "
                             "readers are paired as the same sentence. The null for whether "
                             "the correspondence is what works. Needs --control-position")
    parser.add_argument("--position-only", action="store_true",
                        help="replace every feature vector by smooth functions of when its "
                             "sentence was read, and nothing else. The strongest drift-only "
                             "decoder there is. On position-matched pools it has to land on "
                             "chance; anything above means the matching is too loose and "
                             "every number measured on these pools is suspect")
    parser.add_argument("--shared-view-pcs", type=int, default=60,
                        help="axes kept per reader before fusion. This truncation is where "
                             "the regularisation of MCCA lives")
    parser.add_argument("--shared-ridge", type=float, default=1e2,
                        help="ridge on each reader's map into the shared space")
    parser.add_argument("--shared-iters", type=int, default=5,
                        help="MAXVAR passes after the classical solution. 0 is textbook "
                             "MCCA and nothing else, so it is the baseline for what the "
                             "passes add")
    parser.add_argument("--diagnose-subject", action="store_true",
                        help="report how well a linear readout names the reader from the "
                             "raw bands, from the 1/f parameters and from the residuals, "
                             "then stop without running the ladder. Answers whether the "
                             "reader fingerprint sits where the spectral literature says")
    args = parser.parse_args()

    decoy = args.pools == "decoy"
    effective_pool = 2 if decoy else args.pool_size

    shared_settings: list[dict | None] = [None]
    if args.shared != "none":
        if args.subjects != "disjoint":
            raise SystemExit(
                "--shared braucht --subjects disjoint. Der gemeinsame Raum wird je Fold "
                "ohne die gehaltenen Leser geschaetzt; ohne gehaltene Leser misst er nichts."
            )
        widths = [int(v) for v in args.shared_components.split(",") if v.strip()]
        gammas = [float(v) for v in args.shared_stimulus.split(",") if v.strip()]
        base_setting = {
            "view_pcs": args.shared_view_pcs,
            "ridge": args.shared_ridge,
            "iterations": args.shared_iters,
        }
        shared_settings = []
        if args.shared in ("pca", "all"):
            for width in widths:
                shared_settings.append(
                    {**base_setting, "mode": "pca", "components": width, "stimulus": 0.0,
                     "label": f"pca k={width}"}
                )
        marker = "" if args.shared_null == "none" else "  [NULL verschoben]"
        if args.shared in ("gcca", "all"):
            for width in widths:
                for gamma in gammas:
                    name = "MCCA" if gamma == 0.0 else "SI-GCCA"
                    shared_settings.append(
                        {**base_setting, "mode": "gcca", "components": width,
                         "stimulus": gamma, "null": args.shared_null,
                         "label": f"{name} k={width} gamma={gamma:g}{marker}"}
                    )
        print(f"Gemeinsamer Raum: {len(shared_settings)} Einstellungen, "
              f"{args.shared_view_pcs} Achsen je Leser, Ridge {args.shared_ridge:g}, "
              f"{args.shared_iters} MAXVAR-Durchgaenge\n")

    data = ZuCoEEGEmbeddingDataset(root=args.root, embeddings=args.embeddings)
    print(f"=== Rauschdecke ===\n{data.describe()}")
    if args.subjects == "disjoint" and args.target != "anchor":
        raise SystemExit("--subjects disjoint braucht --target anchor")

    if args.trials == "single" and args.target != "anchor":
        raise SystemExit(
            "--trials single braucht --target anchor: ohne Mittelung gibt es keine zweite "
            "Haelfte, gegen die ein EEG-Ziel gebildet werden koennte."
        )

    readout = args.readout
    if args.target == "anchor" and readout in ("cosine", "both"):
        print("Hinweis: --target anchor laesst nur die Ridge-Auslese zu. EEG und Anker "
              "leben in verschiedenen Raeumen, ein direkter Kosinus zwischen ihnen "
              "vergliche Groessen ohne gemeinsame Achsen.\n")
        readout = "ridge"

    positions = None
    if decoy and args.control_position:
        positions = presentation_position(
            data.subject_ids, data.task_ids, data.sentence_ids
        )
    # The clock is needed by the null and by the position-only run even where it
    # is not used to match the pools, so it is kept apart from ``positions``:
    # mixing them would silently turn the decoy printout into a claim the run
    # does not support.
    clock = positions
    if clock is None and (args.position_only or args.shared_null == "shift"):
        clock = presentation_position(
            data.subject_ids, data.task_ids, data.sentence_ids
        )
    if args.shared_null == "shift" and positions is None:
        raise SystemExit(
            "--shared-null shift braucht --pools decoy --control-position. Ohne "
            "positionsangeglichene Kandidaten misst die Null etwas anderes als der Lauf, "
            "den sie pruefen soll, und die beiden waeren nicht vergleichbar."
        )
    if args.position_only and args.shared != "none":
        raise SystemExit("--position-only misst die Uhr allein, --shared gehoert nicht dazu")
    if decoy:
        matched = "Wortzahl, mittlere Wortlaenge"
        if positions is not None:
            matched += ", Sitzungsposition"
        print(f"Kandidaten: zwei angeglichen auf {matched}")
        served = np.array(sorted(positions or {}), dtype=np.int64)
        if positions is not None and len(served) > 1:
            for label, feats in (
                ("ohne", None),
                ("mit", _decoy_features(data.sentences, served, positions)),
            ):
                pools = build_decoy_pools(
                    data.sentences, served, features=feats, seed=args.pool_seed
                )
                gaps = [
                    abs(positions[pool.sentence_id]
                        - positions[int(pool.candidates[1 - pool.target_position])])
                    for pool in pools
                ]
                print(f"  mittlerer Positionsabstand zum Decoy {label} Kontrolle: "
                      f"{np.mean(gaps):.3f}")
    else:
        print(f"Kandidaten: {args.pool_size} innerhalb +/-{args.tolerance} Woertern")
    print()

    raw = trial_vectors(data)
    groups = readers_by_sentence(data.sentence_ids)

    if args.position_only:
        raw = position_features(data.sentence_ids, clock)
        print(f"Nur die Uhr: {raw.shape[1]} Potenzen der Sitzungsposition, "
              f"sonst nichts. Erwartung auf positionsangeglichenen Kandidaten "
              f"ist Zufall; alles darueber entwertet die Kandidaten.\n")

    if (args.spectral != "none" or args.diagnose_subject) and raw.shape[1] != BAND_FEATURES:
        raise SystemExit(
            f"--spectral und --diagnose-subject setzen die Bandstruktur voraus: "
            f"{BAND_FEATURES} Merkmale als {len(BAND_CENTRES)} Baender x {CHANNELS} "
            f"Kanaele. Diese Einbettung hat {raw.shape[1]}. Auf einer Einbettung ohne "
            f"Bandachse ist die Zerlegung nicht definiert, und sie wuerde trotzdem "
            f"Zahlen liefern, deshalb bricht sie hier ab statt zu raten."
        )

    if args.diagnose_subject:
        aperiodic, residual, lifted = split_spectrum(raw)
        print("=== Wer liest? Leseridentitaet aus den Merkmalen ===")
        if lifted:
            print(f"Hinweis: {lifted} Werte lagen bei oder unter null und wurden vor dem "
                  f"Logarithmus angehoben.")
        print("Ausgelassen werden Saetze, nicht Trials: die Auslese kann nicht gewinnen,")
        print("indem sie einen Satz wiedererkennt und sich erinnert, wer ihn gelesen hat.")
        print("Leserzentrierung ist hier aus, sie zoege genau das ab, was gemessen wird.")
        print()
        for label, block in (
            (f"rohe Baender ({raw.shape[1]})", raw),
            (f"1/f-Parameter ({aperiodic.shape[1]})", aperiodic),
            (f"periodischer Rest ({residual.shape[1]})", residual),
        ):
            found = identify_readers(
                block, data.subject_ids, data.sentence_ids, args.folds, args.seed
            )
            print(f"  {label:<26} Treffer {found['accuracy']:.3f}   "
                  f"Zufall {found['chance']:.3f}   "
                  f"{found['queries']} Anfragen   alpha {found['alpha']:g}")
        print()
        print("Lesart. Liegt der Treffer der 1/f-Parameter nahe an dem der rohen Baender")
        print("und deutlich ueber dem des Rests, dann traegt ein grosser Teil der 840")
        print("Zahlen den Leser und nicht den Satz, und --spectral residual ist der")
        print("naechste Lauf. Liegen alle drei gleichauf, ist diese Erklaerung falsch und")
        print("das Ergebnis spart den teuren Weg ueber die vollen Spektren.")
        print()
        print("Was die Zahl nicht sagt: ein hoher Treffer auf dem Rest schliesst nicht")
        print("aus, dass dort auch Satzinhalt steht. Die beiden Fragen sind getrennt, und")
        print("nur der Durchlauf mit --spectral residual beantwortet die zweite.")
        return

    if args.spectral != "none":
        aperiodic, residual, lifted = split_spectrum(raw)
        raw = residual if args.spectral == "residual" else aperiodic
        print(f"Spektrale Zerlegung: {args.spectral}, {raw.shape[1]} Dimensionen")
        if lifted:
            print(f"  {lifted} Werte lagen bei oder unter null und wurden vor dem "
                  f"Logarithmus angehoben")
        print("  Die Zerlegung logarithmiert bereits, deshalb entfaellt log1p.\n")

    anchors = None
    if args.target == "anchor":
        servable = np.array(
            sorted(s for s, rows in groups.items() if len(rows) >= args.min_readers),
            dtype=np.int64,
        )
        print(f"Anker {args.anchor} fuer {len(servable)} Saetze wird gebaut ...")
        anchors = sentence_anchors(data.sentences, servable, args.anchor)
        width = len(next(iter(anchors.values())))
        print(f"  {width} Dimensionen\n")
    counts = np.array([len(v) for v in groups.values()])
    print(f"Saetze {len(groups)} | Leser je Satz: Median {np.median(counts):.0f}, "
          f"min {counts.min()}, max {counts.max()}")
    print(f"mit mindestens {args.min_readers} Lesern: {int((counts >= args.min_readers).sum())}\n")

    # The spectral split already works in log power, so a second logarithm
    # would compress what it just made linear. One pass, not two.
    transforms = TRANSFORMS if args.spectral == "none" else ("none",)
    for transform in transforms:
        if transform == "log1p" and raw.min() <= -1:
            print(f"--- Transformation: {transform} uebersprungen ---")
            print(f"  Minimum {raw.min():.3f} liegt bei oder unter -1, log1p ist dort "
                  f"nicht definiert.\n")
            continue

        base = np.log1p(raw) if transform == "log1p" else raw.copy()
        if not args.no_reader_centering:
            base = centre_per_reader(base, data.subject_ids)

        variants: list[tuple[str, np.ndarray]] = []
        if args.align == "none":
            variants.append(("ohne Ausrichtung", base))
        elif args.align == "diag":
            variants.append(("Ausrichtung diag", scale_per_reader(base, data.subject_ids)))
        else:
            for raw_value in args.shrinkage.split(","):
                token = raw_value.strip()
                weight = None if token == "lw" else float(token)
                aligned, chosen = whiten_per_reader(base, data.subject_ids, weight)
                name = "lw" if weight is None else f"{weight:g}"
                variants.append((f"weiss s={name} (gewaehlt {np.mean(chosen):.3f})", aligned))

        print(f"--- Transformation: {transform} ---")
        for align_label, vectors in variants:
            print(f"  [[ {align_label} ]]")
            if args.subjects == "disjoint":
                servable = np.array(
                    sorted(s for s, r in groups.items() if len(r) >= args.min_readers),
                    dtype=np.int64,
                )
                for setting in shared_settings:
                    # Same seed per setting, so the settings are compared on the
                    # same folds and the same reader divisions rather than on
                    # differently lucky draws.
                    rng = np.random.default_rng(args.seed)
                    pairs = [
                        subject_split_draws(
                            vectors, data.subject_ids, data.sentence_ids, data.sentences,
                            servable, anchors, rng, args.pool_size, args.tolerance,
                            args.pool_seed, args.folds, decoy, positions, setting, clock,
                        )
                        for _ in range(args.ridge_splits)
                    ]
                    tag = align_label if setting is None else f"{align_label} | {setting['label']}"
                    first_draws = [pair[0] for pair in pairs]
                    print(f"  Trainingszeilen je Fold: {first_draws[0]['n_train']:.0f}, "
                          f"Anfragen je Lauf: {first_draws[0]['n_queries']:.0f}")
                    _report(f"{tag} | probandendisjunkt", first_draws,
                            args.ridge_splits, effective_pool)
                    _report(f"{tag} | groessengleich, geteilt", [pair[1] for pair in pairs],
                            args.ridge_splits, effective_pool)
                continue
            if args.trials == "single":
                rng = np.random.default_rng(args.seed)
                servable = np.array(
                    sorted(s for s, r in groups.items() if len(r) >= args.min_readers),
                    dtype=np.int64,
                )
                draws = [
                    single_trial_split(
                        vectors, data.sentence_ids, data.sentences, servable, anchors, rng,
                        args.pool_size, args.tolerance, args.pool_seed, args.folds,
                        decoy, positions,
                    )
                    for _ in range(args.ridge_splits)
                ]
                print(f"  Anfragen je Lauf: {draws[0]['n_queries']:.0f} Einzeltrials")
                _report(f"{align_label} | Einzeltrial", draws, args.ridge_splits, effective_pool)
                continue
            if readout in ("cosine", "both"):
                rng = np.random.default_rng(args.seed)
                draws = [
                    cosine_split(vectors, groups, data.sentences, rng, args.pool_size,
                                 args.tolerance, args.pool_seed, args.min_readers, decoy,
                                 positions)
                    for _ in range(args.splits)
                ]
                _report(f"{align_label} | cosine", draws, args.splits, effective_pool)
            if readout in ("ridge", "both"):
                rng = np.random.default_rng(args.seed)
                draws = [
                    learned_split(vectors, groups, data.sentences, rng, args.pool_size,
                                  args.tolerance, args.pool_seed, args.min_readers, args.folds,
                                  decoy, positions, anchors)
                    for _ in range(args.ridge_splits)
                ]
                _report(f"{align_label} | ridge -> {args.target}", draws, args.ridge_splits, effective_pool)

    print("Lesart: dies ist ein Bezugspunkt, keine Schranke. Zwei Verzerrungen wirken")
    print("gegeneinander. Grosszuegig: der Schaetzer bekommt mehrere gemittelte Trials")
    print("je Anfrage, ein Modell auf Einzeltrials nur einen. Pessimistisch: Kosinus")
    print("gewichtet alle Dimensionen gleich, ein gelerntes Modell darf die informativen")
    print("hochziehen. Die Ridge-Zeile misst genau diesen zweiten Punkt.")


if __name__ == "__main__":
    main()
