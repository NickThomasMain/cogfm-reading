"""A space shared across readers, and the view that puts a fold into it.

Every alignment the project tried before this one works on a single reader's own
distribution: their mean, their scale, their covariance, the 1/f line through
their bands. Measured on ZuCo, all five moved the subject-disjoint number only by
damaging the reader-sharing control, which is what says the offset between
readers is not a distributional one. What none of them used is that twelve people
read the same seven hundred sentences, so a row of one reader and a row of
another are known to be about the same thing.

That correspondence is what this module estimates. The functions were developed
and verified in ``scripts/measure_noise_ceiling.py`` and live here so the
measurement script and the training pipeline run the same code rather than two
implementations that drift apart.

Measured on ZuCo band power, subject-disjoint, pool 25, ten reader divisions:
recall@1 rises from 0.039 (chance 0.040) to 0.086, and the percentile from 0.487
to 0.747. A reader the model has never seen then performs as well as one it
trained on. Two pre-registered nulls hold: a decoder given only the session clock
lands on chance, and rotating each reader's sentences along the reading order
before fitting drops the result back to the level of a plain projection.
"""

from __future__ import annotations

import numpy as np
import torch

# Ridge strengths the inner validation chooses from, kept identical to the
# measurement script so a number moved between them means the data changed.
ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


def ridge_map(source: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray | None:
    """Linear map from one half to the other, with ridge regularisation.

    The map is what the plain cosine lacks: it may scale the informative
    dimensions up and the noisy ones down instead of weighting all alike.

    Returns None where the system cannot be solved at this strength. With more
    dimensions than sentences the unregularised problem is rank-deficient, so
    the weakest strengths of the grid can fail. A caller that skips those keeps
    the grid wide instead of narrowing it in advance and never learning whether
    the weak end would have won.
    """
    gram = source.T @ source
    gram.flat[:: gram.shape[0] + 1] += alpha
    try:
        weights = np.linalg.solve(gram, source.T @ target)
    except np.linalg.LinAlgError:
        return None
    return weights if np.isfinite(weights).all() else None


def _view_basis(features: np.ndarray, keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre one view and return the transform onto its leading whitened axes.

    This is step one of MCCA as de Cheveigne et al. (2019) describe it: sphere
    each reader separately so none dominates the fusion by having larger units,
    and truncate, because the truncation is where the regularisation lives. With
    840 features and a few hundred sentences per reader the trailing axes are
    estimation noise, and MCCA is documented to latch onto noise that happens to
    be shared. Keeping fewer axes is the published defence.

    The transform is returned rather than the scores, because the same mapping
    has to be applied later to rows the fit never saw.
    """
    centre = features.mean(axis=0, keepdims=True)
    centred = features - centre
    _, values, directions = np.linalg.svd(centred, full_matrices=False)
    usable = int((values > values[0] * 1e-10).sum()) if values.size else 0
    keep = max(1, min(keep, usable))
    transform = directions[:keep].T * (np.sqrt(len(centred)) / values[:keep])
    return centre, transform


def _orthonormal(matrix: np.ndarray) -> np.ndarray:
    """Nearest matrix with orthonormal columns, which MAXVAR requires of it."""
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def shared_space(
    blocks: dict[str, tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
    slots: int,
    components: int,
    view_pcs: int,
    ridge: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]] | None:
    """A space shared across readers, estimated from which sentence is which.

    Every alignment tried so far works on one reader's own distribution: their
    mean, their scale, their covariance, the 1/f line through their bands. All
    five moved the subject-disjoint number only by damaging the reader-sharing
    control. What none of them used is the fact that twelve people read the same
    seven hundred sentences, so a row of one reader and a row of another are
    known to be about the same thing.

    That correspondence is what this estimates. Each view gets its own transform
    into a common space; the space is the thing the views agree on.

    ``blocks`` maps a view name to its features and, per row, the shared slot
    that row belongs to. Slots are sentences. A view may be missing most slots,
    which matters here: only 311 of 700 sentences were read by all twelve, and
    a complete-case fit would throw away more than half the corpus. Below, each
    view is fitted on the rows it has and each slot averages over the views that
    reached it, so all 7,809 trials are used.

    ``iterations`` controls what this is. At zero it is textbook MCCA and
    nothing else: whiten each view, concatenate, take the leading left singular
    vectors, then fit each view's map to that space. Every further pass is a
    MAXVAR alternating least squares step that re-weights the average by what
    each view can actually predict, which is the part that copes with the
    missing slots. Zero is therefore the honest baseline for whatever the
    iterations add.

    ``weights`` scales a view in the slot average. A stimulus view entered at a
    weight above zero makes this stimulus-informed GCCA in the sense of
    Geirnaert et al. (2024), whose stated motivation is that plain GCCA copes
    badly with few subjects. Twelve is few.

    The caveat de Cheveigne et al. put in their own paper applies in full: that
    the components this finds are correlated across views is worth nothing by
    itself, because the method will find agreement in shared noise. Only the
    held-out score decides, which is why nothing here is reported on its own.

    Returns the shared space and, per view, the centre and map that put that
    view into it. None where no view could be solved.
    """
    columns = []
    for name, (features, slot) in blocks.items():
        if len(features) < 2:
            continue
        centre, transform = _view_basis(features, view_pcs)
        block = np.zeros((slots, transform.shape[1]))
        block[slot] = ((features - centre) @ transform) * weights.get(name, 1.0)
        columns.append(block)
    if not columns:
        return None

    stacked = np.hstack(columns)
    left, values, _ = np.linalg.svd(stacked, full_matrices=False)
    width = min(components, int((values > values[0] * 1e-10).sum()))
    if width < 1:
        return None
    space = left[:, :width]

    # Each pass solves the same ridge system per view with only the right-hand
    # side changed, so the expensive half is done once. Inverting rather than
    # re-solving is acceptable only because the ridge is on the diagonal and
    # keeps the system well conditioned; without it this would be the wrong
    # trade. Measured on this corpus it turns the passes from the dominant cost
    # into a rounding error, which is what makes a grid of settings affordable
    # at all.
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (features, slot) in blocks.items():
        centre = features.mean(axis=0, keepdims=True)
        centred = features - centre
        gram = centred.T @ centred
        gram.flat[:: gram.shape[0] + 1] += ridge
        try:
            inverse = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(inverse).all():
            continue
        prepared[name] = (centre, centred, slot, inverse)
    if not prepared:
        return None

    maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for step in range(max(0, iterations) + 1):
        maps = {}
        total = np.zeros((slots, width))
        seen = np.zeros((slots, 1))
        for name, (centre, centred, slot, inverse) in prepared.items():
            mapping = inverse @ (centred.T @ space[slot])
            if not np.isfinite(mapping).all():
                continue
            maps[name] = (centre, mapping)
            share = weights.get(name, 1.0)
            np.add.at(total, slot, share * (centred @ mapping))
            np.add.at(seen, slot, share)
        if not maps:
            return None
        if step == max(0, iterations):
            break
        space = _orthonormal(total / np.clip(seen, 1e-12, None))
    return space, maps


def project_fold(
    signals: np.ndarray,
    row_sentence: np.ndarray,
    row_subject: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    anchors: dict[int, np.ndarray],
    mode: str,
    components: int,
    view_pcs: int,
    ridge: float,
    iterations: int,
    stimulus: float,
    null: str = "none",
    positions: dict[int, float] | None = None,
    null_seed: int = 0,
) -> np.ndarray | None:
    """Put one fold's rows into a space fitted without its held-out half.

    Two rules decide what the fit may see, and both are the point rather than
    bookkeeping. The space is estimated from the training readers only, so a
    held-out reader never contributes to the geometry everyone is measured in.
    And it is estimated on training sentences only, so no sentence that will be
    scored takes part in building the space it is scored in.

    The held-out reader then gets a map of their own, fitted from their
    recordings of the training sentences against the already fixed space. No
    text and no label is touched, only their unlabelled EEG. Those recordings
    are exactly the rows ``make_folds`` discards, because they overlap the
    training set on the sentence axis but not the reader axis. The protocol has
    been leaving them on the floor; here they are what makes a new reader
    readable at all.

    It has to be said plainly in the text what this buys and what it costs: a
    new reader no longer needs their own labelled data, but they do need to have
    read something the training readers also read. The claim narrows from "a new
    reader needs nothing" to "a new reader needs a short calibration".

    ``mode`` selects what is fitted. ``pca`` is the control that has to be run
    beside ``gcca`` and reported with it: one projection onto the leading axes
    of the pooled training rows, same number of dimensions, no use of which
    sentence is which. Without it a gain could just as well come from reducing
    840 dimensions to a few dozen, and the comparison would not be about shared
    structure at all.
    """
    if mode == "pca":
        centre, transform = _view_basis(signals[train], components)
        return (signals - centre) @ transform

    training_sentences = sorted(set(row_sentence[train].tolist()))
    slot_of = {sentence: index for index, sentence in enumerate(training_sentences)}
    on_training = np.array([int(s) in slot_of for s in row_sentence.tolist()])
    held_out = set(row_subject[test].tolist()) - set(row_subject[train].tolist())

    # The null that asks whether the correspondence is doing the work.
    #
    # Each reader's sentences are rotated by a different amount along the
    # reading order, so every reader still contributes the same recordings in
    # the same temporal shape, but no two of them are put side by side as the
    # same sentence any more. What survives is whatever a consistent per-reader
    # linear map can do on its own, which is roughly the pca row.
    #
    # Its limit has to be stated rather than discovered later: this does NOT
    # separate content from the clock. ZuCo shows every reader the same order,
    # so position and sentence identity are one variable, and no relabelling
    # inside this corpus pulls them apart. Rotating readers by different
    # amounts breaks the alignment of both at once. The clock is the job of the
    # position-matched decoy and of position_features above.
    shift_of: dict[str, int] = {}
    if null == "shift":
        if positions is None:
            raise SystemExit(
                "--shared-null shift braucht die Sitzungsposition. Lauf mit "
                "--pools decoy --control-position, sonst waere die Verschiebung "
                "eine Umsortierung nach Satz-ID und nicht nach Lesezeit."
            )
        reading_order = np.argsort([positions[s] for s in training_sentences])
        place = np.empty(len(reading_order), dtype=np.int64)
        place[reading_order] = np.arange(len(reading_order))
        span = len(reading_order)
        draw = np.random.default_rng(null_seed)

    def slots_for(reader: str, rows: np.ndarray) -> np.ndarray:
        slot = np.array([slot_of[int(s)] for s in row_sentence[rows].tolist()])
        if null != "shift":
            return slot
        if reader not in shift_of:
            # Small rotations would leave neighbouring sentences almost paired,
            # which would weaken the null rather than test against it.
            shift_of[reader] = int(draw.integers(span // 8, span - span // 8))
        return reading_order[(place[slot] + shift_of[reader]) % span]

    blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    weights: dict[str, float] = {}
    for reader in sorted(set(row_subject[train].tolist())):
        rows = np.flatnonzero((row_subject == reader) & on_training)
        rows = np.setdiff1d(rows, test, assume_unique=False)
        if len(rows) < 2:
            continue
        blocks[reader] = (signals[rows], slots_for(reader, rows))
        weights[reader] = 1.0

    if stimulus > 0.0:
        blocks["__stimulus__"] = (
            np.stack([anchors[s] for s in training_sentences]),
            np.arange(len(training_sentences)),
        )
        weights["__stimulus__"] = float(stimulus)

    fitted = shared_space(
        blocks, weights, len(training_sentences), components, view_pcs, ridge, iterations
    )
    if fitted is None:
        return None
    space, maps = fitted
    maps.pop("__stimulus__", None)

    # The held-out readers, against the space that was fixed without them.
    for reader in sorted(held_out):
        rows = np.flatnonzero((row_subject == reader) & on_training)
        rows = np.setdiff1d(rows, test, assume_unique=False)
        if len(rows) < 2:
            return None
        centre = signals[rows].mean(axis=0, keepdims=True)
        slot = slots_for(reader, rows)
        mapping = ridge_map(signals[rows] - centre, space[slot], ridge)
        if mapping is None:
            return None
        maps[reader] = (centre, mapping)

    projected = np.zeros((len(signals), space.shape[1]))
    for reader, (centre, mapping) in maps.items():
        rows = np.flatnonzero(row_subject == reader)
        projected[rows] = (signals[rows] - centre) @ mapping
    missing = sorted(set(row_subject.tolist()) - set(maps))
    if missing:
        raise SystemExit(f"ohne Abbildung geblieben: {missing}")
    return projected

class SharedSpaceView:
    """A dataset whose vectors sit in a space fitted without this fold's test half.

    The projection cannot be computed once for the corpus and stored, because the
    space may not see the readers or the sentences it will be scored on. It is
    therefore refitted per fold, and this view is what carries the result to the
    rest of the pipeline without touching the dataset everyone else holds.

    Everything but the vectors is delegated, so folds, pools, texts, subject ids
    and the sentence table stay exactly what they were. Only ``scanpath`` and
    ``embed_dim`` differ.
    """

    def __init__(self, base, vectors: np.ndarray) -> None:
        self._base = base
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.embed_dim = int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self._base)

    def sequence(self, idx: int) -> np.ndarray:
        """One trial as a length-one sequence, matching the mean-file contract."""
        return self.vectors[idx : idx + 1]

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self._base[idx])
        sample["scanpath"] = torch.from_numpy(self.sequence(idx).copy())
        return sample

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def project_dataset(
    dataset,
    fold,
    components: int = 25,
    view_pcs: int = 60,
    ridge: float = 1e2,
    iterations: int = 5,
) -> SharedSpaceView:
    """Fit the shared space on one fold's training half and return the projected view.

    Only the training readers build the space, and only on training sentences, so
    a held-out reader never contributes to the geometry it is measured in and no
    sentence takes part in building the space it is scored in. The held-out
    reader is then given a map of their own, fitted from their recordings of the
    training sentences against the already fixed space: unlabelled EEG only, no
    text. Those recordings are exactly the rows ``make_folds`` discards because
    they overlap the training set on one axis and not the other.

    What that buys and what it costs both belong in the write-up: a new reader
    needs no labelled data, but does need to have read something the training
    readers also read. The claim is a short calibration, not nothing.

    Raises:
        ValueError: on a patch-sequence file, where a trial owns several rows.
            Word-level band power is a planned step and needs the maps applied
            per row rather than per trial; refusing here is better than silently
            averaging the sequence away.
    """
    if getattr(dataset, "is_sequence", False):
        raise ValueError(
            "der gemeinsame Raum ist bisher nur fuer Mittelwert-Dateien gebaut; "
            "eine Sequenzdatei braucht die Abbildung je Zeile statt je Trial"
        )

    features = np.stack([np.asarray(dataset.sequence(i)[0], dtype=float)
                         for i in range(len(dataset))])
    projected = project_fold(
        features,
        np.asarray(dataset.sentence_ids),
        np.asarray(dataset.subject_ids),
        np.asarray(fold.train),
        np.asarray(fold.test),
        {},
        "gcca",
        components,
        view_pcs,
        ridge,
        iterations,
        0.0,
    )
    if projected is None:
        raise ValueError(
            "der gemeinsame Raum liess sich auf diesem Fold nicht schaetzen; "
            "meist zu wenige Trainingssaetze fuer einen gehaltenen Leser"
        )
    return SharedSpaceView(dataset, projected)
