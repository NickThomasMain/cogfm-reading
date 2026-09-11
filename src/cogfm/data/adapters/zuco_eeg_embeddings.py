"""ZuCo 1.0 EEG as the vectors LaBraM has already produced.

Reads what ``scripts/embed_zuco_eeg.py`` wrote. Serving stored vectors rather
than raw EEG is not a shortcut but the only correct way to run this encoder in
a training loop:

* LaBraM is frozen, so a trial's vector never changes. Recomputing it every
  step would run the expensive part hundreds of thousands of times for the same
  answer.
* LaBraM has no time mask. A padded trial resembles its own unpadded self less
  than two different trials resemble each other (0.864 against 0.947, measured).
  Trials must therefore never share a batch with trials of another length --
  which is exactly what a shuffled training loader would do. The embedding
  script sidesteps this by grouping trials of identical length; a loader cannot.

Everything that decides what the connector sees is fixed in the stored file:
reference, channel selection, pooling, and how over-long trials were handled.
Those settings travel with the vectors as metadata, so a run can name the
condition it measured instead of assuming it.

Two shapes of file are read. One holds a vector per trial, the mean over the
whole trial. The other holds one vector per one-second patch, written by
``--pooling time``; a trial is then a sequence and the pooling is left to the
trainable part. Averaging a five-second trial into one vector cannot preserve
what sits in one of its seconds, so which of the two is used is a substantive
choice, not a storage detail. The file says which it is: sequences come with an
``offsets`` array, means do not.

Samples carry the signal under the key ``scanpath``, which is what the batching
and evaluation code reads. The name is the eye-tracking one; keeping it means
both modalities pass through the identical chain -- same folds, same pools, same
permutation null -- so a difference between the two rows cannot come from the
evaluation. A mean is served as a length-one sequence, (1, D), which the padding
machinery leaves untouched; a patch sequence is served as (P, D) and is padded
per batch like any other sequence. That padding is harmless where LaBraM's was
not: the connector receives the mask, the frozen model had no way to take one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path("data/zuco/processed")
DEFAULT_FILE = "eeg_embeddings_all_average_egi62_mean.npz"

DATASET_ID = "zuco1-eeg-emb"

CENTERINGS = ("none", "subject")

# Metadata the embedding script stores as one-element arrays.
METADATA_KEYS = ("reference", "channels", "pooling", "too_long", "checkpoint")


class ZuCoEEGEmbeddingDataset(Dataset):
    """Trials of ZuCo 1.0 as (text, LaBraM vector) pairs.

    Args:
        root: directory holding ``sentences.json`` and the embedding file.
        embeddings: embedding file, absolute or relative to ``root``.
        task: keep only this reading task (``"SR"`` or ``"NR"``); all if None.
        center: ``"none"`` leaves the vectors as stored. ``"subject"`` subtracts
            each reader's own mean vector. Reader identity is the single
            strongest component in this space (+0.452 against +0.077 for the
            sentence), and folds are subject-disjoint, so a reader direction
            learned in training cannot transfer to a test reader anyway.
            Subtracting it is computed per reader over that reader's own trials
            without using any label, but it does look at the test reader's other
            trials, which makes it transductive and has to be reported as such.
            A *global* mean needs no switch: a constant offset on every input is
            absorbed by the first linear layer's bias.

    Attributes:
        sentences: one record per distinct sentence, with id, text, task, n_words.
        subject_ids: subject of each trial, aligned with the dataset index.
        sentence_ids: sentence id of each trial, aligned with the dataset index.
        n_patches: trial length in whole seconds, aligned with the index. Not an
            input; kept because duration is the confounder this modality has to
            be held against.
        embed_dim: width of the stored vectors.
        metadata: the settings the vectors were produced under.
    """

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        embeddings: Path | str = DEFAULT_FILE,
        task: str | None = None,
        center: str = "none",
    ) -> None:
        if center not in CENTERINGS:
            raise ValueError(f"center must be one of {CENTERINGS}, got {center!r}")

        root = Path(root)
        embedding_file = Path(embeddings)
        if not embedding_file.is_absolute() and not embedding_file.is_file():
            embedding_file = root / embedding_file
        sentence_file = root / "sentences.json"

        if not embedding_file.is_file():
            raise FileNotFoundError(
                f"{embedding_file} not found. Run scripts/embed_zuco_eeg.py first."
            )
        if not sentence_file.is_file():
            raise FileNotFoundError(
                f"{sentence_file} not found. Run scripts/extract_zuco_et.py --extract --merge."
            )

        self.sentences: list[dict] = json.loads(sentence_file.read_text(encoding="utf-8"))
        self._text_by_id = {s["id"]: s["text"] for s in self.sentences}

        with np.load(embedding_file, allow_pickle=False) as store:
            self.is_sequence = "offsets" in store.files
            if self.is_sequence:
                vectors = np.asarray(store["tokens"], dtype=np.float32)
                offsets = np.asarray(store["offsets"], dtype=np.int64)
            else:
                vectors = np.asarray(store["vectors"], dtype=np.float32)
                offsets = np.arange(len(vectors) + 1, dtype=np.int64)
            subject = store["subject"]
            task_column = store["task"]
            sentence_id = store["sentence_id"]
            n_patches = store["n_patches"]
            self.metadata = {
                key: str(store[key][0]) for key in METADATA_KEYS if key in store.files
            }
            self.channel_names = (
                [str(name) for name in store["channel_names"]]
                if "channel_names" in store.files
                else []
            )
            self.sample_rate = int(store["sample_rate"][0]) if "sample_rate" in store.files else 0

        unknown = sorted({int(s) for s in sentence_id} - set(self._text_by_id))
        if unknown:
            raise ValueError(
                f"{len(unknown)} sentence ids in {embedding_file.name} have no text; "
                "the embedding file and sentences.json come from different extractions"
            )

        keep = (
            np.ones(len(sentence_id), dtype=bool) if task is None else (task_column == task)
        )
        if not keep.any():
            raise ValueError(f"no trials for task {task!r}; available: {sorted(set(task_column))}")

        if len(offsets) != len(sentence_id) + 1:
            raise ValueError(
                f"{len(offsets) - 1} offsets for {len(sentence_id)} trials in "
                f"{embedding_file.name}"
            )

        self._rows = np.flatnonzero(keep)
        self.subject_ids = subject[self._rows]
        self.task_ids = task_column[self._rows]
        self.sentence_ids = sentence_id[self._rows]
        self.n_patches = n_patches[self._rows]

        # Trials keep their stored rows rather than being copied out, so a
        # task filter costs an index list and not a second copy of the tokens.
        self._spans = np.stack([offsets[self._rows], offsets[self._rows + 1]], axis=1)
        self.vectors = vectors

        self.task = task
        self.center = center
        self.embedding_file = embedding_file
        if center == "subject":
            self.vectors = _center_by_rows(self.vectors, self._spans, self.subject_ids)

        self.embed_dim = int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self._rows)

    def sequence(self, idx: int) -> np.ndarray:
        """One trial's stored embedding, float32, shape (steps, embed_dim).

        ``steps`` is one for a mean file and the trial's whole seconds for a
        patch-sequence file.
        """
        start, end = self._spans[idx]
        return self.vectors[start:end]

    def vector(self, idx: int) -> np.ndarray:
        """One trial's stored embedding as a single vector.

        Raises:
            ValueError: on a patch-sequence file, where a trial is not a point.
        """
        if self.is_sequence:
            raise ValueError("this file holds patch sequences; use sequence(idx)")
        return self.sequence(idx)[0]

    def describe(self) -> str:
        """One line naming the condition these vectors were produced under."""
        settings = ", ".join(f"{key} {value}" for key, value in self.metadata.items())
        steps = self._spans[:, 1] - self._spans[:, 0]
        shape = (
            f"sequences of {steps.min()} to {steps.max()} steps"
            if self.is_sequence
            else "one vector per trial"
        )
        return (
            f"{len(self)} trials, {len(set(self.sentence_ids.tolist()))} sentences, "
            f"{shape}, {self.embed_dim} dims | {settings} | centering {self.center}"
        )

    def __getitem__(self, idx: int) -> dict:
        sentence_id = int(self.sentence_ids[idx])
        return {
            "sample_id": int(self._rows[idx]),
            "subject_id": str(self.subject_ids[idx]),
            "dataset_id": DATASET_ID,
            "task": str(self.task_ids[idx]),
            "sentence_id": sentence_id,
            "text": self._text_by_id[sentence_id],
            # (steps, D). One step for a mean file, so the padding machinery
            # has nothing to do; the trial's seconds for a sequence file.
            "scanpath": torch.from_numpy(np.ascontiguousarray(self.sequence(idx))),
        }


def _center_by_rows(vectors: np.ndarray, spans: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's own mean vector, taken over all its stored rows.

    A trial owns a span of rows rather than a single row, so the mean is over
    every row the group's trials own -- one vector each for a mean file, one per
    second for a sequence file.
    """
    centred = vectors.copy()
    for name in set(groups.tolist()):
        trials = np.flatnonzero(groups == name)
        rows = np.concatenate([np.arange(*spans[i]) for i in trials])
        centred[rows] -= centred[rows].mean(axis=0, keepdims=True)
    return centred
