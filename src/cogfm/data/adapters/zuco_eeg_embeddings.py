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

Samples carry the vector under the key ``scanpath``, which is what the batching
and evaluation code reads. The name is the eye-tracking one; keeping it means
both modalities pass through the identical chain -- same folds, same pools, same
permutation null -- so a difference between the two rows cannot come from the
evaluation. The vector is served as a length-one sequence, (1, D), which the
padding machinery leaves untouched.
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
            vectors = np.asarray(store["vectors"], dtype=np.float32)
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

        self._rows = np.flatnonzero(keep)
        self.subject_ids = subject[self._rows]
        self.task_ids = task_column[self._rows]
        self.sentence_ids = sentence_id[self._rows]
        self.n_patches = n_patches[self._rows]
        self.vectors = vectors[self._rows]

        self.task = task
        self.center = center
        self.embedding_file = embedding_file
        if center == "subject":
            self.vectors = _center_by_group(self.vectors, self.subject_ids)

        self.embed_dim = int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self._rows)

    def vector(self, idx: int) -> np.ndarray:
        """One trial's stored embedding, float32, shape (embed_dim,)."""
        return self.vectors[idx]

    def describe(self) -> str:
        """One line naming the condition these vectors were produced under."""
        settings = ", ".join(f"{key} {value}" for key, value in self.metadata.items())
        return (
            f"{len(self)} trials, {len(set(self.sentence_ids.tolist()))} sentences, "
            f"{self.embed_dim} dims | {settings} | centering {self.center}"
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
            # (1, D): one time step, so the padding machinery has nothing to do.
            "scanpath": torch.from_numpy(self.vectors[idx]).unsqueeze(0),
        }


def _center_by_group(vectors: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's own mean vector."""
    centred = vectors.copy()
    for name in set(groups.tolist()):
        rows = np.flatnonzero(groups == name)
        centred[rows] -= centred[rows].mean(axis=0, keepdims=True)
    return centred
