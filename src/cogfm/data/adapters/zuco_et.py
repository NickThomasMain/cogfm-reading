"""ZuCo 1.0 reading data, loaded from the extracted form.

Reads the two files written by ``scripts/extract_zuco_et.py``: the sentence table
and the flattened scanpath store. The MATLAB sources are not touched here, so
construction costs milliseconds rather than minutes.

Samples carry the same field names as the synthetic dataset, so anything built
against that one works here without change. Scanpath length varies between
trials, which the synthetic data does not exercise; batching therefore needs
padding and a mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path("data/zuco/processed")

DATASET_ID = "zuco1-et"


class ZuCoETDataset(Dataset):
    """Trials of ZuCo 1.0 as (text, scanpath) pairs.

    Args:
        root: directory holding ``sentences.json`` and ``scanpaths.npz``.
        task: keep only this reading task (``"SR"`` or ``"NR"``); all tasks if None.

    Attributes:
        sentences: one record per distinct sentence, with id, text, task, n_words.
        subject_ids: subject of each trial, aligned with the dataset index.
        sentence_ids: sentence id of each trial, aligned with the dataset index.
    """

    def __init__(self, root: Path | str = DEFAULT_ROOT, task: str | None = None) -> None:
        root = Path(root)
        sentence_file = root / "sentences.json"
        scanpath_file = root / "scanpaths.npz"
        for path in (sentence_file, scanpath_file):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. Run scripts/extract_zuco_et.py --extract --merge first."
                )

        self.sentences: list[dict] = json.loads(sentence_file.read_text(encoding="utf-8"))
        self._text_by_id = {s["id"]: s["text"] for s in self.sentences}

        with np.load(scanpath_file, allow_pickle=False) as store:
            subject = store["subject"]
            task_column = store["task"]
            sentence_id = store["sentence_id"]
            offsets = store["offsets"]
            self._fixations = store["fixations"]
            self.features = tuple(str(f) for f in store["features"])

        keep = (
            np.ones(len(sentence_id), dtype=bool)
            if task is None
            else (task_column == task)
        )
        if not keep.any():
            raise ValueError(f"no trials for task {task!r}; available: {sorted(set(task_column))}")

        self._rows = np.flatnonzero(keep)
        self.subject_ids = subject[self._rows]
        self.task_ids = task_column[self._rows]
        self.sentence_ids = sentence_id[self._rows]
        self._starts = offsets[:-1][self._rows]
        self._ends = offsets[1:][self._rows]
        self.task = task

    def __len__(self) -> int:
        return len(self._rows)

    def scanpath(self, idx: int) -> np.ndarray:
        """Fixation sequence of one trial, shape (n_fixations, 3)."""
        return self._fixations[self._starts[idx] : self._ends[idx]]

    def lengths(self) -> np.ndarray:
        """Number of fixations per trial, aligned with the dataset index."""
        return (self._ends - self._starts).astype(np.int64)

    def __getitem__(self, idx: int) -> dict:
        sentence_id = int(self.sentence_ids[idx])
        return {
            "sample_id": int(self._rows[idx]),
            "subject_id": str(self.subject_ids[idx]),
            "dataset_id": DATASET_ID,
            "task": str(self.task_ids[idx]),
            "sentence_id": sentence_id,
            "text": self._text_by_id[sentence_id],
            "scanpath": torch.from_numpy(np.ascontiguousarray(self.scanpath(idx))),
        }
