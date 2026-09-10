"""ZuCo 1.0 sentence-level EEG, loaded from the extracted form.

Reads what ``scripts/extract_zuco_eeg.py`` wrote: one memory-mapped ``.npy`` per
subject and task, plus the shared index ``eeg_index.npz``. Sentence ids come from
the same ``sentences.json`` the eye-tracking adapter uses, so trials of the two
modalities are directly comparable.

Two choices are exposed as switches because the literature settles neither and
both change what an encoder sees.

``reference``
    ``"average"`` (default) subtracts, per time point, the mean over all 105
    stored columns -- *including* the constant Cz column, which is what turns Cz
    back into a usable channel. This is what the ZuCo authors themselves do to
    ``rawData`` (zuco-benchmark, ``matlab/topoplot_nr_tsr_rawEEG.m``, l. 19f.)
    and it matches TUAB, LaBraM's main fine-tuning corpus.
    ``"recording"`` leaves the data as recorded, referenced against Cz. The Cz
    column is then constant zero and is dropped from every selection.

``channels``
    ``"egi62"`` (default) keeps the 10-10 positions whose HydroCel equivalent is
    within EGI's own 1.85 cm criterion; ``"named69"`` keeps every position that
    has a 10-10 name at all, including seven between 2.0 and 2.5 cm; ``"all"``
    keeps every non-constant column, most of which have no 10-10 name and are
    therefore unusable with LaBraM but fine for baselines.

Note that dropping channels loses no information here: ZuCo's ICA stage leaves
each trial with 9 to 42 linear degrees of freedom, so any well-conditioned subset
above that rank spans the same space (measured: R^2 = 1.0000). The selection
decides which *location* an encoder assigns to the signal, not how much signal
there is.

Channel names are drawn from ``zuco_montage.json``, built from Luu & Ferree's EGI
technical note. ``legacy_names`` switches T7/T8/P7/P8 to the older T3/T4/T5/T6,
which is what the TUH corpora use; LaBraM's vocabulary knows both spellings but
holds a separate learned embedding for each.
"""

from __future__ import annotations

import json
from math import gcd
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path("data/zuco/processed")
MONTAGE_FILE = Path(__file__).resolve().parents[1] / "zuco_montage.json"

DATASET_ID = "zuco1-eeg"

REFERENCES = ("average", "recording")
SELECTIONS = ("egi62", "named69", "all")

CZ_COLUMN = 104


class ZuCoEEGDataset(Dataset):
    """Trials of ZuCo 1.0 as (text, EEG) pairs.

    Args:
        root: directory holding ``eeg_index.npz``, ``eeg/`` and ``sentences.json``.
        task: keep only this reading task (``"SR"`` or ``"NR"``); all tasks if None.
        reference: ``"average"`` or ``"recording"``; see module docstring.
        channels: ``"egi62"``, ``"named69"`` or ``"all"``.
        legacy_names: report T3/T4/T5/T6 instead of T7/T8/P7/P8.
        resample_hz: resample to this rate; None keeps the stored 500 Hz.
            LaBraM expects 200 Hz.

    Attributes:
        channel_names: 10-10 name per selected channel, aligned with the feature
            axis; None for columns without a name (only possible with "all").
        sample_rate: rate of the returned signal, after resampling.
    """

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        task: str | None = None,
        reference: str = "average",
        channels: str = "egi62",
        legacy_names: bool = False,
        resample_hz: int | None = None,
    ) -> None:
        if reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
        if channels not in SELECTIONS:
            raise ValueError(f"channels must be one of {SELECTIONS}, got {channels!r}")

        root = Path(root)
        index_file = root / "eeg_index.npz"
        sentence_file = root / "sentences.json"
        for path in (index_file, sentence_file):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} not found. Run scripts/extract_zuco_eeg.py --extract --merge first."
                )

        self.sentences: list[dict] = json.loads(sentence_file.read_text(encoding="utf-8"))
        self._text_by_id = {s["id"]: s["text"] for s in self.sentences}

        with np.load(index_file, allow_pickle=False) as store:
            subject = store["subject"]
            task_column = store["task"]
            sentence_id = store["sentence_id"]
            start = store["start"]
            end = store["end"]
            n_columns = int(store["channels"][0])
            stored_rate = int(store["sample_rate"][0])

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
        self._starts = start[self._rows]
        self._ends = end[self._rows]
        self.task = task
        self.reference = reference
        self.channels = channels
        self.stored_rate = stored_rate
        self._root = root
        self._open_files: dict[tuple[str, str], np.ndarray] = {}

        self._columns, self.channel_names = self._select(channels, reference, legacy_names, n_columns)

        self.resample_hz = resample_hz
        self.sample_rate = resample_hz or stored_rate
        self._ratio: tuple[int, int] | None = None
        if resample_hz is not None and resample_hz != stored_rate:
            try:
                from scipy.signal import resample_poly  # noqa: F401
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError("resample_hz needs scipy; run `uv add scipy`") from exc
            divisor = gcd(stored_rate, resample_hz)
            self._ratio = (resample_hz // divisor, stored_rate // divisor)

    @staticmethod
    def _select(
        selection: str, reference: str, legacy_names: bool, n_columns: int
    ) -> tuple[np.ndarray, list[str | None]]:
        """Column indices and their 10-10 names, ordered by column."""
        montage = json.loads(MONTAGE_FILE.read_text(encoding="utf-8"))
        by_column = {entry["column"]: entry for entry in montage["channels"]}

        if selection == "all":
            wanted = list(range(n_columns))
        else:
            wanted = [
                column
                for column, entry in sorted(by_column.items())
                if selection == "named69" or entry["within_egi_threshold"]
            ]

        # Under the recording reference the Cz column is a constant, so it carries
        # nothing. Only re-referencing turns it into a channel.
        if reference == "recording":
            wanted = [column for column in wanted if column != CZ_COLUMN]

        names: list[str | None] = []
        for column in wanted:
            entry = by_column.get(column)
            if entry is None:
                names.append(None)
            elif legacy_names and entry["legacy"]:
                names.append(entry["legacy"])
            else:
                names.append(entry["name"])
        return np.asarray(wanted, dtype=np.int64), names

    def _signals(self, subject: str, task: str) -> np.ndarray:
        key = (subject, task)
        if key not in self._open_files:
            path = self._root / "eeg" / f"{subject}_{task}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing; extraction incomplete")
            self._open_files[key] = np.load(path, mmap_mode="r")
        return self._open_files[key]

    def __len__(self) -> int:
        return len(self._rows)

    def eeg(self, idx: int) -> np.ndarray:
        """One trial as (time, channels), float32, in microvolts."""
        signals = self._signals(str(self.subject_ids[idx]), str(self.task_ids[idx]))
        block = np.asarray(signals[self._starts[idx] : self._ends[idx]], dtype=np.float32)

        if self.reference == "average":
            # Mean over every stored column, Cz included: that is what recovers Cz
            # and what the ZuCo authors' own code does.
            block = block - block.mean(axis=1, keepdims=True)

        block = block[:, self._columns]

        if self._ratio is not None:
            from scipy.signal import resample_poly

            up, down = self._ratio
            block = resample_poly(block, up, down, axis=0).astype(np.float32)
        return np.ascontiguousarray(block)

    def lengths(self) -> np.ndarray:
        """Samples per trial at the returned rate, aligned with the dataset index."""
        raw = (self._ends - self._starts).astype(np.int64)
        if self._ratio is None:
            return raw
        up, down = self._ratio
        return -(-raw * up // down)

    def __getitem__(self, idx: int) -> dict:
        sentence_id = int(self.sentence_ids[idx])
        return {
            "sample_id": int(self._rows[idx]),
            "subject_id": str(self.subject_ids[idx]),
            "dataset_id": DATASET_ID,
            "task": str(self.task_ids[idx]),
            "sentence_id": sentence_id,
            "text": self._text_by_id[sentence_id],
            "eeg": torch.from_numpy(self.eeg(idx)),
        }
