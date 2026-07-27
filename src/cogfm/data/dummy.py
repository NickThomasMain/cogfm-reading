"""Deterministic synthetic reading data for end-to-end testing.

Produces samples shaped the way the canonical schema is expected to look — text
plus an eye-tracking scanpath plus metadata — so the pipeline can run end-to-end
before real dataset adapters exist.

The scanpath shape here — a sequence of fixations, each carrying (x, y,
duration) — is a deliberate simplifying assumption consistent with a
scanpath-only eye-tracking encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

# A tiny fake vocabulary — meaningless words, only to have text-like tokens.
_FAKE_VOCAB = [
    "lorem",
    "ipsum",
    "dolor",
    "sit",
    "amet",
    "consectetur",
    "adipiscing",
    "elit",
    "sed",
    "eiusmod",
    "tempor",
    "incididunt",
    "labore",
    "magna",
    "aliqua",
    "veniam",
    "nostrud",
    "ullamco",
    "laboris",
    "aliquip",
]

# Each fixation carries these three features, in this order.
SCANPATH_FEATURES = ("x", "y", "duration")


@dataclass
class DummySample:
    """One synthetic reading trial (mirrors the future canonical sample)."""

    sample_id: int
    subject_id: str
    dataset_id: str
    text: list[str]  # sequence of (fake) words
    scanpath: np.ndarray  # shape (n_fixations, 3): x, y, duration; float32


def make_dummy_samples(
    n_samples: int = 64,
    *,
    seed: int = 42,
    n_subjects: int = 4,
    n_fixations: int = 10,
    min_words: int = 5,
    max_words: int = 12,
    dataset_id: str = "dummy",
) -> list[DummySample]:
    """Create a deterministic list of synthetic reading samples.

    Same seed -> identical samples. Each sample gets a text of random length and
    a scanpath with a FIXED number of fixations (equal length across samples, so
    batching needs no padding). Each fixation carries (x, y, duration).
    """
    rng = np.random.default_rng(seed)
    samples: list[DummySample] = []
    for i in range(n_samples):
        n_words = int(rng.integers(min_words, max_words + 1))
        word_idx = rng.integers(0, len(_FAKE_VOCAB), size=n_words)
        text = [_FAKE_VOCAB[j] for j in word_idx]

        # fixed number of fixations per sample -> equal length, no padding needed
        x = rng.uniform(0.0, 1.0, size=n_fixations)  # normalised screen x
        y = rng.uniform(0.0, 1.0, size=n_fixations)  # normalised screen y
        duration = rng.uniform(0.1, 0.4, size=n_fixations)  # seconds
        scanpath = np.stack([x, y, duration], axis=1).astype(np.float32)

        subject_id = f"S{int(rng.integers(0, n_subjects)):02d}"
        samples.append(
            DummySample(
                sample_id=i,
                subject_id=subject_id,
                dataset_id=dataset_id,
                text=text,
                scanpath=scanpath,
            )
        )
    return samples


class DummyReadingDataset(Dataset):
    """Torch Dataset over synthetic samples, for the smoke-run DataLoader.

    __getitem__ returns a dict; the scanpath is a float32 tensor of shape
    (n_fixations, 3). All samples share the same number of fixations, so batching
    stacks them without padding.
    """

    def __init__(self, n_samples: int = 64, *, seed: int = 42, **kwargs) -> None:
        self.samples = make_dummy_samples(n_samples, seed=seed, **kwargs)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "sample_id": s.sample_id,
            "subject_id": s.subject_id,
            "dataset_id": s.dataset_id,
            "text": " ".join(s.text),
            "scanpath": torch.from_numpy(s.scanpath),  # (n_fix, 3)
        }
