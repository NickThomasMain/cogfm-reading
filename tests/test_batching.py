"""Tests for batch_reading_samples: shapes, types, value preservation."""

import torch

from cogfm.data.batching import batch_reading_samples
from cogfm.data.dummy import DummyReadingDataset


def test_batch_shapes_and_types():
    ds = DummyReadingDataset(8, seed=0, n_fixations=10)
    samples = [ds[i] for i in range(4)]
    batch = batch_reading_samples(samples)
    assert batch["scanpath"].shape == (4, 10, 3)
    assert len(batch["text"]) == 4
    assert isinstance(batch["text"][0], str)
    assert len(batch["subject_id"]) == 4


def test_batch_preserves_values():
    ds = DummyReadingDataset(4, seed=1, n_fixations=6)
    samples = [ds[i] for i in range(3)]
    batch = batch_reading_samples(samples)
    assert torch.equal(batch["scanpath"][0], samples[0]["scanpath"])
    assert batch["text"][2] == samples[2]["text"]
