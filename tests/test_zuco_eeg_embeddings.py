"""Tests for the stored-embedding adapter and the passthrough encoder.

The encoder tests need no data and always run. The adapter tests are skipped
unless the embeddings have been computed, following test_zuco_eeg_adapter.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cogfm.data.adapters.zuco_eeg_embeddings import (
    DEFAULT_FILE,
    ZuCoEEGEmbeddingDataset,
)
from cogfm.data.batching import batch_reading_samples
from cogfm.encoders.precomputed import PrecomputedEncoder
from cogfm.registry import ENCODERS

ZUCO_ROOT = Path("data/zuco/processed")
EMBEDDINGS = ZUCO_ROOT / DEFAULT_FILE
needs_embeddings = pytest.mark.skipif(
    not EMBEDDINGS.is_file(),
    reason="EEG embeddings have not been computed; run scripts/embed_zuco_eeg.py",
)


# --- passthrough encoder ----------------------------------------------------

def test_encoder_is_registered_under_its_name():
    encoder = ENCODERS.build("precomputed", embed_dim=8)
    assert isinstance(encoder, PrecomputedEncoder)


def test_encoder_returns_the_stored_vector_unchanged():
    vectors = torch.randn(4, 8)
    encoder = PrecomputedEncoder(embed_dim=8)
    out = encoder(vectors.unsqueeze(1))
    assert out.shape == (4, 8)
    assert torch.equal(out, vectors)


def test_encoder_holds_no_parameters():
    assert list(PrecomputedEncoder(embed_dim=8).parameters()) == []


def test_encoder_ignores_the_mask():
    vectors = torch.randn(3, 8).unsqueeze(1)
    encoder = PrecomputedEncoder(embed_dim=8)
    ones = encoder(vectors, torch.ones(3, 1, dtype=torch.long))
    zeros = encoder(vectors, torch.zeros(3, 1, dtype=torch.long))
    assert torch.equal(ones, zeros), "there is nothing to mask in a stored vector"


def test_encoder_rejects_a_width_that_disagrees_with_the_config():
    with pytest.raises(ValueError, match="does not match"):
        PrecomputedEncoder(embed_dim=8)(torch.randn(2, 1, 16))


def test_encoder_rejects_a_real_sequence():
    with pytest.raises(ValueError, match="one step long"):
        PrecomputedEncoder(embed_dim=8)(torch.randn(2, 5, 8))


# --- adapter ----------------------------------------------------------------

@needs_embeddings
def test_dataset_reports_the_settings_its_vectors_were_made_under():
    data = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    for key in ("reference", "channels", "pooling", "too_long"):
        assert data.metadata[key], f"{key} missing; the file predates the metadata"
    assert data.embed_dim == data.vectors.shape[1]


@needs_embeddings
def test_sample_carries_the_vector_as_a_one_step_sequence():
    data = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    sample = data[0]
    assert sample["scanpath"].shape == (1, data.embed_dim)
    assert torch.equal(sample["scanpath"][0], torch.from_numpy(data.vector(0)))


@needs_embeddings
def test_sample_text_belongs_to_the_sample_sentence():
    data = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    by_id = {s["id"]: s["text"] for s in data.sentences}
    for idx in (0, len(data) // 2, len(data) - 1):
        sample = data[idx]
        assert sample["text"] == by_id[sample["sentence_id"]]
        assert sample["sentence_id"] == int(data.sentence_ids[idx])


@needs_embeddings
def test_batching_leaves_a_stored_vector_untouched():
    data = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    batch = batch_reading_samples([data[i] for i in range(4)])
    assert batch["scanpath"].shape == (4, 1, data.embed_dim)
    assert batch["mask"].sum() == 4, "no padding, so every position is real"


@needs_embeddings
def test_task_filter_keeps_only_that_task():
    both = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    single = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT, task="NR")
    assert set(single.task_ids.tolist()) == {"NR"}
    assert 0 < len(single) < len(both)


@needs_embeddings
def test_subject_centering_removes_each_reader_s_mean():
    plain = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    centred = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT, center="subject")
    assert len(plain) == len(centred)
    for subject in set(centred.subject_ids.tolist()):
        rows = np.flatnonzero(centred.subject_ids == subject)
        mean = centred.vectors[rows].mean(axis=0)
        assert np.abs(mean).max() < 1e-4
    assert np.abs(plain.vectors).max() > 0


@needs_embeddings
def test_centering_does_not_reorder_trials():
    plain = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT)
    centred = ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT, center="subject")
    assert np.array_equal(plain.sentence_ids, centred.sentence_ids)
    assert np.array_equal(plain.subject_ids, centred.subject_ids)


@needs_embeddings
def test_unknown_centering_is_rejected():
    with pytest.raises(ValueError, match="center must be one of"):
        ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT, center="reader")


@needs_embeddings
def test_missing_embedding_file_names_the_script_that_writes_it():
    with pytest.raises(FileNotFoundError, match="embed_zuco_eeg.py"):
        ZuCoEEGEmbeddingDataset(root=ZUCO_ROOT, embeddings="does_not_exist.npz")
