"""Tests for batch_reading_samples: shapes, padding, mask, value preservation."""

import pytest
import torch

from cogfm.data.batching import batch_reading_samples
from cogfm.data.dummy import DummyReadingDataset
from cogfm.encoders.scanpath import ScanpathEncoder


def make_sample(n_fixations: int, value: float = 1.0) -> dict:
    """A sample with a scanpath of the requested length and constant entries."""
    return {
        "sample_id": n_fixations,
        "subject_id": "S00",
        "text": "some words",
        "scanpath": torch.full((n_fixations, 3), value),
    }


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


def test_equal_lengths_give_an_all_ones_mask():
    ds = DummyReadingDataset(4, seed=2, n_fixations=7)
    batch = batch_reading_samples([ds[i] for i in range(3)])
    assert batch["mask"].shape == (3, 7)
    assert int(batch["mask"].sum()) == 3 * 7


def test_padding_reaches_the_longest_sample_only():
    batch = batch_reading_samples([make_sample(3), make_sample(11), make_sample(5)])
    assert batch["scanpath"].shape == (3, 11, 3)
    assert batch["lengths"].tolist() == [3, 11, 5]


def test_mask_marks_real_fixations_and_padding_is_zero():
    batch = batch_reading_samples([make_sample(3), make_sample(11)])
    assert batch["mask"][0].tolist() == [1, 1, 1] + [0] * 8
    assert int(batch["mask"][1].sum()) == 11
    assert torch.all(batch["scanpath"][0, 3:] == 0)


def test_real_positions_survive_padding():
    original = make_sample(4, value=2.5)
    batch = batch_reading_samples([original, make_sample(9)])
    assert torch.equal(batch["scanpath"][0, :4], original["scanpath"])


def test_passthrough_keys_are_optional():
    minimal = [{"scanpath": torch.zeros(2, 3)}, {"scanpath": torch.zeros(6, 3)}]
    batch = batch_reading_samples(minimal)
    assert batch["scanpath"].shape == (2, 6, 3)
    assert "text" not in batch


def test_encoding_does_not_depend_on_batch_neighbours():
    """The point of the mask: padding must not leak into a representation.

    The same trial is encoded twice, once among short neighbours and once among
    long ones, so the amount of padding differs. Pooling over masked positions
    only, both runs have to agree.
    """
    torch.manual_seed(0)
    encoder = ScanpathEncoder(embed_dim=8)
    subject = make_sample(4, value=1.5)

    short_batch = batch_reading_samples([subject, make_sample(5)])
    long_batch = batch_reading_samples([subject, make_sample(40)])

    with torch.no_grad():
        short = encoder(short_batch["scanpath"], short_batch["mask"])[0]
        long = encoder(long_batch["scanpath"], long_batch["mask"])[0]

    assert torch.allclose(short, long, atol=1e-6)


def test_encoding_without_a_mask_does_depend_on_neighbours():
    """Counterpart to the test above: dropping the mask reintroduces the leak."""
    torch.manual_seed(0)
    encoder = ScanpathEncoder(embed_dim=8)
    subject = make_sample(4, value=1.5)

    short_batch = batch_reading_samples([subject, make_sample(5)])
    long_batch = batch_reading_samples([subject, make_sample(40)])

    with torch.no_grad():
        short = encoder(short_batch["scanpath"])[0]
        long = encoder(long_batch["scanpath"])[0]

    assert not torch.allclose(short, long, atol=1e-6)


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        batch_reading_samples([])


def test_feature_mismatch_is_rejected():
    mixed = [{"scanpath": torch.zeros(4, 3)}, {"scanpath": torch.zeros(4, 5)}]
    with pytest.raises(ValueError, match="feature dimension"):
        batch_reading_samples(mixed)
