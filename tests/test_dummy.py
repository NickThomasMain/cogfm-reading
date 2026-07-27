"""Tests for the synthetic data generator: determinism, shapes, dataset wrapper."""

import numpy as np

from cogfm.data.dummy import SCANPATH_FEATURES, make_dummy_samples


def test_same_seed_gives_identical_samples():
    a = make_dummy_samples(8, seed=42)
    b = make_dummy_samples(8, seed=42)
    assert len(a) == len(b) == 8
    for sa, sb in zip(a, b):
        assert sa.text == sb.text
        assert np.array_equal(sa.scanpath, sb.scanpath)


def test_different_seed_differs():
    a = make_dummy_samples(8, seed=1)
    b = make_dummy_samples(8, seed=2)
    assert any(not np.array_equal(sa.scanpath, sb.scanpath) for sa, sb in zip(a, b))


def test_scanpath_shape_and_dtype():
    for s in make_dummy_samples(5, seed=0):
        assert s.scanpath.ndim == 2
        assert s.scanpath.shape[1] == len(SCANPATH_FEATURES)  # x, y, duration
        assert s.scanpath.dtype == np.float32
        assert len(s.text) >= 1


def test_dataset_wrapper():
    from cogfm.data.dummy import DummyReadingDataset

    ds = DummyReadingDataset(6, seed=0)
    assert len(ds) == 6
    item = ds[0]
    assert {"scanpath", "text", "subject_id"} <= set(item)
    assert item["scanpath"].shape[1] == 3
