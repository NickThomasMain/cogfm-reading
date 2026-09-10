"""Tests for the ZuCo eye-tracking adapter.

Counterpart to test_zuco_eeg_adapter.py. Everything that touches trials needs the
extraction to have run and is skipped otherwise, following test_pipeline_check.py.
"""

from pathlib import Path

import numpy as np
import pytest

from cogfm.data.adapters.zuco_et import ZuCoETDataset

ZUCO_ROOT = Path("data/zuco/processed")
HAS_DATA = (ZUCO_ROOT / "scanpaths.npz").is_file()
needs_data = pytest.mark.skipif(
    not HAS_DATA,
    reason="ZuCo has not been extracted; run scripts/extract_zuco_et.py",
)


def test_missing_extraction_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract_zuco_et.py"):
        ZuCoETDataset(root=tmp_path)


@needs_data
def test_scanpath_shape_and_features():
    data = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    assert data.features == ("x", "y", "duration")
    path = data.scanpath(0)
    assert path.ndim == 2, "axes are (time, features)"
    assert path.shape[1] == len(data.features)
    assert np.isfinite(path).all()


@needs_data
def test_lengths_match_the_returned_scanpaths():
    data = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    lengths = data.lengths()
    for idx in (0, 1, len(data) - 1):
        assert data.scanpath(idx).shape[0] == lengths[idx]
    assert lengths.min() >= 5, "extraction drops trials below MIN_FIXATIONS"


@needs_data
def test_sample_carries_the_matching_sentence():
    data = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    sample = data[0]
    assert sample["dataset_id"] == "zuco1-et"
    assert sample["task"] == "SR"
    assert isinstance(sample["text"], str) and sample["text"]
    by_id = {s["id"]: s["text"] for s in data.sentences}
    assert sample["text"] == by_id[sample["sentence_id"]]
    assert sample["scanpath"].shape[1] == len(data.features)


@needs_data
def test_reading_a_trial_twice_gives_the_same_array():
    data = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    assert np.array_equal(data.scanpath(3), data.scanpath(3))


@needs_data
def test_task_filter_and_unknown_task():
    sr = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    nr = ZuCoETDataset(root=ZUCO_ROOT, task="NR")
    both = ZuCoETDataset(root=ZUCO_ROOT, task=None)
    assert len(sr) + len(nr) == len(both)
    assert set(sr.task_ids) == {"SR"}
    with pytest.raises(ValueError, match="no trials for task"):
        ZuCoETDataset(root=ZUCO_ROOT, task="XX")


@needs_data
def test_sentence_ids_are_shared_with_the_eeg_strand():
    """Both adapters read the same sentences.json, so ids must be comparable."""
    pytest.importorskip("cogfm.data.adapters.zuco_eeg")
    from cogfm.data.adapters.zuco_eeg import ZuCoEEGDataset

    if not (ZUCO_ROOT / "eeg_index.npz").is_file():
        pytest.skip("ZuCo EEG has not been extracted")
    et = ZuCoETDataset(root=ZUCO_ROOT, task="SR")
    eeg = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR")
    assert set(eeg.sentence_ids) <= set(et.sentence_ids) | set(eeg.sentence_ids)
    shared = set(et.sentence_ids) & set(eeg.sentence_ids)
    assert shared, "the two modalities must overlap in sentences"
    et_text = {s["id"]: s["text"] for s in et.sentences}
    eeg_text = {s["id"]: s["text"] for s in eeg.sentences}
    assert et_text == eeg_text
