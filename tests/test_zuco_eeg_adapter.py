"""Tests for the ZuCo EEG adapter: channel selection, referencing, resampling.

The montage and selection tests need no data and always run. The rest is skipped
unless the extraction has been done, following test_pipeline_check.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cogfm.data.adapters.zuco_eeg import (
    CZ_COLUMN,
    MONTAGE_FILE,
    ZuCoEEGDataset,
)

ZUCO_ROOT = Path("data/zuco/processed")
HAS_DATA = (ZUCO_ROOT / "eeg_index.npz").is_file()
needs_data = pytest.mark.skipif(
    not HAS_DATA,
    reason="ZuCo EEG has not been extracted; run scripts/extract_zuco_eeg.py",
)

N_STORED_COLUMNS = 105


# --- montage table ----------------------------------------------------------

def test_montage_matches_the_egi_table():
    montage = json.loads(MONTAGE_FILE.read_text(encoding="utf-8"))
    channels = montage["channels"]
    assert len(channels) == 69, "69 ZuCo columns have a 10-10 name"
    within = [c for c in channels if c["within_egi_threshold"]]
    assert len(within) == 62, "62 of them are inside EGI's own 1.85 cm criterion"
    assert montage["egi_threshold_cm"] == pytest.approx(1.8462, abs=1e-3)


def test_montage_columns_are_unique_and_in_range():
    channels = json.loads(MONTAGE_FILE.read_text(encoding="utf-8"))["channels"]
    columns = [c["column"] for c in channels]
    assert len(set(columns)) == len(columns), "each column claimed at most once"
    assert all(0 <= c < N_STORED_COLUMNS for c in columns)
    names = [c["name"] for c in channels]
    assert len(set(names)) == len(names), "no 10-10 position assigned twice"


def test_cz_is_the_last_column():
    channels = json.loads(MONTAGE_FILE.read_text(encoding="utf-8"))["channels"]
    cz = next(c for c in channels if c["name"] == "CZ")
    assert cz["column"] == CZ_COLUMN
    assert cz["hydrocel"] == 129, "Cz is the reference electrode, not one of the 128"


# --- channel selection, no data needed --------------------------------------

@pytest.mark.parametrize(
    ("channels", "reference", "expected"),
    [
        ("egi62", "average", 62),
        ("egi62", "recording", 61),      # Cz is constant zero and dropped
        ("named69", "average", 69),
        ("named69", "recording", 68),
        ("all", "average", N_STORED_COLUMNS),
        ("all", "recording", N_STORED_COLUMNS - 1),
    ],
)
def test_selection_sizes(channels, reference, expected):
    columns, names = ZuCoEEGDataset._select(channels, reference, False, N_STORED_COLUMNS)
    assert len(columns) == expected
    assert len(names) == expected


def test_recording_reference_never_includes_cz():
    for selection in ("egi62", "named69", "all"):
        columns, _ = ZuCoEEGDataset._select(selection, "recording", False, N_STORED_COLUMNS)
        assert CZ_COLUMN not in columns.tolist()


def test_average_reference_includes_cz_in_named_selections():
    for selection in ("egi62", "named69"):
        columns, names = ZuCoEEGDataset._select(selection, "average", False, N_STORED_COLUMNS)
        assert CZ_COLUMN in columns.tolist()
        assert "CZ" in names


def test_columns_are_sorted_and_names_align():
    columns, names = ZuCoEEGDataset._select("egi62", "average", False, N_STORED_COLUMNS)
    assert list(columns) == sorted(columns), "feature axis follows column order"
    assert all(isinstance(n, str) for n in names), "named selections name every channel"


def test_all_selection_leaves_unnamed_columns_without_a_name():
    _, names = ZuCoEEGDataset._select("all", "average", False, N_STORED_COLUMNS)
    assert names.count(None) == N_STORED_COLUMNS - 69


def test_legacy_names_switch_the_four_aliases():
    _, modern = ZuCoEEGDataset._select("named69", "average", False, N_STORED_COLUMNS)
    _, legacy = ZuCoEEGDataset._select("named69", "average", True, N_STORED_COLUMNS)
    assert {"T7", "T8", "P7", "P8"} <= set(modern)
    assert {"T3", "T4", "T5", "T6"} <= set(legacy)
    assert not {"T7", "T8", "P7", "P8"} & set(legacy)
    changed = sum(a != b for a, b in zip(modern, legacy))
    assert changed == 4, "only the four alias positions differ"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"reference": "nope"}, "reference must be"), ({"channels": "nope"}, "channels must be")],
)
def test_invalid_switches_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ZuCoEEGDataset(root=ZUCO_ROOT, **kwargs)


# --- with the extracted data ------------------------------------------------

@needs_data
def test_trial_shape_and_dtype():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", resample_hz=None)
    trial = data.eeg(0)
    assert trial.ndim == 2, "axes are (time, channels)"
    assert trial.shape[1] == 62
    assert trial.dtype == np.float32, "stored as float16, returned as float32"
    assert np.isfinite(trial).all()


@needs_data
def test_lengths_match_the_returned_trials():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", resample_hz=None)
    lengths = data.lengths()
    for idx in (0, 1, len(data) - 1):
        assert data.eeg(idx).shape[0] == lengths[idx]


@needs_data
def test_cz_is_constant_under_the_recording_reference_and_alive_after_averaging():
    """The whole point of the switch, checked on the data rather than argued."""
    raw = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", channels="all", reference="recording",
                         resample_hz=None)
    avg = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", channels="all", reference="average",
                         resample_hz=None)
    assert CZ_COLUMN not in raw._columns.tolist()
    cz = avg.eeg(0)[:, CZ_COLUMN]
    assert cz.std() > 0.1, "Cz carries signal once the reference is moved"


@needs_data
def test_average_reference_zeroes_the_row_mean():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", channels="all", reference="average",
                          resample_hz=None)
    trial = data.eeg(0)
    assert np.abs(trial.mean(axis=1)).max() < 1e-3, "mean over all columns is removed"


@needs_data
def test_recording_reference_leaves_the_data_untouched():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", channels="all", reference="recording",
                          resample_hz=None)
    signals = data._signals(str(data.subject_ids[0]), str(data.task_ids[0]))
    stored = np.asarray(signals[data._starts[0] : data._ends[0]], dtype=np.float32)
    kept = stored[:, data._columns]
    assert np.array_equal(data.eeg(0), kept)


@needs_data
def test_resampling_changes_the_length_by_the_rate_ratio():
    full = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", resample_hz=None)
    down = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR", resample_hz=200)
    assert full.sample_rate == 500 and down.sample_rate == 200
    a, b = full.eeg(0).shape[0], down.eeg(0).shape[0]
    assert b == pytest.approx(a * 2 / 5, rel=0.01)
    assert down.lengths()[0] == b
    assert down.eeg(0).shape[1] == 62, "resampling does not touch the channel axis"


@needs_data
def test_sample_carries_the_matching_sentence():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR")
    sample = data[0]
    assert sample["dataset_id"] == "zuco1-eeg"
    assert sample["task"] == "SR"
    assert isinstance(sample["text"], str) and sample["text"]
    by_id = {s["id"]: s["text"] for s in data.sentences}
    assert sample["text"] == by_id[sample["sentence_id"]]
    assert sample["eeg"].shape[1] == 62


@needs_data
def test_reading_a_trial_twice_gives_the_same_array():
    data = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR")
    assert np.array_equal(data.eeg(3), data.eeg(3))


@needs_data
def test_task_filter_and_unknown_task():
    sr = ZuCoEEGDataset(root=ZUCO_ROOT, task="SR")
    nr = ZuCoEEGDataset(root=ZUCO_ROOT, task="NR")
    both = ZuCoEEGDataset(root=ZUCO_ROOT, task=None)
    assert len(sr) + len(nr) == len(both)
    assert set(sr.task_ids) == {"SR"}
    with pytest.raises(ValueError, match="no trials for task"):
        ZuCoEEGDataset(root=ZUCO_ROOT, task="XX")
