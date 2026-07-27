"""Combine individual reading samples into one batch.

Simplifying assumption for M0: every scanpath has the same number of fixations,
so the scanpaths can be stacked directly with no padding. (Variable lengths and
padding return with real data in M1.)
"""

from __future__ import annotations

import torch


def batch_reading_samples(samples: list[dict]) -> dict:
    """Merge a list of per-sample dicts into one batched dict.

    Each input sample (from DummyReadingDataset) has:
        scanpath: tensor (T, 3)  -- same T for every sample (equal length)
        text: str
        subject_id: str
        sample_id: int

    Returns a dict with:
        scanpath: tensor (B, T, 3)
        text: list[str]        (length B)
        subject_id: list[str]  (length B)
        sample_id: list[int]   (length B)
    """
    scanpath = torch.stack([s["scanpath"] for s in samples], dim=0)  # (B, T, 3)
    return {
        "scanpath": scanpath,
        "text": [s["text"] for s in samples],
        "subject_id": [s["subject_id"] for s in samples],
        "sample_id": [s["sample_id"] for s in samples],
    }
