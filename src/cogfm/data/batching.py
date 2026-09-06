"""Combine individual reading samples into one padded batch.

Scanpaths differ in length from trial to trial, so they cannot be stacked
directly. Every sequence in a batch is padded with zeros up to the longest one
present, and a mask marks which positions hold real fixations.

Padding reaches only as far as the longest sequence in the same batch, not the
longest in the corpus. The corpus maximum is many times the median, so padding
to it would fill most batches with far more padding than data.

The mask is what keeps a trial's representation independent of its neighbours.
Pooling over padded positions would average a short scanpath over whatever
length its batch happened to have, so the same trial would encode differently
depending on who it was batched with. A contrastive loss compares
representations within a batch, and would then partly compare that artefact.
"""

from __future__ import annotations

import torch

# Keys copied straight through from sample to batch, when present.
PASSTHROUGH = ("text", "subject_id", "sample_id", "sentence_id", "task", "dataset_id")


def batch_reading_samples(samples: list[dict]) -> dict:
    """Merge per-sample dicts into one batch with padding and a mask.

    Args:
        samples: dicts holding at least ``scanpath`` of shape (T, F), with T
            free to differ between samples.

    Returns:
        A dict with ``scanpath`` of shape (B, T_max, F), ``mask`` of shape
        (B, T_max) holding 1 for real fixations and 0 for padding, ``lengths``
        of shape (B,), and a list per passthrough key present in the samples.

    Raises:
        ValueError: on an empty batch, or if the feature dimension differs
            between samples.
    """
    if not samples:
        raise ValueError("cannot batch an empty list of samples")

    scanpaths = [s["scanpath"] for s in samples]
    n_features = {int(path.shape[1]) for path in scanpaths}
    if len(n_features) != 1:
        raise ValueError(f"samples disagree on the feature dimension: {sorted(n_features)}")

    lengths = torch.tensor([int(path.shape[0]) for path in scanpaths], dtype=torch.long)
    batch_size = len(scanpaths)
    longest = int(lengths.max())
    features = n_features.pop()

    reference = scanpaths[0]
    padded = torch.zeros(
        (batch_size, longest, features), dtype=reference.dtype, device=reference.device
    )
    mask = torch.zeros((batch_size, longest), dtype=torch.long, device=reference.device)
    for i, path in enumerate(scanpaths):
        end = int(lengths[i])
        padded[i, :end] = path
        mask[i, :end] = 1

    batch = {"scanpath": padded, "mask": mask, "lengths": lengths}
    for key in PASSTHROUGH:
        if key in samples[0]:
            batch[key] = [s[key] for s in samples]
    return batch
