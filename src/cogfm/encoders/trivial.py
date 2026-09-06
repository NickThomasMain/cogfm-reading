"""Scanpath summary statistics, standing in for a learned encoder.

A scanpath carries how long a text is and how it was laid out almost perfectly:
more words means more fixations, more reading time and more return sweeps. A
model with no understanding of language can rank candidates on that alone and
will land well above chance for it. The question a binding result has to answer
is therefore not whether it beats chance, but whether it beats what these few
numbers already achieve.

This encoder computes those numbers and nothing else. It has no parameters and
learns nothing. Because it sits behind the same interface as a real encoder,
the comparison runs through the identical connector, loss, split and pool, so
the only difference between the two conditions is what the encoder saw.

Features are rescaled by fixed, documented transforms rather than by statistics
taken from the data, which keeps the encoder free of any fitting step. Counts
and durations pass through a logarithm because they span orders of magnitude;
distances are divided by a constant to bring them into a similar range. The
point is to give the baseline a fair chance: a baseline crippled by bad scaling
would flatter whatever it is compared against.
"""

from __future__ import annotations

import torch

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS

FEATURE_NAMES = (
    "log_n_fixations",
    "log_total_duration",
    "log_mean_duration",
    "mean_saccade_amplitude",
    "x_range",
    "y_range",
    "log_n_return_sweeps",
)

# Pixel distances are divided by this so they end up near the other features.
DISTANCE_SCALE = 1000.0

# A leftward jump counts as a return sweep when it spans more than this share
# of the trial's horizontal extent, which identifies line breaks without
# needing the line spacing of a particular display.
RETURN_SWEEP_SHARE = 0.5


@ENCODERS.register("trivial")
class TrivialFeatureEncoder(ModalityEncoder):
    """Turns a scanpath into a handful of surface statistics.

    Args:
        embed_dim: must equal the number of features; accepted so the encoder
            can be built through the same config path as the others.

    Raises:
        ValueError: if ``embed_dim`` disagrees with the feature count.
    """

    def __init__(self, embed_dim: int = len(FEATURE_NAMES)) -> None:
        if embed_dim != len(FEATURE_NAMES):
            raise ValueError(
                f"embed_dim must be {len(FEATURE_NAMES)} for the trivial encoder, got {embed_dim}"
            )
        super().__init__(len(FEATURE_NAMES))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        mask = mask.to(x.dtype)

        px, py, duration = x[..., 0], x[..., 1], x[..., 2]
        n_fixations = mask.sum(dim=1)
        safe_n = n_fixations.clamp(min=1.0)

        total_duration = (duration * mask).sum(dim=1)
        mean_duration = total_duration / safe_n

        # Padding must not win a minimum or maximum, so it is pushed outwards.
        far = torch.finfo(x.dtype).max
        x_range = _masked_range(px, mask, far)
        y_range = _masked_range(py, mask, far)

        # A saccade needs two real fixations in a row.
        pair_mask = mask[:, :-1] * mask[:, 1:]
        dx = px[:, 1:] - px[:, :-1]
        dy = py[:, 1:] - py[:, :-1]
        amplitude = torch.sqrt(dx.pow(2) + dy.pow(2))
        n_pairs = pair_mask.sum(dim=1).clamp(min=1.0)
        mean_amplitude = (amplitude * pair_mask).sum(dim=1) / n_pairs

        threshold = -RETURN_SWEEP_SHARE * x_range.unsqueeze(1)
        return_sweeps = ((dx < threshold).to(x.dtype) * pair_mask).sum(dim=1)

        return torch.stack(
            [
                torch.log1p(n_fixations),
                torch.log1p(total_duration),
                torch.log1p(mean_duration),
                mean_amplitude / DISTANCE_SCALE,
                x_range / DISTANCE_SCALE,
                y_range / DISTANCE_SCALE,
                torch.log1p(return_sweeps),
            ],
            dim=1,
        )


def _masked_range(values: torch.Tensor, mask: torch.Tensor, far: float) -> torch.Tensor:
    """Spread between the smallest and largest unmasked value, zero if empty."""
    present = mask > 0
    highest = torch.where(present, values, torch.full_like(values, -far)).max(dim=1).values
    lowest = torch.where(present, values, torch.full_like(values, far)).min(dim=1).values
    spread = highest - lowest
    return torch.where(present.any(dim=1), spread, torch.zeros_like(spread))
