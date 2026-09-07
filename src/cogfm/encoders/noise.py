"""An encoder that answers with noise, used as the floor of the result row.

Every number in the result row is read against a chance level, and two chance
levels are in play at once: the analytic one that follows from the pool size
(one correct candidate among twenty-five gives 0.04), and the measured one the
permutation produces. When those two disagree, the row cannot be read, because
it is unclear which of them a condition is being compared against.

This encoder settles that question. Its output is drawn from a fingerprint of
the trial rather than from how the trial was read: two scanpaths that differ in
a single fixation receive unrelated vectors, so nothing about reading survives
into the embedding and there is nothing for the connector to learn. Whatever
such a condition scores is what the evaluation returns when no signal exists.
Agreement with the analytic level says the ranking is unbiased, and agreement
between its permutation null and the null of the other conditions says the
offset belongs to the design rather than to a fault.

The fingerprint keeps the vector stable across steps and between training and
evaluation, which is what a real encoder does and what keeps the connector from
chasing a target that moves under it. Vectors are Gaussian so that they point
in every direction equally often; a distribution with preferred directions
would let candidates that happen to lie near one win more often than the pool
size allows.
"""

from __future__ import annotations

import torch

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS

# Coordinates and durations are rounded to this many units before they enter
# the fingerprint, so that a value reconstructed slightly differently still
# yields the same vector.
QUANTISATION = 100.0

# Odd multipliers that give a position its own weight in the fingerprint, so
# that the same numbers in a different order do not collide.
MIX_TIME = 0x9E3779B1
MIX_FEATURE = 0x85EBCA6B

# Constants of the standard SplitMix64 scrambler, which turns a counter into a
# seed whose bits no longer follow the counter's.
SCRAMBLE_ODD = 0x9E3779B97F4A7C15
SCRAMBLE_A = 0xBF58476D1CE4E5B9
SCRAMBLE_B = 0x94D049BB133111EB
WORD = 0xFFFFFFFFFFFFFFFF

# Seeds are handed to the generator as non-negative values.
SEED_MASK = 0x7FFFFFFFFFFFFFFF


@ENCODERS.register("noise")
class NoiseEncoder(ModalityEncoder):
    """Returns a fixed random vector per trial, unrelated to the scanpath.

    Args:
        embed_dim: width of the returned vector. Free to choose, since the
            vector carries nothing; matching a real encoder keeps the connector
            the same size across conditions.
        seed: shifts every vector to a different draw, which allows the whole
            condition to be repeated as an independent run.
    """

    def __init__(self, embed_dim: int = 128, seed: int = 0) -> None:
        super().__init__(embed_dim)
        self.seed = int(seed)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)

        generator = torch.Generator(device="cpu")
        rows = []
        for fingerprint in _fingerprints(x, mask).tolist():
            generator.manual_seed(_scramble(fingerprint ^ self.seed))
            rows.append(torch.randn(self.embed_dim, generator=generator))
        return torch.stack(rows).to(device=x.device, dtype=x.dtype)


def _fingerprints(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One integer per trial, mixed from the trial's own numbers.

    Padding is zeroed first so that the same trial in a longer batch keeps its
    fingerprint. The sum is allowed to overflow: wrapping is what turns a
    weighted sum into a hash, and a hash is what makes neighbouring scanpaths
    land far apart instead of close together.
    """
    present = (mask > 0).unsqueeze(-1)
    quantised = torch.where(present, (x * QUANTISATION).round(), torch.zeros_like(x))
    keys = quantised.to(torch.int64)

    steps, features = keys.shape[1], keys.shape[2]
    time_weight = torch.arange(1, steps + 1, device=x.device, dtype=torch.int64) * MIX_TIME
    feature_weight = torch.arange(1, features + 1, device=x.device, dtype=torch.int64) * MIX_FEATURE
    weights = time_weight.unsqueeze(1) ^ feature_weight.unsqueeze(0)

    return (keys * weights).sum(dim=(1, 2))


def _scramble(value: int) -> int:
    """Turn a fingerprint into a seed whose bits no longer track the input.

    Two trials that differ by a small, regular amount produce fingerprints that
    differ by a small, regular amount, and seeding a generator with such a
    series risks streams that are related rather than independent. Passing the
    fingerprint through this mixer removes that structure, so a run over evenly
    spaced trials is as good a draw as a run over arbitrary ones.
    """
    state = (value + SCRAMBLE_ODD) & WORD
    state = ((state ^ (state >> 30)) * SCRAMBLE_A) & WORD
    state = ((state ^ (state >> 27)) * SCRAMBLE_B) & WORD
    return (state ^ (state >> 31)) & SEED_MASK
