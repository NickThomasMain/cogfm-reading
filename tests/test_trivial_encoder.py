"""Tests for the trivial feature encoder: hand-computed values, masking, edges."""

import math

import pytest
import torch

from cogfm.data.batching import batch_reading_samples
from cogfm.encoders.trivial import FEATURE_NAMES, DISTANCE_SCALE, TrivialFeatureEncoder
from cogfm.registry import ENCODERS

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def path(points: list[tuple[float, float, float]]) -> torch.Tensor:
    """One scanpath as (n_fixations, 3) from (x, y, duration) triples."""
    return torch.tensor(points, dtype=torch.float32)


def test_it_is_registered_under_its_name():
    assert isinstance(ENCODERS.build("trivial", embed_dim=7), TrivialFeatureEncoder)


def test_output_has_one_column_per_feature():
    encoder = TrivialFeatureEncoder()
    out = encoder(path([(0.0, 0.0, 100.0)] * 4).unsqueeze(0))
    assert out.shape == (1, len(FEATURE_NAMES))


def test_it_has_no_parameters_to_train():
    assert list(TrivialFeatureEncoder().parameters()) == []


def test_a_mismatched_embed_dim_is_rejected():
    with pytest.raises(ValueError, match="embed_dim"):
        TrivialFeatureEncoder(embed_dim=128)


def test_counts_and_durations_match_a_hand_computed_case():
    scanpath = path([(0.0, 0.0, 100.0), (10.0, 0.0, 200.0), (20.0, 0.0, 300.0)])
    out = TrivialFeatureEncoder()(scanpath.unsqueeze(0))[0]
    assert out[F["log_n_fixations"]].item() == pytest.approx(math.log1p(3))
    assert out[F["log_total_duration"]].item() == pytest.approx(math.log1p(600))
    assert out[F["log_mean_duration"]].item() == pytest.approx(math.log1p(200))


def test_ranges_match_a_hand_computed_case():
    scanpath = path([(10.0, 5.0, 100.0), (60.0, 5.0, 100.0), (30.0, 25.0, 100.0)])
    out = TrivialFeatureEncoder()(scanpath.unsqueeze(0))[0]
    assert out[F["x_range"]].item() == pytest.approx(50.0 / DISTANCE_SCALE)
    assert out[F["y_range"]].item() == pytest.approx(20.0 / DISTANCE_SCALE)


def test_saccade_amplitude_matches_a_hand_computed_case():
    scanpath = path([(0.0, 0.0, 100.0), (3.0, 4.0, 100.0), (3.0, 14.0, 100.0)])
    out = TrivialFeatureEncoder()(scanpath.unsqueeze(0))[0]
    # steps of 5 and 10, mean 7.5
    assert out[F["mean_saccade_amplitude"]].item() == pytest.approx(7.5 / DISTANCE_SCALE)


def test_return_sweeps_count_large_leftward_jumps():
    # two lines: run right, jump far left, run right again
    scanpath = path(
        [(0.0, 0.0, 100.0), (50.0, 0.0, 100.0), (100.0, 0.0, 100.0), (5.0, 20.0, 100.0), (55.0, 20.0, 100.0)]
    )
    out = TrivialFeatureEncoder()(scanpath.unsqueeze(0))[0]
    assert out[F["log_n_return_sweeps"]].item() == pytest.approx(math.log1p(1))


def test_a_single_forward_run_has_no_return_sweeps():
    scanpath = path([(float(i) * 10.0, 0.0, 100.0) for i in range(8)])
    out = TrivialFeatureEncoder()(scanpath.unsqueeze(0))[0]
    assert out[F["log_n_return_sweeps"]].item() == pytest.approx(0.0)


def test_padding_does_not_change_the_features():
    """The whole point of the mask: a padded trial encodes like an unpadded one."""
    short = {"scanpath": path([(0.0, 0.0, 100.0), (10.0, 0.0, 200.0), (20.0, 5.0, 300.0)])}
    long = {"scanpath": path([(float(i), float(i), 50.0) for i in range(30)])}

    alone = batch_reading_samples([short])
    padded = batch_reading_samples([short, long])

    encoder = TrivialFeatureEncoder()
    assert torch.allclose(
        encoder(alone["scanpath"], alone["mask"])[0],
        encoder(padded["scanpath"], padded["mask"])[0],
        atol=1e-5,
    )


def test_a_one_fixation_trial_is_finite():
    out = TrivialFeatureEncoder()(path([(5.0, 5.0, 120.0)]).unsqueeze(0))
    assert torch.isfinite(out).all()
    assert out[0, F["mean_saccade_amplitude"]].item() == pytest.approx(0.0)


def test_longer_texts_produce_larger_counts():
    """The confound the baseline exists to expose, made explicit."""
    encoder = TrivialFeatureEncoder()
    short = {"scanpath": path([(float(i) * 5, 0.0, 200.0) for i in range(6)])}
    long = {"scanpath": path([(float(i) * 5, 0.0, 200.0) for i in range(40)])}
    batch = batch_reading_samples([short, long])
    out = encoder(batch["scanpath"], batch["mask"])
    assert out[1, F["log_n_fixations"]] > out[0, F["log_n_fixations"]]
    assert out[1, F["log_total_duration"]] > out[0, F["log_total_duration"]]
