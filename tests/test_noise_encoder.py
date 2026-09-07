"""Tests for the noise encoder: no information gets through, but the draw is stable."""

import torch

from cogfm.encoders.noise import NoiseEncoder
from cogfm.registry import ENCODERS


def path(points: list[tuple[float, float, float]]) -> torch.Tensor:
    """One scanpath as (n_fixations, 3) from (x, y, duration) triples."""
    return torch.tensor(points, dtype=torch.float32)


def walk(n: int, shift: float = 0.0) -> torch.Tensor:
    """A plausible left-to-right scanpath of n fixations."""
    return path([(30.0 * i + shift, 100.0, 200.0 + i) for i in range(n)])


def test_it_is_registered_under_its_name():
    assert isinstance(ENCODERS.build("noise", embed_dim=16), NoiseEncoder)


def test_output_has_the_requested_width():
    encoder = NoiseEncoder(embed_dim=32)
    assert encoder(walk(5).unsqueeze(0)).shape == (1, 32)


def test_it_has_no_parameters_to_train():
    assert list(NoiseEncoder(embed_dim=8).parameters()) == []


def test_the_same_trial_gets_the_same_vector_every_time():
    encoder = NoiseEncoder(embed_dim=16)
    scanpath = walk(7).unsqueeze(0)
    torch.testing.assert_close(encoder(scanpath), encoder(scanpath))


def test_two_trials_in_one_batch_keep_their_own_vectors():
    encoder = NoiseEncoder(embed_dim=16)
    first, second = walk(6), walk(6, shift=5.0)
    together = encoder(torch.stack([first, second]))
    apart = torch.cat([encoder(first.unsqueeze(0)), encoder(second.unsqueeze(0))])
    torch.testing.assert_close(together, apart)


def test_the_vector_survives_padding_in_a_longer_batch():
    encoder = NoiseEncoder(embed_dim=16)
    short, long = walk(4), walk(9)
    padded = torch.zeros(2, 9, 3)
    padded[0, :4] = short
    padded[1] = long
    mask = torch.zeros(2, 9)
    mask[0, :4] = 1.0
    mask[1] = 1.0

    out = encoder(padded, mask)
    torch.testing.assert_close(out[0], encoder(short.unsqueeze(0))[0])


def test_padding_that_is_not_zero_is_ignored():
    encoder = NoiseEncoder(embed_dim=16)
    scanpath = walk(4)
    padded = torch.full((1, 6, 3), 999.0)
    padded[0, :4] = scanpath
    mask = torch.zeros(1, 6)
    mask[0, :4] = 1.0

    torch.testing.assert_close(encoder(padded, mask), encoder(scanpath.unsqueeze(0)))


def test_a_single_changed_fixation_gives_an_unrelated_vector():
    encoder = NoiseEncoder(embed_dim=64)
    before = walk(8)
    after = before.clone()
    after[3, 0] += 1.0

    a = encoder(before.unsqueeze(0))[0]
    b = encoder(after.unsqueeze(0))[0]
    cosine = torch.dot(a, b) / (a.norm() * b.norm())
    assert abs(float(cosine)) < 0.5


def test_neighbouring_scanpaths_are_no_closer_than_distant_ones():
    """Nearness in the input must not survive into the output at all."""
    encoder = NoiseEncoder(embed_dim=64)
    base = walk(10)
    near = base.clone()
    near[:, 0] += 1.0
    far = base.clone()
    far[:, 0] += 400.0

    vectors = encoder(torch.stack([base, near, far]))
    normed = vectors / vectors.norm(dim=1, keepdim=True)
    close = float(torch.dot(normed[0], normed[1]))
    distant = float(torch.dot(normed[0], normed[2]))
    assert abs(close) < 0.5 and abs(distant) < 0.5


def test_the_same_order_of_fixations_matters():
    encoder = NoiseEncoder(embed_dim=64)
    forward = walk(6)
    backward = torch.flip(forward, dims=[0])

    a = encoder(forward.unsqueeze(0))[0]
    b = encoder(backward.unsqueeze(0))[0]
    assert not torch.allclose(a, b)


def test_a_different_seed_moves_the_whole_condition():
    scanpath = walk(6).unsqueeze(0)
    a = NoiseEncoder(embed_dim=32, seed=0)(scanpath)
    b = NoiseEncoder(embed_dim=32, seed=1)(scanpath)
    assert not torch.allclose(a, b)


def test_vectors_across_many_trials_look_like_noise():
    encoder = NoiseEncoder(embed_dim=8)
    batch = torch.stack([walk(6, shift=float(i)) for i in range(400)])
    out = encoder(batch)

    assert abs(float(out.mean())) < 0.15
    assert 0.8 < float(out.std()) < 1.2


def test_an_empty_trial_still_returns_a_vector():
    encoder = NoiseEncoder(embed_dim=16)
    out = encoder(torch.zeros(1, 5, 3), torch.zeros(1, 5))
    assert out.shape == (1, 16) and torch.isfinite(out).all()


def test_output_follows_the_input_dtype_and_device():
    encoder = NoiseEncoder(embed_dim=16)
    out = encoder(walk(5).unsqueeze(0).to(torch.float64))
    assert out.dtype == torch.float64
