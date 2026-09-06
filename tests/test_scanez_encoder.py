"""Tests for the scanpath transformer: order sensitivity, masking, shapes."""

import pytest
import torch

from cogfm.data.batching import batch_reading_samples
from cogfm.encoders.scanez import ScanEZEncoder, sinusoidal_encoding
from cogfm.encoders.scanpath import ScanpathEncoder
from cogfm.registry import ENCODERS


def path(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=generator)


def encoder(**kwargs) -> ScanEZEncoder:
    torch.manual_seed(0)
    return ScanEZEncoder(embed_dim=32, n_layers=2, n_heads=4, feedforward_dim=64, **kwargs).eval()


def test_it_is_registered_under_its_name():
    torch.manual_seed(0)
    assert isinstance(ENCODERS.build("scanez", embed_dim=32), ScanEZEncoder)


def test_output_shape_follows_embed_dim():
    out = encoder()(path(12).unsqueeze(0))
    assert out.shape == (1, 32)


def test_shuffling_the_scanpath_changes_the_encoding():
    """The property the mean-pooling placeholder lacks."""
    model = encoder()
    original = path(20, seed=1)
    shuffled = original[torch.randperm(20, generator=torch.Generator().manual_seed(2))]
    with torch.no_grad():
        a = model(original.unsqueeze(0))
        b = model(shuffled.unsqueeze(0))
    assert not torch.allclose(a, b, atol=1e-4)


def test_the_placeholder_is_blind_to_order():
    """Documents why the placeholder cannot serve as the architecture control."""
    torch.manual_seed(0)
    model = ScanpathEncoder(embed_dim=32).eval()
    original = path(20, seed=1)
    shuffled = original[torch.randperm(20, generator=torch.Generator().manual_seed(2))]
    with torch.no_grad():
        assert torch.allclose(model(original.unsqueeze(0)), model(shuffled.unsqueeze(0)), atol=1e-5)


def test_two_different_scanpaths_encode_differently():
    model = encoder()
    with torch.no_grad():
        a = model(path(15, seed=1).unsqueeze(0))
        b = model(path(15, seed=2).unsqueeze(0))
    assert not torch.allclose(a, b, atol=1e-4)


def test_padding_does_not_change_the_encoding():
    model = encoder()
    short = {"scanpath": path(6, seed=3)}
    long = {"scanpath": path(40, seed=4)}
    alone = batch_reading_samples([short])
    padded = batch_reading_samples([short, long])
    with torch.no_grad():
        a = model(alone["scanpath"], alone["mask"])[0]
        b = model(padded["scanpath"], padded["mask"])[0]
    assert torch.allclose(a, b, atol=1e-4)


def test_it_is_deterministic_in_eval_mode():
    model = encoder()
    x = path(10, seed=5).unsqueeze(0)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_the_same_seed_gives_the_same_random_weights():
    a, b = encoder(), encoder()
    for left, right in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(left, right)


def test_a_sequence_beyond_max_length_is_rejected():
    model = encoder(max_length=16)
    with pytest.raises(ValueError, match="exceeds max_length"):
        model(path(20).unsqueeze(0))


def test_an_indivisible_width_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        ScanEZEncoder(embed_dim=30, n_heads=4)


def test_a_missing_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="checkpoint not found"):
        ScanEZEncoder(embed_dim=32, checkpoint="does/not/exist.pt")


def test_a_checkpoint_round_trips(tmp_path):
    """Random init and pretrained differ only in the weights that are loaded."""
    torch.manual_seed(0)
    trained = ScanEZEncoder(embed_dim=32, n_layers=2, n_heads=4, feedforward_dim=64)
    target = tmp_path / "scanez.pt"
    torch.save(trained.state_dict(), target)

    torch.manual_seed(99)
    loaded = ScanEZEncoder(
        embed_dim=32, n_layers=2, n_heads=4, feedforward_dim=64, checkpoint=str(target)
    )
    for left, right in zip(trained.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(left, right)


def test_position_codes_are_bounded_and_distinct():
    codes = sinusoidal_encoding(50, 32)
    assert codes.shape == (50, 32)
    assert codes.abs().max() <= 1.0
    assert not torch.allclose(codes[0], codes[1])
