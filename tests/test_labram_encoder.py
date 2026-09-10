"""Tests for the LaBraM wrapper.

The montage lookup and the padding logic need no model and always run. Anything
touching the checkpoint is skipped unless braindecode is installed.
"""

import pytest
import torch

from cogfm.encoders.labram import (
    EXPECTED_TIME_PATCHES,
    PATCH_SAMPLES,
    LaBraMEncoder,
    montage_names,
)

def _has_braindecode() -> bool:
    try:
        import braindecode.models  # noqa: F401
    except ImportError:
        return False
    return True


needs_model = pytest.mark.skipif(
    not _has_braindecode(), reason="braindecode is not installed"
)


# --- channel names, no model needed -----------------------------------------

def test_montage_names_match_the_selection():
    assert len(montage_names("egi62")) == 62
    assert len(montage_names("named69")) == 69


def test_montage_names_are_unique_and_include_cz():
    names = montage_names("egi62")
    assert len(set(names)) == len(names)
    assert "CZ" in names


def test_legacy_names_switch_the_four_aliases():
    modern, legacy = montage_names("named69"), montage_names("named69", legacy_names=True)
    assert {"T7", "T8", "P7", "P8"} <= set(modern)
    assert {"T3", "T4", "T5", "T6"} <= set(legacy)
    assert sum(a != b for a, b in zip(modern, legacy)) == 4


def test_unknown_selection_is_rejected():
    with pytest.raises(ValueError, match="selection must be"):
        montage_names("nope")


# --- padding logic, no model needed -----------------------------------------

def test_patch_weights_without_mask_keep_everything():
    w = LaBraMEncoder._patch_weights(None, 3, 5, torch.device("cpu"), torch.float32)
    assert w.shape == (3, 5)
    assert torch.equal(w, torch.ones(3, 5))


def test_patch_weights_drop_patches_inside_the_padding():
    # Two trials in a batch of 4 patches' width: one full, one half.
    mask = torch.zeros(2, 4 * PATCH_SAMPLES)
    mask[0, :] = 1
    mask[1, : 2 * PATCH_SAMPLES] = 1
    w = LaBraMEncoder._patch_weights(mask, 2, 4, torch.device("cpu"), torch.float32)
    assert torch.equal(w[0], torch.ones(4))
    assert torch.equal(w[1], torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_a_straddling_patch_is_kept():
    """A patch that starts inside the trial counts, even if it ends in padding."""
    mask = torch.zeros(1, 3 * PATCH_SAMPLES)
    mask[0, : PATCH_SAMPLES + 10] = 1        # one full patch plus 10 samples
    w = LaBraMEncoder._patch_weights(mask, 1, 3, torch.device("cpu"), torch.float32)
    assert torch.equal(w[0], torch.tensor([1.0, 1.0, 0.0]))


# --- with braindecode installed ---------------------------------------------

@needs_model
def test_rejects_a_bad_pooling_argument():
    with pytest.raises(ValueError, match="pooling must be"):
        LaBraMEncoder(pooling="nope")


@needs_model
def test_encodes_a_batch_to_one_vector_per_trial():
    encoder = LaBraMEncoder()
    x = torch.randn(2, 4 * PATCH_SAMPLES, 62)
    out = encoder(x)
    assert out.shape == (2, encoder.embed_dim)
    assert torch.isfinite(out).all()


@needs_model
def test_refuses_input_beyond_the_time_embedding():
    encoder = LaBraMEncoder()
    too_long = torch.randn(1, (EXPECTED_TIME_PATCHES + 1) * PATCH_SAMPLES, 62)
    with pytest.raises(ValueError, match="time embedding covers"):
        encoder(too_long)


@needs_model
def test_refuses_a_channel_count_that_disagrees_with_the_names():
    encoder = LaBraMEncoder()
    with pytest.raises(ValueError, match="channels in the batch"):
        encoder(torch.randn(1, PATCH_SAMPLES, 30))


@needs_model
def test_backbone_stays_frozen_and_in_eval():
    encoder = LaBraMEncoder()
    encoder.train()
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())


@needs_model
def test_padding_does_not_change_the_result():
    """A padded batch must give the same vector as the unpadded trial alone."""
    encoder = LaBraMEncoder()
    trial = torch.randn(1, 2 * PATCH_SAMPLES, 62)
    padded = torch.cat([trial, torch.randn(1, 2 * PATCH_SAMPLES, 62)], dim=1)
    mask = torch.zeros(1, 4 * PATCH_SAMPLES)
    mask[0, : 2 * PATCH_SAMPLES] = 1
    assert torch.allclose(encoder(trial), encoder(padded, mask), atol=1e-4)
