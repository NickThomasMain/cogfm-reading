"""Tests for the pooling connector.

The property that matters most is that padding cannot reach the result. A
padded trial and the same trial alone have to produce the same vector, or a
trial's representation would depend on the batch it landed in and the
contrastive loss would partly compare that artefact.
"""

import pytest
import torch

import cogfm.connectors  # noqa: F401  (import triggers registration)
from cogfm.connectors.attention_pool import AttentionPoolConnector
from cogfm.registry import CONNECTORS


def _pad(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(s) for s in sequences)
    padded = torch.zeros(len(sequences), longest, sequences[0].shape[1])
    mask = torch.zeros(len(sequences), longest, dtype=torch.long)
    for i, sequence in enumerate(sequences):
        padded[i, : len(sequence)] = sequence
        mask[i, : len(sequence)] = 1
    return padded, mask


def test_connector_is_registered_under_its_name():
    connector = CONNECTORS.build("attention_pool", in_dim=32, out_dim=64)
    assert isinstance(connector, AttentionPoolConnector)
    assert connector.wants_mask


def test_a_sequence_is_pooled_to_one_vector():
    connector = AttentionPoolConnector(in_dim=32, out_dim=64)
    out = connector(torch.randn(8, 5, 32), torch.ones(8, 5, dtype=torch.long))
    assert out.shape == (8, 64)


def test_a_single_vector_is_accepted_as_one_step():
    connector = AttentionPoolConnector(in_dim=32, out_dim=64)
    assert connector(torch.randn(8, 32)).shape == (8, 64)


def test_padding_does_not_change_a_trial():
    torch.manual_seed(0)
    connector = AttentionPoolConnector(in_dim=8, out_dim=4).eval()
    short, long = torch.randn(3, 8), torch.randn(6, 8)

    padded, mask = _pad([short, long])
    together = connector(padded, mask)
    alone = connector(short.unsqueeze(0), torch.ones(1, 3, dtype=torch.long))

    assert torch.allclose(together[0], alone[0], atol=1e-6)


def test_uniform_scores_reproduce_the_mean():
    """The mean stays reachable, so pooling cannot be worse for lack of it."""
    connector = AttentionPoolConnector(in_dim=8, out_dim=4).eval()
    with torch.no_grad():
        connector.key.weight.zero_()
        connector.key.bias.zero_()

    x = torch.randn(4, 6, 8)
    pooled = connector(x, torch.ones(4, 6, dtype=torch.long))
    assert torch.allclose(pooled, connector.net(x.mean(dim=1)), atol=1e-6)


def test_the_pooling_weights_are_trained():
    connector = AttentionPoolConnector(in_dim=8, out_dim=4)
    out = connector(torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.long))
    out.sum().backward()
    assert connector.query.grad is not None
    assert connector.query.grad.abs().sum() > 0


def test_a_mask_that_does_not_fit_the_batch_is_rejected():
    connector = AttentionPoolConnector(in_dim=8, out_dim=4)
    with pytest.raises(ValueError, match="does not match the batch"):
        connector(torch.randn(2, 5, 8), torch.ones(2, 4, dtype=torch.long))


def test_a_width_that_disagrees_with_the_config_is_rejected():
    connector = AttentionPoolConnector(in_dim=8, out_dim=4)
    with pytest.raises(ValueError, match="does not match the input width"):
        connector(torch.randn(2, 5, 16), torch.ones(2, 5, dtype=torch.long))


def test_the_binding_model_hands_the_mask_to_the_connector():
    import cogfm.anchor  # noqa: F401  (register)
    import cogfm.encoders  # noqa: F401  (register)
    from cogfm.binding.model import BindingModel
    from cogfm.registry import ANCHORS, ENCODERS

    torch.manual_seed(0)
    model = BindingModel(
        ENCODERS.build("precomputed", embed_dim=8, sequence=True),
        AttentionPoolConnector(in_dim=8, out_dim=16),
        ANCHORS.build("standin", dim=16, vocab_size=500),
    ).eval()

    short, long = torch.randn(3, 8), torch.randn(6, 8)
    padded, mask = _pad([short, long])
    together = model.encode_modality(padded, mask)
    alone = model.encode_modality(short.unsqueeze(0), torch.ones(1, 3, dtype=torch.long))

    assert together.shape == (2, 16)
    assert torch.allclose(together[0], alone[0], atol=1e-6), "the mask did not reach the connector"
