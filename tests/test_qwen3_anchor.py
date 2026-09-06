"""Tests for the Qwen3 anchor.

The pooling helpers are checked on their own, without loading a model. The
tests that need the real weights are skipped unless COGFM_ANCHOR_TESTS is set,
so a normal test run stays offline and fast.
"""

import os

import pytest
import torch

from cogfm.anchor.qwen3 import POOLINGS, _pool_words, _word_spans, resolve_device

WITH_MODEL = pytest.mark.skipif(
    not os.environ.get("COGFM_ANCHOR_TESTS"),
    reason="set COGFM_ANCHOR_TESTS=1 to download and run the real model",
)


def test_word_spans_cover_each_word():
    text = "Presents a good case"
    assert _word_spans(text) == [(0, 8), (9, 10), (11, 15), (16, 20)]
    for start, end in _word_spans(text):
        assert text[start:end] in text.split()


def test_word_spans_handle_repeated_words():
    """A naive search would map both occurrences to the first position."""
    text = "the cat and the dog"
    spans = _word_spans(text)
    assert spans[0] != spans[3]
    assert text[spans[3][0] : spans[3][1]] == "the"


def test_word_spans_tolerate_extra_whitespace():
    assert _word_spans("  a   bb ") == [(2, 3), (6, 8)]


def test_pooling_averages_over_words_not_sub_words():
    """A word split into many pieces must not outweigh a word that stayed whole."""
    # "aa" occupies characters 0..2 as two tokens, "b" occupies 3..4 as one.
    states = torch.tensor([[0.0], [2.0], [10.0]])
    offsets = torch.tensor([[0, 1], [1, 2], [3, 4]])
    attention = torch.tensor([1, 1, 1])
    pooled = _pool_words(states, offsets, attention, "aa b")
    # words average to 1.0 and 10.0, the sentence to 5.5
    assert pooled.item() == pytest.approx(5.5)
    # a flat mean over sub-words would have given 4.0
    assert pooled.item() != pytest.approx(4.0)


def test_pooling_ignores_special_tokens():
    states = torch.tensor([[100.0], [1.0], [3.0]])
    offsets = torch.tensor([[0, 0], [0, 1], [2, 3]])  # first token covers nothing
    attention = torch.tensor([1, 1, 1])
    assert _pool_words(states, offsets, attention, "a b").item() == pytest.approx(2.0)


def test_pooling_ignores_padding():
    states = torch.tensor([[1.0], [3.0], [99.0]])
    offsets = torch.tensor([[0, 1], [2, 3], [0, 0]])
    attention = torch.tensor([1, 1, 0])
    assert _pool_words(states, offsets, attention, "a b").item() == pytest.approx(2.0)


def test_pooling_falls_back_when_no_word_survives():
    states = torch.tensor([[4.0], [6.0]])
    offsets = torch.tensor([[0, 0], [0, 0]])
    attention = torch.tensor([1, 1])
    assert _pool_words(states, offsets, attention, "a b").item() == pytest.approx(5.0)


def test_device_resolution_passes_explicit_choices_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "mps", "cuda")


def test_pooling_names_are_the_documented_ones():
    assert POOLINGS == ("word_mean", "token_mean", "last_token")


@WITH_MODEL
def test_unknown_pooling_is_rejected():
    from cogfm.anchor.qwen3 import Qwen3Anchor

    with pytest.raises(ValueError, match="unknown pooling"):
        Qwen3Anchor(pooling="nonsense")


@WITH_MODEL
def test_a_wrong_dim_is_rejected():
    from cogfm.anchor.qwen3 import Qwen3Anchor

    with pytest.raises(ValueError, match="hidden size"):
        Qwen3Anchor(dim=128)


@WITH_MODEL
def test_shape_and_determinism():
    from cogfm.anchor.qwen3 import Qwen3Anchor

    anchor = Qwen3Anchor(device="cpu")
    texts = ["The cat sat on the mat.", "Quarterly revenue exceeded expectations."]
    first = anchor(texts)
    assert first.shape == (2, anchor.dim)
    assert torch.allclose(first, anchor(texts))


@WITH_MODEL
def test_pooling_is_order_sensitive():
    """The reason for contextual states: a bag of words would score these equal."""
    from cogfm.anchor.qwen3 import Qwen3Anchor

    anchor = Qwen3Anchor(device="cpu")
    vectors = anchor(["the dog bit the man", "the man bit the dog"])
    similarity = torch.nn.functional.cosine_similarity(vectors[0], vectors[1], dim=0)
    assert similarity < 0.999


@WITH_MODEL
def test_related_sentences_are_closer_than_unrelated_ones():
    """The property the whole binding idea rests on."""
    from cogfm.anchor.qwen3 import Qwen3Anchor

    anchor = Qwen3Anchor(device="cpu")
    vectors = torch.nn.functional.normalize(
        anchor(
            [
                "The stock market fell sharply on Monday.",
                "Share prices dropped steeply at the start of the week.",
                "She baked a loaf of rye bread.",
            ]
        ),
        dim=-1,
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert related > unrelated

@WITH_MODEL
def test_vectors_come_back_as_cpu_float32():
    """The trainable side runs in float32 on the cpu and must not have to cast."""
    from cogfm.anchor.qwen3 import Qwen3Anchor

    anchor = Qwen3Anchor()
    out = anchor(["A short sentence."])
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


@WITH_MODEL
def test_the_backbone_stays_in_eval_mode():
    """A train() call on the outer model must not reach the frozen anchor."""
    from cogfm.anchor.qwen3 import Qwen3Anchor

    anchor = Qwen3Anchor(device="cpu")
    anchor.train()
    assert not anchor.model.training
    texts = ["Determinism must survive a train call."]
    anchor._cache.clear()
    first = anchor(texts).clone()
    anchor._cache.clear()
    assert torch.allclose(first, anchor(texts))
