"""A frozen decoder language model as the anchor space.

The text a signal has to be bound to is represented by the hidden states of the
model's top layer. Static token embeddings would be a lookup: the token ``bank``
would receive the same vector in every sentence, and a sentence vector built by
averaging them would be order-invariant, a bag of words that uses only the
embedding matrix and none of the model. Contextual states depend on the whole
sentence through self-attention and stay order-sensitive after pooling.

Pooling runs in two steps, sub-word to word and word to sentence, with one mean
each. The tokenizer reports the character span of every token, so a token
belongs to the word whose span it overlaps. Averaging over words rather than
over all sub-words makes every word count equally regardless of how many pieces
it was split into, and it matches the operation used on the modality side.

The last-token read-out is kept as an ablation. It corresponds to the point CLIP
reads its text feature from, and under causal attention that position has seen
the entire sequence, but it carries a recency bias that grows with input length.

Vectors are cached by text. The model is frozen and evaluated without dropout,
so a sentence always yields the same vector, and a corpus of a few hundred
sentences is encoded once no matter how many training steps read from it.

The backbone runs wherever it is fastest, but vectors always come back as
float32 on the cpu, where the trainable connector lives. Keeping the read-out
in float32 matters for a measuring instrument: reduced precision collapses
similarities that are close together into ties, and ties move ranks.
"""

from __future__ import annotations

import logging

import torch

from cogfm.anchor.base import AnchorEncoder
from cogfm.registry import ANCHORS

log = logging.getLogger(__name__)

POOLINGS = ("word_mean", "token_mean", "last_token")


def resolve_device(device: str) -> str:
    """Pick a device, preferring Apple silicon then CUDA when asked to choose."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@ANCHORS.register("qwen3")
class Qwen3Anchor(AnchorEncoder):
    """Frozen Qwen3 producing one vector per text.

    Args:
        dim: expected width; must match the model's hidden size.
        model_name: any causal language model on the Hugging Face hub.
        pooling: ``word_mean`` for the two-step hierarchy, ``token_mean`` for a
            flat mean over sub-words, ``last_token`` for the CLIP-style read-out.
        device: ``auto`` picks mps, then cuda, then cpu.
        max_length: texts longer than this are truncated.
        batch_size: texts encoded per forward pass.

    Raises:
        ValueError: on an unknown pooling name, or when ``dim`` disagrees with
            the loaded model.
    """

    def __init__(
        self,
        dim: int = 1024,
        model_name: str = "Qwen/Qwen3-0.6B",
        pooling: str = "word_mean",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 32,
        **_ignored,
    ) -> None:
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}; expected one of {POOLINGS}")

        from transformers import AutoModel, AutoTokenizer

        self.pooling = pooling
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = resolve_device(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        hidden = int(model.config.hidden_size)
        if dim != hidden:
            raise ValueError(
                f"anchor dim {dim} does not match {model_name} hidden size {hidden}; "
                f"set anchor.dim to {hidden}"
            )

        super().__init__(hidden)
        self.tokenizer = tokenizer
        # Checkpoints often ship in bfloat16; the read-out is cast up so the
        # similarities feeding the ranking keep full single precision.
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}
        log.info(
            "anchor %s on %s, pooling %s, hidden %d", model_name, self.device, pooling, hidden
        )

    def train(self, mode: bool = True) -> "Qwen3Anchor":
        """Keep the backbone in eval mode however the surrounding model is set.

        The anchor is a submodule of the binding model, so a call to ``train()``
        on the outer model would otherwise reach it and enable dropout, and the
        cache would hold vectors that a second run could not reproduce.
        """
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, texts: list[str]) -> torch.Tensor:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            for text, vector in zip(chunk, self._encode(chunk), strict=True):
                self._cache[text] = vector
        return torch.stack([self._cache[text] for text in texts])

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """Run the model over a chunk and pool each text into one vector."""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            states = self.model(**encoded).last_hidden_state

        attention = encoded["attention_mask"]
        if self.pooling == "last_token":
            last = attention.sum(dim=1) - 1
            return states[torch.arange(len(texts), device=self.device), last].cpu()
        if self.pooling == "token_mean":
            weights = (attention * _is_content(offsets).to(self.device)).unsqueeze(-1)
            return _weighted_mean(states, weights).cpu()

        pooled = [
            _pool_words(states[i], offsets[i], attention[i], texts[i]) for i in range(len(texts))
        ]
        return torch.stack(pooled).cpu()


def _is_content(offsets: torch.Tensor) -> torch.Tensor:
    """Mark tokens that cover characters; special tokens report an empty span."""
    return (offsets[..., 1] > offsets[..., 0]).long()


def _weighted_mean(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Character span of every whitespace-separated word."""
    spans = []
    position = 0
    for word in text.split():
        start = text.index(word, position)
        spans.append((start, start + len(word)))
        position = start + len(word)
    return spans


def _pool_words(
    states: torch.Tensor, offsets: torch.Tensor, attention: torch.Tensor, text: str
) -> torch.Tensor:
    """Mean over word vectors, each of which is a mean over its sub-words.

    A token belongs to a word when their character spans overlap, which handles
    both the leading space a byte-pair tokenizer keeps inside a token and the
    punctuation it splits off. Words that survive truncation contribute; a text
    whose words are all cut falls back to the mean over its content tokens.
    """
    keep = (attention.bool()) & (offsets[:, 1] > offsets[:, 0]).to(attention.device)
    if not bool(keep.any()):
        return states.mean(dim=0)

    starts = offsets[:, 0]
    ends = offsets[:, 1]
    word_vectors = []
    for word_start, word_end in _word_spans(text):
        overlap = keep & (starts < word_end).to(keep.device) & (ends > word_start).to(keep.device)
        if bool(overlap.any()):
            word_vectors.append(states[overlap].mean(dim=0))
    if not word_vectors:
        return states[keep].mean(dim=0)
    return torch.stack(word_vectors).mean(dim=0)
