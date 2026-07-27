"""Lightweight placeholder anchor.

NOT a language model — it splits text into words, maps each word to a stable id
via a content hash, looks up an embedding, and mean-pools per sample. This mimics
the anchor interface (text in, one pooled vector per sample out), so a frozen LLM
anchor (e.g. Qwen3) can drop in behind the same class without changing the rest
of the pipeline.

The hash is content-based (hashlib), so it is stable across processes — unlike
Python's built-in hash(), which is randomised per run.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn

from cogfm.anchor.base import AnchorEncoder
from cogfm.registry import ANCHORS


@ANCHORS.register("standin")
class StandInAnchor(AnchorEncoder):
    def __init__(self, dim: int = 128, vocab_size: int = 16384) -> None:
        super().__init__(dim)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)

    def _word_id(self, word: str) -> int:
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.vocab_size

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.embed.weight.device
        pooled = []
        for text in texts:
            words = text.split()
            if not words:  # empty string -> a single id-0 token
                ids = torch.zeros(1, dtype=torch.long, device=device)
            else:
                ids = torch.tensor(
                    [self._word_id(w) for w in words],
                    dtype=torch.long,
                    device=device,
                )
            pooled.append(self.embed(ids).mean(dim=0))
        return torch.stack(pooled, dim=0)  # (B, dim)
