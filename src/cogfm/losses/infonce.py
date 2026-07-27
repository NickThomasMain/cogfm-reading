"""Symmetric InfoNCE (CLIP-style) contrastive loss.

Both embeddings are L2-normalised, then compared as a scaled cosine-similarity
matrix. The loss pushes each modality embedding towards its matching text
embedding (the diagonal) and away from the other texts in the batch, in both
directions (modality->text and text->modality).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cogfm.losses.base import BindingLoss
from cogfm.registry import LOSSES


@LOSSES.register("infonce")
class InfoNCELoss(BindingLoss):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, modality: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = F.normalize(modality, dim=-1)
        a = F.normalize(anchor, dim=-1)
        logits = (m @ a.t()) / self.temperature  # (B, B) cosine similarities / T
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, targets)      # modality -> text
            + F.cross_entropy(logits.t(), targets)  # text -> modality
        )
        return loss, logits
