"""LaBraM as a frozen EEG encoder.

Wraps ``braindecode.models.Labram`` and translates our data into its dialect.
The adapter already delivers clean ZuCo data — selected channels, chosen
reference, 200 Hz, axes (time, channels). Four things are still LaBraM's own
conventions rather than properties of the data, and they live here:

1. **Axes.** We store every modality as (time, features); braindecode wants
   (batch, channels, time). Transposed on the way in.
2. **Unit.** The paper normalises "by setting the unit to 0.1 mV", i.e. values
   in microvolts divided by 100. Our data has std ~4.4 uV, so this lands near
   0.04 -- well inside the range the model saw.
3. **Channel identity.** LaBraM looks every channel name up in its own
   ``LABRAM_CHANNEL_ORDER`` and uses the position as an index into a learned
   table of electrode locations. Our 10-10 names come from the adapter; the
   lookup is done by braindecode when ``ch_names`` is passed to ``forward``.
4. **Length.** The published checkpoint carries a learned absolute time
   embedding with 16 rows, one per one-second patch of 200 samples, so 16
   seconds. braindecode does NOT raise on longer input -- it silently clamps
   with ``min(...)`` -- so this wrapper checks the length itself.

Two checkpoints are in circulation and they differ. ``from_pretrained`` gives
128 channel positions and 16 time patches; the ``.pt`` file named in
braindecode's docstring example gives 64 and 8. Only the former is intended
here, which is why the expected shapes are asserted after loading.

Length: LaBraM handles varying lengths -- the original reads the patch count off
the input -- but braindecode's port hard-codes it to a construction-time
constant. The forward pass works around that; see the comment there. Within one
batch all trials still share a length, because a batch is one tensor.

Padding: batches mix trial lengths, and LaBraM has no notion of a time mask.
Padded samples would therefore enter the patch embedding as if they were
signal. The wrapper pools over patch tokens and drops the patches that lie
entirely inside the padding, so a padded tail cannot reach the output. Patches
that straddle the boundary are kept; at one second per patch the contamination
is bounded by the padding within a single patch.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import torch

from cogfm.encoders.base import ModalityEncoder
from cogfm.registry import ENCODERS

log = logging.getLogger(__name__)

MONTAGE_FILE = Path(__file__).resolve().parents[1] / "data" / "zuco_montage.json"

DEFAULT_CHECKPOINT = "braindecode/labram-pretrained"

# What the intended checkpoint must carry, verified against the loaded weights
# on 10.09.2026: position (1, 129, 200), temporal (1, 16, 200).
EXPECTED_POSITIONS = 129
EXPECTED_TIME_PATCHES = 16

PATCH_SAMPLES = 200  # braindecode default patch_size; one second at 200 Hz
UNIT_DIVISOR = 100.0  # microvolts -> the paper's 0.1 mV unit


def montage_names(selection: str = "egi62", legacy_names: bool = False) -> list[str]:
    """The 10-10 names of one channel selection, in column order.

    Mirrors ``ZuCoEEGDataset``; pass the dataset's ``channel_names`` instead
    whenever they are at hand, so the two cannot drift apart.
    """
    montage = json.loads(MONTAGE_FILE.read_text(encoding="utf-8"))
    entries = sorted(montage["channels"], key=lambda e: e["column"])
    if selection == "egi62":
        entries = [e for e in entries if e["within_egi_threshold"]]
    elif selection != "named69":
        raise ValueError(f"selection must be 'egi62' or 'named69', got {selection!r}")
    return [e["legacy"] if legacy_names and e["legacy"] else e["name"] for e in entries]


@ENCODERS.register("labram")
class LaBraMEncoder(ModalityEncoder):
    """Frozen LaBraM over sentence-level EEG.

    Args:
        embed_dim: width of the returned vector; LaBraM base uses 200.
        channel_names: 10-10 name per input channel, aligned with the feature
            axis. Falls back to ``montage_names(channels)`` when None.
        channels: which montage selection to fall back to.
        legacy_names: use T3/T4/T5/T6 instead of T7/T8/P7/P8.
        checkpoint: Hugging Face id passed to ``Labram.from_pretrained``.
        max_patches: refuse inputs longer than this many one-second patches.
        pooling: ``"mean"`` over valid patch tokens, ``"cls"``, or ``"time"``.
            ``"time"`` averages the channel axis but keeps the time axis, so it
            returns one vector per one-second patch, shape (batch, patches, dim),
            and the caller decides how to pool. The other two return (batch, dim).
        layer: read the representation off this transformer block instead
            of the last one. The final block of a foundation model is often
            the most specialised to its pretraining objective, so an earlier
            one can carry more general information. None uses the model's
            own output. 0-based; the checkpoint has 12 blocks.

    Raises:
        ImportError: if braindecode is not installed.
        ValueError: on a checkpoint with unexpected table sizes, on a channel
            count that disagrees with the names, or on input beyond the length
            the checkpoint's time embedding covers.
    """

    def __init__(
        self,
        embed_dim: int = 200,
        channel_names: list[str] | None = None,
        channels: str = "egi62",
        legacy_names: bool = False,
        checkpoint: str = DEFAULT_CHECKPOINT,
        max_patches: int = EXPECTED_TIME_PATCHES,
        pooling: str = "mean",
        layer: int | None = None,
    ) -> None:
        super().__init__(embed_dim)
        if pooling not in ("mean", "cls", "time"):
            raise ValueError(f"pooling must be 'mean', 'cls' or 'time', got {pooling!r}")

        try:
            from braindecode.models import Labram
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "the labram encoder needs braindecode; run `uv add 'braindecode[hub]'`"
            ) from exc

        self.channel_names = list(channel_names) if channel_names else montage_names(
            channels, legacy_names
        )
        self.pooling = pooling
        self.max_patches = max_patches

        model = Labram.from_pretrained(checkpoint)
        self._check_checkpoint(model, checkpoint)
        self._install_length_fix(model)

        self.layer = layer
        self._captured: torch.Tensor | None = None
        if layer is not None:
            if not 0 <= layer < len(model.blocks):
                raise ValueError(
                    f"layer must be in 0..{len(model.blocks) - 1}, got {layer}"
                )

            def capture(_module, _inputs, output):
                self._captured = output

            model.blocks[layer].register_forward_hook(capture)

        # Frozen: no gradients, and eval mode kept even when the outer model
        # is switched to train, so dropout and norm statistics stay fixed.
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

        log.info(
            "labram: %s, %d channels, up to %d patches (%d s)",
            checkpoint, len(self.channel_names), max_patches, max_patches,
        )

    @staticmethod
    def _install_length_fix(model) -> None:
        """Restore the original's handling of varying input length.

        braindecode ignores the actual patch count in two places and both must
        be corrected, or the position and time embeddings are built for a
        different length than the input has:

        * ``_adj_position_embedding`` reads ``patch_embed[0].n_patchs``, fixed
          when the model was constructed. Set per call in ``forward``.
        * ``_adj_temporal_embedding`` computes
          ``min(dim_embed, temporal_embedding.shape[1] - 1)``, but the caller
          passes the embedding WIDTH (200) as ``dim_embed``, so the result is
          always ``shape[1] - 1`` -- 15 for the published checkpoint. Replaced
          here.

        The replacement is the original line, verbatim in effect
        (``modeling_finetune.py``: ``self.time_embed[:, 0:input_time_window, :]
        .unsqueeze(1).expand(batch_size, nc, -1, -1).flatten(1, 2)``), with
        ``input_time_window`` taken from the batch being encoded.
        """

        def _adj_temporal_embedding(model_self, num_ch, batch_size, dim_embed=None):
            n_patches = getattr(model_self, "_cogfm_n_patches", None)
            if n_patches is None:
                raise RuntimeError(
                    "the labram encoder must set _cogfm_n_patches before the forward pass"
                )
            embedding = model_self.temporal_embedding[:, 0:n_patches, :]
            return (
                embedding.unsqueeze(1)
                .expand(batch_size, num_ch, -1, -1)
                .flatten(1, 2)
            )

        model._adj_temporal_embedding = types.MethodType(_adj_temporal_embedding, model)

    @staticmethod
    def _check_checkpoint(model, checkpoint: str) -> None:
        """Guard against the smaller 64-channel / 8-second checkpoint."""
        positions = model.position_embedding.shape[1]
        patches = model.temporal_embedding.shape[1]
        if positions != EXPECTED_POSITIONS or patches != EXPECTED_TIME_PATCHES:
            raise ValueError(
                f"{checkpoint} carries position_embedding {positions} and "
                f"temporal_embedding {patches}; expected {EXPECTED_POSITIONS} and "
                f"{EXPECTED_TIME_PATCHES}. The .pt file from braindecode's docstring "
                "example is the 64-channel / 8-second variant -- use from_pretrained."
            )

    def train(self, mode: bool = True):  # noqa: D102 - keep the backbone frozen
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, n_times, n_channels = x.shape
        if n_channels != len(self.channel_names):
            raise ValueError(
                f"{n_channels} channels in the batch but {len(self.channel_names)} names; "
                "the encoder's channel selection must match the adapter's"
            )

        n_patches = n_times // PATCH_SAMPLES
        if n_patches > self.max_patches:
            raise ValueError(
                f"{n_times} samples are {n_patches} patches of {PATCH_SAMPLES}; the "
                f"checkpoint's time embedding covers {self.max_patches}. braindecode would "
                "clamp this silently -- shorten, crop or window the trial before encoding."
            )
        if n_patches == 0:
            raise ValueError(f"{n_times} samples are shorter than one {PATCH_SAMPLES}-sample patch")

        usable = n_patches * PATCH_SAMPLES
        signal = x[:, :usable, :].transpose(1, 2) / UNIT_DIVISOR  # (B, C, T)

        # braindecode bug (labram.py, _adj_position_embedding): it expands the
        # position embedding to `patch_embed[0].n_patchs`, a constant fixed when
        # the model was built, while the segmentation itself works off the actual
        # input. The original derives the count from the input shape instead
        # (`input_time_window = a if t == self.patch_size else t`,
        # modeling_finetune.py). Setting the attribute per call restores that and
        # is what allows batches of differing length. Only read in that one place
        # after construction, so this has no other effect.
        self.model.patch_embed[0].n_patchs = n_patches
        self.model._cogfm_n_patches = n_patches

        with torch.no_grad():
            out = self.model(
                signal, ch_names=self.channel_names, return_features=True
            )
        if self.layer is None:
            tokens, cls_token = out["features"], out["cls_token"]
        else:
            # The hook holds this block's output, still carrying the CLS token at
            # position 0. The model's final norm is applied so the scale matches
            # what the last block's output would have.
            captured = self.model.norm(self._captured)
            tokens, cls_token = captured[:, 1:, :], captured[:, 0, :]

        if self.pooling == "cls":
            return cls_token

        # tokens are (B, C * P, D), channel-major: all patches of channel 0 first.
        tokens = tokens.reshape(batch, n_channels, n_patches, -1)

        if self.pooling == "time":
            # Keep the time axis and average only over channels. Nothing is
            # dropped here even under a mask: a caller asking for the sequence
            # gets every patch and decides itself which ones count, which is
            # what a trainable pooling layer downstream needs.
            return tokens.mean(dim=1)

        weights = self._patch_weights(mask, batch, n_patches, x.device, tokens.dtype)
        weights = weights.view(batch, 1, n_patches, 1)
        total = (tokens * weights).sum(dim=(1, 2))
        return total / (n_channels * weights.sum(dim=(1, 2, 3)).clamp(min=1e-6)).unsqueeze(1)

    @staticmethod
    def _patch_weights(mask, batch, n_patches, device, dtype) -> torch.Tensor:
        """1 for patches holding real signal, 0 for those inside the padding."""
        if mask is None:
            return torch.ones(batch, n_patches, device=device, dtype=dtype)
        lengths = mask.to(torch.long).sum(dim=1)
        # A patch counts when it starts before the trial ends.
        starts = torch.arange(n_patches, device=device) * PATCH_SAMPLES
        keep = starts.unsqueeze(0) < lengths.unsqueeze(1)
        return keep.to(dtype)
