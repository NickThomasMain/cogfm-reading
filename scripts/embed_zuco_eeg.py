"""Encode every ZuCo trial with the frozen LaBraM once and store the vectors.

The encoder is frozen, so a trial's vector never changes. Recomputing it in
every training step would run the expensive part hundreds of thousands of times
for the same result; this script runs it once and writes the vectors to disk.
The connector then trains on the stored vectors and LaBraM is not needed again.

That also settles the padding problem. Measured on real trials, a padded trial
resembles its own unpadded self LESS than two entirely different trials resemble
each other (0.864 against 0.947) -- LaBraM has no time mask, so the filler zeros
reach the attention layers. Trials must therefore never be padded.

Here they are not. LaBraM works in one-second patches, so after truncation to
whole patches only sixteen distinct lengths exist. Trials are grouped by that
length and each group is encoded as its own batch: every trial in a batch has
exactly the same length, so no padding is needed and batching stays efficient.

Truncating to whole patches costs the tail of a trial -- half a second in the
median, just under one at worst. That is unavoidable: the model has no finer
time resolution.

Trials beyond the checkpoint's 16 patches need a decision that is deliberately
NOT made here. ``--too-long skip`` (default) leaves them out and reports how
many; ``--too-long crop`` keeps their first 16 seconds. Whichever is chosen ends
up in the output file's metadata.

``--pooling time`` writes something different: one vector per one-second patch
instead of one per trial, so the trial stays a sequence and the pooling can be
learned downstream instead of being fixed here. Averaging a five-second trial
into a single vector cannot preserve anything that sits in one of its seconds,
which is the reason to keep the option open.

Read only apart from the output file. Run:

    uv run python scripts/embed_zuco_eeg.py
    uv run python scripts/embed_zuco_eeg.py --pooling time
    uv run python scripts/embed_zuco_eeg.py --reference recording --too-long crop
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cogfm.data.adapters.zuco_eeg import ZuCoEEGDataset
from cogfm.encoders.labram import (
    EXPECTED_TIME_PATCHES,
    PATCH_SAMPLES,
    LaBraMEncoder,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/zuco/processed")
    parser.add_argument("--task", default=None, help="SR, NR, or both when omitted")
    parser.add_argument("--reference", default="average", choices=("average", "recording"))
    parser.add_argument("--channels", default="egi62", choices=("egi62", "named69", "all"))
    parser.add_argument("--pooling", default="mean", choices=("mean", "cls", "time"))
    parser.add_argument("--too-long", default="skip", choices=("skip", "crop"),
                        help="trials beyond the checkpoint's 16 patches")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data = ZuCoEEGDataset(root=args.root, task=args.task, reference=args.reference,
                          channels=args.channels, resample_hz=200)
    encoder = LaBraMEncoder(channel_names=data.channel_names, pooling=args.pooling)
    encoder.eval()

    lengths = data.lengths()
    patches = lengths // PATCH_SAMPLES

    too_short = int((patches < 1).sum())
    too_long = int((patches > EXPECTED_TIME_PATCHES).sum())
    keep = patches >= 1
    if args.too_long == "skip":
        keep &= patches <= EXPECTED_TIME_PATCHES
    effective = np.minimum(patches, EXPECTED_TIME_PATCHES)

    print(f"{len(data)} trials | task {args.task or 'SR+NR'} | reference {args.reference} | "
          f"{len(data.channel_names)} channels | pooling {args.pooling}")
    print(f"  shorter than one patch, dropped : {too_short}")
    print(f"  longer than {EXPECTED_TIME_PATCHES} patches            : {too_long}"
          f"  ({'dropped' if args.too_long == 'skip' else 'cropped to 16 s'})")
    print(f"  encoding                        : {int(keep.sum())}\n")

    groups: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(keep):
        groups[int(effective[index])].append(int(index))

    # "time" pooling returns one vector per one-second patch, so a trial is a
    # sequence rather than a point and the trials differ in length. They are
    # collected per trial here and stored flat with offsets, which keeps the
    # file free of padding: what the connector later masks is padding added
    # inside a batch, never padding baked into the data.
    sequence = args.pooling == "time"
    pieces: dict[int, np.ndarray] = {}
    vectors = np.zeros((len(data), encoder.embed_dim), dtype=np.float32)

    done, started = 0, time.time()
    for n_patches in sorted(groups):
        indices = groups[n_patches]
        cut = n_patches * PATCH_SAMPLES
        for start in range(0, len(indices), args.batch_size):
            chunk = indices[start : start + args.batch_size]
            # Every trial in this batch has exactly n_patches -- no padding, no mask.
            batch = torch.stack([torch.from_numpy(data.eeg(i)[:cut]) for i in chunk])
            with torch.no_grad():
                encoded = encoder(batch).numpy()
            if sequence:
                for row, index in enumerate(chunk):
                    pieces[index] = encoded[row]
            else:
                vectors[chunk] = encoded
            done += len(chunk)
        print(f"  {n_patches:2d} s : {len(indices):5d} trials   ({done}/{int(keep.sum())}, "
              f"{time.time() - started:.0f} s)")

    order = np.flatnonzero(keep)
    if sequence:
        lengths = np.array([len(pieces[int(i)]) for i in order], dtype=np.int64)
        payload = {
            "tokens": np.concatenate([pieces[int(i)] for i in order], axis=0),
            "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        }
        finite = np.isfinite(payload["tokens"]).all()
    else:
        payload = {"vectors": vectors[keep]}
        finite = np.isfinite(payload["vectors"]).all()
    if not finite:
        raise SystemExit("non-finite vectors produced; refusing to write")

    out = Path(args.out) if args.out else Path(args.root) / (
        f"eeg_embeddings_{args.task or 'all'}_{args.reference}_{args.channels}_"
        f"{args.pooling}.npz"
    )
    np.savez(
        out,
        **payload,
        subject=data.subject_ids[keep],
        task=data.task_ids[keep],
        sentence_id=data.sentence_ids[keep],
        n_patches=effective[keep],
        # Metadata, so a stored file can never be mistaken for another setting.
        reference=np.array([args.reference]),
        channels=np.array([args.channels]),
        channel_names=np.array(encoder.channel_names),
        pooling=np.array([args.pooling]),
        too_long=np.array([args.too_long]),
        checkpoint=np.array([encoder.model.__class__.__name__]),
        sample_rate=np.array([200]),
    )
    size = out.stat().st_size / 1e6
    shape = "sequences" if sequence else "vectors"
    print(f"\nwrote {out}  ({int(keep.sum())} {shape}, {encoder.embed_dim} dims, {size:.1f} MB)")
    print(f"total {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
