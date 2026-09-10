"""What does the frozen LaBraM actually encode -- and can a variant do better?

A first measurement on all 7,664 embeddings was discouraging. Relative to a
baseline of two trials differing in both sentence and reader, the vectors carried:

    sentence signal  +0.077     the one we want
    reader signal    +0.452     six times stronger
    duration signal  +0.398     five times stronger

and centring per reader, then per duration, left the sentence signal at +0.003 --
so what looked like "same sentence" was almost entirely "same reading time".

This script asks whether a different read-out of the same frozen model does
better. Three knobs, measured against the same three signals:

    layer     the last block versus an earlier one. The final block of a
              foundation model is usually the most specialised to its pretraining
              objective, and LaBraM was pretrained on motor imagery, emotion,
              epilepsy and resting state -- never on reading. An earlier block may
              hold more general structure.
    pooling   mean over patch tokens versus the CLS token.
    centring  none, or per reader. Per-reader centring is already known to remove
              the reader component entirely and to raise the sentence signal.

Sampling is by SENTENCE, not by trial: all trials of a few hundred sentences are
taken, so that "same sentence, different reader" pairs are plentiful.

Read only. Run:

    uv run python scripts/probe_labram_variants.py
    uv run python scripts/probe_labram_variants.py --n-sentences 200 --layers 5 8 11
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cogfm.data.adapters.zuco_eeg import ZuCoEEGDataset
from cogfm.encoders.labram import EXPECTED_TIME_PATCHES, PATCH_SAMPLES, LaBraMEncoder


def build_pairs(sentence: np.ndarray, subject: np.ndarray, patches: np.ndarray, seed: int = 0):
    """Index pairs for the three comparisons plus the baseline."""
    rng = np.random.default_rng(seed)

    by_sentence: dict[int, list[int]] = {}
    for i, s in enumerate(sentence):
        by_sentence.setdefault(int(s), []).append(i)
    same_sentence = np.array(
        [(ix[a], ix[b]) for ix in by_sentence.values()
         for a in range(len(ix)) for b in range(a + 1, len(ix))]
    )

    def cross(groups, forbid_same_sentence=True):
        out = []
        for ix in groups:
            if len(ix) < 2:
                continue
            a = rng.choice(ix, size=min(2000, len(ix) * 4))
            b = rng.choice(ix, size=len(a))
            keep = a != b
            if forbid_same_sentence:
                keep &= sentence[a] != sentence[b]
            out.extend(zip(a[keep], b[keep]))
        return np.array(out)

    by_subject: dict[str, list[int]] = {}
    for i, s in enumerate(subject):
        by_subject.setdefault(str(s), []).append(i)
    same_subject = cross(by_subject.values())

    by_length: dict[int, list[int]] = {}
    for i, p in enumerate(patches):
        by_length.setdefault(int(p), []).append(i)
    same_length = np.array(
        [(a, b) for a, b in cross(by_length.values()) if subject[a] != subject[b]]
    )

    a = rng.integers(0, len(sentence), 60000)
    b = rng.integers(0, len(sentence), 60000)
    keep = (sentence[a] != sentence[b]) & (subject[a] != subject[b])
    baseline = np.c_[a[keep], b[keep]]
    return same_sentence, same_subject, same_length, baseline


def signals(vectors: np.ndarray, pairs, subject: np.ndarray, mode: str,
            patches: np.ndarray | None = None) -> tuple:
    """Signal strengths under one centring scheme.

    ``mode`` is "global", "reader" (subtract each reader's own mean) or
    "reader+length" (then subtract each duration group's mean as well). The last
    one is the decisive test: it removes everything the duration could explain,
    so whatever sentence signal survives is genuinely about content. On the last
    layer with mean pooling it collapsed the sentence signal from +0.103 to
    +0.003 -- if a variant holds up here, that is the real find.
    """
    v = vectors.astype(np.float64).copy()
    if mode in ("reader", "reader+length"):
        for s in set(subject.tolist()):
            index = np.flatnonzero(subject == s)
            v[index] -= v[index].mean(axis=0, keepdims=True)
    if mode == "reader+length":
        for p in set(patches.tolist()):
            index = np.flatnonzero(patches == p)
            if len(index) > 1:
                v[index] -= v[index].mean(axis=0, keepdims=True)
    v -= v.mean(axis=0, keepdims=True)
    v /= np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    median = lambda p: float(np.median((v[p[:, 0]] * v[p[:, 1]]).sum(axis=1)))
    same_sentence, same_subject, same_length, baseline = pairs
    base = median(baseline)
    return (median(same_sentence) - base, median(same_subject) - base,
            median(same_length) - base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/zuco/processed")
    parser.add_argument("--task", default=None)
    parser.add_argument("--reference", default="average")
    parser.add_argument("--n-sentences", type=int, default=150)
    parser.add_argument("--layers", type=int, nargs="*", default=[5, 8])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data = ZuCoEEGDataset(root=args.root, task=args.task, reference=args.reference,
                          resample_hz=200)
    patches_all = data.lengths() // PATCH_SAMPLES
    usable = np.flatnonzero((patches_all >= 1) & (patches_all <= EXPECTED_TIME_PATCHES))

    rng = np.random.default_rng(0)
    sentences = rng.choice(np.unique(data.sentence_ids[usable]),
                           size=args.n_sentences, replace=False)
    chosen = usable[np.isin(data.sentence_ids[usable], sentences)]
    sentence, subject = data.sentence_ids[chosen], data.subject_ids[chosen]
    patches = patches_all[chosen]
    print(f"{len(chosen)} trials from {args.n_sentences} sentences, "
          f"{len(set(subject.tolist()))} readers, reference {args.reference}")

    pairs = build_pairs(sentence, subject, patches)
    print(f"pairs: sentence {len(pairs[0])} | reader {len(pairs[1])} | "
          f"length {len(pairs[2])} | baseline {len(pairs[3])}\n")

    # Group by length once; every variant reuses the same grouping and slicing.
    groups: dict[int, list[int]] = {}
    for position, index in enumerate(chosen):
        groups.setdefault(int(patches[position]), []).append(position)

    print(f"{'layer':>6} {'pooling':>8} {'centring':>13} | {'sentence':>9} "
          f"{'reader':>8} {'duration':>9}")
    print("-" * 63)
    started = time.time()
    for layer in [None, *args.layers]:
        for pooling in ("mean", "cls"):
            encoder = LaBraMEncoder(channel_names=data.channel_names,
                                    pooling=pooling, layer=layer)
            encoder.eval()
            vectors = np.zeros((len(chosen), encoder.embed_dim), dtype=np.float32)
            for n_patches, positions in groups.items():
                cut = n_patches * PATCH_SAMPLES
                for start in range(0, len(positions), args.batch_size):
                    block = positions[start : start + args.batch_size]
                    batch = torch.stack(
                        [torch.from_numpy(data.eeg(int(chosen[p]))[:cut]) for p in block]
                    )
                    with torch.no_grad():
                        vectors[block] = encoder(batch).numpy()
            for mode in ("global", "reader", "reader+length"):
                s, r, d = signals(vectors, pairs, subject, mode, patches)
                name = "last" if layer is None else str(layer)
                print(f"{name:>6} {pooling:>8} {mode:>13} | {s:>+9.4f} {r:>+8.4f} {d:>+9.4f}")
            del encoder
    print(f"\n{time.time() - started:.0f} s")
    print("\nsentence higher is better; reader and duration lower is better.")
    print("The row that decides it is 'reader+length': it removes everything duration")
    print("could account for. On the full set that left the last layer at +0.003.")


if __name__ == "__main__":
    main()
