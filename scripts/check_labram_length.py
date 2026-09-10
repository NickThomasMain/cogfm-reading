"""Does LaBraM's output follow the trial's content or its duration?

The encoder returns one vector per trial. If that vector is driven by how long
the trial is rather than by what is in it, a retrieval result would be explained
by sentence length instead of by what was read. This has to be measured before
the windowing and pooling decisions are made.

Measured on real ZuCo trials, not on noise. Trials long enough to be truncated
are encoded at several lengths, always from their start, and three similarities
are compared:

    A  same trial, different length        -> is a trial recognisable across lengths?
    B  different trial, same length        -> do trials cluster by duration?
    C  different trial, different length   -> the baseline both are judged against

Content wins when A clearly exceeds both B and C. Duration wins when B exceeds C
by a wide margin, because then sharing a length matters more than sharing content.

Everything is reported twice. **Raw** cosine similarity is dominated by the mean
vector all embeddings share -- a first run put every pair between 0.94 and 0.998,
which says more about that common component than about the trials. **Centred**
subtracts the mean over all vectors first, so what remains is the variance that
actually distinguishes trials. The centred block is the one to read.

Both pooling modes are shown side by side: the mean over patch tokens, and the
CLS token. A first run showed trials growing *more* alike as they grow longer
(0.940 at 4 s to 0.968 at 12 s), which is what averaging over more tokens does,
so the CLS token may hold up better.

Read only. Run:

    uv run python scripts/check_labram_length.py
    uv run python scripts/check_labram_length.py --n-trials 32 --task NR
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import torch

from cogfm.data.adapters.zuco_eeg import ZuCoEEGDataset
from cogfm.encoders.labram import PATCH_SAMPLES, LaBraMEncoder

LENGTHS = (4, 6, 8, 10, 12)  # patches, i.e. seconds at 200 Hz


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def similarities(vectors: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """The three comparisons, over every applicable pair."""
    same_trial_diff_length, diff_trial_same_length, diff_trial_diff_length = [], [], []
    n_trials = len(next(iter(vectors.values())))

    for short, long in combinations(LENGTHS, 2):
        a, b = _unit(vectors[short]), _unit(vectors[long])
        same_trial_diff_length.append((a * b).sum(axis=1))
        cross = a @ b.T
        off_diagonal = ~np.eye(n_trials, dtype=bool)
        diff_trial_diff_length.append(cross[off_diagonal])

    for length in LENGTHS:
        a = _unit(vectors[length])
        matrix = a @ a.T
        diff_trial_same_length.append(matrix[np.triu_indices(n_trials, k=1)])

    return {
        "A same trial, diff length": np.concatenate(same_trial_diff_length),
        "B diff trial, same length": np.concatenate(diff_trial_same_length),
        "C diff trial, diff length": np.concatenate(diff_trial_diff_length),
    }


def report(title: str, vectors: dict[int, np.ndarray]) -> tuple[float, float, float]:
    print(f"\n  {title}")
    stats = similarities(vectors)
    for name, values in stats.items():
        print(f"    {name}: median {np.median(values):+.4f}   "
              f"p10 {np.percentile(values, 10):+.4f}   p90 {np.percentile(values, 90):+.4f}")
    a, b, c = (float(np.median(stats[k])) for k in stats)
    print(f"    content margin  A - C = {a - c:+.4f}   (higher is better)")
    print(f"    length  margin  B - C = {b - c:+.4f}   (lower is better)")
    return a, b, c


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--task", default="SR")
    parser.add_argument("--root", default="data/zuco/processed")
    parser.add_argument("--reference", default="average")
    args = parser.parse_args()

    data = ZuCoEEGDataset(root=args.root, task=args.task, reference=args.reference,
                          resample_hz=200)
    encoder = LaBraMEncoder(channel_names=data.channel_names)
    encoder.eval()

    needed = max(LENGTHS) * PATCH_SAMPLES
    long_enough = np.flatnonzero(data.lengths() >= needed)
    if len(long_enough) < args.n_trials:
        raise SystemExit(f"only {len(long_enough)} trials reach {max(LENGTHS)} s")
    chosen = np.random.default_rng(0).choice(long_enough, size=args.n_trials, replace=False)
    print(f"{args.n_trials} trials, task {args.task}, reference '{args.reference}', "
          f"{len(data.channel_names)} channels")

    batches = {
        n: torch.stack([torch.from_numpy(data.eeg(int(i))[: n * PATCH_SAMPLES]) for i in chosen])
        for n in LENGTHS
    }

    summary = {}
    for pooling in ("mean", "cls"):
        encoder.pooling = pooling
        raw = {}
        for n, batch in batches.items():
            with torch.no_grad():
                raw[n] = encoder(batch).numpy()
        print(f"\n=== pooling: {pooling} ===")
        report("raw (dominated by the shared mean vector)", raw)

        stacked = np.concatenate([raw[n] for n in LENGTHS], axis=0)
        centre = stacked.mean(axis=0, keepdims=True)
        centred = {n: raw[n] - centre for n in LENGTHS}
        summary[pooling] = report("centred (read this one)", centred)

    print("\n=== verdict ===")
    for pooling, (a, b, c) in summary.items():
        content, length = a - c, b - c
        print(f"  {pooling:>4}: content margin {content:+.4f} | length margin {length:+.4f}")
    best = max(summary, key=lambda p: (summary[p][0] - summary[p][2]))
    a, b, c = summary[best]
    print(f"\n  strongest content margin: {best} pooling")
    if a - c > 0.10 and (b - c) < (a - c) / 2:
        print("  -> content clearly wins. Variable trial length is defensible;")
        print("     encode each trial at its own length, no padding, no cropping.")
    elif b - c >= a - c:
        print("  -> DURATION WINS: trials group by length rather than content.")
        print("     Fixed-length windows are required, and this is the justification.")
    else:
        print("  -> content leads but not decisively; note the margin in the thesis")
        print("     and keep trial duration as a control variable in the evaluation.")

    print("\n=== what padding costs (control, already known to be large) ===")
    encoder.pooling = "mean"
    short = LENGTHS[0]
    cut = short * PATCH_SAMPLES
    alone = batches[short]
    padded = torch.zeros(len(alone), 3 * cut, alone.shape[2])
    padded[:, :cut, :] = alone
    mask = torch.zeros(len(alone), 3 * cut)
    mask[:, :cut] = 1
    with torch.no_grad():
        a_vec, b_vec = encoder(alone).numpy(), encoder(padded, mask).numpy()
    similarity = (_unit(a_vec) * _unit(b_vec)).sum(axis=1)
    print(f"  {short} s alone vs padded to {3 * short} s: "
          f"median {np.median(similarity):+.4f}   min {similarity.min():+.4f}")


if __name__ == "__main__":
    main()
