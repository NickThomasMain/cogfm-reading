"""Write embeddings that provably carry the sentence, as a control for the chain.

A negative result answers two questions at once and does not say which: either
the signal is not in the data, or the chain cannot find a signal that is there.
This script settles the second half. It writes a file in the same format the
real embeddings use, with the same trials in the same order, but with vectors
that are a known linear image of the sentence plus noise:

    vector(trial) = P . anchor(text of its sentence) + noise * N(0, 1)

``P`` is a fixed random projection from the anchor's width to the encoder's, so
the ideal connector is a linear map and an MLP finds it quickly. Nothing about
the trial other than its sentence enters the vector, so every reader of a
sentence receives the same signal and only the noise differs -- which is the
shape a perfect EEG encoder would produce.

Reading the result:

* At ``--noise 0`` the run has to land far above chance. If it does not, the
  fault is in the chain, not in the data, and the negative EEG result cannot be
  read until that is found.
* Raising the noise shows how much signal the chain needs before it stops
  finding anything, which puts a number on what the real result rules out.

The folds are item-disjoint, so the sentences in the test split never appear in
training. The projection is what makes the control fair: a connector cannot
memorise a lookup table, it has to learn the mapping and apply it to sentences
it has never seen.

Run, from the repo root:

    uv run python scripts/make_probe_embeddings.py --noise 0
    uv run python scripts/run_binding_eval.py --config-name eval_eeg \\
        data.embeddings=eeg_embeddings_probe_noise0.0.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

import cogfm.anchor  # noqa: F401  (registers the anchors)
from cogfm.registry import ANCHORS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/zuco/processed")
    parser.add_argument("--like", default="eeg_embeddings_all_average_egi62_mean.npz",
                        help="file whose trial layout is copied")
    parser.add_argument("--anchor", default="qwen3", help="config name under configs/anchor")
    parser.add_argument("--noise", type=float, default=1.0,
                        help="standard deviations of noise per unit of signal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    with np.load(root / args.like, allow_pickle=False) as store:
        subject, task = store["subject"], store["task"]
        sentence_id, n_patches = store["sentence_id"], store["n_patches"]
        width = int(store["vectors"].shape[1]) if "vectors" in store.files else 200

    sentences = json.loads((root / "sentences.json").read_text(encoding="utf-8"))
    text_by_id = {int(s["id"]): s["text"] for s in sentences}
    wanted = sorted({int(i) for i in sentence_id})

    cfg = OmegaConf.load(CONFIG_DIR / "anchor" / f"{args.anchor}.yaml")
    params = {k: v for k, v in cfg.items() if k not in ("name", "dim")}
    anchor = ANCHORS.build(cfg.name, dim=cfg.dim, **params)
    anchor.requires_grad_(False)
    with torch.no_grad():
        embedded = anchor([text_by_id[i] for i in wanted]).cpu().numpy().astype(np.float64)

    rng = np.random.default_rng(args.seed)
    projection = rng.normal(size=(embedded.shape[1], width)) / np.sqrt(embedded.shape[1])
    signal = embedded @ projection

    # Standardise per dimension so that "noise 1" means one standard deviation
    # of the signal, independent of the anchor's own scale.
    signal = (signal - signal.mean(axis=0)) / signal.std(axis=0).clip(min=1e-8)
    row_of = {sentence: row for row, sentence in enumerate(wanted)}

    vectors = signal[[row_of[int(i)] for i in sentence_id]]
    if args.noise > 0:
        vectors = vectors + rng.normal(scale=args.noise, size=vectors.shape)
    vectors = vectors.astype(np.float32)

    out = Path(args.out) if args.out else root / f"eeg_embeddings_probe_noise{args.noise}.npz"
    np.savez(
        out,
        vectors=vectors,
        subject=subject,
        task=task,
        sentence_id=sentence_id,
        n_patches=n_patches,
        reference=np.array(["probe"]),
        channels=np.array(["probe"]),
        pooling=np.array(["probe"]),
        too_long=np.array(["probe"]),
        checkpoint=np.array([str(cfg.name)]),
        probe_noise=np.array([args.noise]),
        sample_rate=np.array([0]),
    )
    print(f"{len(vectors)} trials over {len(wanted)} sentences, {width} dims, "
          f"noise {args.noise}")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
