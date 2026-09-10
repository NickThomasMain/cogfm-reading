"""Extract sentence texts and eye-tracking scanpaths from the ZuCo 1.0 MATLAB files.

The distributed ``results<SUBJECT>_<TASK>.mat`` files carry the full recording,
including per-word EEG and band powers, and take tens of seconds each to open.
Only two fields are needed to bind a scanpath to the text that produced it: the
sentence string and the fixation sequence. This script pulls out those two and
writes them to a compact form that loads in milliseconds.

Extraction runs per source file and caches its result, so an interrupted run
resumes instead of starting over. The merge step combines the caches, writes the
two output files, and reports the verification checks.

Usage:
    python scripts/extract_zuco_et.py --extract      # process source files (resumable)
    python scripts/extract_zuco_et.py --merge        # combine caches, write output, verify
    python scripts/extract_zuco_et.py --extract --merge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

# Trials with fewer fixations than this carry too little signal to encode and are
# dropped. Trials with no eye-tracking data at all fall out through the same rule.
MIN_FIXATIONS = 5

# Fixation features, in the column order used throughout the project.
FEATURES = ("x", "y", "duration")

TASKS = {"SR": "task1_SR", "NR": "task2_NR"}


def source_files(data_root: Path) -> list[tuple[str, str, Path]]:
    """Locate every per-subject MATLAB file, as (subject, task, path)."""
    found = []
    for task, folder in TASKS.items():
        directory = data_root / folder / "Matlab files"
        if not directory.is_dir():
            raise SystemExit(f"missing directory: {directory}")
        for path in sorted(directory.glob(f"results*_{task}.mat")):
            subject = path.stem.replace("results", "").replace(f"_{task}", "")
            found.append((subject, task, path))
    return found


def extract_one(path: Path) -> dict:
    """Read one source file into plain arrays.

    Returns the sentence strings in file order, and for every trial that clears
    the fixation threshold its sentence position plus its fixation array.
    """
    raw = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["sentenceData"]
    raw = np.atleast_1d(raw)

    texts: list[str] = []
    positions: list[int] = []
    scanpaths: list[np.ndarray] = []

    for position, entry in enumerate(raw):
        text = getattr(entry, "content", "")
        texts.append(text.strip() if isinstance(text, str) else "")

        fixations = getattr(entry, "allFixations", None)
        if fixations is None or not hasattr(fixations, "x"):
            continue
        columns = [np.atleast_1d(getattr(fixations, name)).astype(np.float32) for name in FEATURES]
        if len(columns[0]) < MIN_FIXATIONS:
            continue
        scanpaths.append(np.stack(columns, axis=1))
        positions.append(position)

    return {"texts": texts, "positions": positions, "scanpaths": scanpaths}


def cache_path(cache_dir: Path, subject: str, task: str) -> Path:
    return cache_dir / f"{subject}_{task}.npz"


def run_extract(data_root: Path, cache_dir: Path) -> None:
    """Extract every source file that has no cache entry yet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subject, task, path in source_files(data_root):
        target = cache_path(cache_dir, subject, task)
        if target.exists():
            print(f"  skip    {subject}_{task}")
            continue
        result = extract_one(path)
        lengths = np.array([len(s) for s in result["scanpaths"]], dtype=np.int64)
        stacked = (
            np.concatenate(result["scanpaths"], axis=0)
            if result["scanpaths"]
            else np.zeros((0, len(FEATURES)), dtype=np.float32)
        )
        np.savez_compressed(
            target,
            texts=np.array(result["texts"], dtype=object),
            positions=np.array(result["positions"], dtype=np.int64),
            lengths=lengths,
            fixations=stacked,
        )
        print(f"  done    {subject}_{task}  {len(result['scanpaths'])} trials")


def load_caches(cache_dir: Path) -> list[dict]:
    """Read every cache entry, newest state of the extraction."""
    entries = []
    for path in sorted(cache_dir.glob("*.npz")):
        subject, task = path.stem.rsplit("_", 1)
        with np.load(path, allow_pickle=True) as data:
            entries.append(
                {
                    "subject": subject,
                    "task": task,
                    "texts": [str(t) for t in data["texts"]],
                    "positions": data["positions"],
                    "lengths": data["lengths"],
                    "fixations": data["fixations"],
                }
            )
    return entries


def build_sentence_table(entries: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Assign a stable id to every distinct sentence.

    Ids run task by task and follow the presentation order within a task, which
    is identical across subjects. Sentence text is used as the identity key, so
    a repeated text across tasks would collapse two stimuli into one; the caller
    verifies that this does not happen.
    """
    by_task: dict[str, list[str]] = {}
    for entry in entries:
        known = by_task.setdefault(entry["task"], entry["texts"])
        if entry["texts"] != known:
            raise SystemExit(
                f"sentence list of {entry['subject']}_{entry['task']} differs from the others"
            )

    sentences: list[dict] = []
    index: dict[str, int] = {}
    for task in sorted(by_task):
        for text in by_task[task]:
            if not text:
                continue
            if text in index:
                raise SystemExit(f"sentence text is not unique across tasks: {text[:60]!r}")
            index[text] = len(sentences)
            sentences.append(
                {
                    "id": len(sentences),
                    "text": text,
                    "task": task,
                    "n_words": len(text.split()),
                }
            )
    return sentences, index


def build_trials(entries: list[dict], index: dict[str, int]) -> dict:
    """Flatten all cached trials into one fixation array plus an offset table."""
    subjects: list[str] = []
    tasks: list[str] = []
    sentence_ids: list[int] = []
    lengths: list[int] = []
    blocks: list[np.ndarray] = []

    for entry in entries:
        cursor = 0
        for position, length in zip(entry["positions"], entry["lengths"], strict=True):
            block = entry["fixations"][cursor : cursor + length]
            cursor += length
            subjects.append(entry["subject"])
            tasks.append(entry["task"])
            sentence_ids.append(index[entry["texts"][position]])
            lengths.append(int(length))
            blocks.append(block)

    lengths_array = np.array(lengths, dtype=np.int64)
    return {
        "subject": np.array(subjects),
        "task": np.array(tasks),
        "sentence_id": np.array(sentence_ids, dtype=np.int64),
        "offsets": np.concatenate([[0], np.cumsum(lengths_array)]).astype(np.int64),
        "fixations": np.concatenate(blocks, axis=0).astype(np.float32),
    }


def trial_view(trials: dict, i: int) -> np.ndarray:
    """Return trial i as an (n_fixations, 3) array."""
    start, end = trials["offsets"][i], trials["offsets"][i + 1]
    return trials["fixations"][start:end]


def report_counts(entries: list[dict], sentences: list[dict], trials: dict) -> None:
    print("\n--- 1. Zaehlabgleich ---")
    print(f"{'Proband':10s} {'Task':>5s} {'nutzbar':>9s}")
    for entry in sorted(entries, key=lambda e: (e["task"], e["subject"])):
        print(f"{entry['subject']:10s} {entry['task']:>5s} {len(entry['positions']):9d}")
    print(f"\n  verschiedene Saetze : {len(sentences)}")
    print(f"  Trials gesamt       : {len(trials['sentence_id'])}")
    print(f"  Fixationen gesamt   : {len(trials['fixations'])}")

    pairs = set(zip(trials["subject"], trials["sentence_id"], strict=True))
    print(f"  eindeutige (Proband, Satz)-Paare: {len(pairs)}", end="  ")
    print("OK" if len(pairs) == len(trials["sentence_id"]) else "FEHLER: Duplikate")


def report_sample_check(
    data_root: Path, entries: list[dict], trials: dict, n_files: int, rng: np.random.Generator
) -> None:
    """Reopen a few source files and compare their trials against the output."""
    print(f"\n--- 2. Stichprobe gegen die Quelldateien ({n_files} Dateien) ---")
    available = source_files(data_root)
    chosen = [available[i] for i in rng.choice(len(available), n_files, replace=False)]

    for subject, task, path in chosen:
        raw = np.atleast_1d(
            sio.loadmat(path, squeeze_me=True, struct_as_record=False)["sentenceData"]
        )
        entry = next(e for e in entries if e["subject"] == subject and e["task"] == task)
        mask = (trials["subject"] == subject) & (trials["task"] == task)
        rows = np.flatnonzero(mask)

        mismatches = 0
        checked = 0
        for local, row in enumerate(rows):
            position = int(entry["positions"][local])
            source = getattr(raw[position], "allFixations")
            expected = np.stack(
                [np.atleast_1d(getattr(source, name)).astype(np.float32) for name in FEATURES],
                axis=1,
            )
            got = trial_view(trials, int(row))
            checked += 1
            if expected.shape != got.shape or not np.array_equal(expected, got):
                mismatches += 1
        status = "OK" if mismatches == 0 else f"FEHLER: {mismatches} Abweichungen"
        print(f"  {subject}_{task}: {checked} Trials verglichen  {status}")


def report_ranges(trials: dict) -> None:
    print("\n--- 3. Wertebereiche ---")
    for i, name in enumerate(FEATURES):
        column = trials["fixations"][:, i]
        finite = np.isfinite(column)
        print(
            f"  {name:9s} min {np.nanmin(column):9.1f}  median {np.nanmedian(column):8.1f}  "
            f"max {np.nanmax(column):9.1f}  nicht-endlich {np.count_nonzero(~finite)}"
        )
    duration = trials["fixations"][:, 2]
    implausible = np.count_nonzero((duration <= 0) | (duration > 2000))
    share = 100 * implausible / len(duration)
    print(f"  Dauern ausserhalb 0 bis 2000 ms: {implausible} ({share:.2f} %)")


def report_ordering(trials: dict, rng: np.random.Generator, n_trials: int = 2000) -> None:
    """Test whether the fixation order looks like reading rather than shuffle.

    Within a line the gaze moves rightwards far more often than leftwards, and
    across a trial the vertical position drifts downwards. Both signatures
    vanish if the stored order is not chronological. A shuffled control from the
    same data provides the comparison value.
    """
    print("\n--- 4. Reihenfolge der Fixationen ---")
    n = len(trials["sentence_id"])
    rows = rng.choice(n, min(n_trials, n), replace=False)

    def statistics(shuffle: bool) -> tuple[float, float]:
        rightwards = 0
        steps = 0
        downwards = []
        for row in rows:
            path = trial_view(trials, int(row))
            if shuffle:
                path = path[rng.permutation(len(path))]
            dx = np.diff(path[:, 0])
            dy = np.diff(path[:, 1])
            same_line = np.abs(dy) < 10.0
            rightwards += int(np.count_nonzero(dx[same_line] > 0))
            steps += int(np.count_nonzero(same_line))
            order = np.arange(len(path))
            if len(path) > 2 and np.std(path[:, 1]) > 0:
                downwards.append(float(np.corrcoef(order, path[:, 1])[0, 1]))
        return rightwards / max(steps, 1), float(np.mean(downwards)) if downwards else float("nan")

    real_right, real_down = statistics(shuffle=False)
    control_right, control_down = statistics(shuffle=True)
    print(f"  Anteil Vorwaertsspruenge in der Zeile : {real_right:.3f}   (gemischt {control_right:.3f})")
    print(f"  Korrelation Fixationsindex zu y       : {real_down:+.3f}   (gemischt {control_down:+.3f})")
    verdict = "chronologisch plausibel" if real_right > 0.6 and real_down > 0.2 else "AUFFAELLIG"
    print(f"  Befund: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/zuco"))
    parser.add_argument("--extract", action="store_true", help="process source files")
    parser.add_argument("--merge", action="store_true", help="combine caches and verify")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-files", type=int, default=3)
    args = parser.parse_args()

    if not args.extract and not args.merge:
        parser.error("choose --extract, --merge, or both")

    processed = args.data_root / "processed"
    cache_dir = processed / "_cache"

    if args.extract:
        print("=== Extraktion ===")
        run_extract(args.data_root, cache_dir)

    if not args.merge:
        return

    entries = load_caches(cache_dir)
    expected = len(source_files(args.data_root))
    if len(entries) != expected:
        raise SystemExit(f"only {len(entries)} of {expected} source files extracted so far")

    sentences, index = build_sentence_table(entries)
    trials = build_trials(entries, index)

    processed.mkdir(parents=True, exist_ok=True)
    (processed / "sentences.json").write_text(
        json.dumps(sentences, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.savez_compressed(
        processed / "scanpaths.npz",
        subject=trials["subject"],
        task=trials["task"],
        sentence_id=trials["sentence_id"],
        offsets=trials["offsets"],
        fixations=trials["fixations"],
        features=np.array(FEATURES),
    )

    rng = np.random.default_rng(args.seed)
    print("\n=== Pruefungen ===")
    report_counts(entries, sentences, trials)
    report_sample_check(args.data_root, entries, trials, args.sample_files, rng)
    report_ranges(trials)
    report_ordering(trials, rng)

    written = [processed / "sentences.json", processed / "scanpaths.npz"]
    print("\n=== Geschrieben ===")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
