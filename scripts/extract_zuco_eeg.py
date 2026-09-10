"""Extract sentence-level EEG from the ZuCo 1.0 MATLAB files.

The ``sentenceData`` struct carries a ``rawData`` field per sentence: the
preprocessed EEG of the whole trial as (channels, samples). It is already cut to
the sentence boundaries, so no alignment work is needed and the separately
distributed continuous recordings stay untouched.

Unlike the eye-tracking strecke, the signal does not shrink to megabytes. One
trial is 105 channels by roughly 2000 samples, so the extraction writes one
memory-mappable array per source file and a small index that addresses trials
inside it. Nothing loads the full corpus into memory.

The survey stage answers the questions that decide how to store and encode the
signal -- trial yield, overlap with the eye-tracking trials, sampling rate,
field-name consistency -- without writing any signal data.

Usage:
    python scripts/extract_zuco_eeg.py --survey     # counts only, writes a small json
    python scripts/extract_zuco_eeg.py --extract    # write signals (resumable)
    python scripts/extract_zuco_eeg.py --merge      # build the index, verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

# Scalp channels retained by the ZuCo preprocessing. A file that reports a
# different count has been produced by a different pipeline and is not usable
# here without revisiting every downstream assumption.
CHANNELS = 105

# Trials shorter than this carry too little signal to encode. At the sampling
# rate implied by the files this is roughly a fifth of a second.
MIN_SAMPLES = 200

# Trials are stored as float16. Measured amplitudes stay inside a few tens of
# microvolts, where float16 resolves far below the noise floor of the recording.
STORAGE_DTYPE = np.float16

# Recording rate of the corpus. The eye-tracking duration fields count samples of
# the same clock rather than milliseconds, which is why a word segment holds about
# as many samples as its duration value. Two checks agree with the published rate:
# the median fixation lasts 208 ms, and reading runs at 3.3 words per second.
SOURCE_HZ = 500

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


def load_sentence_data(path: Path) -> np.ndarray:
    return np.atleast_1d(sio.loadmat(path, squeeze_me=True, struct_as_record=False)["sentenceData"])


def signal_of(entry) -> np.ndarray | None:
    """Return the trial signal as (samples, channels), or None if unusable.

    The field holds NaN for trials the recording pipeline discarded. Trials that
    survive are transposed into the (time, features) convention the batching
    layer expects, and rejected if they are too short, have an unexpected
    channel count, or contain non-finite samples an encoder cannot consume.
    """
    raw = getattr(entry, "rawData", None)
    if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.size == 0:
        return None
    if raw.dtype.kind != "f":
        return None
    if raw.shape[0] != CHANNELS or raw.shape[1] < MIN_SAMPLES:
        return None
    signal = np.ascontiguousarray(raw.T, dtype=np.float32)
    if not np.isfinite(signal).all():
        return None
    return signal


def fixation_duration_sum(entry) -> float:
    """Total time the gaze rested on the sentence, in milliseconds."""
    fixations = getattr(entry, "allFixations", None)
    if fixations is None or not hasattr(fixations, "duration"):
        return float("nan")
    return float(np.sum(np.atleast_1d(fixations.duration)))


def survey_one(path: Path) -> dict:
    """Measure one source file without keeping any signal."""
    raw = load_sentence_data(path)
    fields = sorted(raw[0]._fieldnames) if len(raw) else []
    word_fields: list[str] = []
    for entry in raw:
        words = getattr(entry, "word", None)
        if isinstance(words, np.ndarray) and words.size:
            word_fields = sorted(np.atleast_1d(words)[0]._fieldnames)
            break

    texts, usable, samples, ratios = [], [], [], []
    rejected = {"missing": 0, "short": 0, "channels": 0, "nonfinite": 0}
    for entry in raw:
        text = getattr(entry, "content", "")
        texts.append(text.strip() if isinstance(text, str) else "")

        candidate = getattr(entry, "rawData", None)
        if (
            not isinstance(candidate, np.ndarray)
            or candidate.ndim != 2
            or candidate.size == 0
            or candidate.dtype.kind != "f"
        ):
            rejected["missing"] += 1
        elif candidate.shape[0] != CHANNELS:
            rejected["channels"] += 1
        elif candidate.shape[1] < MIN_SAMPLES:
            rejected["short"] += 1
        elif not np.isfinite(candidate).all():
            rejected["nonfinite"] += 1

        signal = signal_of(entry)
        if signal is None:
            continue
        usable.append(len(texts) - 1)
        samples.append(len(signal))
        duration = fixation_duration_sum(entry)
        if duration == duration and duration > 0:
            ratios.append(len(signal) / duration)

    return {
        "n_sentences": int(len(raw)),
        "sentence_fields": fields,
        "word_fields": word_fields,
        "texts": texts,
        "usable_positions": usable,
        "samples": samples,
        "samples_per_ms": ratios,
        "rejected": rejected,
    }


def sentence_index(processed: Path) -> dict[str, int] | None:
    """Map sentence text to the id the eye-tracking extraction assigned."""
    path = processed / "sentences.json"
    if not path.exists():
        return None
    return {row["text"]: row["id"] for row in json.loads(path.read_text(encoding="utf-8"))}


def report_survey(results: dict[tuple[str, str], dict], processed: Path) -> None:
    print("\n--- 1. Ausbeute je Datei ---")
    print(f"{'Proband':10s} {'Task':>5s} {'Saetze':>7s} {'nutzbar':>8s} {'fehlend':>8s}"
          f" {'kurz':>6s} {'NaN':>6s}")
    totals = {"n": 0, "usable": 0}
    for (subject, task), result in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        r = result["rejected"]
        usable = len(result["usable_positions"])
        print(f"{subject:10s} {task:>5s} {result['n_sentences']:7d} {usable:8d}"
              f" {r['missing']:8d} {r['short']:6d} {r['nonfinite']:6d}")
        totals["n"] += result["n_sentences"]
        totals["usable"] += len(result["usable_positions"])
    share = 100 * totals["usable"] / max(totals["n"], 1)
    print(f"\n  Trials gesamt: {totals['usable']} von {totals['n']} praesentierten ({share:.1f} %)")

    print("\n--- 2. Feldnamen ---")
    # The two tasks differ by design: only the question-answering task carries
    # the answer_* and *_sec fields. Consistency therefore has to hold within a
    # task, while across tasks only the fields this script reads must be shared.
    for level in ("sentence_fields", "word_fields"):
        for task in sorted(TASKS):
            variants = {tuple(r[level]) for (_, t), r in results.items() if t == task}
            print(f"  {level:16s} {task}: {len(variants)} Variante(n)", end="  ")
            print("OK" if len(variants) == 1 else "FEHLER: Dateien sind nicht gleich aufgebaut")
    required = {"content", "rawData", "allFixations"}
    missing = {
        f"{subject}_{task}": sorted(required - set(r["sentence_fields"]))
        for (subject, task), r in results.items()
        if not required.issubset(r["sentence_fields"])
    }
    print(f"  benoetigte Felder {sorted(required)} in allen Dateien", end="  ")
    print("OK" if not missing else f"FEHLER: {missing}")

    print("\n--- 3. Abtastrate ---")
    ratios = np.concatenate([np.array(r["samples_per_ms"]) for r in results.values()])
    print(f"  Samples je ms Fixationsdauer: median {np.median(ratios):.3f}"
          f"  p10 {np.percentile(ratios, 10):.3f}  p90 {np.percentile(ratios, 90):.3f}")
    print("  Die Fixationsdauer deckt nur einen Teil der Lesezeit ab, die Abtastrate")
    print(f"  liegt also unter {np.median(ratios) * 1000:.0f} Hz. Genauer nur mit Satzdauern.")

    print("\n--- 4. Groesse ---")
    samples = int(sum(sum(r["samples"]) for r in results.values()))
    per_dtype = samples * CHANNELS
    print(f"  Zeitpunkte gesamt: {samples:,}".replace(",", "."))
    print(f"  als float32: {per_dtype * 4 / 1e9:.1f} GB"
          f"   als float16: {per_dtype * 2 / 1e9:.1f} GB")

    print("\n--- 5. Schnittmenge mit den Eye-Tracking-Trials ---")
    index = sentence_index(processed)
    scanpaths = processed / "scanpaths.npz"
    if index is None or not scanpaths.exists():
        print("  uebersprungen: sentences.json oder scanpaths.npz fehlt")
        return
    eeg_keys = set()
    unknown = 0
    for (subject, task), result in results.items():
        for position in result["usable_positions"]:
            text = result["texts"][position]
            if text in index:
                eeg_keys.add((subject, task, index[text]))
            else:
                unknown += 1
    with np.load(scanpaths, allow_pickle=True) as data:
        et_keys = set(
            zip(data["subject"], data["task"], data["sentence_id"].tolist(), strict=True)
        )
    both = eeg_keys & et_keys
    print(f"  EEG-Trials {len(eeg_keys)} | ET-Trials {len(et_keys)} | in beiden {len(both)}")
    print(f"  nur EEG {len(eeg_keys - et_keys)} | nur ET {len(et_keys - eeg_keys)}")
    if unknown:
        print(f"  FEHLER: {unknown} Saetze ohne Eintrag in sentences.json")


def run_survey(data_root: Path, processed: Path, budget: int) -> None:
    """Measure every source file, caching per file so an interrupted run resumes.

    Reading one file takes seconds and the whole corpus takes minutes, which is
    longer than some shells allow. The budget caps how many uncached files a
    single invocation opens; the report is only produced once all are present.
    """
    cache_dir = processed / "_cache_eeg_survey"
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = source_files(data_root)
    opened = 0

    for subject, task, path in files:
        target = cache_dir / f"{subject}_{task}.json"
        if target.exists():
            continue
        if budget and opened >= budget:
            break
        result = survey_one(path)
        target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        opened += 1
        print(f"  gelesen  {subject}_{task}  {len(result['usable_positions'])} nutzbar")

    results: dict[tuple[str, str], dict] = {}
    for subject, task, _ in files:
        target = cache_dir / f"{subject}_{task}.json"
        if target.exists():
            results[(subject, task)] = json.loads(target.read_text(encoding="utf-8"))
    if len(results) < len(files):
        print(f"\n  {len(results)} von {len(files)} Dateien erhoben, erneut aufrufen")
        return

    report_survey(results, processed)

    processed.mkdir(parents=True, exist_ok=True)
    summary = {
        f"{subject}_{task}": {
            "n_sentences": r["n_sentences"],
            "n_usable": len(r["usable_positions"]),
            "rejected": r["rejected"],
            "median_samples": int(np.median(r["samples"])) if r["samples"] else 0,
        }
        for (subject, task), r in results.items()
    }
    target = processed / "eeg_survey.json"
    target.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n  geschrieben: {target}")


def resample(signal: np.ndarray, target_hz: int) -> np.ndarray:
    """Convert to another sampling rate with an anti-aliasing polyphase filter.

    The ratio between the corpus rate and a model's expected rate is not
    generally an integer, so decimation by a whole factor is not enough.
    """
    if not target_hz or target_hz == SOURCE_HZ:
        return signal
    common = np.gcd(target_hz, SOURCE_HZ)
    return resample_poly(signal, target_hz // common, SOURCE_HZ // common, axis=0)


def extract_one(path: Path, target_hz: int) -> dict:
    """Read one source file into a stacked signal array plus its trial table."""
    raw = load_sentence_data(path)
    texts: list[str] = []
    positions: list[int] = []
    blocks: list[np.ndarray] = []

    for position, entry in enumerate(raw):
        text = getattr(entry, "content", "")
        texts.append(text.strip() if isinstance(text, str) else "")
        signal = signal_of(entry)
        if signal is None:
            continue
        blocks.append(resample(signal, target_hz).astype(STORAGE_DTYPE))
        positions.append(position)

    stacked = (
        np.concatenate(blocks, axis=0)
        if blocks
        else np.zeros((0, CHANNELS), dtype=STORAGE_DTYPE)
    )
    lengths = np.array([len(b) for b in blocks], dtype=np.int64)
    return {"texts": texts, "positions": positions, "lengths": lengths, "signals": stacked}


def run_extract(data_root: Path, signal_dir: Path, target_hz: int, budget: int) -> None:
    """Extract every source file that has no signal file yet.

    A file counts as done only once its table is written, so a run interrupted
    while saving redoes that file instead of leaving a truncated array behind.
    """
    signal_dir.mkdir(parents=True, exist_ok=True)
    opened = 0
    for subject, task, path in source_files(data_root):
        target = signal_dir / f"{subject}_{task}.npy"
        table = signal_dir / f"{subject}_{task}.json"
        if target.exists() and table.exists():
            continue
        if budget and opened >= budget:
            print(f"  Budget erreicht, erneut aufrufen")
            return
        opened += 1
        result = extract_one(path, target_hz)
        np.save(target, result["signals"])
        table.write_text(
            json.dumps(
                {
                    "texts": result["texts"],
                    "positions": [int(p) for p in result["positions"]],
                    "lengths": [int(n) for n in result["lengths"]],
                    "sample_rate": target_hz or SOURCE_HZ,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        size = target.stat().st_size / 1e6
        print(f"  done    {subject}_{task}  {len(result['positions'])} trials  {size:.0f} MB")


def build_index(signal_dir: Path, index: dict[str, int]) -> dict:
    """Address every trial as a slice into its subject's signal file."""
    subjects, tasks, sentence_ids, starts, ends = [], [], [], [], []
    for table in sorted(signal_dir.glob("*.json")):
        subject, task = table.stem.rsplit("_", 1)
        meta = json.loads(table.read_text(encoding="utf-8"))
        cursor = 0
        for position, length in zip(meta["positions"], meta["lengths"], strict=True):
            text = meta["texts"][position]
            if text not in index:
                raise SystemExit(f"sentence of {subject}_{task} missing in sentences.json")
            subjects.append(subject)
            tasks.append(task)
            sentence_ids.append(index[text])
            starts.append(cursor)
            ends.append(cursor + length)
            cursor += length
    return {
        "subject": np.array(subjects),
        "task": np.array(tasks),
        "sentence_id": np.array(sentence_ids, dtype=np.int64),
        "start": np.array(starts, dtype=np.int64),
        "end": np.array(ends, dtype=np.int64),
    }


def report_merge(trials: dict, signal_dir: Path, processed: Path, rng: np.random.Generator) -> None:
    print("\n--- 1. Zaehlabgleich ---")
    print(f"  Trials gesamt : {len(trials['sentence_id'])}")
    print(f"  Zeitpunkte    : {int(np.sum(trials['end'] - trials['start'])):,}".replace(",", "."))
    pairs = set(zip(trials["subject"], trials["task"], trials["sentence_id"], strict=True))
    print(f"  eindeutige (Proband, Task, Satz)-Tripel: {len(pairs)}", end="  ")
    print("OK" if len(pairs) == len(trials["sentence_id"]) else "FEHLER: Duplikate")

    print("\n--- 2. Wertebereiche (Stichprobe) ---")
    n_trials = len(trials["sentence_id"])
    rows = rng.choice(n_trials, min(200, n_trials), replace=False)
    values = []
    for row in rows:
        key = f"{trials['subject'][row]}_{trials['task'][row]}"
        signals = np.load(signal_dir / f"{key}.npy", mmap_mode="r")
        values.append(np.asarray(signals[trials["start"][row] : trials["end"][row]], np.float32))
    flat = np.concatenate([v.ravel() for v in values])
    broken = np.count_nonzero(~np.isfinite(flat))
    print(f"  std {flat.std():.2f} uV | p1 {np.percentile(flat, 1):.1f} | "
          f"p99 {np.percentile(flat, 99):.1f} | nicht-endlich {broken}")

    print("\n--- 3. Schnittmenge mit den Eye-Tracking-Trials ---")
    scanpaths = processed / "scanpaths.npz"
    if not scanpaths.exists():
        print("  uebersprungen: scanpaths.npz fehlt")
        return
    with np.load(scanpaths, allow_pickle=True) as data:
        et_keys = set(
            zip(data["subject"], data["task"], data["sentence_id"].tolist(), strict=True)
        )
    eeg_keys = {(s, t, int(i)) for s, t, i in
                zip(trials["subject"], trials["task"], trials["sentence_id"], strict=True)}
    print(f"  EEG {len(eeg_keys)} | ET {len(et_keys)} | in beiden {len(eeg_keys & et_keys)}")
    print(f"  nur EEG {len(eeg_keys - et_keys)} | nur ET {len(et_keys - eeg_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/zuco"))
    parser.add_argument("--survey", action="store_true", help="count trials, write no signal")
    parser.add_argument("--extract", action="store_true", help="write signal files")
    parser.add_argument("--merge", action="store_true", help="build the index and verify")
    parser.add_argument("--resample-hz", type=int, default=0,
                        help=f"store at this rate instead of the recorded {SOURCE_HZ} Hz")
    parser.add_argument("--budget", type=int, default=0,
                        help="max files to open per invocation, 0 for all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (args.survey or args.extract or args.merge):
        parser.error("choose --survey, --extract, --merge, or a combination")
    if args.resample_hz < 0:
        parser.error("--resample-hz must not be negative")

    processed = args.data_root / "processed"
    signal_dir = processed / "eeg"

    if args.survey:
        print("=== Erhebung ===")
        run_survey(args.data_root, processed, args.budget)

    if args.extract:
        print("\n=== Extraktion ===")
        run_extract(args.data_root, signal_dir, args.resample_hz, args.budget)

    if not args.merge:
        return

    index = sentence_index(processed)
    if index is None:
        raise SystemExit("sentences.json missing; run scripts/extract_zuco_et.py --merge first")
    expected = len(source_files(args.data_root))
    written = len(list(signal_dir.glob("*.npy")))
    if written != expected:
        raise SystemExit(f"only {written} of {expected} source files extracted so far")

    trials = build_index(signal_dir, index)
    target = processed / "eeg_index.npz"
    np.savez_compressed(
        target,
        subject=trials["subject"],
        task=trials["task"],
        sentence_id=trials["sentence_id"],
        start=trials["start"],
        end=trials["end"],
        channels=np.array([CHANNELS]),
        sample_rate=np.array([args.resample_hz or SOURCE_HZ]),
    )

    print("\n=== Pruefungen ===")
    report_merge(trials, signal_dir, processed, np.random.default_rng(args.seed))
    print(f"\n=== Geschrieben ===\n  {target}  ({target.stat().st_size / 1e6:.1f} MB)")
    total = sum(item.stat().st_size for item in signal_dir.glob("*.npy"))
    print(f"  {signal_dir}/*.npy  ({total / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
