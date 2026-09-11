"""Extract ZuCo's own word-level frequency band features.

The corpus ships more than the raw signal. Every sentence carries a mean band
power per channel, and every word carries the same for five fixation measures:
first fixation duration, single fixation duration, gaze duration, go-past time
and total reading time. Each field holds one value per channel, so eight bands
across 105 channels give 840 numbers per word.

That representation is what every published ZuCo result rests on, and it is
computed by the corpus pipeline rather than by any model here: a Hilbert
transform per band, averaged over the fixation window the measure defines.
Nothing is learned, so nothing has to be trained or loaded to obtain it.

Two files are written from one pass over the sources, because both levels come
from the same structure and reading it twice would double the cost:

    <out>_sentence.npz   one 840-vector per trial, from the sentence-level means
    <out>_words.npz      one 840-vector per word, as a sequence per trial

Both use the layout ``scripts/embed_zuco_eeg.py`` writes, so the existing
embedding adapter reads them unchanged: the first as a mean file, the second as
a sequence file with offsets.

Coverage is partial and that is a property of the corpus, not of this script.
Words the recording pipeline could not resolve carry no band values at all;
measured on one file, 67.8 per cent of words have them, and all eight bands are
present or absent together. Words without values are left out of the sequence
rather than filled in, so a trial's length is its count of usable words.

Nothing here is normalised. Band power is raw power and usually skewed, but
whether to take a logarithm or scale robustly is a separate decision; --survey
reports the range and whether negative values occur, which is what that decision
needs.

Usage:
    python scripts/extract_zuco_bands.py --survey    # coverage and ranges only
    python scripts/extract_zuco_bands.py --extract   # write both files
    python scripts/extract_zuco_bands.py --extract --measure FFD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

# Scalp channels the ZuCo preprocessing retains. A file reporting a different
# count came from another pipeline and is not usable without revisiting every
# assumption downstream.
CHANNELS = 105

# The eight bands the corpus provides, in the order they enter the feature
# vector. Theta, alpha, beta and gamma, each split in two.
BANDS = ("t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2")

# Fixation measures the band power can be averaged over. TRT covers every
# fixation on a word and is what the published work uses; the others isolate
# earlier stages of processing and are kept reachable for an ablation.
MEASURES = ("TRT", "FFD", "GD", "GPT", "SFD")

FEATURES = len(BANDS) * CHANNELS  # 840

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
    return np.atleast_1d(
        sio.loadmat(path, squeeze_me=True, struct_as_record=False)["sentenceData"]
    )


def channel_values(entry, field: str) -> np.ndarray | None:
    """One band field as 105 finite floats, or None where the corpus has none.

    Absent values arrive as an empty array or an object array rather than as
    NaN, so the shape and the dtype both have to be checked before the contents.
    """
    value = getattr(entry, field, None)
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.size != CHANNELS:
        return None
    array = array.astype(np.float32).ravel()
    if not np.isfinite(array).all():
        return None
    return array


def band_vector(entry, prefix: str) -> np.ndarray | None:
    """The eight bands of one measure as a single 840-vector.

    Returns None unless every band is present. The bands are written together by
    the corpus pipeline, so a partial set indicates a different problem than a
    missing word and is not worth patching over.
    """
    parts = [channel_values(entry, f"{prefix}_{band}") for band in BANDS]
    if any(part is None for part in parts):
        return None
    return np.concatenate(parts)


def sentence_vector(sentence) -> np.ndarray | None:
    """The sentence-level band means as one 840-vector."""
    parts = [channel_values(sentence, f"mean_{band}") for band in BANDS]
    if any(part is None for part in parts):
        return None
    return np.concatenate(parts)


def words_of(sentence) -> list:
    """The word entries of one sentence, empty when the field is absent."""
    words = getattr(sentence, "word", None)
    if words is None:
        return []
    return [w for w in np.atleast_1d(words) if hasattr(w, "content")]


def extract_one(path: Path, measure: str) -> dict:
    """Read one source file into per-trial sentence vectors and word sequences."""
    data = load_sentence_data(path)
    texts: list[str] = []
    sentence_rows: list[np.ndarray | None] = []
    word_rows: list[list[np.ndarray]] = []
    word_texts: list[list[str]] = []
    seen_words = 0

    for sentence in data:
        text = getattr(sentence, "content", "")
        texts.append(text.strip() if isinstance(text, str) else "")
        sentence_rows.append(sentence_vector(sentence))

        vectors: list[np.ndarray] = []
        strings: list[str] = []
        for entry in words_of(sentence):
            seen_words += 1
            vector = band_vector(entry, measure)
            if vector is None:
                continue
            vectors.append(vector)
            content = getattr(entry, "content", "")
            strings.append(content.strip() if isinstance(content, str) else "")
        word_rows.append(vectors)
        word_texts.append(strings)

    return {
        "texts": texts,
        "sentence_rows": sentence_rows,
        "word_rows": word_rows,
        "word_texts": word_texts,
        "n_words_seen": seen_words,
    }


def sentence_index(processed: Path) -> dict[str, int]:
    """Map sentence text to the id the eye-tracking extraction assigned."""
    path = processed / "sentences.json"
    if not path.is_file():
        raise SystemExit(f"{path} missing; run scripts/extract_zuco_et.py --merge first")
    return {row["text"]: row["id"] for row in json.loads(path.read_text(encoding="utf-8"))}


def feature_names() -> np.ndarray:
    """Name per feature column, so a stored file documents its own layout."""
    return np.array([f"{band}_ch{channel:03d}" for band in BANDS for channel in range(CHANNELS)])


def run(data_root: Path, processed: Path, measure: str, min_words: int,
        survey_only: bool, budget: int, out_stem: str) -> None:
    index = sentence_index(processed)
    files = source_files(data_root)

    subjects: list[str] = []
    tasks: list[str] = []
    sentence_ids: list[int] = []
    sentence_vectors: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    sequence_words: list[list[str]] = []

    seen_words = 0
    kept_words = 0
    unknown_text = 0
    no_sentence_mean = 0
    too_few_words = 0
    truncated = False

    print(f"=== Bandmerkmale, Mass {measure} ===")
    for opened, (subject, task, path) in enumerate(files):
        if budget and opened >= budget:
            truncated = True
            print(f"  Budget erreicht, {opened} von {len(files)} Dateien gelesen")
            break
        result = extract_one(path, measure)
        seen_words += result["n_words_seen"]

        usable = 0
        for position, text in enumerate(result["texts"]):
            if text not in index:
                unknown_text += 1
                continue
            mean_vector = result["sentence_rows"][position]
            if mean_vector is None:
                no_sentence_mean += 1
                continue
            vectors = result["word_rows"][position]
            if len(vectors) < min_words:
                too_few_words += 1
                continue

            subjects.append(subject)
            tasks.append(task)
            sentence_ids.append(index[text])
            sentence_vectors.append(mean_vector)
            sequences.append(np.stack(vectors))
            sequence_words.append(result["word_texts"][position])
            kept_words += len(vectors)
            usable += 1

        print(f"  {subject}_{task:2s}  {usable:4d} Trials")

    if not subjects:
        raise SystemExit("kein einziger Trial nutzbar; Eingabepfade pruefen")

    stacked = np.stack(sentence_vectors)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)

    print("\n=== Pruefungen ===")
    print(f"  Trials                      : {len(subjects)}")
    print(f"  Woerter gesehen             : {seen_words}")
    print(f"  Woerter mit allen 8 Baendern: {kept_words}  "
          f"({100 * kept_words / max(seen_words, 1):.1f} %)")
    print(f"  Woerter je Trial            : Median {np.median(lengths):.0f}, "
          f"min {lengths.min()}, max {lengths.max()}")
    print(f"  verworfen: {unknown_text} ohne Satz-ID, {no_sentence_mean} ohne Satzmittel, "
          f"{too_few_words} unter {min_words} Woertern")
    print(f"  Wertebereich Satzebene      : min {stacked.min():.4f}, "
          f"p50 {np.median(stacked):.4f}, max {stacked.max():.4f}")
    negative = int((stacked < 0).sum())
    print(f"  negative Werte              : {negative}  "
          f"({'Log-Transformation nicht direkt moeglich' if negative else 'Log moeglich'})")

    if survey_only:
        print("\n  --survey: nichts geschrieben")
        return

    # A truncated pass would produce a file that looks complete and is not.
    if truncated:
        raise SystemExit(
            "--budget hat den Durchlauf abgeschnitten; eine Teildatei zu schreiben waere "
            "nicht von einer vollstaendigen zu unterscheiden. Ohne --budget erneut aufrufen, "
            "oder --survey nutzen, um nur zu messen."
        )

    metadata = dict(
        subject=np.array(subjects),
        task=np.array(tasks),
        sentence_id=np.array(sentence_ids, dtype=np.int64),
        reference=np.array(["zuco-band"]),
        channels=np.array(["all105"]),
        too_long=np.array(["n/a"]),
        checkpoint=np.array([f"zuco-hilbert-{measure}"]),
        channel_names=feature_names(),
        sample_rate=np.array([500]),
    )

    flat_out = processed / f"{out_stem}_{measure}_sentence.npz"
    np.savez(
        flat_out,
        vectors=stacked.astype(np.float32),
        n_patches=lengths,
        pooling=np.array(["sentence-mean"]),
        **metadata,
    )

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    sequence_out = processed / f"{out_stem}_{measure}_words.npz"
    np.savez(
        sequence_out,
        tokens=np.concatenate(sequences, axis=0).astype(np.float32),
        offsets=offsets,
        n_patches=lengths,
        pooling=np.array(["word-sequence"]),
        word=np.array([w for words in sequence_words for w in words]),
        **metadata,
    )

    print("\n=== Geschrieben ===")
    for target in (flat_out, sequence_out):
        print(f"  {target}  ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/zuco"))
    parser.add_argument("--measure", default="TRT", choices=MEASURES)
    parser.add_argument("--min-words", type=int, default=3,
                        help="trials with fewer usable words are dropped")
    parser.add_argument("--survey", action="store_true", help="report only, write nothing")
    parser.add_argument("--extract", action="store_true", help="write both files")
    parser.add_argument("--budget", type=int, default=0,
                        help="max files to open per invocation, 0 for all")
    parser.add_argument("--out-stem", default="eeg_bands")
    args = parser.parse_args()

    if not (args.survey or args.extract):
        parser.error("choose --survey or --extract")

    run(
        data_root=args.data_root,
        processed=args.data_root / "processed",
        measure=args.measure,
        min_words=args.min_words,
        survey_only=not args.extract,
        budget=args.budget,
        out_stem=args.out_stem,
    )


if __name__ == "__main__":
    main()
