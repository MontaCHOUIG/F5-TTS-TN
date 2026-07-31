"""Prepare a Tunisian Arabic dataset while preserving Habibi-TTS token indices.

Input metadata is pipe-separated without a header:
    wavs/clip_0001.wav|نحكيلك حاجة توا

The script normalizes paths for the checked-out CSV preparer, optionally wraps
text in the Unified model's TUN tag, validates every resulting token against a
provided Habibi vocabulary, and writes raw.arrow, duration.json, and vocab.txt.
"""

import argparse
import csv
import shutil
import tempfile
from pathlib import Path

from f5_tts.train.datasets.prepare_csv_wavs import prepare_csv_wavs_dir, save_prepped_dataset


TUN_TAG = "⑨"


def wrap_tunisian(text: str) -> str:
    if text.startswith(f"{TUN_TAG}〈") and text.endswith("〉"):
        return text
    return f"{TUN_TAG}〈{text}〉"


def normalize_metadata(metadata_path: Path, dataset_dir: Path, output_path: Path, use_tun_tag: bool) -> int:
    rows: list[tuple[str, str]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.reader(metadata_file, delimiter="|")
        for line_number, row in enumerate(reader, start=1):
            if line_number == 1 and len(row) >= 2 and row[0].strip() == "audio_file" and row[1].strip() == "text":
                continue
            if len(row) != 2:
                raise ValueError(f"{metadata_path}:{line_number}: expected exactly 'audio_path|text'")

            audio_value, text = (value.strip() for value in row)
            if not audio_value or not text:
                raise ValueError(f"{metadata_path}:{line_number}: audio path and text must be non-empty")

            audio_path = Path(audio_value).expanduser()
            if not audio_path.is_absolute():
                audio_path = dataset_dir / audio_path
            text = wrap_tunisian(text) if use_tun_tag else text
            rows.append((audio_path.resolve().as_posix(), text))

    if not rows:
        raise ValueError(f"{metadata_path}: no training rows found")

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="|", lineterminator="\n")
        writer.writerow(("audio_file", "text"))
        writer.writerows(rows)
    return len(rows)


def load_vocab(vocab_path: Path) -> set[str]:
    with vocab_path.open("r", encoding="utf-8") as vocab_file:
        ordered_tokens = [line.rstrip("\r\n") for line in vocab_file]
    if not ordered_tokens or ordered_tokens[0] != " ":
        raise ValueError(f"{vocab_path}: token 0 must be a space; this does not look like an F5-TTS char vocabulary")
    return set(ordered_tokens)


def validate_tokens(processed_rows: list[dict], vocab: set[str]) -> None:
    used_tokens = {token for row in processed_rows for token in row["text"]}
    missing = sorted(used_tokens - vocab)
    if missing:
        rendered = ", ".join(repr(token) for token in missing)
        raise ValueError(
            "Dataset contains tokens absent from the Habibi MSA vocabulary: "
            f"{rendered}. Normalize them or deliberately extend the checkpoint and vocabulary."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TunisianTTS_char for Habibi-TTS fine-tuning.")
    parser.add_argument("dataset_dir", type=Path, help="Directory containing metadata.csv and wavs/")
    parser.add_argument(
        "--pretrained-vocab",
        type=Path,
        required=True,
        help="Downloaded Specialized/MSA/vocab.txt; it will become the dataset vocab.txt",
    )
    parser.add_argument("--metadata", default="metadata.csv", help="Metadata filename inside dataset_dir")
    parser.add_argument("--workers", type=int, default=None, help="Audio metadata worker count")
    parser.add_argument(
        "--dialect-tag",
        choices=("none", "TUN"),
        default="none",
        help="Use TUN only for an intentional tag-conditioned experiment; Specialized/MSA defaults to none",
    )
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    metadata_path = dataset_dir / args.metadata
    pretrained_vocab = args.pretrained_vocab.expanduser().resolve()

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not pretrained_vocab.is_file():
        raise FileNotFoundError(pretrained_vocab)
    if pretrained_vocab == (dataset_dir / "vocab.txt").resolve():
        raise ValueError("--pretrained-vocab must be a separate source file because preparation replaces vocab.txt")
    if args.min_duration <= 0 or args.max_duration < args.min_duration:
        raise ValueError("Require 0 < min-duration <= max-duration")

    vocab = load_vocab(pretrained_vocab)
    with tempfile.TemporaryDirectory(prefix="f5_tts_tunisian_") as temp_dir:
        normalized_csv = Path(temp_dir) / "metadata.absolute.csv"
        row_count = normalize_metadata(
            metadata_path,
            dataset_dir,
            normalized_csv,
            use_tun_tag=args.dialect_tag == "TUN",
        )
        processed_rows, durations, generated_vocab = prepare_csv_wavs_dir(normalized_csv, args.workers)

    invalid_durations = [
        (row["audio_path"], duration)
        for row, duration in zip(processed_rows, durations)
        if not args.min_duration <= duration <= args.max_duration
    ]
    if invalid_durations:
        examples = ", ".join(f"{path} ({duration:.2f}s)" for path, duration in invalid_durations[:5])
        raise ValueError(f"{len(invalid_durations)} clips are outside the duration limits; examples: {examples}")

    validate_tokens(processed_rows, vocab)
    save_prepped_dataset(dataset_dir, processed_rows, durations, generated_vocab, is_finetune=False)
    shutil.copy2(pretrained_vocab, dataset_dir / "vocab.txt")
    print(f"Prepared {row_count} Tunisian clips with the Habibi MSA vocabulary.")


if __name__ == "__main__":
    main()
