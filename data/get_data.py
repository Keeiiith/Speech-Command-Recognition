"""Utilities for downloading and preparing the Speech Commands dataset.

This script downloads Google Speech Commands v0.02, keeps a configurable set of
labels, and writes CSV manifests that the training pipeline can consume.
"""

from __future__ import annotations
from tqdm import tqdm

import argparse
import csv
import json
import random

# PrepConfig keeps all important paths and preprocessing switches in one place,
# so each pipeline function stays simple and testable.
import tarfile
import urllib.request
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
    # Use the data folder beside this script by default so the command works
    # immediately after cloning the repository.


DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DEFAULT_ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"
DEFAULT_LABEL_MAP = {
    "go": "play",
    "off": "pause",
    "right": "next",
    "left": "previous",
    "stop": "stop",
}
DEFAULT_SPLIT_COUNTS = {"train": 600, "val": 120, "test": 120}
BACKGROUND_NOISE_DIR = "_background_noise_"


@dataclass(slots=True)
class PrepConfig:
    """Configuration for dataset download and manifest generation."""

    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    archive_path: Path
    sample_rate: int = 16000
    clip_duration_seconds: float = 1.0
    seed: int = 42
    include_unknown: bool = True
    unknown_ratio: float = 0.5
    label_map: dict[str, str] | None = None

    def resolved_label_map(self) -> dict[str, str]:
        return self.label_map or DEFAULT_LABEL_MAP.copy()


def parse_args() -> PrepConfig:
    """Parse CLI arguments into a strongly typed configuration object."""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download and prepare Speech Commands manifests.")
    parser.add_argument("--data-dir", type=Path, default=script_dir, help="Base data directory.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=1.0,
        help="Fixed clip duration in seconds used by the model pipeline.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--unknown-ratio",
        type=float,
        default=0.5,
        help="Unknown examples per split as a ratio of target examples.",
    )
    parser.add_argument(
        "--disable-unknown",
        action="store_true",
        help="Skip sampling non-target labels into an 'unknown' class.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        help="JSON object mapping dataset labels to project intent labels.",
    )
    args = parser.parse_args()

    label_map = json.loads(args.label_map) if args.label_map else None
    data_dir = args.data_dir.resolve()
    return PrepConfig(
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        archive_path=data_dir / DEFAULT_ARCHIVE_NAME,
        sample_rate=args.sample_rate,
        clip_duration_seconds=args.clip_duration,
        seed=args.seed,
        include_unknown=not args.disable_unknown,
        unknown_ratio=args.unknown_ratio,
        label_map=label_map,
    )


def download_archive(url: str, destination: Path) -> None:
    """Download the dataset archive with a progress bar."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Archive already present at {destination}")
        return

    print(f"Downloading dataset archive to {destination}...")

    response = urllib.request.urlopen(url)
    total_size = int(response.getheader("Content-Length").strip())

    with open(destination, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading dataset",
    ) as pbar:
        while True:
            chunk = response.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract the archive with a progress bar."""

    validation_file = output_dir / "validation_list.txt"
    if validation_file.exists():
        print(f"Dataset already extracted in {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} into {output_dir}...")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        for member in tqdm(members, desc="Extracting files"):
            archive.extract(member, output_dir)


def read_official_splits(raw_dir: Path) -> tuple[set[str], set[str]]:
    """Load the validation and test files shipped with the dataset."""

    validation = raw_dir / "validation_list.txt"
    testing = raw_dir / "testing_list.txt"
    validation_items = {line.strip().replace("\\", "/") for line in validation.read_text().splitlines() if line.strip()}
    testing_items = {line.strip().replace("\\", "/") for line in testing.read_text().splitlines() if line.strip()}
    return validation_items, testing_items


def assign_split(relative_path: str, validation_items: set[str], testing_items: set[str]) -> str:
    """Assign an example to the official train, validation, or test split."""

    if relative_path in validation_items:
        return "val"
    if relative_path in testing_items:
        return "test"
    return "train"


def parse_filename(path: Path) -> tuple[str, str]:
    """Extract the speaker id and utterance id from dataset filenames."""

    stem = path.stem
    if "_nohash_" not in stem:
        return "unknown_speaker", stem
    speaker_id, utterance_id = stem.split("_nohash_", maxsplit=1)
    return speaker_id, utterance_id


def iter_audio_files(raw_dir: Path) -> Iterable[Path]:
    """Yield all WAV files except the background-noise folder."""

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name == BACKGROUND_NOISE_DIR:
            continue
        yield from sorted(label_dir.glob("*.wav"))


def build_manifest_records(config: PrepConfig) -> dict[str, list[dict[str, str | float | int]]]:
    """Create manifest rows for target labels, plus optional unknown examples."""

    label_map = config.resolved_label_map()
    target_labels = set(label_map)
    validation_items, testing_items = read_official_splits(config.raw_dir)
    split_records: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    unknown_pool: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)

    for wav_path in tqdm(iter_audio_files(config.raw_dir), desc="Processing audio files"):
        relative_path = wav_path.relative_to(config.raw_dir).as_posix()
        raw_label = wav_path.parent.name
        split = assign_split(relative_path, validation_items, testing_items)
        speaker_id, utterance_id = parse_filename(wav_path)
        record = {
            "path": relative_path,
            "raw_label": raw_label,
            "label": label_map.get(raw_label, "unknown"),
            "split": split,
            "speaker_id": speaker_id,
            "utterance_id": utterance_id,
            "is_unknown": int(raw_label not in target_labels),
            "is_silence": 0,
            "segment_start": 0.0,
            "segment_duration": config.clip_duration_seconds,
        }

        if raw_label in target_labels:
            split_records[split].append(record)
        elif config.include_unknown:
            unknown_pool[split].append(record)

    if config.include_unknown:
        add_unknown_records(split_records, unknown_pool, config.unknown_ratio, config.seed)

    add_silence_records(split_records, config)
    return split_records


def add_unknown_records(
    split_records: dict[str, list[dict[str, str | float | int]]],
    unknown_pool: dict[str, list[dict[str, str | float | int]]],
    unknown_ratio: float,
    seed: int,
) -> None:
    """Sample a bounded number of non-target examples into an unknown class."""

    random_generator = random.Random(seed)
    for split_name, pool in unknown_pool.items():
        target_count = len(split_records[split_name])
        if target_count == 0 or not pool:
            continue

        sample_size = min(len(pool), max(1, int(target_count * unknown_ratio)))
        for record in tqdm(random_generator.sample(pool, sample_size), desc=f"Sampling unknown ({split_name})"):
            split_records[split_name].append(record)


def add_silence_records(
    split_records: dict[str, list[dict[str, str | float | int]]],
    config: PrepConfig,
) -> None:
    """Create pseudo-silence examples from the dataset's background noise clips."""

    noise_dir = config.raw_dir / BACKGROUND_NOISE_DIR
    if not noise_dir.exists():
        print("Background noise directory not found. Skipping silence generation.")
        return

    candidate_segments: list[dict[str, str | float | int]] = []
    for noise_path in tqdm(sorted(noise_dir.glob("*.wav")), desc="Generating silence"):
        duration_seconds = get_wave_duration_seconds(noise_path)
        max_offset = max(0.0, duration_seconds - config.clip_duration_seconds)
        segment_index = 0
        current_offset = 0.0
        while current_offset <= max_offset:
            candidate_segments.append(
                {
                    "path": noise_path.relative_to(config.raw_dir).as_posix(),
                    "raw_label": "silence",
                    "label": "silence",
                    "split": "unassigned",
                    "speaker_id": "background_noise",
                    "utterance_id": f"{noise_path.stem}_{segment_index}",
                    "is_unknown": 0,
                    "is_silence": 1,
                    "segment_start": round(current_offset, 4),
                    "segment_duration": config.clip_duration_seconds,
                }
            )
            current_offset += config.clip_duration_seconds
            segment_index += 1

    random_generator = random.Random(config.seed)
    random_generator.shuffle(candidate_segments)
    cursor = 0
    for split_name in ("train", "val", "test"):
        count = DEFAULT_SPLIT_COUNTS[split_name]
        split_records[split_name].extend(candidate_segments[cursor:cursor + count])
        cursor += count


def get_wave_duration_seconds(path: Path) -> float:
    """Read WAV metadata without loading the full file into memory."""

    with wave.open(str(path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
    return frame_count / float(frame_rate)


def write_manifests(
    split_records: dict[str, list[dict[str, str | float | int]]],
    processed_dir: Path,
) -> None:
    """Write one CSV per split plus a metadata summary JSON file."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "raw_label",
        "label",
        "split",
        "speaker_id",
        "utterance_id",
        "is_unknown",
        "is_silence",
        "segment_start",
        "segment_duration",
    ]

    metadata: dict[str, object] = {
        "labels": [],
        "examples_per_split": {},
        "label_distribution": {},
    }

    all_labels: set[str] = set()
    for split_name, records in tqdm(split_records.items(), desc="Writing manifests"):
        records.sort(key=lambda record: (str(record["label"]), str(record["path"]), str(record["utterance_id"])))
        manifest_path = processed_dir / f"{split_name}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        # ✅ UPDATED: remove silence from metadata counts
        label_counter = Counter(
            str(record["label"])
            for record in records
            if record["label"] != "silence"
        )

        metadata["examples_per_split"][split_name] = len(records)
        metadata["label_distribution"][split_name] = dict(sorted(label_counter.items()))
        all_labels.update(label_counter)
        print(f"Wrote {manifest_path} with {len(records)} rows")

    # ✅ UPDATED: remove silence from labels list
    metadata["labels"] = sorted(
        label for label in all_labels if label != "silence"
    )

    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata summary to {metadata_path}")


def main() -> None:
    """Entry point used from the command line."""

    config = parse_args()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    download_archive(DATASET_URL, config.archive_path)
    extract_archive(config.archive_path, config.raw_dir)
    manifests = build_manifest_records(config)
    write_manifests(manifests, config.processed_dir)

    summary = {
        "config": asdict(config) | {"label_map": config.resolved_label_map()},
        "processed_dir": str(config.processed_dir),
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()