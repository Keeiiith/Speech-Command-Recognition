"""
Data loading utilities for the speech command recognition project.

This module reads CSV manifests and exposes a PyTorch Dataset that returns
log-Mel spectrograms plus label indices.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
    from torchaudio import functional as audio_functional
    from torchaudio import transforms as audio_transforms
except ImportError as exc:
    raise ImportError(
        "torchaudio is required for the audio pipeline. Install torch and torchaudio first."
    ) from exc


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16000
    clip_duration_seconds: float = 1.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 20.0
    f_max: float | None = 7600.0
    batch_size: int = 32
    num_workers: int = 0
    normalize_waveform: bool = True
    add_training_noise: bool = False
    noise_scale: float = 0.005
    time_mask_param: int = 24
    frequency_mask_param: int = 8

    @property
    def clip_num_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_seconds)


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest = Path(manifest_path)
    with manifest.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader]


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def build_label_index(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(labels))}


class SpeechCommandsDataset(Dataset[tuple[torch.Tensor, int, dict[str, Any]]]):

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: str | Path,
        label_to_index: dict[str, int],
        config: AudioConfig | None = None,
        augment: bool = False,
    ) -> None:
        self.records = load_manifest(manifest_path)
        self.audio_root = Path(audio_root)
        self.label_to_index = label_to_index
        self.config = config or AudioConfig()
        self.augment = augment

        self.mel_transform = audio_transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            center=True,
            power=2.0,
        )
        self.db_transform = audio_transforms.AmplitudeToDB(stype="power")
        self.time_mask = audio_transforms.TimeMasking(time_mask_param=self.config.time_mask_param)
        self.freq_mask = audio_transforms.FrequencyMasking(freq_mask_param=self.config.frequency_mask_param)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        waveform = self._load_waveform(record)
        spectrogram = self._waveform_to_features(waveform)

        # ✅ FIX: safe label mapping
        label = record["label"]
        if label not in self.label_to_index:
            label = "unknown"

        label_index = self.label_to_index[label]

        metadata = {
            "path": record["path"],
            "raw_label": record["raw_label"],
            "label": label,
            "speaker_id": record["speaker_id"],
            "utterance_id": record["utterance_id"],
            "is_unknown": bool(int(record["is_unknown"])),
            "is_silence": bool(int(record["is_silence"])),
        }

        return spectrogram, label_index, metadata

    def _load_waveform(self, record: dict[str, str]) -> torch.Tensor:
        if bool(int(record["is_silence"])) and not record["path"]:
            return torch.zeros(1, self.config.clip_num_samples)

        audio_path = self.audio_root / record["path"]
        waveform, sr = torchaudio.load(str(audio_path))

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.config.sample_rate:
            waveform = audio_functional.resample(waveform, sr, self.config.sample_rate)

        waveform = self._slice_segment(waveform, record)
        waveform = self._fit_to_fixed_length(waveform)

        if self.config.normalize_waveform:
            peak = waveform.abs().max().clamp(min=1e-6)
            waveform = waveform / peak

        if self.augment and self.config.add_training_noise:
            waveform = waveform + torch.randn_like(waveform) * self.config.noise_scale

        return waveform

    def _slice_segment(self, waveform: torch.Tensor, record: dict[str, str]) -> torch.Tensor:
        start = float(record.get("segment_start", 0.0) or 0.0)
        duration = float(record.get("segment_duration", self.config.clip_duration_seconds))

        if start <= 0.0:
            return waveform

        start_idx = int(start * self.config.sample_rate)
        end_idx = start_idx + int(duration * self.config.sample_rate)

        return waveform[:, start_idx:end_idx]

    def _fit_to_fixed_length(self, waveform: torch.Tensor) -> torch.Tensor:
        target = self.config.clip_num_samples
        current = waveform.size(-1)

        if current > target:
            return waveform[:, :target]
        if current < target:
            return torch.nn.functional.pad(waveform, (0, target - current))

        return waveform

    def _waveform_to_features(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)

        if self.augment:
            spec = self.time_mask(spec)
            spec = self.freq_mask(spec)

        mean = spec.mean()
        std = spec.std().clamp(min=1e-6)
        return (spec - mean) / std


def create_datasets(
    manifest_dir: str | Path,
    audio_root: str | Path,
    config: AudioConfig | None = None,
):
    manifest_root = Path(manifest_dir)
    metadata = load_metadata(manifest_root / "metadata.json")

    labels = metadata.get("labels", [])

    # ✅ Important: ensure "unknown" exists
    if "unknown" not in labels:
        labels.append("unknown")

    label_to_index = build_label_index(labels)

    cfg = config or AudioConfig()

    datasets = {
        "train": SpeechCommandsDataset(manifest_root / "train_manifest.csv", audio_root, label_to_index, cfg, augment=True),
        "val": SpeechCommandsDataset(manifest_root / "val_manifest.csv", audio_root, label_to_index, cfg, augment=False),
        "test": SpeechCommandsDataset(manifest_root / "test_manifest.csv", audio_root, label_to_index, cfg, augment=False),
    }

    return datasets, label_to_index


def create_dataloaders(
    manifest_dir: str | Path,
    audio_root: str | Path,
    config: AudioConfig | None = None,
):
    cfg = config or AudioConfig()
    datasets, label_to_index = create_datasets(manifest_dir, audio_root, cfg)

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }

    return loaders, label_to_index