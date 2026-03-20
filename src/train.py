"""
Training entry-point for speech command classification.

This module trains a lightweight CNN on Mel-spectrogram inputs produced by the
data_pipeline module and saves checkpoints plus metric logs for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm  # ✅ ADDED ONLY

from data_pipeline import AudioConfig, create_dataloaders


@dataclass(slots=True)
class TrainConfig:
    """Config values used to control model training and outputs."""

    manifest_dir: Path
    audio_root: Path
    output_dir: Path
    logs_dir: Path
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    patience: int = 4


class BaselineSpeechCNN(nn.Module):
    """A compact CNN baseline for classifying log-Mel spectrograms."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # The feature extractor learns time-frequency patterns from spectrograms.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.15),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),
        )

        # Adaptive pooling removes dependency on exact spectrogram dimensions.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(features))


def parse_args() -> TrainConfig:
    """Convert CLI options to a strongly typed configuration object."""

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Train the baseline speech command CNN model.")
    parser.add_argument("--manifest-dir", type=Path, default=root_dir / "data" / "processed")
    parser.add_argument("--audio-root", type=Path, default=root_dir / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "experiments" / "checkpoints")
    parser.add_argument("--logs-dir", type=Path, default=root_dir / "experiments" / "logs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    return TrainConfig(
        manifest_dir=args.manifest_dir.resolve(),
        audio_root=args.audio_root.resolve(),
        output_dir=args.output_dir.resolve(),
        logs_dir=args.logs_dir.resolve(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
    )


def set_seed(seed: int) -> None:
    """Set random seeds so runs are as reproducible as practical."""

    random.seed(seed)
    torch.manual_seed(seed)
    # This call is safe on CPU-only systems and helps when CUDA is available.
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_macro_f1(
    predicted_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-F1 without third-party metrics dependencies."""

    f1_scores: list[float] = []
    for class_index in range(num_classes):
        predicted_positive = predicted_labels == class_index
        actual_positive = true_labels == class_index

        true_positive = torch.logical_and(predicted_positive, actual_positive).sum().item()
        false_positive = torch.logical_and(predicted_positive, ~actual_positive).sum().item()
        false_negative = torch.logical_and(~predicted_positive, actual_positive).sum().item()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator > 0 else 0.0
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))

    return float(sum(f1_scores) / max(1, len(f1_scores)))


def run_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and return aggregate metrics."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    loop = tqdm(
        data_loader,
        desc="Training" if optimizer is not None else "Validation",
        leave=False
    )

    for features, labels, _ in loop:
        # Move tensors to CPU/GPU and ensure correct dtype for training.
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass creates class logits for each audio clip.
        logits = model(features)
        loss = criterion(logits, labels)
        loop.set_postfix(loss=loss.item())

        if is_training:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        # Convert logits into hard class predictions for metrics tracking.
        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(labels.detach().cpu())

    if not all_targets:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0}

    targets = torch.cat(all_targets)
    predictions = torch.cat(all_predictions)
    accuracy = (predictions == targets).float().mean().item()
    macro_f1 = compute_macro_f1(predictions, targets, num_classes=model.classifier[-1].out_features)
    average_loss = running_loss / len(targets)

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def save_history(history: list[dict[str, float]], logs_dir: Path) -> Path:
    """Persist epoch metrics in CSV format for plotting and analysis."""

    logs_dir.mkdir(parents=True, exist_ok=True)
    history_path = logs_dir / "train_history.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]

    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return history_path


def save_run_summary(
    config: TrainConfig,
    label_to_index: dict[str, int],
    best_metrics: dict[str, float],
    logs_dir: Path,
) -> Path:
    """Write a compact JSON summary that documents the training run."""

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "labels": label_to_index,
        "best_validation_metrics": best_metrics,
    }

    summary_path = logs_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def train(config: TrainConfig) -> dict[str, Any]:
    """Train the CNN, save artifacts, and return final run information."""

    # 1) Initialize reproducible environment and output folders.
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    # 2) Build data pipeline from manifests.
    audio_config = AudioConfig(batch_size=config.batch_size, num_workers=config.num_workers)
    data_loaders, label_to_index = create_dataloaders(
        manifest_dir=config.manifest_dir,
        audio_root=config.audio_root,
        config=audio_config,
    )

    # 3) Build model + optimization components.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineSpeechCNN(num_classes=len(label_to_index)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = config.output_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        # 4) Train on train split, then evaluate on validation split.
        train_metrics = run_epoch(
            model=model,
            data_loader=data_loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model=model,
            data_loader=data_loaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_macro_f1": round(train_metrics["macro_f1"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"train_f1={epoch_metrics['train_macro_f1']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_f1={epoch_metrics['val_macro_f1']:.4f}"
        )

        current_val_f1 = val_metrics["macro_f1"]
        # 5) Track best checkpoint by validation macro-F1.
        if current_val_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "label_to_index": label_to_index,
                    "audio_config": asdict(audio_config),
                    "train_config": asdict(config),
                    "metrics": {
                        "train": train_metrics,
                        "val": val_metrics,
                    },
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        # 6) Early stopping protects against overfitting and wasted compute.
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch} after {config.patience} epochs without improvement.")
            break

    history_path = save_history(history, config.logs_dir)
    summary_path = save_run_summary(
        config=config,
        label_to_index=label_to_index,
        best_metrics={
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_macro_f1, 6),
        },
        logs_dir=config.logs_dir,
    )

    return {
        "best_model_path": best_model_path,
        "history_path": history_path,
        "summary_path": summary_path,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
    }


def main() -> None:
    """CLI entry point for model training."""

    config = parse_args()
    results = train(config)

    printable_results = {
        key: str(value) if isinstance(value, Path) else value for key, value in results.items()
    }
    print(json.dumps(printable_results, indent=2))


if __name__ == "__main__":
    main()
