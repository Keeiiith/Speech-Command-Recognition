"""Evaluation entry-point for speech command recognition.

This module loads a trained checkpoint, runs inference on validation or test
splits, and exports report-friendly metrics and artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from data_pipeline import AudioConfig, create_dataloaders
from train import BaselineSpeechCNN, compute_macro_f1


@dataclass(slots=True)
class EvalConfig:
    """Configuration used to evaluate the trained model."""

    checkpoint_path: Path
    manifest_dir: Path
    audio_root: Path
    output_dir: Path
    split: str = "test"
    batch_size: int = 32
    num_workers: int = 0
    threshold: float = 0.5
    false_positive_cost: float = 5.0
    false_negative_cost: float = 2.0
    command_mismatch_cost: float = 1.0


def parse_args() -> EvalConfig:
    """Parse command line arguments into a typed evaluation config."""

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Evaluate a trained speech command model.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root_dir / "experiments" / "checkpoints" / "best_model.pt",
        help="Path to model checkpoint saved by train.py.",
    )
    parser.add_argument("--manifest-dir", type=Path, default=root_dir / "data" / "processed")
    parser.add_argument("--audio-root", type=Path, default=root_dir / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "experiments" / "results")
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum confidence to accept a predicted command.",
    )
    parser.add_argument("--false-positive-cost", type=float, default=5.0)
    parser.add_argument("--false-negative-cost", type=float, default=2.0)
    parser.add_argument("--command-mismatch-cost", type=float, default=1.0)
    args = parser.parse_args()

    return EvalConfig(
        checkpoint_path=args.checkpoint.resolve(),
        manifest_dir=args.manifest_dir.resolve(),
        audio_root=args.audio_root.resolve(),
        output_dir=args.output_dir.resolve(),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        threshold=args.threshold,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        command_mismatch_cost=args.command_mismatch_cost,
    )


def inverse_label_map(label_to_index: dict[str, int]) -> dict[int, str]:
    """Build reverse mapping from integer index to class label."""

    return {index: label for label, index in label_to_index.items()}


@torch.no_grad()
def run_inference(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float,
    silence_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run inference and apply confidence thresholding to predictions."""

    model.eval()
    all_probabilities: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for features, labels, _ in data_loader:
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # Model outputs are converted to probabilities for threshold logic.
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        max_probabilities, predicted_labels = torch.max(probabilities, dim=1)

        # If confidence is too low, force the output to silence/non-trigger.
        # Low-confidence predictions are redirected to silence to reduce
        # accidental command triggers in noisy conditions.
        low_confidence_mask = max_probabilities < threshold
        predicted_labels = predicted_labels.clone()
        predicted_labels[low_confidence_mask] = silence_index

        all_probabilities.append(probabilities.cpu())
        all_predictions.append(predicted_labels.cpu())
        all_targets.append(labels.cpu())

    return (
        torch.cat(all_probabilities),
        torch.cat(all_predictions),
        torch.cat(all_targets),
    )


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Create a dense confusion matrix of shape [num_classes, num_classes]."""

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true_label, predicted_label in zip(targets.tolist(), predictions.tolist()):
        confusion[true_label, predicted_label] += 1
    return confusion


def compute_expected_cost(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    index_to_label: dict[int, str],
    false_positive_cost: float,
    false_negative_cost: float,
    command_mismatch_cost: float,
) -> float:
    """Compute expected cost under asymmetric command-detection penalties."""

    trigger_labels = {label for label in index_to_label.values() if label not in {"silence", "unknown"}}

    total_cost = 0.0
    total_examples = max(1, len(targets))

    for true_index, pred_index in zip(targets.tolist(), predictions.tolist()):
        true_label = index_to_label[true_index]
        predicted_label = index_to_label[pred_index]

        true_is_trigger = true_label in trigger_labels
        pred_is_trigger = predicted_label in trigger_labels

        # False activation of a command is highest penalty by design.
        if pred_is_trigger and not true_is_trigger:
            total_cost += false_positive_cost
        elif true_is_trigger and not pred_is_trigger:
            total_cost += false_negative_cost
        elif true_is_trigger and pred_is_trigger and true_label != predicted_label:
            total_cost += command_mismatch_cost

    return total_cost / total_examples


def compute_per_class_metrics(confusion: torch.Tensor, index_to_label: dict[int, str]) -> list[dict[str, float | str]]:
    """Compute precision, recall, and F1 for every class from confusion matrix."""

    metrics: list[dict[str, float | str]] = []
    for class_index, label_name in sorted(index_to_label.items()):
        true_positive = confusion[class_index, class_index].item()
        false_positive = confusion[:, class_index].sum().item() - true_positive
        false_negative = confusion[class_index, :].sum().item() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(
            {
                "label": label_name,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
        )
    return metrics


def save_confusion_matrix_csv(confusion: torch.Tensor, index_to_label: dict[int, str], output_path: Path) -> None:
    """Save confusion matrix as CSV for reports and spreadsheet analysis."""

    label_order = [index_to_label[index] for index in sorted(index_to_label)]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true\\pred"] + label_order)
        for row_index, row_label in enumerate(label_order):
            row_values = confusion[row_index].tolist()
            writer.writerow([row_label] + row_values)


def evaluate(config: EvalConfig) -> dict[str, Any]:
    """Load checkpoint, run evaluation, and write result artifacts."""

    # 1) Restore label mapping and model weights from training checkpoint.
    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = inverse_label_map(label_to_index)

    # If silence does not exist for any reason, fallback to unknown then 0.
    silence_index = label_to_index.get("silence", label_to_index.get("unknown", 0))

    # 2) Rebuild dataloaders for the selected split.
    audio_config = AudioConfig(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    data_loaders, _ = create_dataloaders(
        manifest_dir=config.manifest_dir,
        audio_root=config.audio_root,
        config=audio_config,
    )

    model = BaselineSpeechCNN(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Run thresholded inference and gather prediction tensors.
    probabilities, predictions, targets = run_inference(
        model=model,
        data_loader=data_loaders[config.split],
        device=device,
        threshold=config.threshold,
        silence_index=silence_index,
    )

    # 4) Compute summary metrics and class-wise diagnostics.
    confusion = compute_confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=len(label_to_index),
    )

    accuracy = (predictions == targets).float().mean().item()
    macro_f1 = compute_macro_f1(
        predicted_labels=predictions,
        true_labels=targets,
        num_classes=len(label_to_index),
    )
    expected_cost = compute_expected_cost(
        predictions=predictions,
        targets=targets,
        index_to_label=index_to_label,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
        command_mismatch_cost=config.command_mismatch_cost,
    )

    per_class = compute_per_class_metrics(confusion, index_to_label)

    # 5) Persist machine-readable artifacts for reports and error analysis.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = config.output_dir / f"{config.split}_confusion_matrix.csv"
    save_confusion_matrix_csv(confusion, index_to_label, confusion_path)

    results = {
        "split": config.split,
        "num_examples": int(len(targets)),
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "expected_cost": round(float(expected_cost), 6),
        "threshold": config.threshold,
        "label_to_index": label_to_index,
        "per_class_metrics": per_class,
        "output_files": {
            "confusion_matrix_csv": str(confusion_path),
        },
    }

    results_path = config.output_dir / f"{config.split}_metrics.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Save prediction-level details for error analysis and slicing later.
    predictions_path = config.output_dir / f"{config.split}_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true_label", "predicted_label", "max_probability"])
        max_probabilities = probabilities.max(dim=1).values
        for true_idx, pred_idx, max_prob in zip(targets.tolist(), predictions.tolist(), max_probabilities.tolist()):
            writer.writerow([index_to_label[true_idx], index_to_label[pred_idx], round(float(max_prob), 6)])

    results["output_files"]["metrics_json"] = str(results_path)
    results["output_files"]["predictions_csv"] = str(predictions_path)
    return results


def main() -> None:
    """CLI entry point for running evaluation."""

    config = parse_args()
    metrics = evaluate(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
