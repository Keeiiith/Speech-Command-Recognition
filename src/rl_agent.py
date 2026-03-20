"""Reinforcement learning threshold tuning for speech-command inference.

This module trains a lightweight Q-learning agent that learns which confidence
threshold to apply based on model uncertainty signals, minimizing asymmetric
error costs (especially false activations).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from data_pipeline import AudioConfig, create_dataloaders
from train import BaselineSpeechCNN


@dataclass(slots=True)
class RLConfig:
    """Configuration for RL-based threshold tuning."""

    checkpoint_path: Path
    manifest_dir: Path
    audio_root: Path
    output_dir: Path
    train_split: str = "val"
    eval_split: str = "test"
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 42
    episodes: int = 25
    learning_rate: float = 0.15
    discount_factor: float = 0.9
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    confidence_bins: int = 10
    entropy_bins: int = 6
    false_positive_cost: float = 5.0
    false_negative_cost: float = 2.0
    mismatch_cost: float = 1.0


@dataclass(slots=True)
class CostWeights:
    """Asymmetric penalties used by the reward and expected-cost objectives."""

    false_positive: float
    false_negative: float
    mismatch: float


@dataclass(slots=True)
class Sample:
    """Single model inference sample used by the RL environment."""

    true_index: int
    predicted_index: int
    confidence: float
    entropy: float


def parse_args() -> RLConfig:
    """Parse CLI options for RL threshold tuning."""

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Train an RL agent for adaptive threshold tuning.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root_dir / "experiments" / "checkpoints" / "best_model.pt",
    )
    parser.add_argument("--manifest-dir", type=Path, default=root_dir / "data" / "processed")
    parser.add_argument("--audio-root", type=Path, default=root_dir / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=root_dir / "experiments" / "results")
    parser.add_argument("--train-split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--eval-split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--confidence-bins", type=int, default=10)
    parser.add_argument("--entropy-bins", type=int, default=6)
    parser.add_argument("--false-positive-cost", type=float, default=5.0)
    parser.add_argument("--false-negative-cost", type=float, default=2.0)
    parser.add_argument("--mismatch-cost", type=float, default=1.0)
    args = parser.parse_args()

    return RLConfig(
        checkpoint_path=args.checkpoint.resolve(),
        manifest_dir=args.manifest_dir.resolve(),
        audio_root=args.audio_root.resolve(),
        output_dir=args.output_dir.resolve(),
        train_split=args.train_split,
        eval_split=args.eval_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        confidence_bins=args.confidence_bins,
        entropy_bins=args.entropy_bins,
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        mismatch_cost=args.mismatch_cost,
    )


def set_seed(seed: int) -> None:
    """Set deterministic seeds for repeatable RL tuning runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    # Keeps behavior stable across GPU runs when CUDA is available.
    torch.cuda.manual_seed_all(seed)


def inverse_label_map(label_to_index: dict[str, int]) -> dict[int, str]:
    """Build reverse mapping from class index to class name."""

    return {index: label for label, index in label_to_index.items()}


@torch.no_grad()
def collect_samples(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> list[Sample]:
    """Run model inference once and convert outputs to RL training samples."""

    model.eval()
    samples: list[Sample] = []
    for features, labels, _ in data_loader:
        features = features.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # We collect softmax confidence + entropy because they summarize model
        # certainty and are compact enough for tabular RL states.
        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

        # Entropy is used as a compact uncertainty signal for the RL state.
        entropy = -(probabilities * torch.log(probabilities.clamp(min=1e-9))).sum(dim=1)

        for true_idx, pred_idx, conf, ent in zip(
            labels.tolist(), predictions.tolist(), confidence.tolist(), entropy.tolist()
        ):
            samples.append(
                Sample(
                    true_index=int(true_idx),
                    predicted_index=int(pred_idx),
                    confidence=float(conf),
                    entropy=float(ent),
                )
            )

    return samples


def create_actions() -> list[float]:
    """Discrete thresholds that the RL policy can choose from."""

    return [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def build_state(sample: Sample, confidence_bins: int, entropy_bins: int, max_entropy: float) -> tuple[int, int]:
    """Map continuous confidence/entropy into a compact discrete RL state."""

    confidence_index = min(confidence_bins - 1, int(sample.confidence * confidence_bins))

    normalized_entropy = sample.entropy / max(max_entropy, 1e-8)
    entropy_index = min(entropy_bins - 1, int(normalized_entropy * entropy_bins))
    return confidence_index, entropy_index


def compute_cost_for_decision(
    true_label: str,
    predicted_label: str,
    accepted: bool,
    trigger_labels: set[str],
    cost_weights: CostWeights,
) -> float:
    """Cost function aligned with project requirement on asymmetric errors."""

    true_is_trigger = true_label in trigger_labels

    if accepted:
        if not true_is_trigger:
            return cost_weights.false_positive
        if predicted_label != true_label:
            return cost_weights.mismatch
        return 0.0

    if true_is_trigger:
        return cost_weights.false_negative
    return 0.0


def train_q_learning_policy(
    samples: list[Sample],
    index_to_label: dict[int, str],
    config: RLConfig,
    actions: list[float],
) -> tuple[dict[tuple[int, int], list[float]], list[dict[str, float]]]:
    """Train a tabular Q-learning policy for threshold selection."""

    trigger_labels = {label for label in index_to_label.values() if label not in {"silence", "unknown"}}
    cost_weights = CostWeights(
        false_positive=config.false_positive_cost,
        false_negative=config.false_negative_cost,
        mismatch=config.mismatch_cost,
    )

    if not samples:
        return {}, []

    max_entropy = max(sample.entropy for sample in samples)
    q_table: dict[tuple[int, int], list[float]] = {}
    history: list[dict[str, float]] = []

    for episode in range(1, config.episodes + 1):
        # Linearly anneal epsilon from exploration-heavy to policy-focused.
        epsilon_fraction = (episode - 1) / max(1, config.episodes - 1)
        epsilon = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * epsilon_fraction

        shuffled = samples.copy()
        random.shuffle(shuffled)

        total_reward = 0.0
        for sample in shuffled:
            state = build_state(sample, config.confidence_bins, config.entropy_bins, max_entropy)
            q_values = q_table.setdefault(state, [0.0 for _ in actions])

            # Epsilon-greedy action selection balances exploration and exploitation.
            if random.random() < epsilon:
                action_index = random.randrange(len(actions))
            else:
                action_index = int(max(range(len(actions)), key=lambda idx: q_values[idx]))

            # Action == chosen confidence threshold for this uncertainty state.
            threshold = actions[action_index]
            accepted = sample.confidence >= threshold
            true_label = index_to_label[sample.true_index]
            predicted_label = index_to_label[sample.predicted_index]

            immediate_cost = compute_cost_for_decision(
                true_label=true_label,
                predicted_label=predicted_label,
                accepted=accepted,
                trigger_labels=trigger_labels,
                cost_weights=cost_weights,
            )
            reward = -immediate_cost
            total_reward += reward

            # Single-step environment: use state's best value as bootstrap target.
            best_future_q = max(q_values)
            old_q = q_values[action_index]
            q_values[action_index] = old_q + config.learning_rate * (
                reward + config.discount_factor * best_future_q - old_q
            )

        average_reward = total_reward / max(1, len(shuffled))
        history.append(
            {
                "episode": float(episode),
                "epsilon": round(float(epsilon), 6),
                "average_reward": round(float(average_reward), 6),
            }
        )
        print(f"Episode {episode:02d}/{config.episodes} | epsilon={epsilon:.3f} | avg_reward={average_reward:.4f}")

    return q_table, history


def policy_threshold_for_sample(
    sample: Sample,
    q_table: dict[tuple[int, int], list[float]],
    actions: list[float],
    confidence_bins: int,
    entropy_bins: int,
    max_entropy: float,
    fallback_threshold: float,
) -> float:
    """Select threshold from policy for one sample, falling back if unseen state."""

    state = build_state(sample, confidence_bins, entropy_bins, max_entropy)
    if state not in q_table:
        return fallback_threshold

    best_action_index = int(max(range(len(actions)), key=lambda idx: q_table[state][idx]))
    return actions[best_action_index]


def evaluate_policy(
    samples: list[Sample],
    index_to_label: dict[int, str],
    q_table: dict[tuple[int, int], list[float]],
    actions: list[float],
    config: RLConfig,
    fallback_threshold: float,
) -> dict[str, float]:
    """Compare RL policy against fixed threshold baseline on expected cost."""

    trigger_labels = {label for label in index_to_label.values() if label not in {"silence", "unknown"}}
    cost_weights = CostWeights(
        false_positive=config.false_positive_cost,
        false_negative=config.false_negative_cost,
        mismatch=config.mismatch_cost,
    )

    if not samples:
        return {
            "rl_expected_cost": 0.0,
            "baseline_expected_cost": 0.0,
            "cost_reduction": 0.0,
            "rl_accept_rate": 0.0,
            "baseline_accept_rate": 0.0,
        }

    max_entropy = max(sample.entropy for sample in samples)

    rl_total_cost = 0.0
    baseline_total_cost = 0.0
    rl_accept_count = 0
    baseline_accept_count = 0

    for sample in samples:
        true_label = index_to_label[sample.true_index]
        predicted_label = index_to_label[sample.predicted_index]

        # RL policy can vary threshold per state; baseline is a fixed 0.5 gate.
        rl_threshold = policy_threshold_for_sample(
            sample=sample,
            q_table=q_table,
            actions=actions,
            confidence_bins=config.confidence_bins,
            entropy_bins=config.entropy_bins,
            max_entropy=max_entropy,
            fallback_threshold=fallback_threshold,
        )
        rl_accept = sample.confidence >= rl_threshold
        baseline_accept = sample.confidence >= fallback_threshold

        if rl_accept:
            rl_accept_count += 1
        if baseline_accept:
            baseline_accept_count += 1

        rl_total_cost += compute_cost_for_decision(
            true_label=true_label,
            predicted_label=predicted_label,
            accepted=rl_accept,
            trigger_labels=trigger_labels,
            cost_weights=cost_weights,
        )
        baseline_total_cost += compute_cost_for_decision(
            true_label=true_label,
            predicted_label=predicted_label,
            accepted=baseline_accept,
            trigger_labels=trigger_labels,
            cost_weights=cost_weights,
        )

    sample_count = len(samples)
    rl_expected_cost = rl_total_cost / sample_count
    baseline_expected_cost = baseline_total_cost / sample_count

    return {
        "rl_expected_cost": round(float(rl_expected_cost), 6),
        "baseline_expected_cost": round(float(baseline_expected_cost), 6),
        "cost_reduction": round(float(baseline_expected_cost - rl_expected_cost), 6),
        "rl_accept_rate": round(float(rl_accept_count / sample_count), 6),
        "baseline_accept_rate": round(float(baseline_accept_count / sample_count), 6),
    }


def write_training_curve(history: list[dict[str, float]], output_path: Path) -> None:
    """Save RL learning curve values for plotting in reports/slides."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["episode", "epsilon", "average_reward"])
        writer.writeheader()
        writer.writerows(history)


def serialize_policy(
    q_table: dict[tuple[int, int], list[float]],
    actions: list[float],
) -> dict[str, Any]:
    """Convert Q-table to JSON-friendly format for reuse in inference."""

    serialized_states = []
    for (confidence_idx, entropy_idx), q_values in sorted(q_table.items()):
        best_action_index = int(max(range(len(actions)), key=lambda idx: q_values[idx]))
        serialized_states.append(
            {
                "state": {
                    "confidence_bin": confidence_idx,
                    "entropy_bin": entropy_idx,
                },
                "q_values": [round(float(value), 6) for value in q_values],
                "best_threshold": actions[best_action_index],
            }
        )

    return {
        "actions": actions,
        "states": serialized_states,
    }


def main() -> None:
    """CLI entry point for RL threshold tuning and evaluation."""

    # 1) Load checkpoint and data.
    config = parse_args()
    set_seed(config.seed)

    checkpoint = torch.load(config.checkpoint_path, map_location="cpu")
    label_to_index: dict[str, int] = checkpoint["label_to_index"]
    index_to_label = inverse_label_map(label_to_index)

    audio_config = AudioConfig(batch_size=config.batch_size, num_workers=config.num_workers)
    data_loaders, _ = create_dataloaders(
        manifest_dir=config.manifest_dir,
        audio_root=config.audio_root,
        config=audio_config,
    )

    model = BaselineSpeechCNN(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use validation split for policy learning, and test split for holdout evaluation.
    # 2) Collect model outputs as RL environment samples.
    training_samples = collect_samples(model, data_loaders[config.train_split], device)
    evaluation_samples = collect_samples(model, data_loaders[config.eval_split], device)

    # 3) Learn adaptive threshold policy and evaluate against baseline.
    actions = create_actions()
    q_table, history = train_q_learning_policy(
        samples=training_samples,
        index_to_label=index_to_label,
        config=config,
        actions=actions,
    )

    baseline_threshold = 0.5
    metrics = evaluate_policy(
        samples=evaluation_samples,
        index_to_label=index_to_label,
        q_table=q_table,
        actions=actions,
        config=config,
        fallback_threshold=baseline_threshold,
    )

    # 4) Save policy, learning curve, and summary metrics.
    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = config.output_dir / "rl_threshold_policy.json"
    curve_path = config.output_dir / "rl_learning_curve.csv"
    summary_path = config.output_dir / "rl_metrics.json"

    policy_payload = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in asdict(config).items()},
        "label_to_index": label_to_index,
        "policy": serialize_policy(q_table, actions),
    }
    policy_path.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")

    write_training_curve(history, curve_path)

    summary = {
        "train_split": config.train_split,
        "eval_split": config.eval_split,
        "num_training_samples": len(training_samples),
        "num_evaluation_samples": len(evaluation_samples),
        "baseline_threshold": baseline_threshold,
        "metrics": metrics,
        "output_files": {
            "policy_json": str(policy_path),
            "learning_curve_csv": str(curve_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
