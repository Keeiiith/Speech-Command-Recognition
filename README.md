# Speech Command Recognition for Hands-Free Music Control

## Project Overview

This project implements a hands-free voice interface for music playback designed to operate reliably in noisy environments. By utilizing a CNN to interpret audio spectrograms and a Reinforcement Learning (RL) agent to tune activation thresholds, the system minimizes "false triggers" (accidental activations) which are often more disruptive than a missed command.

## Technical Components

The system integrates three distinct AI paradigms as required by the course specifications:

* **CNN Component (Core):** A lightweight CNN (e.g., MobileNet or a custom architecture) trained on Mel-Spectrograms to classify four specific vocal commands.
* **NLP Component (Auxiliary):** A **Metadata Search** layer that validates the "Play" command by cross-referencing recognized intent with a library of song metadata to ensure contextually valid actions.
* **RL Component:** A **Q-learning** or **Contextual Bandit** agent that optimizes the decision threshold $(\tau)$. The agent learns to adjust sensitivity based on environmental noise to minimize the "Expected Cost" of errors.



## Dataset

* **Source:** Google Speech Commands Dataset V2.
* **Governance:** The dataset is open-source (CC BY 4.0) and contains no PII (Personally Identifiable Information).
* **Splits:** Data is divided into Train, Validation, and Test sets using stratified sampling to prevent leakage.
* **Preprocessing:** Audio files are converted into 2D Mel-Spectrograms before being fed into the CNN.



## Supported Commands

The system follows a specific state-based logic for playback control:

| Command | Action | Functional Logic |
| --- | --- | --- |
| **Play** | Resume | Resumes audio if paused. |
| **Play + [Song Name]** | Search & Play | Triggers **NLP Metadata Search** to find and play a specific track. |
| **Pause** | Stop | Halts the current track immediately. |
| **Next** | Skip | Moves to the next track in the current queue. |
| **Previous** | Go Back | Moves to the previous track in the current queue. |
| **Stop** | Terminate | Ends playback session and clears the current state. |



## System Pipeline

1. **Audio Input:** Captured real-time or from the Google Speech dataset.
2. **Spectrogram Conversion:** Raw audio is processed via STFT into a Mel-Spectrogram.
3. **CNN Inference:** The model outputs probability scores for each supported command.
4. **RL Thresholding:** The RL agent evaluates the confidence score and decides if the threshold is met based on learned costs.
5. **NLP Verification:** If "Play" is detected, the system performs a metadata search to finalize the music request.
6. **Execution:** The validated command is sent to the music control interface.



## Success Metrics

* **Classification:** Accuracy and **Macro-F1 Score** to ensure performance across all command classes.
* **RL Performance:** **Expected Cost (EC)** and Reward learning curves.



## Ethics and Policy

The project includes a formal **Ethics Impact Statement** and **Model Card**:

* **Privacy:** Implementation of data minimization; audio is processed in-memory and not stored.
* **Fairness:** Evaluation includes **Slice Analysis** to detect performance variance across different accents or background noise conditions.
* **Safety:** A clear disclaimer is provided stating the model is not intended for safety-critical or clinical use.


## Repository Pipeline (End-to-End)

1. Prepare data: download Speech Commands v0.02, map labels, and generate manifests.
2. Build features: load audio clips and convert to normalized log-Mel spectrograms.
3. Train classifier: train a compact CNN baseline using train/val splits.
4. Evaluate model: compute accuracy, macro-F1, confusion matrix, and expected cost.
5. Tune threshold with RL: train a Q-learning policy to adapt confidence threshold.
6. Resolve song text with NLP: parse play text and match to song metadata.


## File Responsibilities

- [data/get_data.py](data/get_data.py): downloads/extracts Speech Commands, maps raw labels to intents, writes train/val/test manifests and metadata, and adds sampled unknown plus generated silence examples.
- [src/data_pipeline.py](src/data_pipeline.py): loads manifests, builds datasets/dataloaders, resamples/pads audio, computes normalized log-Mel spectrograms, and supports simple training augmentation.
- [src/train.py](src/train.py): trains the lightweight CNN baseline, tracks train/val loss/accuracy/macro-F1, and saves best checkpoint and logs.
- [src/eval.py](src/eval.py): loads a trained checkpoint, evaluates val/test split with thresholded predictions, exports metrics JSON, confusion matrix CSV, and prediction CSV.
- [src/rl_agent.py](src/rl_agent.py): learns adaptive thresholds via tabular Q-learning, optimizes asymmetric costs, and compares RL policy vs fixed-threshold baseline.
- [src/nlp_metadata.py](src/nlp_metadata.py): detects intent, extracts song queries for play commands, and ranks metadata catalog with lightweight fuzzy/token scoring.
- [src/player_ui.py](src/player_ui.py): Tkinter simulation UI for text/voice commands and local playback demo.


## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```


## Run Commands

From project root (Speech-Command-Recognition):

### 1) Data preparation
```bash
python data/get_data.py
```

### 2) Train baseline CNN
```bash
python src/train.py
```

### 3) Evaluate trained model
```bash
python src/eval.py --split test --threshold 0.5
```

### 4) Train/evaluate RL threshold policy
```bash
python src/rl_agent.py --train-split val --eval-split test
```

### 5) Run NLP metadata-search demo
```bash
python src/nlp_metadata.py --text "play shape of you"
```

With custom catalog JSON:
```bash
python src/nlp_metadata.py --text "play blinding lights" --catalog data/song_catalog.json
```

Catalog format example:
```json
[
	{"title": "Shape of You", "artist": "Ed Sheeran", "album": "Divide"},
	{"title": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours"}
]
```


## Output Artifacts

- experiments/checkpoints/best_model.pt
- experiments/logs/train_history.csv
- experiments/logs/run_summary.json
- experiments/results/test_metrics.json
- experiments/results/test_confusion_matrix.csv
- experiments/results/test_predictions.csv
- experiments/results/rl_threshold_policy.json
- experiments/results/rl_learning_curve.csv
- experiments/results/rl_metrics.json


## Required Metrics Coverage

- Audio classifier: accuracy, macro-F1, confusion matrix.
- RL component: expected cost and learning curve.
- NLP component: intent parsing and metadata retrieval for song-title commands.


## Ethics Summary

See [docs/ethics_statement.md](docs/ethics_statement.md) for detailed privacy, fairness, and risk-mitigation guidelines.


## Demo Music Source and License

- Sample playback tracks in data/music/ are sourced from Artlist.io and treated as royalty-free based on licensed access.
- These tracks are used for UI simulation and command-testing only, not for CNN training data.
- Usage and redistribution remain subject to the active Artlist license terms of the account holder.
- If redistributing this repository publicly, verify your license allows sharing these files; otherwise remove or replace them.


## Tkinter Simulation UI

A lightweight visual simulator lives in [src/player_ui.py](src/player_ui.py).

Features:
- Text command input (play, pause, next, stop)
- Optional microphone voice command input
- NLP intent parsing and song metadata matching
- Player state transitions with local audio playback from data/music/

Run:
```bash
python src/player_ui.py
```

Optional music folder override:
```bash
python src/player_ui.py --music-dir data/music
```

Notes:
- Microphone input needs SpeechRecognition.
- If microphone backend is missing on Windows, text input still works for demo/testing.
