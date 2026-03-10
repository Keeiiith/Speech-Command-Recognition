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
