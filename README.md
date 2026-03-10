# Speech Command Recognition for Hands-Free Music Control
This project focuses on building a hands-free voice control system for music playback. By converting audio into visual spectrograms and applying Convolutional Neural Networks (CNN), the system identifies specific commands. To solve the common issue of "false triggers" (accidental activations), we utilize a Reinforcement Learning (RL) agent to dynamically optimize decision thresholds.

## Project Overview

In environments with background noise or ambient conversation, standard voice recognition often fails by either ignoring a command or triggering one accidentally. This system addresses this by treating the "confidence threshold" not as a fixed number, but as a dynamic variable learned through Reinforcement Learning.

## Core Technologies

- **Audio Processing:** Short-Time Fourier Transform (STFT) to generate Mel-Spectrograms.
- **Deep Learning:** A lightweight CNN architecture (optimized for mobile/consumer devices) for 4-class command classification.
- **Optimization:** A Reinforcement Learning Agent that adjusts the activation sensitivity based on asymmetric error costs (prioritizing the reduction of annoying false starts).

## Supported Commands

| Command | Action |
|---|---|
| Play | Initiates or resumes audio playback. |
| Pause | Temporarily halts the current track. |
| Next | Skips to the next track in the queue. |
| Stop | Terminates playback and clears the current state. |

## System Pipeline

1. **Audio Input:** Captured via a real-time microphone stream.
2. **Spectrogram Conversion:** Raw .wav data is transformed into a 2D frequency-over-time image.
3. **CNN Inference:** The model predicts the probability of each command.
4. **RL Thresholding:** The RL agent evaluates the prediction confidence and environmental "noise" to decide if the command is legitimate.
5. **Execution:** If verified, the command is sent to the music interface.

## Success Metrics

We evaluate our system beyond simple accuracy to ensure real-world usability:

- **Macro-F1 Score:** To ensure the model doesn't favor one command over others.
- **Expected Cost ($EC$):** A custom metric where False Positives (accidental plays) are penalized more heavily than False Negatives (missed commands).
- **Inference Latency:** Ensuring the "Voice-to-Action" delay is minimal for a seamless user experience.
