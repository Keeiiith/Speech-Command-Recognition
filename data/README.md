# Data Folder Notes

## Purpose
This folder stores both machine-learning dataset artifacts and local demo music resources.

## Subfolders
- `raw/`: Extracted Google Speech Commands dataset files used for model training/evaluation.
- `processed/`: Generated manifests and metadata (`train/val/test` CSV + `metadata.json`).
- `music/`: Local sample songs used by the Tkinter player UI for playback simulation.

## Music Source and License Clarification
- Songs currently in `data/music/` are sourced from **Artlist.io** and treated as royalty-free under the team's licensed usage.
- These files are used for UI/demo playback and command testing, not as model training labels.
- Redistribution of these specific files depends on the active Artlist license terms; remove or replace them before public sharing if needed.

## Privacy and Compliance
- Do not store private microphone recordings in this folder.
- Keep only approved and licensed audio assets.
