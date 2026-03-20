# Ethics Statement

## Project Context
This project builds a speech-command recognition system for hands-free music control using:
- a CNN audio classifier on Mel-spectrograms,
- an RL threshold-tuning agent to reduce harmful false activations,
- a lightweight NLP metadata-search layer for `play <song name>` requests.

The intended use is convenience control for non-critical media playback. The system is **not** intended for medical, legal, security, emergency, or other safety-critical decision contexts.

## Core Ethical Risks and Mitigations

### 1) Privacy and Unintended Audio Capture
**Risk:** Microphone-based systems may capture speech that users did not intend to submit as commands.

**Mitigations:**
- Process audio in-memory whenever possible; avoid long-term raw audio retention by default.
- Store only aggregate experiment metrics and anonymized model outputs for research/reporting.
- If any clips must be logged for debugging, require explicit opt-in and provide deletion controls.
- Show clear user-facing notice that microphone input is being used.

### 2) Accent/Dialect Fairness
**Risk:** Performance can vary across accents, dialects, speaking pace, and voice pitch, causing unequal usability.

**Mitigations:**
- Report macro-F1 (not only accuracy) to avoid hiding minority-class and subgroup failures.
- Perform slice analysis on available speaker metadata and noise conditions.
- Use data augmentation and threshold-tuning policies that reduce over-confident triggering on unfamiliar speech.
- Document known performance gaps in the Model Card and include usage caveats.

### 3) False Activations and User Trust
**Risk:** Accidental command execution (false positives) can disrupt user experience and reduce trust.

**Mitigations:**
- Use asymmetric error costs that penalize false positives more than false negatives.
- Tune confidence threshold with RL (`src/rl_agent.py`) to adapt to uncertain/noisy contexts.
- Include `silence` and `unknown` handling to reduce unnecessary command triggering.
- Evaluate expected cost in addition to accuracy/F1 for realistic operational behavior.

### 4) Metadata and Content Search Misfires
**Risk:** The NLP layer may match the wrong song title and execute unintended playback.

**Mitigations:**
- Require a minimum confidence/score before accepting metadata match.
- Use fallback behavior (ask for repeat/clarification) when confidence is low.
- Keep search logic transparent and lightweight (`src/nlp_metadata.py`) so it is auditable.

## Data Governance
- Dataset source: Google Speech Commands v0.02 (public research dataset).
- Demo playback source: local songs in `data/music/` are sourced from Artlist.io under royalty-free licensed use by the team.
- Licensing: Follow dataset license terms and attribution requirements, and respect Artlist account license conditions for music usage/redistribution.
- Personal data: Do not add personally identifiable information (PII) to repository artifacts.
- Storage hygiene: Do not commit private recordings or raw user microphone captures.
## Transparency and Documentation Commitments
- Provide reproducible training/evaluation commands in `README.md`.
- Publish model limitations, risks, and non-intended uses in final report and model card.
- Maintain clear versioned experiment outputs in `experiments/` for auditability.

## Responsible Use Policy
Users and evaluators should treat system output as assistive automation for media control only. If model confidence is low or behavior appears unstable in noisy environments, commands should be retried or manually overridden.

## Known Limitations
- Limited command vocabulary may not generalize to unconstrained speech.
- Performance may degrade in very noisy environments or for underrepresented accents.
- RL threshold policy is optimized for expected-cost tradeoffs, not perfect semantic understanding.

## Continuous Improvement Plan
- Expand slice analysis and fairness reporting as more speaker-condition metadata becomes available.
- Compare RL policy against multiple baselines and document tradeoffs.
- Regularly review and adjust asymmetric cost weights with user-centered testing.
