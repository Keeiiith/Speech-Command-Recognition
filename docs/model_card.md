# 📄 Model Card: Speech Command Recognition for Hands-Free Music Control

## Model Details
- **Model Type:** Spectrogram-based Convolutional Neural Network (CNN) with Reinforcement Learning (RL) threshold optimization  
- **Task:** Speech command classification  
- **Commands:** play, pause, next, stop, unknown, silence  
- **Input:** Log-Mel spectrograms derived from audio signals  
- **Output:** Command class with confidence score  
- **Additional Components:**
  - Reinforcement Learning for threshold tuning  
  - NLP-based metadata search for song queries  

---

## Model Summary (Performance)
- **Test Samples:** 2,416  
- **Accuracy:** 83.11%  
- **Macro F1-Score:** 72.45%  
- **Expected Cost:** 0.4139  
- **Decision Threshold:** 0.5  

The model performs well overall, but performance is uneven across classes, as reflected in the lower Macro F1-score compared to accuracy.

---

## Intended Use
- Hands-free music control systems  
- Voice command recognition applications  
- Educational and research purposes  

**Not intended for:**
- Safety-critical systems  
- Medical or emergency applications  

---

## Training Data
- **Dataset:** Google Speech Commands Dataset V2  
- **License:** Creative Commons (CC BY 4.0)  
- **Preprocessing:**
  - Audio normalization  
  - Conversion to Mel-spectrograms  
  - Resampling and padding  
- **Special Classes:**
  - unknown: sampled from non-target words  
  - silence: generated from low-energy audio  

---

## Evaluation Results

### Per-Class Performance
- **Strong classes:**
  - stop (F1 = 0.9402)  
  - pause (F1 = 0.9019)  
  - next (F1 = 0.8996)  

- **Moderate classes:**
  - play (F1 = 0.8084, lower recall)  
  - unknown (F1 = 0.7971)  

- **Weak class:**
  - silence (F1 = 0.0000)

---

## Limitations
- Poor performance on the silence class  
- Lower recall for the play command  
- Uneven performance across classes  
- Sensitive to:
  - Background noise  
  - Speaker variation  
  - Acoustic conditions  

---

## Ethical Considerations
- **Privacy:**  
  The system uses real-time audio input, which may unintentionally capture background conversations. All processing is done locally, and no audio is stored.

- **Fairness:**  
  Performance may vary across accents and dialects due to dataset limitations.

- **Reliability:**  
  Reinforcement learning is used to reduce false activations by minimizing expected cost.

---

## Reinforcement Learning Component
- **Method:** Q-learning  
- **Purpose:** Optimize decision threshold  
- **Objective:** Minimize expected cost  
- **Behavior:**
  - Penalizes false positives more than false negatives  
  - Adapts threshold based on environment  

---

## NLP Component
- Handles commands like: `play <song name>`  
- Extracts intent and query  
- Matches against a music metadata catalog  
- Uses lightweight similarity-based matching  

---

## Future Improvements
- Improve silence detection  
- Increase dataset diversity  
- Enhance NLP matching  
- Improve RL reward design  
- Reduce class imbalance effects  