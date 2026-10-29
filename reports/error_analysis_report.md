# VoxShield Error Analysis & Disagreement Report

**Test Population Evaluated**: 153 held-out samples.

- **False Positives (Legitimate flagged as Scam)**: 0
- **False Negatives (Scam misclassified as Legitimate)**: 0
- **High Confidence False Positives (Prob > 0.80)**: 0
- **High Confidence False Negatives (Prob < 0.20)**: 0

## 1. Qualitative Review of False Positives (Legitimate Calls Flagged as Suspicious)
*No false positives encountered on held-out test split.*

## 2. Qualitative Review of False Negatives (Missed Scam Calls)
*No false negatives encountered on held-out test split.*

## 3. Disagreement Between Text and Audio Modalities
Found 1 samples where text and audio model probabilities diverged by > 40%.

- **Sample `lot15_en_legit_1455`** (True: `legitimate`): Text Prob=0.21, Audio Prob=0.89
  *Transcript*: "Congratulations! Your new loan application for $15,000 has been approved at a fixed rate of 5%. The funds will be disbur..."
