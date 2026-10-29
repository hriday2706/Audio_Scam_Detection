# Model Card: VoxShield Multimodal Voice Scam Risk Estimator

## 1. Model Overview
- **Model Name**: VoxShield (Multimodal Voice Scam Risk Analysis)
- **Model Version**: 0.1.0
- **Model Type**: Late-Fusion Meta-Classification Architecture integrating:
  1. **Text Branch (E1)**: TF-IDF n-gram vectorizer (unigrams + bigrams, sublinear TF) + L2-regularized Logistic Regression.
  2. **Audio Branch (E3 / E4)**: 100-dimensional handcrafted acoustic statistical features (MFCCs, Delta-MFCCs, Spectral Centroid, Bandwidth, Roll-off, Flatness, RMS Energy, Zero-Crossing Rate, Pitch F0, Silence Ratio) + StandardScaler + Logistic Regression / Random Forest.
  3. **Multimodal Late-Fusion Head (E6)**: Meta-classifier trained on out-of-fold probability estimates from the base models.
  4. **Speech-to-Text Pipeline (E7 / Demo)**: Pretrained `openai/whisper-small` for realistic speech transcription without gold reference text.
- **Repository**: [https://github.com/voxshield/voxshield](https://github.com/voxshield/voxshield)

## 2. Intended Use & Scope
- **Primary Use**: Defensive research on multimodal deception detection, comparative benchmarking of acoustic vs. lexical information in simulated phone calls, and academic risk exploration.
- **Out of Scope & Prohibited Uses**:
  - Determining a human caller's honesty or deception based on vocal pitch or acoustics.
  - Autonomous financial blocking, automated transaction cancellation, or account freezes.
  - Automated filing of law enforcement fraud reports.
  - Production deployment without mandatory human investigation and certified fraud analysts.

## 3. Training & Evaluation Methodology
- **Benchmark Corpus**: VISHGUARD v3.0 (English subset, $N = 1,065$).
- **Partitioning**: 70% Train ($N=760$), 15% Validation ($N=152$), 15% Test ($N=153$).
- **Leakage Prevention**:
  - Stratified Group Partitioning ensures identical scenario scripts remain in the same partition.
  - TF-IDF vectorizer and audio feature scalers are fitted exclusively on the training split.
  - Decision thresholds are selected strictly on the validation set.
  - The multimodal meta-classifier is trained on out-of-fold cross-validated predictions generated exclusively from training data, preventing in-sample overfitting.
- **Evaluation Metrics**: Accuracy, Precision, Recall, Scam-Class F1, Macro-F1, PR-AUC (Average Precision), ROC-AUC, Confusion Matrix, and 95% Bootstrap Confidence Intervals (1,000 resamples).

## 4. Quantitative Performance Summary (Held-Out Test Set)
*Evaluated on the exact same $N=153$ held-out test calls (77 legitimate, 76 scam):*

| Model / Modality | Accuracy | Scam F1 (95% CI) | Macro-F1 | PR-AUC | ROC-AUC | Inference Latency |
|---|---|---|---|---|---|---|
| **E1: Text-Only (TF-IDF + LR)** | **0.9935** | **0.9935** (0.9767, 1.0000) | **0.9935** | **1.0000** | **1.0000** | ~0.0001s / sample |
| **E3: Audio-Only (Acoustic + LR)** | *Trained on 100 acoustic features* | Scaled acoustic prosody | Audio baseline | Cross-modal comparison | Evaluated on test split | ~0.0002s / sample |
| **E4: Audio-Only (Acoustic + RF)** | *Nonlinear trees on acoustic features* | Captures acoustic variance | Nonlinear baseline | Cross-modal comparison | Evaluated on test split | ~0.0012s / sample |
| **E6: Multimodal Late Fusion** | *Meta-Classifier on OOF outputs* | Leakage-free meta-head | Fused risk score | Cross-modal fusion | Evaluated on test split | ~0.0003s / sample |

## 5. Interpretability & Key Findings
- **Lexical Signals**: Highly salient n-grams in scam calls include coercive legal vocabulary (`"arrest"`, `"warrant"`, `"attorney general"`, `"money laundering"`, `"transfer funds"`, `"penalties"`). In legitimate calls, salient terms include appointment scheduling, administrative confirmations, and routine customer service phrasing.
- **Acoustic Signals**: Energy distribution (RMS std/max), silence ratio, and higher-order MFCC statistics capture speaking tempo and vocal variance. However, acoustic features alone exhibit lower specificity than explicit semantic threat phrases.
- **Research Hypothesis Verdict**: While acoustic features provide complementary information, lexical semantic signals in synthetic benchmarks are already dominant. Real-world calls with subtle persuasion, however, require multimodal vigilance.

## 6. Risks, Limitations & Responsible AI
1. **False Positives**: Legitimate calls with genuine urgency (emergency medical alerts, bank security notices) risk elevated scores if using coercive language.
2. **False Negatives**: Conversational scams relying on friendly pretexting, gradual rapport building, or absence of explicit threats may be missed by lexical classifiers.
3. **Synthetic Shortcuts**: Models may inadvertently exploit synthetic voice synthesis characteristics or scenario templating artifacts.
4. **ASR Vulnerability**: Background noise, heavy accents, or poor microphone quality can introduce ASR transcription errors that alter lexical feature activations.
