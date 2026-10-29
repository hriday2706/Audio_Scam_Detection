# VoxShield Exploratory Data Analysis (EDA) Report

## 1. Executive Data Summary
- **Dataset**: VISHGUARD v3.0 (Nature Scientific Data 2026)
- **Total Records in Release**: 3,059 across English (`en`), French (`fr`), and Modern Standard Arabic (`ar`).
- **Primary Operational Benchmark**: English subset (1,065 calls).
- **Target Label Balance (English)**:
  - Legitimate calls: 536 (50.3%)
  - Fraudulent / Scam calls: 529 (49.7%)

## 2. Descriptive Audio & Lexical Statistics (English Subset)
| Feature | Scam (Mean ± Std) | Legitimate (Mean ± Std) | Statistical Comparison (Welch's t-test) |
|---|---|---|---|
| **Audio Duration (s)** | 50.28 ± 7.77 | 22.49 ± 2.64 | t = 77.941, p = 0.0000e+00 |
| **Word Count** | 117.99 ± 17.94 | 52.70 ± 5.08 | t = 80.558, p = 0.0000e+00 |

## 3. Potential Confounders and Dataset Shortcuts
1. **Duration Discrepancy**: Notice whether call duration exhibits minor systematic differences between scam and legitimate scenarios. A model relying strictly on length could overfit to generation templates.
2. **Lexical Patterns**: Synthetic scenarios exhibit recurrent persuasion triggers (e.g., threats of arrest, urgency, bank verification) in fraudulent calls that are absent from legitimate calls.
3. **Voice Synthesis**: Because recordings are synthesized, acoustic features (pitch, jitter, shimmer) may capture synthetic voice profile artifacts rather than human deceptive intent.
4. **Zero-Leakage Group Partitioning**: Multiple scenarios share generation templates. Our stratified group split strictly groups identical transcripts into the same split, eliminating cross-partition text memorization.

## 4. Top Persuasion and Emotional Signals (Scam Calls)
- **Top Persuasion Strategies**: {'authority,threat,urgency,reward,scarcity': 133, 'authority,threat,urgency,scarcity': 100, 'reward,scarcity,urgency': 94, 'authority,threat,urgency,scarcity,reward': 45, 'threat,scarcity,urgency': 35}
- **Top Emotional Tone Tags**: {'authority,fear': 277, 'excitement,urgency': 96, 'authority,anxiety': 75, 'urgency,anxiety': 36, 'fear,urgency': 10}

## 5. Saved Visualizations
- `reports/figures/class_distribution.png`: Overall binary class balance.
- `reports/figures/language_class_distribution.png`: Multi-language parity.
- `reports/figures/audio_duration_distribution.png`: Audio duration histogram by class.
- `reports/figures/transcript_length_distribution.png`: Lexical length distribution.
