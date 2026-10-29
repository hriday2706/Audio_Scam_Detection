# Dataset Card: VISHGUARD v3.0 (English Subset)

## 1. Dataset Description
- **Dataset Name**: VISHGUARD (Voice Phishing & Social Engineering Guard)
- **Version**: Initial Data Release: Vish-Guard v3.0
- **DOI / Record**: [10.5281/zenodo.20429763](https://doi.org/10.5281/zenodo.20429763)
- **Associated Publication**: Nature Scientific Data ([https://www.nature.com/articles/s41597-026-07724-z](https://www.nature.com/articles/s41597-026-07724-z))
- **Release Date**: 2026
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)
- **Language**: English (`en`), with extensions for French (`fr`) and Modern Standard Arabic (`ar`).
- **Target Task**: Binary classification of voice phone calls into `scam` (fraudulent social engineering / vishing) vs. `legitimate`.

## 2. Dataset Composition & Schema
The dataset consists of paired audio recordings, transcripts, and behavioral annotation metadata:
- **Audio Format**: 16 kHz / 44.1 kHz PCM WAV mono/stereo audio recordings (durations ~10–90 seconds).
- **Total Corpus Size**: 3,059 simulated calls across 3 languages (6.45 GB archive).
- **Primary Operational Benchmark (English)**:
  - Total records: 1,065 calls.
  - Class distribution: 536 legitimate calls (50.3%) and 529 fraudulent calls (49.7%).
- **Metadata Fields**:
  - `sample_id`: Unique sample identifier (e.g. `lot1_en_fraud_001`).
  - `label`: Standardized target class (`scam` vs `legitimate`).
  - `original_type`: Source annotation (`fraudulent` vs `legitimate`).
  - `language`: Two-letter ISO language code (`en`).
  - `transcript`: Verbatim spoken script.
  - `duration`: Audio duration in seconds.
  - `words_count`: Total word token count.
  - `emotion`: Perceived vocal emotion tags (e.g. `fear`, `authority`, `neutral`, `urgency`).
  - `tone`: Spoken communication tone tags (e.g. `threat`, `authoritative`, `polite`).
  - `persuasion_patterns`: Social engineering persuasion taxonomy (e.g. `threat`, `urgency`, `authority`, `pretexting`).

## 3. Data Split & Leakage Prevention Strategy
To ensure reproducible and leak-free evaluation:
- **Splitting Strategy**: Stratified Group Partitioning using unique transcript text hashes (`group_id`).
- **Split Ratios**: 70% Train (760 samples), 15% Validation (152 samples), 15% Test (153 samples).
- **Leakage Auditing**:
  - Exact duplicates and shared scenario templates are strictly isolated within individual splits.
  - Disjoint sample IDs and transcript text hashes verified between Train, Val, and Test ($Train \cap Test = \emptyset$, $Train \cap Val = \emptyset$, $Val \cap Test = \emptyset$).
  - Hyperparameter tuning and decision thresholds are optimized strictly on the validation set; the test set is touched only once for unbiased final evaluation.

## 4. Key Limitations & Confounders
1. **Synthetic-to-Real Domain Gap**: VISHGUARD consists of simulated, synthetic scenario calls. It does not contain spontaneous real-world phone calls with human hesitation, real telephone line codecs (e.g., AMR-NB, G.711), or packet dropouts.
2. **Templated Persuasion Cues**: Synthetic fraud scenarios exhibit strong, recurrent lexical cues (threats of arrest, immediate bank account suspension) that are readily memorized by lexical models.
3. **Synthetic Voice Artifacts**: Acoustic properties reflect synthetic speech synthesis models rather than human physiological vocal stress.
4. **Non-Generalizability**: High in-benchmark test accuracy should not be interpreted as evidence of operational efficacy against evolving real-world vishing threats.

## 5. Ethical Considerations & Intended Use
- **Intended Use**: Defensive cybersecurity research, fraud-awareness training, multimodal model comparison, and academic study.
- **Prohibited / Non-Goals**: Automated account freezing, real-time telephony blocking, automated police reports, detecting human honesty or stress, or deploying without certified human review.
- **Privacy**: No private real-person voice calls or sensitive personally identifiable financial data are committed or distributed.
