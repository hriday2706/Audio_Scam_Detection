# VoxShield: Multimodal Voice Scam Risk Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License: CC-BY-4.0](https://img.shields.io/badge/License-CC--BY--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen.svg)]()

> **Academic Research Project**: An empirical investigation into whether acoustic voice features enhance the detection of voice phishing (vishing) calls beyond speech transcripts.

---

## ⚠️ Research Disclaimer
**FOR ACADEMIC & RESEARCH EVALUATION ONLY.**  
VoxShield provides an experimental research risk score trained on synthetic conversational scenario data. It is **NOT** an authoritative fraud verdict. It must **NEVER** be used for autonomous financial blocking, automated account freezes, legal accusations, or determining human honesty without certified human review.

---

## 1. Project Overview & Research Question

### Central Research Question
> **Does combining acoustic features with transcript features improve scam-versus-legitimate conversation classification compared with transcript-only or audio-only models?**

- **$H_1$ (Alternative Hypothesis)**: Fusing acoustic prosody with lexical transcripts yields superior classification performance on held-out test calls.
- **$H_0$ (Null Hypothesis)**: Multimodal fusion does not produce meaningful performance gains over a strong transcript-only model.

VoxShield follows rigorous data science practices:
- **Zero-Leakage Group Partitioning**: Identical scenario scripts are strictly isolated within individual partitions to eliminate scenario memorization.
- **Out-of-Fold (OOF) Late Fusion**: The multimodal meta-classifier is trained exclusively on cross-validated out-of-fold predictions, preventing in-sample optimistic leakage.
- **Unbiased Single-Pass Test Evaluation**: Hyperparameters and decision thresholds are tuned strictly on the validation set; the held-out test set is evaluated once.

---

## 2. Repository Structure

```
voxshield/
├── README.md                          # Master project documentation
├── pyproject.toml                     # Package dependencies & build configuration
├── requirements.txt                   # Pinned environment requirements
├── .env.example                       # Environment configuration template
├── configs/
│   ├── default.yaml                   # Global path and hardware settings
│   ├── data.yaml                      # VISHGUARD dataset & split parameters
│   └── experiments.yaml               # Model architectures and hyperparameters
├── data/
│   ├── README.md                      # Dataset provenance, checksums, and license
│   ├── raw/                           # Raw audio and JSON archives (git-ignored)
│   ├── processed/                     # Parquet feature caches & schemas (git-ignored)
│   └── manifests/                     # Manifests & leak-free split tables
│       ├── vishguard_full_manifest.csv
│       ├── vishguard_en_manifest.csv
│       ├── splits_en.csv
│       └── smoke_test_manifest.csv
├── src/
│   └── voxshield/
│       ├── __init__.py
│       ├── config.py                  # Hardware audit & config loader
│       ├── data/
│       │   ├── load_dataset.py        # Zenodo downloader & remote range stream
│       │   ├── validate_data.py       # Audio/text QA and clipping/silence checks
│       │   ├── manifest.py            # Manifest builder & label standardizer
│       │   └── split_data.py          # Leak-free Stratified Group Splitting
│       ├── features/
│       │   ├── audio_features.py      # 100 deterministic handcrafted acoustic features
│       │   ├── text_features.py       # TF-IDF lexical vectorization
│       │   ├── speech_to_text.py      # Whisper ASR transcription & WER/CER
│       │   └── speech_embeddings.py   # Pretrained frozen HuBERT embeddings
│       ├── models/
│       │   ├── train_text.py          # E1 Text baseline (TF-IDF + LR)
│       │   ├── train_audio.py         # E3 (LR) & E4 (Random Forest) audio baselines
│       │   ├── train_fusion.py        # E6 Multimodal late fusion meta-model
│       │   ├── predict.py             # Single-file multimodal inference engine
│       │   └── evaluate.py            # Bootstrap CI & binary metrics calculation
│       ├── reporting/
│       │   ├── plots.py               # Publication-grade visualization charts
│       │   └── error_analysis.py      # FP/FN review and cross-modal disagreements
│       └── utils/
│           ├── logging_utils.py       # Structured logging
│           ├── paths.py               # Path definitions & cache redirection
│           └── reproducibility.py     # Deterministic seed management
├── app/
│   └── streamlit_app.py               # Interactive research prototype UI
├── reports/
│   ├── figures/                       # Publication plots (PNG, 300 DPI)
│   ├── metrics/                       # JSON metrics & Markdown tables
│   ├── final_report.md                # Comprehensive research report & findings
│   ├── dataset_card.md                # Dataset card (provenance, schema, bias)
│   ├── model_card.md                  # Model card (architectures, metrics, use)
│   └── hardware_environment.txt       # Hardware diagnostic dump
├── scripts/
│   ├── audit_dataset.py               # Audit dataset and build manifests
│   ├── run_eda.py                     # Generate EDA figures & statistical tests
│   ├── extract_audio_features.py      # Extract and cache acoustic statistics
│   ├── transcribe_audio.py            # Run Whisper ASR and compute WER/CER
│   ├── train_baselines.py             # Train E1, E3, E4, and E6
│   ├── run_experiments.py             # Unified experiment runner
│   └── evaluate_models.py             # Compile comparison matrix & error analysis
└── tests/
    ├── test_data_validation.py        # QA & silence/clipping tests
    ├── test_audio_features.py         # Acoustic feature shape & NaN tests
    ├── test_splits.py                 # Split disjointness & leakage prevention tests
    ├── test_metrics.py                # Metric formulas & bootstrap CI tests
    └── test_prediction_pipeline.py    # End-to-end inference verification
```

---

## 3. Quickstart & Reproducibility Guide

### Step 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/voxshield/voxshield.git
cd voxshield

# Create virtual environment with Python 3.11
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Step 2: Validate Hardware & Environment
Inspect GPU, CUDA runtime, VRAM, and disk space:
```bash
python -m voxshield.config --check-environment
```

### Step 3: Audit Dataset & Create Manifests
Download metadata from Zenodo record `10.5281/zenodo.20429763` and verify schema:
```bash
python scripts/audit_dataset.py --config configs/data.yaml
```

### Step 4: Generate Zero-Leakage Splits
Partitions English calls into 70% Train, 15% Validation, and 15% Test with zero scenario leakage:
```bash
python -m voxshield.data.split_data --config configs/data.yaml
```

### Step 5: Run Exploratory Data Analysis (EDA)
Generates distribution figures in `reports/figures/` and statistical tests in `reports/eda_report.md`:
```bash
python scripts/run_eda.py --config configs/data.yaml
```

### Step 6: Extract Handcrafted Acoustic Features
Extracts 100 deterministic features per audio recording:
```bash
python scripts/extract_audio_features.py --config configs/experiments.yaml
```

### Step 7: Train Baseline Models & Multimodal Fusion
Trains text baseline (E1), audio models (E3 & E4), and late-fusion meta-classifier (E6):
```bash
python scripts/train_baselines.py --config configs/experiments.yaml
```

### Step 8: Compile Evaluation & Error Analysis
Compiles cross-modality comparison table and error analysis:
```bash
python scripts/evaluate_models.py --config configs/experiments.yaml
```

### Step 9: Launch Interactive Research Demo
```bash
streamlit run app/streamlit_app.py
```

### Step 10: Run Automated Tests
```bash
python -m pytest tests/
```

---

## 4. Key Experimental Results (Held-Out Test Set, N=153)

All models evaluated under identical held-out test samples ($N=153$, 77 legitimate, 76 scam):

| Model ID | Architecture | Modality | Input Condition | Accuracy | Scam F1 (95% CI) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **E1** | TF-IDF + Logistic Regression | Text | Reference Transcript | **0.9935** | **0.9935** [0.9767, 1.0000] | **1.0000** | **1.0000** |
| **E3** | 100 Acoustic Stats + Logistic Regression | Audio | Handcrafted Prosody | *Evaluated* | Audio-LR F1 | Acoustic PR | Acoustic ROC |
| **E4** | 100 Acoustic Stats + Random Forest | Audio | Handcrafted Prosody | *Evaluated* | Audio-RF F1 | Acoustic PR | Acoustic ROC |
| **E6** | Late Fusion Meta-Classifier | Multimodal | Reference Text + Audio | **0.9935** | **0.9935** [0.9767, 1.0000] | **1.0000** | **1.0000** |

---

## 5. License & Citation
- **Code License**: MIT License
- **Dataset License**: CC-BY-4.0 (VISHGUARD v3.0, Nature Scientific Data 2026)
- **Citation**:
```bibtex
@article{hmimou2026vishguard,
  title={VISHGUARD: A Multilingual Audio-Text Dataset for Voice Phishing and Persuasion Detection},
  journal={Scientific Data},
  year={2026},
  doi={10.5281/zenodo.20429763}
}
```
