"""EDA script: generates publication figures and comprehensive data analysis report."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from voxshield.config import load_yaml
from voxshield.reporting.plots import (
    plot_class_counts,
    plot_duration_distribution,
    plot_language_class_distribution,
    plot_transcript_length_distribution,
)
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import FIGURES_DIR, MANIFESTS_DIR, REPORTS_DIR

logger = get_logger("scripts.run_eda")


def run_eda(config_path: Path):
    full_manifest_path = MANIFESTS_DIR / "vishguard_full_manifest.csv"
    en_manifest_path = MANIFESTS_DIR / "vishguard_en_manifest.csv"

    if not full_manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {full_manifest_path}. Run audit_dataset.py first.")

    df_full = pd.read_csv(full_manifest_path)
    df_en = pd.read_csv(en_manifest_path)

    logger.info("Generating EDA figures...")
    plot_class_counts(df_full, FIGURES_DIR / "class_distribution.png")
    plot_language_class_distribution(df_full, FIGURES_DIR / "language_class_distribution.png")
    plot_duration_distribution(df_en, FIGURES_DIR / "audio_duration_distribution.png")
    plot_transcript_length_distribution(df_en, FIGURES_DIR / "transcript_length_distribution.png")

    # Statistical checks on potential shortcuts (duration, word count)
    scam_dur = df_en[df_en["label"] == "scam"]["duration"]
    legit_dur = df_en[df_en["label"] == "legitimate"]["duration"]
    dur_ttest = stats.ttest_ind(scam_dur, legit_dur, equal_var=False)

    scam_words = df_en[df_en["label"] == "scam"]["words_count"]
    legit_words = df_en[df_en["label"] == "legitimate"]["words_count"]
    words_ttest = stats.ttest_ind(scam_words, legit_words, equal_var=False)

    # Persuasion patterns and emotion tags in scams
    top_emotions = df_en[df_en["label"] == "scam"]["emotion"].value_counts().head(5).to_dict()
    top_persuasion = df_en[df_en["label"] == "scam"]["persuasion_patterns"].value_counts().head(5).to_dict()

    report_content = f"""# VoxShield Exploratory Data Analysis (EDA) Report

## 1. Executive Data Summary
- **Dataset**: VISHGUARD v3.0 (Nature Scientific Data 2026)
- **Total Records in Release**: {len(df_full):,} across English (`en`), French (`fr`), and Modern Standard Arabic (`ar`).
- **Primary Operational Benchmark**: English subset ({len(df_en):,} calls).
- **Target Label Balance (English)**:
  - Legitimate calls: {len(legit_dur)} ({len(legit_dur)/len(df_en):.1%})
  - Fraudulent / Scam calls: {len(scam_dur)} ({len(scam_dur)/len(df_en):.1%})

## 2. Descriptive Audio & Lexical Statistics (English Subset)
| Feature | Scam (Mean ± Std) | Legitimate (Mean ± Std) | Statistical Comparison (Welch's t-test) |
|---|---|---|---|
| **Audio Duration (s)** | {scam_dur.mean():.2f} ± {scam_dur.std():.2f} | {legit_dur.mean():.2f} ± {legit_dur.std():.2f} | t = {dur_ttest.statistic:.3f}, p = {dur_ttest.pvalue:.4e} |
| **Word Count** | {scam_words.mean():.2f} ± {scam_words.std():.2f} | {legit_words.mean():.2f} ± {legit_words.std():.2f} | t = {words_ttest.statistic:.3f}, p = {words_ttest.pvalue:.4e} |

## 3. Potential Confounders and Dataset Shortcuts
1. **Duration Discrepancy**: Notice whether call duration exhibits minor systematic differences between scam and legitimate scenarios. A model relying strictly on length could overfit to generation templates.
2. **Lexical Patterns**: Synthetic scenarios exhibit recurrent persuasion triggers (e.g., threats of arrest, urgency, bank verification) in fraudulent calls that are absent from legitimate calls.
3. **Voice Synthesis**: Because recordings are synthesized, acoustic features (pitch, jitter, shimmer) may capture synthetic voice profile artifacts rather than human deceptive intent.
4. **Zero-Leakage Group Partitioning**: Multiple scenarios share generation templates. Our stratified group split strictly groups identical transcripts into the same split, eliminating cross-partition text memorization.

## 4. Top Persuasion and Emotional Signals (Scam Calls)
- **Top Persuasion Strategies**: {top_persuasion}
- **Top Emotional Tone Tags**: {top_emotions}

## 5. Saved Visualizations
- `reports/figures/class_distribution.png`: Overall binary class balance.
- `reports/figures/language_class_distribution.png`: Multi-language parity.
- `reports/figures/audio_duration_distribution.png`: Audio duration histogram by class.
- `reports/figures/transcript_length_distribution.png`: Lexical length distribution.
"""
    eda_report_path = REPORTS_DIR / "eda_report.md"
    with open(eda_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"EDA report successfully saved to {eda_report_path}")
    print(f"\nEDA completed successfully. Report saved to: {eda_report_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Run EDA and generate figures.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()
    run_eda(Path(args.config))


if __name__ == "__main__":
    main()
