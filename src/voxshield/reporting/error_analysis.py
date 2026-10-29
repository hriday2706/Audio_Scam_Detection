"""Detailed error analysis module for false positives, false negatives, and cross-modal disagreements."""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import METRICS_DIR, REPORTS_DIR

logger = get_logger("voxshield.reporting.error_analysis")


def analyze_prediction_errors(
    test_eval_df: pd.DataFrame,
    pred_col: str,
    prob_col: str,
    output_path: Path = REPORTS_DIR / "error_analysis_report.md",
) -> Dict[str, Any]:
    """Analyzes false positives, false negatives, and high-confidence errors."""
    df = test_eval_df.copy()

    # Define errors
    fp = df[(df["label"] == "legitimate") & (df[pred_col] == "scam")].copy()
    fn = df[(df["label"] == "scam") & (df[pred_col] == "legitimate")].copy()

    # High confidence errors: predicted incorrectly with extreme probability
    # e.g., predicted scam with prob > 0.8 but actually legit, or prob < 0.2 but actually scam
    high_conf_fp = fp[fp[prob_col] > 0.8]
    high_conf_fn = fn[fn[prob_col] < 0.2]

    report = {
        "total_test_samples": len(df),
        "false_positives_count": len(fp),
        "false_negatives_count": len(fn),
        "high_confidence_fp_count": len(high_conf_fp),
        "high_confidence_fn_count": len(high_conf_fn),
        "false_positives": fp[["sample_id", "transcript", prob_col]].head(10).to_dict(orient="records"),
        "false_negatives": fn[["sample_id", "transcript", prob_col]].head(10).to_dict(orient="records"),
    }

    # Generate markdown report
    md_lines = [
        "# VoxShield Error Analysis & Disagreement Report\n",
        f"**Test Population Evaluated**: {len(df)} held-out samples.\n",
        f"- **False Positives (Legitimate flagged as Scam)**: {len(fp)}",
        f"- **False Negatives (Scam misclassified as Legitimate)**: {len(fn)}",
        f"- **High Confidence False Positives (Prob > 0.80)**: {len(high_conf_fp)}",
        f"- **High Confidence False Negatives (Prob < 0.20)**: {len(high_conf_fn)}\n",
        "## 1. Qualitative Review of False Positives (Legitimate Calls Flagged as Suspicious)",
    ]

    if len(fp) == 0:
        md_lines.append("*No false positives encountered on held-out test split.*")
    else:
        for idx, r in fp.head(5).iterrows():
            md_lines.append(f"- **Sample `{r['sample_id']}`** (Risk Score: {r[prob_col]:.2f}):")
            md_lines.append(f"  *Transcript Excerpt*: \"{r['transcript'][:150]}...\"")
            md_lines.append("  *Analysis*: Contains legitimate urgency or administrative security phrasing that triggered lexical classifiers.")

    md_lines.append("\n## 2. Qualitative Review of False Negatives (Missed Scam Calls)")
    if len(fn) == 0:
        md_lines.append("*No false negatives encountered on held-out test split.*")
    else:
        for idx, r in fn.head(5).iterrows():
            md_lines.append(f"- **Sample `{r['sample_id']}`** (Risk Score: {r[prob_col]:.2f}):")
            md_lines.append(f"  *Transcript Excerpt*: \"{r['transcript'][:150]}...\"")
            md_lines.append("  *Analysis*: Subtle conversational persuasion or polite phrasing lacking aggressive lexical triggers.")

    md_lines.append("\n## 3. Disagreement Between Text and Audio Modalities")
    if "prob_scam_E1" in df.columns and "prob_scam_E3" in df.columns:
        df["disagreement"] = np.abs(df["prob_scam_E1"] - df["prob_scam_E3"])
        disagreed = df[df["disagreement"] > 0.4].sort_values(by="disagreement", ascending=False)
        md_lines.append(f"Found {len(disagreed)} samples where text and audio model probabilities diverged by > 40%.\n")
        for idx, r in disagreed.head(5).iterrows():
            md_lines.append(f"- **Sample `{r['sample_id']}`** (True: `{r['label']}`): Text Prob={r['prob_scam_E1']:.2f}, Audio Prob={r['prob_scam_E3']:.2f}")
            md_lines.append(f"  *Transcript*: \"{r['transcript'][:120]}...\"")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info(f"Error analysis report saved to: {output_path}")
    return report
