"""Unified model evaluation script: compiles metrics table, generates comparison plot, and runs error analysis."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from voxshield.reporting.error_analysis import analyze_prediction_errors
from voxshield.reporting.plots import plot_model_comparisons
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import FIGURES_DIR, MANIFESTS_DIR, METRICS_DIR, REPORTS_DIR

logger = get_logger("scripts.evaluate_models")


def compile_evaluation_table() -> pd.DataFrame:
    """Reads all completed experiment metric files and formats a unified markdown/CSV comparison table."""
    records = []
    for metric_file in METRICS_DIR.glob("*_metrics.json"):
        if metric_file.name == "asr_metrics.json":
            continue
        try:
            with open(metric_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            exp_id = data.get("experiment_id", metric_file.stem.replace("_metrics", ""))
            test_m = data.get("test_metrics", {})
            val_m = data.get("val_metrics", {})

            records.append({
                "experiment_id": exp_id,
                "model_description": data.get("model", exp_id),
                "modality": data.get("modality", "unknown"),
                "input_source": data.get("input", data.get("text_source", "default")),
                "test_samples": test_m.get("sample_count", 0),
                "accuracy": test_m.get("accuracy", 0.0),
                "precision_scam": test_m.get("precision_scam", 0.0),
                "recall_scam": test_m.get("recall_scam", 0.0),
                "f1_scam": test_m.get("f1_scam", 0.0),
                "f1_scam_95_ci": str(test_m.get("f1_scam_ci", ("N/A", "N/A"))),
                "macro_f1": test_m.get("macro_f1", 0.0),
                "pr_auc": test_m.get("pr_auc", 0.0),
                "roc_auc": test_m.get("roc_auc", 0.0),
                "train_time_sec": test_m.get("training_time_seconds", 0.0),
                "inference_per_sample_sec": test_m.get("per_sample_inference_seconds", 0.0),
            })
        except Exception as e:
            logger.warning(f"Error loading {metric_file}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df.sort_values(by="experiment_id", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save CSV
        csv_out = METRICS_DIR / "unified_experiment_comparison.csv"
        df.to_csv(csv_out, index=False)
        logger.info(f"Unified metrics saved to: {csv_out}")

        # Save Markdown table
        md_out = METRICS_DIR / "unified_experiment_comparison.md"
        with open(md_out, "w", encoding="utf-8") as f:
            f.write("# VoxShield Unified Model Performance Matrix\n\n")
            f.write(df.to_markdown(index=False) + "\n")
        logger.info(f"Markdown comparison table saved to: {md_out}")

        # Generate comparison plot
        plot_model_comparisons(df, FIGURES_DIR / "model_comparison.png")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate all completed models and compile report.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    df = compile_evaluation_table()
    if df.empty:
        logger.warning("No experiment metrics found to compile.")
        return

    print("\n" + "=" * 90)
    print("VOXSHIELD UNIFIED EXPERIMENTAL RESULTS (HELD-OUT TEST SET)")
    print("=" * 90)
    print(df[["experiment_id", "modality", "accuracy", "f1_scam", "f1_scam_95_ci", "pr_auc", "roc_auc"]].to_string(index=False))
    print("=" * 90 + "\n")

    # Run error analysis on predictions if available
    preds_e1 = MANIFESTS_DIR / "preds_E1.parquet"
    preds_e3 = MANIFESTS_DIR / "preds_E3.parquet"
    if preds_e1.exists():
        df_e1 = pd.read_parquet(preds_e1)
        test_e1 = df_e1[df_e1["split"] == "test"].copy()
        test_e1["pred_label"] = (test_e1["prob_scam_E1"] >= 0.5).map({0: "legitimate", 1: "scam"})

        # Load transcripts
        splits_df = pd.read_csv(MANIFESTS_DIR / "splits_en.csv")
        test_e1 = pd.merge(test_e1, splits_df[["sample_id", "transcript"]], on="sample_id", how="left")

        if preds_e3.exists():
            df_e3 = pd.read_parquet(preds_e3)
            test_e1 = pd.merge(test_e1, df_e3[["sample_id", "prob_scam_E3"]], on="sample_id", how="left")

        analyze_prediction_errors(test_e1, pred_col="pred_label", prob_col="prob_scam_E1")


if __name__ == "__main__":
    main()
