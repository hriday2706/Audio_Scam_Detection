"""Unified experiment runner for VoxShield."""

import argparse
from pathlib import Path
import pandas as pd

from voxshield.models.train_audio import run_experiment_e3_e4
from voxshield.models.train_fusion import run_fusion_experiment
from voxshield.models.train_text import run_experiment_e1
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR, METRICS_DIR, PROCESSED_DATA_DIR

logger = get_logger("scripts.run_experiments")


def run_all(config_path: Path):
    logger.info("=" * 60)
    logger.info("RUNNING VOXSHIELD EXPERIMENTAL MATRIX")
    logger.info("=" * 60)

    # 1. Text Baseline E1
    logger.info("\n--- Experiment E1: TF-IDF + Logistic Regression (Reference Transcript) ---")
    run_experiment_e1(config_path)

    # 2. Audio Baselines E3 & E4
    features_path = PROCESSED_DATA_DIR / "audio_features_en.parquet"
    if features_path.exists():
        logger.info("\n--- Experiment E3 (LR) & E4 (RF): Acoustic Features ---")
        run_experiment_e3_e4(features_path)

        # 3. Multimodal Fusion E6
        logger.info("\n--- Experiment E6: Multimodal Late Fusion (E1 Text + E3 Audio) ---")
        text_preds = MANIFESTS_DIR / "preds_E1.parquet"
        audio_preds = MANIFESTS_DIR / "preds_E3.parquet"
        if text_preds.exists() and audio_preds.exists():
            run_fusion_experiment(text_preds, audio_preds, experiment_id="E6_fusion_ref")
    else:
        logger.warning(f"Audio features {features_path} not found. Skipping audio & fusion experiments until extracted.")


def main():
    parser = argparse.ArgumentParser(description="Run VoxShield experiments.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "text", "audio", "fusion"])
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.experiment == "text":
        run_experiment_e1(config_path)
    elif args.experiment == "audio":
        run_experiment_e3_e4(PROCESSED_DATA_DIR / "audio_features_en.parquet")
    elif args.experiment == "fusion":
        run_fusion_experiment(
            MANIFESTS_DIR / "preds_E1.parquet",
            MANIFESTS_DIR / "preds_E3.parquet",
            experiment_id="E6_fusion_ref"
        )
    else:
        run_all(config_path)


if __name__ == "__main__":
    main()
