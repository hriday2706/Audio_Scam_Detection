"""Script to train the core baseline models: E1 (text), E3 (audio LR), E4 (audio RF), and E6 (fusion)."""

import argparse
from pathlib import Path

from voxshield.models.train_audio import run_experiment_e3_e4
from voxshield.models.train_fusion import run_fusion_experiment
from voxshield.models.train_text import run_experiment_e1
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR, PROCESSED_DATA_DIR

logger = get_logger("scripts.train_baselines")


def main():
    parser = argparse.ArgumentParser(description="Train all mandatory baseline models (E1, E3, E4, E6).")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)

    # 1. Train E1
    logger.info("=" * 60)
    logger.info("STEP 1: Training Text Baseline E1 (TF-IDF + Logistic Regression)")
    logger.info("=" * 60)
    run_experiment_e1(config_path)

    # 2. Train E3 and E4
    features_path = PROCESSED_DATA_DIR / "audio_features_en.parquet"
    if not features_path.exists():
        logger.error(
            f"Audio features not found at {features_path}. "
            "Please run 'python scripts/extract_audio_features.py' first."
        )
        return

    logger.info("=" * 60)
    logger.info("STEP 2: Training Audio Baselines E3 (LR) and E4 (Random Forest)")
    logger.info("=" * 60)
    run_experiment_e3_e4(features_path)

    # 3. Train E6
    logger.info("=" * 60)
    logger.info("STEP 3: Training Multimodal Late Fusion E6 (E1 Text + E3 Audio)")
    logger.info("=" * 60)
    run_fusion_experiment(
        MANIFESTS_DIR / "preds_E1.parquet",
        MANIFESTS_DIR / "preds_E3.parquet",
        experiment_id="E6_fusion_ref",
    )

    logger.info("=" * 60)
    logger.info("ALL MANDATORY BASELINES (E1, E3, E4, E6) TRAINED AND EVALUATED SUCCESSFULLY.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
