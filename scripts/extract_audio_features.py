"""Script to extract and cache deterministic handcrafted acoustic features for all English samples."""

import argparse
from pathlib import Path
import pandas as pd

from voxshield.config import load_yaml
from voxshield.data.load_dataset import download_selected_samples
from voxshield.features.audio_features import extract_batch_features
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = get_logger("scripts.extract_audio_features")


def main():
    parser = argparse.ArgumentParser(description="Extract handcrafted acoustic features.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run on smoke test subset only.")
    parser.add_argument("--download-workers", type=int, default=10, help="Parallel download workers.")
    args = parser.parse_args()

    if args.smoke_test:
        manifest_path = MANIFESTS_DIR / "smoke_test_manifest.csv"
        out_parquet = PROCESSED_DATA_DIR / "smoke_audio_features.parquet"
    else:
        manifest_path = MANIFESTS_DIR / "splits_en.csv"
        out_parquet = PROCESSED_DATA_DIR / "audio_features_en.parquet"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run audit_dataset and split_data first.")

    manifest_df = pd.read_csv(manifest_path)
    audio_dir = RAW_DATA_DIR / "audio"

    logger.info(f"Target manifest has {len(manifest_df)} records.")
    sample_ids = manifest_df["sample_id"].tolist()

    # Step 1: Ensure audio files exist on disk
    download_selected_samples(sample_ids, raw_dir=RAW_DATA_DIR, max_workers=args.download_workers)

    # Step 2: Extract audio features
    feat_df = extract_batch_features(manifest_df, audio_dir, output_parquet=out_parquet)

    logger.info("=" * 50)
    logger.info(f"ACOUSTIC FEATURE EXTRACTION COMPLETED")
    logger.info(f"Total samples with features: {len(feat_df)} / {len(manifest_df)}")
    logger.info(f"Feature dimensions: {len(feat_df.columns) - 3} acoustic statistics")
    logger.info(f"Saved cache to: {out_parquet}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
