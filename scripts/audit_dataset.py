"""Script to audit dataset, verify files, and generate manifests."""

import argparse
import json
from pathlib import Path
import pandas as pd

from voxshield.config import load_yaml
from voxshield.data.load_dataset import fetch_vishguard_metadata
from voxshield.data.manifest import create_and_save_manifests
from voxshield.data.validate_data import audit_dataset_manifest
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR, RAW_DATA_DIR, REPORTS_DIR

logger = get_logger("scripts.audit_dataset")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset and create manifests.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data config YAML.")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    raw_dir = RAW_DATA_DIR

    logger.info("Step 1: Fetching VISHGUARD dataset metadata...")
    records = fetch_vishguard_metadata(raw_dir=raw_dir)

    logger.info("Step 2: Building and saving manifests...")
    manifest_paths = create_and_save_manifests(records, manifests_dir=MANIFESTS_DIR)

    logger.info("Step 3: Auditing English manifest data quality...")
    df_en = pd.read_csv(manifest_paths["en"])
    audio_dir = raw_dir / "audio"
    report, cleaned_df = audit_dataset_manifest(df_en, check_audio_files=True, audio_dir=audio_dir)

    audit_report_path = REPORTS_DIR / "data_audit_report.json"
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Data audit report saved to: {audit_report_path}")
    print("\n" + "=" * 50)
    print("VISHGUARD DATA AUDIT SUMMARY")
    print("=" * 50)
    print(f"Total input English records : {report['total_input_records']}")
    print(f"Valid records verified      : {report['valid_records_count']}")
    print(f"Excluded records            : {report['excluded_records_count']}")
    print(f"Unique transcripts count    : {report['total_input_records'] - report['repeated_transcripts_count']}")
    print(f"Repeated transcripts count  : {report['repeated_transcripts_count']}")
    print(f"Class balance               : {report['class_balance']}")
    print(f"Audio checked               : {report['audio_checks']['checked_count']}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
