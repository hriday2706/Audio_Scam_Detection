"""Data splitting module with strict leakage prevention for VoxShield."""

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR
from voxshield.utils.reproducibility import set_seed

logger = get_logger("voxshield.data.split_data")


def create_group_identifier(text: str) -> str:
    """Creates a deterministic group ID based on normalized transcript content to prevent scenario leakage."""
    normalized = " ".join(text.lower().strip().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]


def split_dataset(
    manifest_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Performs stratified group splitting to guarantee zero transcript/scenario leakage across train, val, and test.

    Args:
        manifest_df: DataFrame containing sample_id, label, transcript, etc.
        train_ratio: Target train fraction (default: 0.70).
        val_ratio: Target validation fraction (default: 0.15).
        test_ratio: Target test fraction (default: 0.15).
        random_seed: Random state seed.

    Returns:
        DataFrame with an added 'split' column ('train', 'val', 'test').
    """
    set_seed(random_seed)
    df = manifest_df.copy()

    # Assign group ID based on unique transcript text
    df["group_id"] = df["transcript"].apply(create_group_identifier)

    # We use StratifiedGroupKFold to cleanly partition groups while maintaining label stratification
    # With ~70/15/15, a 7-fold setup (5 folds train ~71%, 1 fold val ~14.3%, 1 fold test ~14.3%) is mathematically ideal
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=random_seed)

    splits = list(sgkf.split(df, y=df["label"], groups=df["group_id"]))

    # Fold 0 -> test, Fold 1 -> val, Folds 2..6 -> train
    test_idx = splits[0][1]
    val_idx = splits[1][1]
    train_idx = np.concatenate([splits[i][1] for i in range(2, 7)])

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Leakage verification asserts
    train_groups = set(df[df["split"] == "train"]["group_id"])
    val_groups = set(df[df["split"] == "val"]["group_id"])
    test_groups = set(df[df["split"] == "test"]["group_id"])

    assert len(train_groups & val_groups) == 0, "Leakage detected between train and val groups!"
    assert len(train_groups & test_groups) == 0, "Leakage detected between train and test groups!"
    assert len(val_groups & test_groups) == 0, "Leakage detected between val and test groups!"

    train_ids = set(df[df["split"] == "train"]["sample_id"])
    val_ids = set(df[df["split"] == "val"]["sample_id"])
    test_ids = set(df[df["split"] == "test"]["sample_id"])

    assert len(train_ids & val_ids) == 0, "Sample ID overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Sample ID overlap between train and test!"
    assert len(val_ids & test_ids) == 0, "Sample ID overlap between val and test!"

    logger.info("Stratified group split verified with zero scenario leakage.")
    logger.info(f"Split counts: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    for s in ["train", "val", "test"]:
        dist = df[df["split"] == s]["label"].value_counts().to_dict()
        logger.info(f"  Split '{s}' class balance: {dist}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate and validate data splits.")
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    en_manifest_path = MANIFESTS_DIR / "vishguard_en_manifest.csv"
    if not en_manifest_path.exists():
        logger.error(f"English manifest not found at {en_manifest_path}. Run audit_dataset.py first.")
        return

    df_en = pd.read_csv(en_manifest_path)
    df_splits = split_dataset(df_en)

    out_path = MANIFESTS_DIR / "splits_en.csv"
    df_splits.to_csv(out_path, index=False)
    logger.info(f"Splits saved to {out_path}")


if __name__ == "__main__":
    main()
