"""Unit tests for dataset splitting and leakage prevention."""

import pandas as pd
import pytest

from voxshield.data.split_data import create_group_identifier, split_dataset


def test_create_group_identifier():
    t1 = "Hello, your bank account is locked!"
    t2 = "  hello,  YOUR BANK ACCOUNT is locked! "
    t3 = "A completely different scenario."
    assert create_group_identifier(t1) == create_group_identifier(t2)
    assert create_group_identifier(t1) != create_group_identifier(t3)


def test_split_dataset_zero_leakage():
    # Construct synthetic dataset with duplicated scenarios across variants
    records = []
    for i in range(100):
        group_text = f"Scenario pattern text number {i % 20}"
        records.append({
            "sample_id": f"sample_{i}",
            "label": "scam" if (i % 2 == 0) else "legitimate",
            "transcript": group_text,
        })
    df = pd.DataFrame(records)

    splits_df = split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    train_ids = set(splits_df[splits_df["split"] == "train"]["sample_id"])
    val_ids = set(splits_df[splits_df["split"] == "val"]["sample_id"])
    test_ids = set(splits_df[splits_df["split"] == "test"]["sample_id"])

    # Disjoint sample IDs
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    # Disjoint group transcripts (zero scenario leakage)
    train_groups = set(splits_df[splits_df["split"] == "train"]["group_id"])
    val_groups = set(splits_df[splits_df["split"] == "val"]["group_id"])
    test_groups = set(splits_df[splits_df["split"] == "test"]["group_id"])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0
