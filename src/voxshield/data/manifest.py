"""Dataset manifest creation and management for VoxShield."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import DATA_DIR, MANIFESTS_DIR, RAW_DATA_DIR

logger = get_logger("voxshield.data.manifest")


def build_manifest_from_raw_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Transforms raw VISHGUARD metadata records into a structured manifest DataFrame."""
    rows = []
    for r in records:
        sid = r.get("id")
        orig_type = r.get("type", "").lower()
        # Standardize target label: 'scam' vs 'legitimate'
        label = "scam" if orig_type in ["fraudulent", "fraud", "scam"] else "legitimate"
        lang = r.get("language", "")
        script = r.get("script", "")
        duration = float(r.get("duration", 0.0))
        metadata = r.get("metadata", {})
        words_count = int(metadata.get("words_count", len(script.split())))

        # Scenario / lot grouping extracted from sample_id (e.g., 'lot1' from 'lot1_en_fraud_001')
        scenario_group = sid.split("_")[0] if "_" in sid else "default"

        emotion = ",".join(r.get("emotion", [])) if isinstance(r.get("emotion"), list) else str(r.get("emotion", ""))
        tone = ",".join(r.get("tone", [])) if isinstance(r.get("tone"), list) else str(r.get("tone", ""))
        persuasion = ",".join(r.get("persuasion_patterns", [])) if isinstance(r.get("persuasion_patterns"), list) else str(r.get("persuasion_patterns", ""))
        behavior = ",".join(r.get("behavior", [])) if isinstance(r.get("behavior"), list) else str(r.get("behavior", ""))

        rows.append({
            "sample_id": sid,
            "label": label,
            "original_type": orig_type,
            "language": lang,
            "transcript": script,
            "duration": duration,
            "words_count": words_count,
            "scenario_group": scenario_group,
            "relative_audio_path": f"audio/{sid}.wav",
            "relative_transcript_path": f"transcripts/{sid}.txt",
            "emotion": emotion,
            "tone": tone,
            "persuasion_patterns": persuasion,
            "behavior": behavior,
        })

    df = pd.DataFrame(rows)
    df.sort_values(by="sample_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_and_save_manifests(
    raw_records: List[Dict[str, Any]],
    manifests_dir: Path = MANIFESTS_DIR,
) -> Dict[str, Path]:
    """Builds and saves full manifest and English manifest to disk."""
    manifests_dir.mkdir(parents=True, exist_ok=True)
    df_full = build_manifest_from_raw_records(raw_records)

    full_path = manifests_dir / "vishguard_full_manifest.csv"
    df_full.to_csv(full_path, index=False)
    logger.info(f"Full manifest saved: {full_path} ({len(df_full)} rows)")

    # English subset manifest
    df_en = df_full[df_full["language"] == "en"].copy().reset_index(drop=True)
    en_path = manifests_dir / "vishguard_en_manifest.csv"
    df_en.to_csv(en_path, index=False)
    logger.info(f"English manifest saved: {en_path} ({len(df_en)} rows)")

    # 30-sample balanced smoke test manifest
    en_scam = df_en[df_en["label"] == "scam"].head(15)
    en_legit = df_en[df_en["label"] == "legitimate"].head(15)
    smoke_df = pd.concat([en_scam, en_legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    smoke_path = manifests_dir / "smoke_test_manifest.csv"
    smoke_df.to_csv(smoke_path, index=False)
    logger.info(f"Smoke test manifest saved: {smoke_path} ({len(smoke_df)} rows)")

    return {
        "full": full_path,
        "en": en_path,
        "smoke": smoke_path,
    }
