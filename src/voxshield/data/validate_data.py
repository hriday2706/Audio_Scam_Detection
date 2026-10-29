"""Data validation and quality audit module for VoxShield."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import soundfile as sf

from voxshield.utils.logging_utils import get_logger

logger = get_logger("voxshield.data.validate_data")


def validate_single_audio(file_path: Path) -> Dict[str, Any]:
    """Validates an audio file: checks readability, duration, sample rate, channels, silence, and clipping."""
    if not file_path.exists():
        return {"valid": False, "error": "file_not_found"}

    try:
        info = sf.info(str(file_path))
    except Exception as e:
        return {"valid": False, "error": f"soundfile_header_error: {str(e)}"}

    if info.frames == 0 or info.duration == 0:
        return {"valid": False, "error": "empty_audio"}

    try:
        data, sr = sf.read(str(file_path), always_2d=False)
    except Exception as e:
        return {"valid": False, "error": f"soundfile_read_error: {str(e)}"}

    if np.isnan(data).any() or np.isinf(data).any():
        return {"valid": False, "error": "contains_nan_or_inf"}

    # Check for complete silence
    rms = np.sqrt(np.mean(data**2))
    if rms < 1e-6:
        return {"valid": False, "error": "silent_audio"}

    # Check for clipping (peak >= 0.999 in normalized float)
    peak = np.max(np.abs(data))
    is_clipped = bool(peak >= 0.999)

    return {
        "valid": True,
        "sample_rate": sr,
        "channels": info.channels,
        "duration": info.duration,
        "frames": info.frames,
        "rms": float(rms),
        "peak": float(peak),
        "is_clipped": is_clipped,
    }


def audit_dataset_manifest(
    manifest_df: pd.DataFrame,
    check_audio_files: bool = True,
    audio_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Audits the full manifest and returns a structured audit report and cleaned dataframe."""
    total_records = len(manifest_df)
    logger.info(f"Auditing manifest with {total_records} records...")

    exclusions: List[Dict[str, Any]] = []
    valid_indices: List[int] = []

    # Check for duplicate IDs
    id_counts = manifest_df["sample_id"].value_counts()
    duplicate_ids = set(id_counts[id_counts > 1].index)

    # Check for repeated transcripts
    transcript_counts = manifest_df["transcript"].value_counts()
    repeated_transcripts = set(transcript_counts[transcript_counts > 1].index)

    audio_checks: Dict[str, Any] = {
        "checked_count": 0,
        "readable_count": 0,
        "sample_rate_counts": {},
        "channel_counts": {},
        "clipped_count": 0,
    }

    for idx, row in manifest_df.iterrows():
        sid = row["sample_id"]
        label = row.get("label")
        transcript = str(row.get("transcript", "")).strip()

        # Check duplicate ID
        if sid in duplicate_ids:
            exclusions.append({"sample_id": sid, "reason": "duplicate_sample_id"})
            continue

        # Check missing label
        if pd.isna(label) or str(label).lower() not in ["scam", "legitimate", "fraudulent"]:
            exclusions.append({"sample_id": sid, "reason": "missing_or_invalid_label"})
            continue

        # Check missing transcript
        if not transcript:
            exclusions.append({"sample_id": sid, "reason": "empty_transcript"})
            continue

        # Check audio file if requested
        if check_audio_files:
            audio_path_str = row.get("audio_path")
            if audio_dir and audio_path_str:
                audio_file = audio_dir / f"{sid}.wav"
            elif audio_path_str:
                audio_file = Path(audio_path_str)
            else:
                audio_file = None

            if audio_file and audio_file.exists():
                audio_checks["checked_count"] += 1
                audio_res = validate_single_audio(audio_file)
                if not audio_res["valid"]:
                    exclusions.append({"sample_id": sid, "reason": f"audio_{audio_res.get('error')}"})
                    continue
                audio_checks["readable_count"] += 1
                sr = audio_res["sample_rate"]
                ch = audio_res["channels"]
                audio_checks["sample_rate_counts"][sr] = audio_checks["sample_rate_counts"].get(sr, 0) + 1
                audio_checks["channel_counts"][ch] = audio_checks["channel_counts"].get(ch, 0) + 1
                if audio_res["is_clipped"]:
                    audio_checks["clipped_count"] += 1

        valid_indices.append(idx)

    cleaned_df = manifest_df.loc[valid_indices].copy()
    cleaned_df.reset_index(drop=True, inplace=True)

    report = {
        "total_input_records": total_records,
        "valid_records_count": len(cleaned_df),
        "excluded_records_count": len(exclusions),
        "exclusions": exclusions,
        "duplicate_sample_ids_count": len(duplicate_ids),
        "repeated_transcripts_count": len(repeated_transcripts),
        "class_balance": cleaned_df["label"].value_counts().to_dict() if "label" in cleaned_df else {},
        "language_distribution": cleaned_df["language"].value_counts().to_dict() if "language" in cleaned_df else {},
        "audio_checks": audio_checks,
    }

    logger.info(f"Audit completed: {len(cleaned_df)} valid records, {len(exclusions)} exclusions.")
    return report, cleaned_df
