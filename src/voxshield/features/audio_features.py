"""Deterministic handcrafted acoustic feature extraction using Librosa and SoundFile."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import PROCESSED_DATA_DIR

logger = get_logger("voxshield.features.audio_features")

DEFAULT_SR = 16000
N_MFCC = 20


def extract_features_from_audio(
    file_path: Path,
    target_sr: int = DEFAULT_SR,
    n_mfcc: int = N_MFCC,
) -> Dict[str, float]:
    """Extracts a comprehensive set of deterministic acoustic statistics from a single audio file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Load and resample to 16 kHz mono
    y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
    duration = float(len(y) / sr)

    if len(y) == 0:
        raise ValueError(f"Audio file is empty: {file_path}")

    features: Dict[str, float] = {}
    features["duration_sec"] = duration

    # 1. RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_max"] = float(np.max(rms)) if len(rms) > 0 else 0.0

    # Silence proportion (frames with energy < 10% of mean energy)
    silence_threshold = 0.1 * (features["rms_mean"] + 1e-9)
    silence_frames = np.sum(rms < silence_threshold)
    features["silence_ratio"] = float(silence_frames / len(rms)) if len(rms) > 0 else 0.0

    # 2. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 3. Spectral Features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_std"] = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(sb))
    features["spectral_bandwidth_std"] = float(np.std(sb))

    srf = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    features["spectral_rolloff_mean"] = float(np.mean(srf))
    features["spectral_rolloff_std"] = float(np.std(srf))

    sf_feat = librosa.feature.spectral_flatness(y=y)[0]
    features["spectral_flatness_mean"] = float(np.mean(sf_feat))
    features["spectral_flatness_std"] = float(np.std(sf_feat))

    # 4. MFCCs and Delta-MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)

    for i in range(n_mfcc):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        features[f"mfcc_delta_{i+1}_std"] = float(np.std(delta_mfcc[i]))

    # 5. Fundamental Frequency (F0 / Pitch) using YIN
    try:
        f0 = librosa.yin(y=y, fmin=50, fmax=500, sr=sr)
        valid_f0 = f0[(f0 >= 50) & (f0 <= 500) & (~np.isnan(f0))]
        if len(valid_f0) > 0:
            features["pitch_f0_mean"] = float(np.mean(valid_f0))
            features["pitch_f0_std"] = float(np.std(valid_f0))
        else:
            features["pitch_f0_mean"] = 0.0
            features["pitch_f0_std"] = 0.0
    except Exception:
        features["pitch_f0_mean"] = 0.0
        features["pitch_f0_std"] = 0.0

    # Ensure no NaN or Inf
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features


def extract_batch_features(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    output_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """Extracts features for audio files with incremental caching."""
    existing_df = None
    already_done_ids = set()
    if output_parquet and output_parquet.exists():
        existing_df = pd.read_parquet(output_parquet)
        already_done_ids = set(existing_df["sample_id"])
        logger.info(f"Loaded {len(existing_df)} previously cached feature rows.")

    needed_df = manifest_df[~manifest_df["sample_id"].isin(already_done_ids)].copy()

    rows = []
    failed = []

    logger.info(f"Extracting features for {len(needed_df)} samples...")
    for _, row in tqdm(needed_df.iterrows(), total=len(needed_df), desc="Extracting audio features"):
        sid = row["sample_id"]
        audio_path = audio_dir / f"{sid}.wav"

        if not audio_path.exists():
            failed.append((sid, "file_missing"))
            continue

        try:
            feats = extract_features_from_audio(audio_path)
            feats["sample_id"] = sid
            feats["label"] = row.get("label")
            feats["split"] = row.get("split")
            rows.append(feats)
        except Exception as e:
            failed.append((sid, str(e)))

    if failed:
        logger.warning(f"{len(failed)} samples skipped in feature extraction.")

    new_feat_df = pd.DataFrame(rows)
    if existing_df is not None and not existing_df.empty:
        feat_df = pd.concat([existing_df, new_feat_df], ignore_index=True)
    else:
        feat_df = new_feat_df

    if not feat_df.empty:
        cols = ["sample_id", "label", "split"] + [c for c in feat_df.columns if c not in ["sample_id", "label", "split"]]
        feat_df = feat_df[cols]

    if output_parquet:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(output_parquet, index=False)
        schema_path = output_parquet.parent / "audio_feature_schema.json"
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_features": len(feat_df.columns) - 3,
                "feature_columns": [c for c in feat_df.columns if c not in ["sample_id", "label", "split"]],
                "sample_rate": DEFAULT_SR,
                "n_mfcc": N_MFCC,
            }, f, indent=2)
        logger.info(f"Cached audio features ({len(feat_df)} rows) to: {output_parquet}")

    return feat_df
