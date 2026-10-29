"""Unit tests for dataset audit and audio validation."""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pytest

from voxshield.data.validate_data import audit_dataset_manifest, validate_single_audio


def test_validate_single_audio_nonexistent():
    res = validate_single_audio(Path("non_existent_file.wav"))
    assert res["valid"] is False
    assert res["error"] == "file_not_found"


def test_validate_single_audio_synthetic(tmp_path):
    # Generate clean 1-sec 16kHz sine wave
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test_sine.wav"
    sf.write(str(wav_path), sig, sr)

    res = validate_single_audio(wav_path)
    assert res["valid"] is True
    assert res["sample_rate"] == sr
    assert res["channels"] == 1
    assert pytest.approx(res["duration"], 0.05) == 1.0
    assert res["is_clipped"] is False


def test_validate_silent_audio(tmp_path):
    sr = 16000
    sig = np.zeros(sr, dtype=np.float32)
    wav_path = tmp_path / "test_silent.wav"
    sf.write(str(wav_path), sig, sr)

    res = validate_single_audio(wav_path)
    assert res["valid"] is False
    assert res["error"] == "silent_audio"


def test_audit_dataset_manifest_exclusions():
    df = pd.DataFrame([
        {"sample_id": "s1", "label": "scam", "transcript": "Urgent bank alert."},
        {"sample_id": "s2", "label": "legitimate", "transcript": "Hello doctor."},
        {"sample_id": "s1", "label": "scam", "transcript": "Duplicate ID row."},  # duplicate
        {"sample_id": "s3", "label": None, "transcript": "Missing label."},       # missing label
        {"sample_id": "s4", "label": "scam", "transcript": ""},                   # empty transcript
    ])

    report, cleaned = audit_dataset_manifest(df, check_audio_files=False)
    assert report["total_input_records"] == 5
    assert len(cleaned) == 1  # only s2 is valid; both s1 rows are excluded for duplicate ID
    assert report["excluded_records_count"] == 4
    assert report["duplicate_sample_ids_count"] == 1
