"""Unit tests for prediction pipeline."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from voxshield.models.predict import VoxShieldPredictor


def test_predict_pipeline_mock(tmp_path):
    # Create synthetic test wav
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 300 * t)
    wav_path = tmp_path / "test_sample.wav"
    sf.write(str(wav_path), sig, sr)

    predictor = VoxShieldPredictor()
    res = predictor.predict(
        audio_path=wav_path,
        reference_transcript="Your account has been locked. Urgent action required.",
        threshold=0.5,
    )

    assert "verdict" in res
    assert "fusion_risk_score" in res
    assert "text_risk_score" in res
    assert "audio_risk_score" in res
    assert "acoustic_evidence" in res
    assert "disclaimer" in res
    assert 0.0 <= res["fusion_risk_score"] <= 1.0
    assert isinstance(res["is_scam"], bool)
