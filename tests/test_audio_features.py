"""Unit tests for acoustic feature extraction."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from voxshield.features.audio_features import extract_features_from_audio


def test_extract_features_shape_and_no_nan(tmp_path):
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # 440 Hz tone with noise
    sig = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), sig, sr)

    feats = extract_features_from_audio(wav_path)
    assert isinstance(feats, dict)
    assert len(feats) >= 90  # 100 features
    assert "duration_sec" in feats
    assert "rms_mean" in feats
    assert "spectral_centroid_mean" in feats
    assert "mfcc_1_mean" in feats
    assert "mfcc_20_std" in feats
    assert "pitch_f0_mean" in feats

    # Verify no NaN or Inf
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} contains NaN!"
        assert not np.isinf(v), f"Feature {k} contains Inf!"
