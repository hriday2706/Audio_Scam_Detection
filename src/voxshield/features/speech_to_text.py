"""Speech-to-text pipeline using Hugging Face Whisper with caching and WER/CER analysis."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import CACHE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = get_logger("voxshield.features.speech_to_text")

DEFAULT_WHISPER_MODEL = "openai/whisper-small"


def compute_wer_cer(ref: str, hyp: str) -> Tuple[float, float]:
    """Computes Word Error Rate (WER) and Character Error Rate (CER) via dynamic programming."""
    ref_words = ref.strip().lower().split()
    hyp_words = hyp.strip().lower().split()

    # Levenshtein distance on words
    d_word = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d_word[i][0] = i
    for j in range(len(hyp_words) + 1):
        d_word[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d_word[i][j] = d_word[i - 1][j - 1]
            else:
                d_word[i][j] = min(
                    d_word[i - 1][j] + 1,      # deletion
                    d_word[i][j - 1] + 1,      # insertion
                    d_word[i - 1][j - 1] + 1,  # substitution
                )
    wer = float(d_word[len(ref_words)][len(hyp_words)] / max(1, len(ref_words)))

    # Levenshtein distance on characters
    ref_chars = list("".join(ref.strip().lower().split()))
    hyp_chars = list("".join(hyp.strip().lower().split()))
    d_char = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=int)
    for i in range(len(ref_chars) + 1):
        d_char[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d_char[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d_char[i][j] = d_char[i - 1][j - 1]
            else:
                d_char[i][j] = min(
                    d_char[i - 1][j] + 1,
                    d_char[i][j - 1] + 1,
                    d_char[i - 1][j - 1] + 1,
                )
    cer = float(d_char[len(ref_chars)][len(hyp_chars)] / max(1, len(ref_chars)))

    return round(wer, 4), round(cer, 4)


class WhisperTranscriber:
    """Encapsulates Whisper model loading and single/batch audio transcription."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Whisper pipeline: {model_name} on device '{self.device}'...")
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            torch_dtype=torch_dtype,
            device=self.device,
            chunk_length_s=30,
        )
        logger.info("Whisper model loaded successfully.")

    def transcribe_file(self, audio_path: Path, language: str = "en") -> Dict[str, Any]:
        """Transcribes a single audio file and returns text and latency."""
        t0 = time.time()
        # Librosa loads as 16kHz mono float32 for Whisper input
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        generate_kwargs = {"language": language, "task": "transcribe"}
        result = self.asr_pipeline({"raw": y, "sampling_rate": sr}, generate_kwargs=generate_kwargs)
        latency = round(time.time() - t0, 3)
        return {
            "asr_transcript": result["text"].strip(),
            "latency_seconds": latency,
        }


def transcribe_manifest_dataset(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    output_parquet: Path = PROCESSED_DATA_DIR / "whisper_transcripts_en.parquet",
    model_name: str = DEFAULT_WHISPER_MODEL,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Transcribes dataset audio files, records ASR transcripts, calculates WER/CER, and caches output."""
    existing_df = None
    if output_parquet.exists():
        existing_df = pd.read_parquet(output_parquet)
        done_ids = set(existing_df["sample_id"])
    else:
        done_ids = set()

    target_df = manifest_df[~manifest_df["sample_id"].isin(done_ids)].copy()
    if limit is not None:
        target_df = target_df.head(limit)

    logger.info(f"Transcribing {len(target_df)} audio files (already cached: {len(done_ids)})...")
    if len(target_df) == 0:
        return existing_df if existing_df is not None else pd.DataFrame()

    transcriber = WhisperTranscriber(model_name=model_name)
    rows = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Whisper ASR"):
        sid = row["sample_id"]
        ref_text = str(row.get("transcript", ""))
        audio_path = audio_dir / f"{sid}.wav"

        if not audio_path.exists():
            continue

        try:
            res = transcriber.transcribe_file(audio_path, language="en")
            asr_text = res["asr_transcript"]
            wer, cer = compute_wer_cer(ref_text, asr_text)
            rows.append({
                "sample_id": sid,
                "split": row.get("split"),
                "label": row.get("label"),
                "reference_transcript": ref_text,
                "asr_transcript": asr_text,
                "wer": wer,
                "cer": cer,
                "asr_latency": res["latency_seconds"],
                "model_name": model_name,
            })
        except Exception as e:
            logger.warning(f"ASR failed for {sid}: {e}")

    new_df = pd.DataFrame(rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(combined_df)} ASR transcripts to {output_parquet}")
    return combined_df
