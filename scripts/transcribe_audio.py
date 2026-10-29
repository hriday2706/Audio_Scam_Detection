"""Script to transcribe audio files with Whisper and evaluate transcription error rate."""

import argparse
import json
from pathlib import Path
import pandas as pd

from voxshield.features.speech_to_text import transcribe_manifest_dataset
from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import MANIFESTS_DIR, METRICS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = get_logger("scripts.transcribe_audio")


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional max files to transcribe.")
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    args = parser.parse_args()

    splits_path = MANIFESTS_DIR / "splits_en.csv"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits not found at {splits_path}")

    df_splits = pd.read_csv(splits_path)
    audio_dir = RAW_DATA_DIR / "audio"

    asr_df = transcribe_manifest_dataset(
        df_splits,
        audio_dir=audio_dir,
        output_parquet=PROCESSED_DATA_DIR / "whisper_transcripts_en.parquet",
        model_name=args.model,
        limit=args.limit,
    )

    if not asr_df.empty:
        mean_wer = float(asr_df["wer"].mean())
        median_wer = float(asr_df["wer"].median())
        mean_cer = float(asr_df["cer"].mean())
        logger.info(f"ASR Evaluation: Mean WER={mean_wer:.4f}, Median WER={median_wer:.4f}, Mean CER={mean_cer:.4f}")

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_DIR / "asr_metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": args.model,
                "transcribed_count": len(asr_df),
                "mean_wer": round(mean_wer, 4),
                "median_wer": round(median_wer, 4),
                "mean_cer": round(mean_cer, 4),
            }, f, indent=2)


if __name__ == "__main__":
    main()
