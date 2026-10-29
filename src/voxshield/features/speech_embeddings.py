"""Pretrained self-supervised speech representation extraction using frozen HuBERT."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import PROCESSED_DATA_DIR

logger = get_logger("voxshield.features.speech_embeddings")

DEFAULT_HUBERT_MODEL = "facebook/hubert-base-ls960"


class HuBERTEmbeddingExtractor:
    """Extracts mean-pooled frame representations from audio using frozen HuBERT weights."""

    def __init__(self, model_name: str = DEFAULT_HUBERT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing HuBERT extractor: {model_name} on device '{self.device}'...")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("HuBERT model loaded and frozen successfully.")

    @torch.inference_mode()
    def extract_embedding(self, audio_path: Path) -> np.ndarray:
        """Loads audio at 16 kHz and returns a 768-dim mean-pooled embedding."""
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if len(y) == 0:
            return np.zeros(768, dtype=np.float32)

        # Truncate or chunk if excessively long (e.g., max 60 seconds = 960,000 samples)
        max_samples = 16000 * 60
        if len(y) > max_samples:
            y = y[:max_samples]

        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        hidden_states = outputs.last_hidden_state  # shape [1, T, 768]
        mean_pooled = torch.mean(hidden_states, dim=1).squeeze(0).cpu().numpy()
        return mean_pooled


def extract_manifest_hubert_embeddings(
    manifest_df: pd.DataFrame,
    audio_dir: Path,
    output_parquet: Path = PROCESSED_DATA_DIR / "hubert_embeddings_en.parquet",
    model_name: str = DEFAULT_HUBERT_MODEL,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Extracts and caches HuBERT embeddings for all manifest audio files."""
    existing_df = None
    if output_parquet.exists():
        existing_df = pd.read_parquet(output_parquet)
        done_ids = set(existing_df["sample_id"])
    else:
        done_ids = set()

    target_df = manifest_df[~manifest_df["sample_id"].isin(done_ids)].copy()
    if limit is not None:
        target_df = target_df.head(limit)

    logger.info(f"Extracting HuBERT embeddings for {len(target_df)} audio files (already cached: {len(done_ids)})...")
    if len(target_df) == 0:
        return existing_df if existing_df is not None else pd.DataFrame()

    extractor = HuBERTEmbeddingExtractor(model_name=model_name)
    rows = []

    for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="HuBERT embeddings"):
        sid = row["sample_id"]
        audio_path = audio_dir / f"{sid}.wav"
        if not audio_path.exists():
            continue

        try:
            emb = extractor.extract_embedding(audio_path)
            row_dict = {
                "sample_id": sid,
                "split": row.get("split"),
                "label": row.get("label"),
            }
            for i, val in enumerate(emb):
                row_dict[f"emb_{i}"] = float(val)
            rows.append(row_dict)
        except Exception as e:
            logger.warning(f"HuBERT extraction failed for {sid}: {e}")

    new_df = pd.DataFrame(rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True) if existing_df is not None else new_df
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved {len(combined_df)} HuBERT embeddings to {output_parquet}")
    return combined_df
