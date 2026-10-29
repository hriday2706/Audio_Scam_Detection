"""Dataset downloader and loader for VISHGUARD via Zenodo."""

import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import struct
import time
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

from voxshield.utils.logging_utils import get_logger
from voxshield.utils.paths import DATA_DIR, RAW_DATA_DIR

logger = get_logger("voxshield.data.load_dataset")

ZENODO_RECORD_URL = "https://zenodo.org/api/records/20429763"
ZENODO_ZIP_URL = "https://zenodo.org/api/records/20429763/files/vishguard_dataset.zip/content"
EXPECTED_MD5 = "b6bf4c29812159ddecf7e6042fe0958f"


class RemoteZipStream(io.RawIOBase):
    """File-like object that reads arbitrary slices of a remote HTTP resource via HTTP Range requests."""

    def __init__(self, url: str, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=20)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers.get("content-length", 0))
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        headers = {"Range": f"bytes={self.pos}-{self.pos + size - 1}"}
        resp = self.session.get(self.url, headers=headers, timeout=30)
        resp.raise_for_status()
        chunk = resp.content
        self.pos += len(chunk)
        return chunk

    def seekable(self) -> bool:
        return True


def fetch_vishguard_metadata(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, Any]:
    """Downloads vishguard_data.json from Zenodo remote archive if not already cached."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_json = raw_dir / "vishguard_data.json"
    if local_json.exists():
        logger.info(f"Using cached metadata: {local_json}")
        with open(local_json, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Connecting to remote VISHGUARD Zenodo archive to retrieve metadata...")
    stream = RemoteZipStream(ZENODO_ZIP_URL)
    with zipfile.ZipFile(stream) as zf:
        json_content = zf.read("vishguard_dataset/vishguard_data.json")
        with open(local_json, "wb") as f:
            f.write(json_content)

    logger.info(f"Saved metadata to {local_json}")
    with open(local_json, "r", encoding="utf-8") as f:
        return json.load(f)


def download_selected_samples(
    sample_ids: List[str],
    raw_dir: Path = RAW_DATA_DIR,
    include_transcripts: bool = True,
    max_workers: int = 10,
) -> int:
    """Downloads audio WAV files from Zenodo archive using direct single-request range extraction with multithreading."""
    audio_dir = raw_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = raw_dir / "transcripts"
    if include_transcripts:
        transcripts_dir.mkdir(parents=True, exist_ok=True)

    needed_audio = [sid for sid in sample_ids if not (audio_dir / f"{sid}.wav").exists()]
    if not needed_audio:
        logger.info(f"All {len(sample_ids)} audio files already present in {audio_dir}")
        return len(sample_ids)

    logger.info(f"Querying central directory for {len(needed_audio)} needed audio files...")
    stream = RemoteZipStream(ZENODO_ZIP_URL)
    with zipfile.ZipFile(stream) as zf:
        audio_info_map = {}
        txt_info_map = {}
        for sid in needed_audio:
            try:
                audio_info_map[sid] = zf.getinfo(f"vishguard_dataset/audio/{sid}.wav")
            except KeyError:
                pass
            if include_transcripts:
                try:
                    txt_info_map[sid] = zf.getinfo(f"vishguard_dataset/transcripts/{sid}.txt")
                except KeyError:
                    pass

    direct_url = stream.url

    def download_worker(sid: str) -> bool:
        if sid not in audio_info_map:
            return False
        info = audio_info_map[sid]
        start_pos = info.header_offset
        end_pos = start_pos + 30 + len(info.filename) + 1024 + info.compress_size

        for attempt in range(3):
            try:
                headers = {"Range": f"bytes={start_pos}-{end_pos}"}
                resp = requests.get(direct_url, headers=headers, timeout=45)
                raw = resp.content
                if len(raw) < 30:
                    time.sleep(0.5)
                    continue

                sig, ver, flags, comp, mtime, mdate, crc, csize, usize, fn_len, extra_len = struct.unpack("<4sHHHHHIIIHH", raw[:30])
                data_offset = 30 + fn_len + extra_len

                if len(raw) < data_offset + info.compress_size:
                    # Request was short, fetch exact data range
                    actual_data_start = start_pos + data_offset
                    d_headers = {"Range": f"bytes={actual_data_start}-{actual_data_start + info.compress_size - 1}"}
                    d_resp = requests.get(direct_url, headers=d_headers, timeout=45)
                    compressed = d_resp.content
                else:
                    compressed = raw[data_offset : data_offset + info.compress_size]

                decomp = zlib.decompress(compressed, -15)
                with open(audio_dir / f"{sid}.wav", "wb") as f:
                    f.write(decomp)

                if include_transcripts and sid in txt_info_map:
                    t_info = txt_info_map[sid]
                    t_start = t_info.header_offset
                    t_end = t_start + 30 + len(t_info.filename) + 512 + t_info.compress_size
                    t_resp = requests.get(direct_url, headers={"Range": f"bytes={t_start}-{t_end}"}, timeout=20)
                    t_raw = t_resp.content
                    t_sig, t_ver, t_flags, t_comp, t_mtime, t_mdate, t_crc, t_csize, t_usize, t_fn_len, t_extra_len = struct.unpack("<4sHHHHHIIIHH", t_raw[:30])
                    t_offset = 30 + t_fn_len + t_extra_len
                    t_comp_data = t_raw[t_offset : t_offset + t_info.compress_size]
                    if t_comp == 8:
                        t_decomp = zlib.decompress(t_comp_data, -15)
                    else:
                        t_decomp = t_comp_data
                    with open(transcripts_dir / f"{sid}.txt", "wb") as tf:
                        tf.write(t_decomp)

                return True
            except Exception as e:
                time.sleep(1.0)

        logger.warning(f"Failed to download {sid} after 3 attempts.")
        return False

    logger.info(f"Downloading {len(needed_audio)} audio files in parallel ({max_workers} threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_worker, sid): sid for sid in needed_audio}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(needed_audio), desc="Downloading audio"):
            future.result()

    downloaded = sum(1 for sid in sample_ids if (audio_dir / f"{sid}.wav").exists())
    logger.info(f"Total audio files available in {audio_dir}: {downloaded}/{len(sample_ids)}")
    return downloaded
