"""VoxShield: Multimodal Voice Scam Risk Analysis package."""

import os
from pathlib import Path

# Redirect large neural caches to Drive E project cache
_cache_dir = Path("E:/Audio_Scam_Detection/.cache")
os.environ.setdefault("HF_HOME", str(_cache_dir / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(_cache_dir / "torch"))

__version__ = "0.1.0"
