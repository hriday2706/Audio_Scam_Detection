"""Path management utilities for VoxShield."""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Returns the root directory of the VoxShield project."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return Path("E:/Audio_Scam_Detection").resolve()


PROJECT_ROOT = get_project_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Set cache directories to project root on drive E: to preserve drive C: space
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
