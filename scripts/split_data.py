"""Script to perform leak-free stratified group splitting."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxshield.data.split_data import main

if __name__ == "__main__":
    main()
