"""Script to check and audit the hardware and software execution environment."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxshield.config import inspect_environment, save_hardware_audit


def main():
    info = inspect_environment()
    print("\n--- System & Hardware Environment Diagnostic ---")
    for k, v in info.items():
        print(f"  {k:<22}: {v}")
    out_file = save_hardware_audit()
    print(f"\nSaved diagnostic report to: {out_file}\n")


if __name__ == "__main__":
    main()
