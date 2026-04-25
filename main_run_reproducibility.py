#!/usr/bin/env python3
"""
Reproducibility driver (thin entrypoint).

Implementation lives under autonuggetizer/reproducibility_*.py. Dataset JSON paths
are relative to the project root (e.g. data/reproducibility_Qampari_PARAM.json).

Usage (from the project root):
  python3 main_run_reproducibility.py --dataset Qampari_PARAM --provider openai --limit 2 --dry-run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autonuggetizer.reproducibility_cli import main

if __name__ == "__main__":
    main()
