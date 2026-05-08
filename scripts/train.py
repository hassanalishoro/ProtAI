"""Wrapper for `python -m protai.training.train`."""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protai.training.train import main

if __name__ == "__main__":
    main()
