"""Wrapper for `python -m protai.data.preshard` so users can run from repo root."""
import sys
from pathlib import Path

# Make the package importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from protai.data.preshard import _cli

if __name__ == "__main__":
    _cli()
