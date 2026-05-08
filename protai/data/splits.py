"""Load PDB-id splits from the MISATO split files."""
from __future__ import annotations

from pathlib import Path
from typing import List


def load_split(path: str | Path) -> List[str]:
    """Read a split file (one PDB id per line) and return sorted unique ids.

    Stripping whitespace handles both LF and CRLF endings cleanly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise ValueError(f"Split file is empty: {path}")
    return sorted(set(ids))
