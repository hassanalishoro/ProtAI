"""Convert a PDBbind-style affinity index into the canonical CSV ProtAI expects.

ProtAI's pre-shard step joins each MISATO complex with its experimental
binding affinity using a single CSV file at `data/MD/affinity.csv` with the
schema:

    pdb_id,neg_log_k,affinity_type
    1A1B,7.49,Ki
    10GS,5.62,Kd
    ...

`neg_log_k` is -log10(K) in molar units (so a tighter binder has a larger
value; values typically fall in 0-14). `affinity_type` is one of {Kd, Ki,
IC50} and tracks which experimental assay produced the constant.

This script handles two input formats:

    1. PDBbind's raw `INDEX_general_PL_data.YYYY` text file (whitespace
       delimited, comment lines start with `#`). This is the standard
       format published by the PDBbind authors at
       http://www.pdbbind-plus.org.cn/. Free for academic use.

    2. A pre-existing CSV with the canonical schema above. In that case the
       script just verifies and copies.

If neither path exists, you need to either:
  - Download `INDEX_general_PL_data.YYYY` from PDBbind and place it under
    `data/MD/pdbbind_index.txt`, then run this script with no arguments.
  - Or hand-write the CSV at `data/MD/affinity.csv` (one row per complex).

The script reports:
  - How many records it parsed
  - How many overlap with the complexes present in `data/processed/*.pt`
  - The split of affinity types (Kd / Ki / IC50)

Usage:
    py -3.11 scripts/build_affinity_csv.py
    py -3.11 scripts/build_affinity_csv.py --in data/MD/pdbbind_index.txt
    py -3.11 scripts/build_affinity_csv.py --in some/other/index.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CANDIDATES = [
    # Where ProtAI expects the file to be dropped — checked first.
    REPO_ROOT / "data" / "MD" / "pdbbind_index.txt",
    # Author-named files (PDBbind release naming conventions).
    REPO_ROOT / "data" / "MD" / "INDEX_general_PL_data.2020",
    REPO_ROOT / "data" / "MD" / "INDEX_general_PL_data.2019",
    REPO_ROOT / "data" / "MD" / "INDEX_general_PL.2020R1.lst",
    # Sibling DATASETS folder (the user's actual download location).
    Path("U:/FYP/DATASETS/PDB bind/Index files/INDEX_general_PL.2020R1.lst"),
    Path("U:/FYP/DATASETS/PDB bind/Index files/INDEX_general_PL_data.2020"),
    REPO_ROOT / "data" / "MD" / "affinity_raw.csv",
]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "MD" / "affinity.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Regex that pulls (kind, value, unit) from PDBbind's measurement column.
# Examples we need to handle:
#   Kd=63nM       Kd<10uM       Ki<=17nM       IC50>=300nM       Kd~50uM
# Operator group accepts: '=' '<' '>' '~' '<=' '>=' '==' '~='. Order in the
# alternation matters — the longer multi-char operators must come first so
# the regex engine doesn't greedily match '=' alone and bail on the trailing '='.
_MEAS_RE = re.compile(
    r"(?P<kind>Kd|Ki|IC50)\s*(?:<=|>=|==|~=|=|<|>|~)\s*"
    r"(?P<val>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"(?P<unit>[fpnumM]M)",
    re.IGNORECASE,
)
_UNIT_TO_LOG = {
    "fM": -15, "pM": -12, "nM": -9, "uM": -6, "mM": -3, "M": 0,
}


def _meas_to_neg_log_k(meas: str) -> Optional[Tuple[float, str]]:
    """Parse "Kd=63nM" → (7.20, 'Kd'). Returns None on parse failure."""
    m = _MEAS_RE.search(meas)
    if not m:
        return None
    kind = m.group("kind")
    # Normalise capitalisation: PDBbind sometimes writes "kd" / "KI".
    if kind.lower() == "kd":
        kind = "Kd"
    elif kind.lower() == "ki":
        kind = "Ki"
    else:
        kind = "IC50"
    try:
        val = float(m.group("val"))
    except ValueError:
        return None
    unit = m.group("unit")
    # Normalise unit casing: micro is often spelled "uM"; tolerate "UM".
    canonical_unit = None
    for k in _UNIT_TO_LOG:
        if unit.lower() == k.lower():
            canonical_unit = k
            break
    if canonical_unit is None or val <= 0:
        return None
    log_exp = _UNIT_TO_LOG[canonical_unit]
    # K in molar = val * 10^log_exp; -log10(K) = -log10(val) - log_exp
    neg_log_k = -math.log10(val) - log_exp
    return neg_log_k, kind


def _parse_pdbbind_index(path: Path) -> Dict[str, Tuple[float, str]]:
    """Parse PDBbind's whitespace-delimited INDEX file.

    Handles two known column layouts:

      OLD `INDEX_general_PL_data.YYYY` (e.g. 2019, 2020):
          PDB  resolution  year  -logK   measurement   //  reference
          3zzf 2.20        2012  7.20    Kd=63nM       //  3zzf.pdf

      NEW `INDEX_general_PL.YYYYR1.lst` (2020R1, refined 2024):
          PDB  resolution  year  measurement   //  reference
          2tpi 2.10        1982  Kd=49uM       //  2tpi.pdf (2-mer)

    The first format has the precomputed -logK in column 4; the second
    skips it and puts the measurement string there directly. We compute
    -logK from the measurement when the precomputed column is absent.
    """
    out: Dict[str, Tuple[float, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_id = parts[0].upper()

            # Pull (kind, neg_log_k_from_meas) from anywhere in the line —
            # this works for both layouts because the measurement string
            # always appears, just at different column offsets.
            meas_parsed = _meas_to_neg_log_k(line)
            if meas_parsed is None:
                # No parseable measurement → row is unusable for our target.
                continue
            neg_log_k_from_meas, kind = meas_parsed

            # Prefer the precomputed -logK if column 4 is a plain float
            # (old layout). Otherwise use the value derived from the
            # measurement string.
            try:
                neg_log_k = float(parts[3])
            except ValueError:
                neg_log_k = neg_log_k_from_meas

            out[pdb_id] = (neg_log_k, kind)
    return out


def _parse_csv(path: Path) -> Dict[str, Tuple[float, str]]:
    """Parse a pre-canonical CSV with header `pdb_id,neg_log_k,affinity_type`."""
    out: Dict[str, Tuple[float, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("pdb_id") or "").strip().upper()
            if not pid:
                continue
            try:
                neg_log_k = float(row.get("neg_log_k", ""))
            except ValueError:
                continue
            kind = (row.get("affinity_type") or "Kd").strip()
            out[pid] = (neg_log_k, kind)
    return out


def _resolve_input(arg: Optional[str]) -> Path:
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            sys.exit(f"[fatal] input path does not exist: {p}")
        return p
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    sys.exit(
        "[fatal] no affinity index file found. Tried:\n  "
        + "\n  ".join(str(c) for c in DEFAULT_INPUT_CANDIDATES)
        + "\n\n"
        "Download PDBbind's INDEX_general_PL_data.<year> from\n"
        "  http://www.pdbbind-plus.org.cn/ (free academic registration)\n"
        "and place it at any of the paths above. Or pass --in <path>."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="src", default=None,
                   help="Path to PDBbind INDEX file or pre-canonical CSV.")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT),
                   help=f"Output CSV (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})")
    args = p.parse_args()

    src = _resolve_input(args.src)
    print(f"[in]  {src.relative_to(REPO_ROOT) if REPO_ROOT in src.parents else src}")

    # Detect format by suffix + a quick peek.
    if src.suffix.lower() == ".csv":
        with src.open("r", encoding="utf-8") as f:
            head = f.readline()
        if "neg_log_k" in head.lower():
            records = _parse_csv(src)
        else:
            sys.exit(f"[fatal] CSV header missing 'neg_log_k' column: {head!r}")
    else:
        records = _parse_pdbbind_index(src)

    if not records:
        sys.exit("[fatal] parsed zero records — file format may not match PDBbind's INDEX layout.")

    # Stats: type breakdown.
    types: Dict[str, int] = {}
    for _, (_, kind) in records.items():
        types[kind] = types.get(kind, 0) + 1

    # Overlap with the processed shards.
    overlap = 0
    if PROCESSED_DIR.exists():
        existing_ids = {p.stem.upper() for p in PROCESSED_DIR.glob("*.pt")}
        overlap = len(records.keys() & existing_ids)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "neg_log_k", "affinity_type"])
        for pid in sorted(records.keys()):
            neg_log_k, kind = records[pid]
            w.writerow([pid, f"{neg_log_k:.3f}", kind])

    print(f"[out] {out_path.relative_to(REPO_ROOT)}")
    print(f"      records: {len(records):,}")
    for k in sorted(types):
        print(f"        {k:5s}: {types[k]:,}")
    if PROCESSED_DIR.exists():
        pct = 100.0 * overlap / max(len(records), 1)
        print(f"      overlap with data/processed: {overlap:,} ({pct:.1f}%)")
        # Also report MISATO-side coverage (records / total MISATO complexes).
        total_misato = len(list(PROCESSED_DIR.glob("*.pt")))
        misato_pct = 100.0 * overlap / max(total_misato, 1)
        print(f"      coverage of MISATO complexes: {misato_pct:.1f}% ({overlap:,}/{total_misato:,})")
    print("[done] next step: re-run preshard with --affinity-csv to inject labels into .pt files.")


if __name__ == "__main__":
    main()
