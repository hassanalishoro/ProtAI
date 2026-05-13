"""Filter existing splits to the subset of complexes that have a PDBbind log K label.

Reads `data/MD/affinity.csv` (the canonical affinity index produced by
`build_affinity_csv.py`) and the existing `train_random.txt`,
`val_random.txt`, `test_random.txt` (or any other split family). Writes
filtered counterparts with a `_logk` suffix:

    train_random_logk.txt
    val_random_logk.txt
    test_random_logk.txt

The new splits preserve the original train/val/test partitioning — they
just drop complexes that don't have a measured affinity. This keeps the
trajectory-vs-static comparison directly comparable to the original
y_energy_mean experiments (same protein family overlap, same proportions).

Usage:
    py -3.11 scripts/build_logk_splits.py
    py -3.11 scripts/build_logk_splits.py --base random
    py -3.11 scripts/build_logk_splits.py --base MD     # for similarity splits
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "MD" / "splits"
AFFINITY_CSV = REPO_ROOT / "data" / "MD" / "affinity.csv"


def load_affinity_table(csv_path: Path) -> Dict[str, str]:
    """Returns {pdb_id_upper: affinity_type}. Drops rows with non-finite K."""
    if not csv_path.exists():
        raise SystemExit(
            f"[fatal] affinity CSV not found: {csv_path}\n"
            f"Build it first with `py -3.11 scripts/build_affinity_csv.py`."
        )
    out: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = (row.get("pdb_id") or "").strip().upper()
            try:
                v = float(row.get("neg_log_k", ""))
            except ValueError:
                continue
            if pid and v == v:  # NaN-safe
                out[pid] = (row.get("affinity_type") or "Unknown").strip()
    return out


def filter_split(src: Path, table: Dict[str, str]) -> tuple[list[str], Dict[str, int]]:
    """Returns (kept_ids, per-type counts) for the kept subset."""
    if not src.exists():
        raise SystemExit(f"[fatal] split not found: {src}")
    ids = [line.strip() for line in src.read_text().splitlines() if line.strip()]
    kept = [i for i in ids if i.upper() in table]
    counts: Dict[str, int] = {}
    for i in kept:
        t = table[i.upper()]
        counts[t] = counts.get(t, 0) + 1
    return kept, counts


def _fmt_counts(counts: Dict[str, int], total: int) -> str:
    if total == 0:
        return "(empty)"
    parts = []
    for k in sorted(counts):
        pct = 100.0 * counts[k] / total
        parts.append(f"{k}={counts[k]} ({pct:.1f}%)")
    return ", ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="random",
                   help="Split family suffix (default: 'random' → train_random.txt etc). "
                        "Use 'MD' for the MISATO sequence-similarity splits.")
    p.add_argument("--out-suffix", default="logk",
                   help="Suffix appended to the output filename (default: 'logk').")
    p.add_argument("--affinity-csv", default=str(AFFINITY_CSV),
                   help=f"Affinity CSV (default: {AFFINITY_CSV.relative_to(REPO_ROOT)})")
    args = p.parse_args()

    table = load_affinity_table(Path(args.affinity_csv))
    print(f"[in]  affinity table: {len(table):,} labelled complexes")

    base = args.base
    suffix = args.out_suffix
    pairs = [
        (SPLITS_DIR / f"train_{base}.txt", SPLITS_DIR / f"train_{base}_{suffix}.txt"),
        (SPLITS_DIR / f"val_{base}.txt",   SPLITS_DIR / f"val_{base}_{suffix}.txt"),
        (SPLITS_DIR / f"test_{base}.txt",  SPLITS_DIR / f"test_{base}_{suffix}.txt"),
    ]

    print("\n[verify] split sizes + affinity-type distribution after filtering")
    print("─" * 78)
    grand_counts: Dict[str, int] = {}
    grand_total = 0
    for src, dst in pairs:
        kept, counts = filter_split(src, table)
        dst.write_text("\n".join(kept) + "\n")
        n_orig = len([l for l in src.read_text().splitlines() if l.strip()])
        kept_pct = 100.0 * len(kept) / max(n_orig, 1)
        print(f"  {src.name:25s}  {n_orig:>6,} → {len(kept):>6,} ({kept_pct:5.1f}%)")
        print(f"      {dst.name}")
        print(f"      types: {_fmt_counts(counts, len(kept))}")
        for k, v in counts.items():
            grand_counts[k] = grand_counts.get(k, 0) + v
        grand_total += len(kept)

    print("─" * 78)
    print(f"  TOTAL (train+val+test): {grand_total:,}")
    print(f"      types: {_fmt_counts(grand_counts, grand_total)}")
    # Covariate-shift sanity check: warn if test type distribution drifts from train.
    train_kept, train_counts = filter_split(pairs[0][0], table)
    test_kept, test_counts = filter_split(pairs[2][0], table)
    if train_kept and test_kept:
        types = sorted(set(train_counts) | set(test_counts))
        print("\n[verify] train-vs-test type drift (large deltas = covariate shift)")
        for t in types:
            tr_pct = 100.0 * train_counts.get(t, 0) / len(train_kept)
            te_pct = 100.0 * test_counts.get(t, 0) / len(test_kept)
            delta = te_pct - tr_pct
            flag = "  ⚠" if abs(delta) > 5.0 else ""
            print(f"  {t:6s} train={tr_pct:5.1f}%  test={te_pct:5.1f}%  Δ={delta:+5.1f}pp{flag}")
    print("\n[done]")


if __name__ == "__main__":
    main()
