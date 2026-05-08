"""Verify the presharded dataset is intact and aligned with splits.

Prints: file count, total size, sample record schema, splits coverage.
Run from repo root:
    py -3.11 scripts/verify_preshard.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

PROCESSED = REPO_ROOT / "data" / "processed"
SPLITS = REPO_ROOT / "data" / "MD" / "splits"


def main():
    print("=" * 70)
    print("Preshard verification")
    print("=" * 70)

    # 1. File count + total size
    files = sorted(PROCESSED.glob("*.pt"))
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\n[counts]")
    print(f"  files: {len(files):,}")
    print(f"  total size: {total_bytes / 1e9:.2f} GB")
    print(f"  avg file size: {total_bytes / len(files) / 1e6:.1f} MB" if files else "")

    # 2. Sample record schema (pick first that exists)
    sample_id = "1A1B"
    sample_path = PROCESSED / f"{sample_id}.pt"
    if not sample_path.exists() and files:
        sample_path = files[0]
        sample_id = sample_path.stem
    print(f"\n[sample: {sample_id}]")
    rec = torch.load(sample_path, weights_only=False)
    for k, v in rec.items():
        if hasattr(v, "shape"):
            shape_str = str(tuple(v.shape))
            print(f"  {k:22s} shape={shape_str:<25} dtype={v.dtype}")
        else:
            print(f"  {k:22s} {v}")

    # 3. Splits coverage
    print(f"\n[splits coverage]")
    on_disk = {f.stem for f in files}
    for split_name in ("train_MD", "val_MD", "test_MD"):
        split_file = SPLITS / f"{split_name}.txt"
        if not split_file.exists():
            print(f"  {split_name:12s} (split file missing)")
            continue
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        have = sum(1 for i in ids if i in on_disk)
        missing = len(ids) - have
        print(f"  {split_name:12s} split={len(ids):>6}  on_disk={have:>6}  missing={missing}")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
