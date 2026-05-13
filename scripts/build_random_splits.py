"""Generate random train/val/test splits matching MISATO's official proportions.

The MISATO release ships sequence-similarity splits at
`data/MD/splits/{train,val,test}_MD.txt` (13,765 / 1,595 / 1,612). Those
are intentionally hard — protein families in the test set are absent from
training, which makes them an out-of-distribution evaluation.

For within-distribution evaluation (comparable to standard PDBbind
benchmarks), this script builds a matched random partition: same total
sizes, but the membership is a `random.shuffle` of the full PDB ID list
seeded with --seed. Default seed is 42 to match the experiments reported
in the ProtAI thesis.

Output files (overwrites if they already exist):
    data/MD/splits/train_random.txt   13,765 ids
    data/MD/splits/val_random.txt      1,595 ids
    data/MD/splits/test_random.txt     1,612 ids

Source of IDs: by default reads from `data/processed/*.pt` (so the splits
only contain complexes that survived preshard). Override with --source-h5
to read directly from MISATO's HDF5 file instead.

Usage:
    py -3.11 scripts/build_random_splits.py
    py -3.11 scripts/build_random_splits.py --seed 1337
    py -3.11 scripts/build_random_splits.py --source-h5 data/MD/h5_files/MD.hdf5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "MD" / "splits"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Match MISATO's published proportions exactly so the random and similarity
# split families are directly comparable.
DEFAULT_N_TRAIN = 13_765
DEFAULT_N_VAL = 1_595
DEFAULT_N_TEST = 1_612


def _ids_from_processed(processed_dir: Path) -> list[str]:
    if not processed_dir.exists():
        return []
    return sorted(p.stem.upper() for p in processed_dir.glob("*.pt"))


def _ids_from_h5(h5_path: Path) -> list[str]:
    import h5py
    with h5py.File(h5_path, "r") as f:
        return sorted(k.upper() for k in f.keys())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    p.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    p.add_argument("--n-val", type=int, default=DEFAULT_N_VAL)
    p.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    p.add_argument("--source-h5", default=None,
                   help="Read PDB IDs from MD.hdf5 instead of data/processed/*.pt")
    p.add_argument("--out-suffix", default="random",
                   help="Suffix on output filenames (default: 'random' → "
                        "train_random.txt etc).")
    args = p.parse_args()

    if args.source_h5:
        ids = _ids_from_h5(Path(args.source_h5))
        src_label = f"MD.hdf5 ({args.source_h5})"
    else:
        ids = _ids_from_processed(PROCESSED_DIR)
        src_label = f"data/processed/*.pt"
        if not ids:
            sys.exit(
                f"[fatal] {PROCESSED_DIR} has no .pt files. Either pre-shard "
                f"first or pass --source-h5 to read PDB IDs directly from the "
                f"HDF5 dataset."
            )

    total_requested = args.n_train + args.n_val + args.n_test
    if len(ids) < total_requested:
        print(
            f"[warn] only {len(ids):,} complexes available, less than the "
            f"{total_requested:,} requested. Sizes will be scaled down "
            f"proportionally."
        )
        # Scale all three down proportionally; round to keep the total in [N-1, N+1]
        scale = len(ids) / total_requested
        args.n_train = int(args.n_train * scale)
        args.n_val = int(args.n_val * scale)
        args.n_test = len(ids) - args.n_train - args.n_val

    rng = random.Random(args.seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    train = shuffled[: args.n_train]
    val = shuffled[args.n_train : args.n_train + args.n_val]
    test = shuffled[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [
        (f"train_{args.out_suffix}.txt", train),
        (f"val_{args.out_suffix}.txt", val),
        (f"test_{args.out_suffix}.txt", test),
    ]
    print(f"[in]  source: {src_label} ({len(ids):,} ids)")
    print(f"      seed:   {args.seed}")
    print(f"[out] {SPLITS_DIR.relative_to(REPO_ROOT)}/")
    for name, members in pairs:
        path = SPLITS_DIR / name
        path.write_text("\n".join(members) + "\n")
        print(f"  {name:25s} {len(members):>6,} ids")
    print("[done]")


if __name__ == "__main__":
    main()
