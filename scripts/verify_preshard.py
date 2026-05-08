"""Verify the presharded dataset is intact, well-aligned with splits, and
free of common silent-corruption modes (NaN, out-of-range indices, etc).

Two modes:

    py -3.11 scripts/verify_preshard.py
        Fast: file count, schema of one record, splits coverage. ~1 second.

    py -3.11 scripts/verify_preshard.py --deep
        Slow: also opens a random sample of 200 files and runs value-range
        and integrity checks on each (NaN, edge_index validity, atomic
        number range, adaptability ≥ 0, finite energies). ~30-60 seconds.

Run --deep at least once after any fresh preshard. The cheap path is enough
for sanity-checking that the preshard directory is the right one before
launching training.

Optional: pass --processed-dir <path> to verify a non-default location
(e.g. data/processed_lite from the --no-traj preshard).
"""
import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
SPLITS = REPO_ROOT / "data" / "MD" / "splits"


# ---------------------------------------------------------------------------
# Cheap checks (always run)
# ---------------------------------------------------------------------------

def _print_counts(files):
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"\n[counts]")
    print(f"  files:         {len(files):,}")
    print(f"  total size:    {total_bytes / 1e9:.2f} GB")
    if files:
        print(f"  avg file size: {total_bytes / len(files) / 1e6:.2f} MB")


def _print_sample_schema(files):
    if not files:
        return None
    sample_path = files[0]
    rec = torch.load(sample_path, weights_only=False)
    print(f"\n[sample schema: {sample_path.stem}]")
    for k, v in rec.items():
        if hasattr(v, "shape"):
            shape_str = str(tuple(v.shape))
            print(f"  {k:22s} shape={shape_str:<25} dtype={v.dtype}")
        else:
            print(f"  {k:22s} {v}")
    return rec


def _print_splits_coverage(on_disk_ids):
    print(f"\n[splits coverage]")
    for split_name in ("train_MD", "val_MD", "test_MD"):
        split_file = SPLITS / f"{split_name}.txt"
        if not split_file.exists():
            print(f"  {split_name:12s} (split file missing)")
            continue
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        have = sum(1 for i in ids if i in on_disk_ids)
        missing = len(ids) - have
        print(f"  {split_name:12s} split={len(ids):>6}  on_disk={have:>6}  missing={missing}")


# ---------------------------------------------------------------------------
# Deep checks (run with --deep)
# ---------------------------------------------------------------------------

def _check_record(rec: dict, pdb_id: str) -> list[str]:
    """Return a list of integrity issues for this record. Empty list = clean."""
    issues = []

    pos = rec.get("pos")
    z = rec.get("z")
    edge_index = rec.get("edge_index")
    edge_attr = rec.get("edge_attr")
    adaptability = rec.get("adaptability")
    y_energy_mean = rec.get("y_energy_mean")
    y_energy_per_frame = rec.get("y_energy_per_frame")

    # 1. Tensors are present.
    for name in ("pos", "z", "edge_index", "edge_attr", "adaptability",
                 "y_energy_mean", "y_energy_per_frame"):
        if rec.get(name) is None:
            issues.append(f"missing field: {name}")

    if pos is None or z is None:
        return issues  # can't run further checks without these

    n = pos.shape[0]

    # 2. NaN / Inf finiteness on every floating tensor.
    for name, t in (
        ("pos", pos),
        ("edge_attr", edge_attr),
        ("adaptability", adaptability),
        ("y_energy_mean", y_energy_mean),
        ("y_energy_per_frame", y_energy_per_frame),
    ):
        if t is None:
            continue
        if torch.is_floating_point(t) and not torch.isfinite(t).all():
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            issues.append(f"{name}: {n_nan} NaN, {n_inf} Inf")

    # 3. Atomic numbers in plausible range [1, 100]. Heavy atoms only after
    #    H stripping → so range is actually [3, ~83] for our data, but [1, 100]
    #    is the sanity envelope.
    if z is not None:
        zmin, zmax = int(z.min()), int(z.max())
        if zmin < 1 or zmax > 100:
            issues.append(f"z out of [1, 100]: min={zmin} max={zmax}")

    # 4. Adaptability is a distance, must be ≥ 0.
    if adaptability is not None:
        if (adaptability < 0).any():
            n_neg = int((adaptability < 0).sum())
            issues.append(f"adaptability has {n_neg} negative values")
        if adaptability.shape[0] != n:
            issues.append(f"adaptability shape {tuple(adaptability.shape)} != ({n},)")

    # 5. edge_index validity: integer, shape (2, E), values in [0, n).
    if edge_index is not None:
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            issues.append(f"edge_index shape {tuple(edge_index.shape)} != (2, E)")
        else:
            ei_min, ei_max = int(edge_index.min()) if edge_index.numel() else 0, \
                             int(edge_index.max()) if edge_index.numel() else -1
            if edge_index.numel() > 0 and (ei_min < 0 or ei_max >= n):
                issues.append(f"edge_index values out of [0, {n}): min={ei_min} max={ei_max}")
            # Self-loops: row[i] == col[i]. We exclude these in build_radius_graph;
            # any present here is a bug.
            if edge_index.numel() > 0 and (edge_index[0] == edge_index[1]).any():
                n_self = int((edge_index[0] == edge_index[1]).sum())
                issues.append(f"edge_index has {n_self} self-loops")

    # 6. edge_attr shape matches edge_index.
    if edge_index is not None and edge_attr is not None:
        if edge_attr.shape[0] != edge_index.shape[1]:
            issues.append(
                f"edge_attr length {edge_attr.shape[0]} != E={edge_index.shape[1]}"
            )

    # 7. Energy sanity: protein-ligand interaction energies are typically
    #    in [-300, +50] kcal/mol. Way outside that = corruption.
    if y_energy_mean is not None:
        m = float(y_energy_mean)
        if not (-500 < m < 100):
            issues.append(f"y_energy_mean {m:.2f} outside expected kcal/mol range")

    # 8. pos_traj (if present) shape consistency.
    pos_traj = rec.get("pos_traj")
    if pos_traj is not None:
        if pos_traj.dim() != 3 or pos_traj.shape[1] != n or pos_traj.shape[2] != 3:
            issues.append(
                f"pos_traj shape {tuple(pos_traj.shape)} not (T, {n}, 3)"
            )
        if torch.is_floating_point(pos_traj) and not torch.isfinite(pos_traj).all():
            issues.append("pos_traj has NaN or Inf")

    return issues


def _deep_check(files: list[Path], n_sample: int = 200, seed: int = 42):
    """Sample n_sample files and run integrity checks on each."""
    print(f"\n[deep check: {min(n_sample, len(files))} random samples]")
    rng = random.Random(seed)
    sample = rng.sample(files, min(n_sample, len(files)))

    schema_keys = None
    bad_files: list[tuple[str, list[str]]] = []
    n_clean = 0
    # Aggregated stats
    n_atoms = []
    n_edges = []
    energies = []

    for path in sample:
        try:
            rec = torch.load(path, weights_only=False)
        except Exception as e:
            bad_files.append((path.stem, [f"load failed: {type(e).__name__}: {e}"]))
            continue

        # 0. Schema consistency across the sample.
        keys = set(rec.keys())
        if schema_keys is None:
            schema_keys = keys
        elif keys != schema_keys:
            extra = keys - schema_keys
            missing = schema_keys - keys
            bad_files.append((
                path.stem,
                [f"schema mismatch: extra={sorted(extra)} missing={sorted(missing)}"],
            ))
            continue

        issues = _check_record(rec, path.stem)
        if issues:
            bad_files.append((path.stem, issues))
        else:
            n_clean += 1

        # Collect stats from this file regardless of issues.
        if "pos" in rec:
            n_atoms.append(rec["pos"].shape[0])
        if "edge_index" in rec and rec["edge_index"] is not None:
            n_edges.append(rec["edge_index"].shape[1])
        if "y_energy_mean" in rec and rec["y_energy_mean"] is not None:
            energies.append(float(rec["y_energy_mean"]))

    print(f"  clean: {n_clean}/{len(sample)}")
    if bad_files:
        print(f"  issues found in {len(bad_files)} files (showing first 10):")
        for pdb_id, issues in bad_files[:10]:
            print(f"    {pdb_id}:")
            for issue in issues:
                print(f"      - {issue}")
        if len(bad_files) > 10:
            print(f"    ... and {len(bad_files) - 10} more")

    # Aggregated stats — useful for spotting outliers.
    if n_atoms:
        import statistics
        print(f"\n[aggregate stats over sample]")
        print(f"  atoms per complex:  median={statistics.median(n_atoms):.0f} "
              f"min={min(n_atoms)} max={max(n_atoms)}")
        print(f"  edges per complex:  median={statistics.median(n_edges):.0f} "
              f"min={min(n_edges)} max={max(n_edges)}")
        if energies:
            print(f"  y_energy_mean:      median={statistics.median(energies):.2f} "
                  f"min={min(energies):.2f} max={max(energies):.2f} (kcal/mol expected)")

    return len(bad_files) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=str(DEFAULT_PROCESSED),
                    help="Directory of {pdb_id}.pt files (default: data/processed)")
    ap.add_argument("--deep", action="store_true",
                    help="Open a random sample and run value-range / integrity checks")
    ap.add_argument("--sample-size", type=int, default=200,
                    help="How many files to deep-check (default: 200)")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    if not processed.exists():
        print(f"[error] {processed} does not exist")
        sys.exit(1)

    print("=" * 70)
    print(f"Preshard verification: {processed}")
    print("=" * 70)

    files = sorted(processed.glob("*.pt"))
    if not files:
        print(f"[error] no .pt files in {processed}")
        sys.exit(1)

    _print_counts(files)
    _print_sample_schema(files)
    _print_splits_coverage({f.stem for f in files})

    if args.deep:
        ok = _deep_check(files, n_sample=args.sample_size)
    else:
        ok = True
        print("\n[deep check skipped — run with --deep to validate values]")

    print("\n" + "=" * 70)
    if ok:
        print("DONE — no issues found")
    else:
        print("DONE — issues found above; investigate before training")
        sys.exit(2)


if __name__ == "__main__":
    main()
