"""One-time pre-sharding: walk MD.hdf5 and emit one .pt file per complex.

Why: training reads pre-built graphs ~5-10x faster than re-parsing the H5
every batch, and side-steps the fork-safety problem with h5py + DataLoader workers.

Per-complex output (a dict of tensors saved with torch.save):
    pdb_id              : str
    pos                 : (N, 3)   float32  reference frame coordinates
    pos_traj (optional) : (T, N, 3) float32  full trajectory (if keep_trajectory)
    z                   : (N,)     int64    real atomic numbers (for SchNet)
    element_idx         : (N,)     int64    MISATO 1-based element index (for GNN_MD)
    edge_index          : (2, E)   int64    radius graph at cutoff, both directions
    edge_attr           : (E,)     float32  inverse distances aligned with edge_index
    y_energy_per_frame  : (T,)     float32  per-frame interaction energy (kcal/mol)
    y_energy_mean       : ()       float32  mean over frames (for graph-level target)
    y_logk              : ()       float32  experimental -log10(K) from PDBbind (NaN if absent)
    affinity_type       : str               'Kd' | 'Ki' | 'IC50' | 'Unknown'
    adaptability        : (N,)     float32  per-atom flexibility from trajectory variance
    mol_idx             : (3,)     int64    [0, ligand_start, solvent_start]

Hydrogens and solvent are stripped by default. All floats stored as float32.
The optional `--affinity-csv` argument joins each complex with an external
PDBbind-style affinity index (see `scripts/build_affinity_csv.py`). Without
that argument, `y_logk` defaults to NaN and `affinity_type` to 'Unknown',
which is enough for the training pipeline to skip log-K-target rows
gracefully via `PreShardedDataset` filtering.
"""
from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ..config import H_ATOMIC_NUMBER, REPO_ROOT, resolve_path
from .graph import build_radius_graph
from .splits import load_split


def load_affinity_csv(path: Path) -> Dict[str, Tuple[float, str]]:
    """Read `data/MD/affinity.csv` into {pdb_id_upper: (neg_log_k, affinity_type)}.

    Tolerates extra columns and case-insensitive PDB IDs. Schema produced by
    `scripts/build_affinity_csv.py`:
        pdb_id,neg_log_k,affinity_type
    """
    out: Dict[str, Tuple[float, str]] = {}
    if not path.exists():
        return out
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
            if not math.isfinite(neg_log_k):
                continue
            kind = (row.get("affinity_type") or "Kd").strip() or "Kd"
            out[pid] = (neg_log_k, kind)
    return out


def _compute_adaptability(traj: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Per-atom flexibility: mean pairwise inter-frame distance.

    For each atom, average the Euclidean distance across all unordered pairs
    of frames (i, j) with i < j. Equivalently, the mean of the strict upper
    triangle of the per-atom (T, T) frame-distance matrix, with denominator
    T*(T-1)/2.

    This matches MISATO's preprocessed `feature_atoms_adaptability` exactly.
    Earlier versions of this function divided by T**2 instead of T*(T-1)/2,
    which silently scaled adaptability values by (T-1)/T (~1% low for T=100)
    and included T spurious zeros from the diagonal in the average. Fixed
    2026-05-08.

    Args:
        traj: (T, N, 3) trajectory coordinates.
        device: torch device. CUDA uses batched torch.cdist (~5-10x faster
                than numpy on typical N≈3k complexes).

    Returns:
        (N,) per-atom adaptability (numpy float32, units of Å).
    """
    T = traj.shape[0]
    if T < 2:
        # Single frame — adaptability is undefined; return zeros.
        return np.zeros(traj.shape[1], dtype=np.float32)

    if device is not None and device.type == "cuda":
        # (T, N, 3) -> (N, T, 3); torch.cdist batched: per-atom (T, T) distances.
        coords = torch.from_numpy(traj).to(device).transpose(0, 1)
        dists = torch.cdist(coords, coords)  # (N, T, T)
        # Upper triangle only — exclude diagonal (zeros) and lower triangle (dup).
        mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        # dists[:, mask] -> (N, T*(T-1)/2). Mean is over the unique frame pairs.
        return dists[:, mask].mean(dim=1).cpu().numpy().astype(np.float32)

    # Numpy fallback (CPU).
    coords = np.transpose(traj, (1, 0, 2)).astype(np.float32)
    diffs = coords[:, :, None, :] - coords[:, None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))  # (N, T, T)
    iu = np.triu_indices(T, k=1)  # upper triangle indices
    return dists[:, iu[0], iu[1]].mean(axis=1).astype(np.float32)


def process_complex(
    h5_path: Path,
    pdb_id: str,
    out_dir: Path,
    edge_cutoff: float = 4.5,
    keep_trajectory: bool = True,
    strip_hydrogens: bool = True,
    strip_solvent: bool = True,
    overwrite: bool = False,
    device: torch.device | None = None,
    affinity: Optional[Tuple[float, str]] = None,
) -> Dict[str, str]:
    """Process a single complex from MD.hdf5 → out_dir/{pdb_id}.pt.

    If `device` is CUDA, adaptability and graph construction run on GPU.

    Returns a status dict {pdb_id, status, error?}.
    """
    out_path = out_dir / f"{pdb_id}.pt"
    if out_path.exists() and not overwrite:
        return {"pdb_id": pdb_id, "status": "skipped"}

    try:
        with h5py.File(h5_path, "r") as f:
            if pdb_id not in f:
                return {"pdb_id": pdb_id, "status": "missing", "error": "not in H5"}
            grp = f[pdb_id]
            atoms_element = grp["atoms_element"][:]
            atoms_number = grp["atoms_number"][:]
            mol_idx = grp["molecules_begin_atom_index"][:]
            traj = grp["trajectory_coordinates"][:]            # (T, N, 3)
            energy = grp["frames_interaction_energy"][:]        # (T,)

        n_atoms_total = atoms_element.shape[0]
        keep_mask = np.ones(n_atoms_total, dtype=bool)
        if strip_solvent and len(mol_idx) >= 1:
            keep_mask[int(mol_idx[-1]):] = False
        if strip_hydrogens:
            keep_mask &= (atoms_number != H_ATOMIC_NUMBER)

        if keep_mask.sum() == 0:
            return {"pdb_id": pdb_id, "status": "empty", "error": "no atoms after filtering"}

        # Filter atoms.
        atoms_element_f = atoms_element[keep_mask].astype(np.int64)
        atoms_number_f = atoms_number[keep_mask].astype(np.int64)
        traj_f = traj[:, keep_mask, :].astype(np.float32)
        pos = traj_f[0]  # reference frame

        # Recompute mol_idx for the filtered array.
        # Original mol_idx is into the unfiltered array; rebuild by counting kept atoms
        # before each original boundary.
        cumkeep = np.cumsum(keep_mask)
        mol_idx_f = np.array(
            [
                int(cumkeep[mol_idx[i] - 1]) if mol_idx[i] > 0 else 0
                for i in range(len(mol_idx))
            ],
            dtype=np.int64,
        )
        # If solvent was stripped, the trailing boundary is now N (end of array).
        if strip_solvent:
            mol_idx_f = mol_idx_f[:2].tolist()
            mol_idx_f.append(int(keep_mask.sum()))
            mol_idx_f = np.array(mol_idx_f, dtype=np.int64)

        # Adaptability from filtered trajectory (GPU if device is CUDA).
        adaptability = _compute_adaptability(traj_f, device=device)

        # Build the reference-frame radius graph (GPU if device is CUDA).
        pos_t = torch.from_numpy(pos)
        if device is not None and device.type == "cuda":
            pos_t_gpu = pos_t.to(device)
            edge_index, edge_attr = build_radius_graph(pos_t_gpu, cutoff=edge_cutoff)
            edge_index = edge_index.cpu()
            edge_attr = edge_attr.cpu()
        else:
            edge_index, edge_attr = build_radius_graph(pos_t, cutoff=edge_cutoff)

        # Experimental affinity: NaN when the complex isn't in PDBbind. The
        # dataset class filters NaN-target rows when `target` is log_k or a
        # multitask variant that requires it.
        if affinity is not None:
            y_logk = float(affinity[0])
            affinity_type = str(affinity[1])
        else:
            y_logk = float("nan")
            affinity_type = "Unknown"

        record: Dict = {
            "pdb_id": pdb_id,
            "pos": pos_t,
            "z": torch.from_numpy(atoms_number_f),
            "element_idx": torch.from_numpy(atoms_element_f),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y_energy_per_frame": torch.from_numpy(energy.astype(np.float32)),
            "y_energy_mean": torch.tensor(float(energy.mean()), dtype=torch.float32),
            "y_logk": torch.tensor(y_logk, dtype=torch.float32),
            "affinity_type": affinity_type,
            "adaptability": torch.from_numpy(adaptability),
            "mol_idx": torch.from_numpy(mol_idx_f),
        }
        if keep_trajectory:
            record["pos_traj"] = torch.from_numpy(traj_f)

        out_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file then rename.
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(record, tmp)
        tmp.replace(out_path)
        return {"pdb_id": pdb_id, "status": "ok"}

    except Exception as e:  # narrow surfaces, but log the error not silently swallow
        return {"pdb_id": pdb_id, "status": "error", "error": f"{type(e).__name__}: {e}"}


def _worker(args):
    return process_complex(*args)


def annotate_affinity_in_place(
    processed_dir: Path,
    affinity: Dict[str, Tuple[float, str]],
) -> Dict[str, int]:
    """Inject `y_logk` and `affinity_type` into every existing .pt without
    re-doing the full preshard. Idempotent: re-running with the same CSV is a
    no-op on already-annotated files. ~1 minute for 16,972 complexes vs ~28
    minutes for a full re-preshard.

    Returns counts: {ok, missing, error}.
    """
    counts = {"labelled": 0, "missing": 0, "error": 0, "unchanged": 0}
    files = sorted(Path(processed_dir).glob("*.pt"))
    for f in tqdm(files, desc="annotating", unit="cplx"):
        try:
            rec = torch.load(f, weights_only=False)
            pid = (rec.get("pdb_id") or f.stem).upper()
            hit = affinity.get(pid)
            new_logk = float(hit[0]) if hit else float("nan")
            new_type = str(hit[1]) if hit else "Unknown"

            existing_logk = rec.get("y_logk")
            existing_type = rec.get("affinity_type")
            existing_logk_val = (
                float(existing_logk) if isinstance(existing_logk, torch.Tensor) else
                float(existing_logk) if existing_logk is not None else float("nan")
            )
            if (
                ((math.isnan(existing_logk_val) and math.isnan(new_logk)) or
                 existing_logk_val == new_logk)
                and existing_type == new_type
            ):
                counts["unchanged"] += 1
                continue

            rec["y_logk"] = torch.tensor(new_logk, dtype=torch.float32)
            rec["affinity_type"] = new_type
            tmp = f.with_suffix(".pt.tmp")
            torch.save(rec, tmp)
            tmp.replace(f)
            if hit:
                counts["labelled"] += 1
            else:
                counts["missing"] += 1
        except Exception as e:
            counts["error"] += 1
            print(f"[!] {f.name}: {type(e).__name__}: {e}")
    return counts


def _resolve_device(spec: str | None) -> torch.device:
    """Pick a torch device. 'auto' → cuda if available, else cpu."""
    if spec is None or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def preshard(
    h5_path: str | Path,
    out_dir: str | Path,
    pdb_ids: Optional[List[str]] = None,
    edge_cutoff: float = 4.5,
    keep_trajectory: bool = True,
    strip_hydrogens: bool = True,
    strip_solvent: bool = True,
    workers: int = 1,
    overwrite: bool = False,
    device: str | torch.device | None = "auto",
    affinity: Optional[Dict[str, Tuple[float, str]]] = None,
) -> Dict[str, int]:
    """Pre-shard MD.hdf5 into one .pt per complex. Returns a status summary.

    Multi-process workers force CPU-only (each worker would otherwise need its
    own CUDA context, which fights for GPU memory). For GPU acceleration, use
    workers=1 — single-process GPU is typically faster than multi-process CPU
    for this workload because the per-complex compute is small relative to I/O.
    """
    h5_path = resolve_path(h5_path)
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = _resolve_device(device) if isinstance(device, str) or device is None else device
    if workers > 1 and dev.type == "cuda":
        print(f"[preshard] workers={workers} with CUDA → falling back to CPU per-worker (multi-process GPU thrashes memory)")
        per_worker_device = torch.device("cpu")
    else:
        per_worker_device = dev

    if pdb_ids is None:
        with h5py.File(h5_path, "r") as f:
            pdb_ids = sorted(f.keys())

    aff_map = affinity or {}
    job_args = [
        (h5_path, pid, out_dir, edge_cutoff, keep_trajectory,
         strip_hydrogens, strip_solvent, overwrite, per_worker_device,
         aff_map.get(pid.upper()))
        for pid in pdb_ids
    ]

    print(f"[preshard] {len(pdb_ids)} complexes -> {out_dir}  device={per_worker_device}  workers={workers}")

    results: List[Dict] = []
    if workers <= 1:
        for a in tqdm(job_args, desc="presharding", unit="cplx"):
            results.append(_worker(a))
    else:
        with mp.Pool(workers) as pool:
            for r in tqdm(pool.imap_unordered(_worker, job_args), total=len(job_args), desc="presharding", unit="cplx"):
                results.append(r)

    # Summary.
    counts: Dict[str, int] = {}
    errors: List[Dict] = []
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "error":
            errors.append(r)
    if errors:
        print(f"\n[!] {len(errors)} errors. First 5:")
        for e in errors[:5]:
            print(f"    {e['pdb_id']}: {e.get('error', '?')}")
    return counts


def _cli():
    p = argparse.ArgumentParser(description="Pre-shard MD.hdf5 into per-complex .pt files")
    p.add_argument("--h5", default="data/MD/h5_files/MD.hdf5", help="Path to MD.hdf5")
    p.add_argument("--out", default="data/processed", help="Output directory for .pt files")
    p.add_argument("--split", default=None, help="Optional split file to limit which complexes to process")
    p.add_argument("--cutoff", type=float, default=4.5, help="Radius graph cutoff in Å")
    p.add_argument("--no-traj", action="store_true", help="Do not store full trajectory (only frame 0)")
    p.add_argument("--keep-h", action="store_true", help="Do not strip hydrogen atoms")
    p.add_argument("--keep-solvent", action="store_true", help="Do not strip solvent atoms")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (auto-falls-back to CPU if >1)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt files")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu (auto picks cuda if available)")
    p.add_argument("--affinity-csv", default="data/MD/affinity.csv",
                   help="Optional PDBbind affinity CSV (built by scripts/build_affinity_csv.py). "
                        "If file does not exist, all .pt files get NaN y_logk.")
    p.add_argument("--annotate-only", action="store_true",
                   help="Skip the H5 walk and just inject y_logk + affinity_type into "
                        "the existing .pt files in --out using --affinity-csv. ~1 min for 16,972 complexes.")
    args = p.parse_args()

    aff_path = resolve_path(args.affinity_csv) if args.affinity_csv else None
    affinity = load_affinity_csv(aff_path) if aff_path else {}
    if aff_path and not aff_path.exists():
        print(f"[preshard] no affinity CSV at {aff_path} — y_logk will be NaN for every complex.")
    elif affinity:
        print(f"[preshard] affinity CSV: {len(affinity):,} records loaded from {aff_path.relative_to(REPO_ROOT) if REPO_ROOT in aff_path.parents else aff_path}")

    if args.annotate_only:
        if not affinity:
            raise SystemExit("[preshard] --annotate-only requires a non-empty --affinity-csv.")
        out_dir = resolve_path(args.out)
        if not out_dir.exists():
            raise SystemExit(f"[preshard] --annotate-only target dir does not exist: {out_dir}")
        summary = annotate_affinity_in_place(out_dir, affinity)
        print("\nSummary (annotate-only):")
        for k, v in summary.items():
            print(f"  {k:10s} {v}")
        return

    pdb_ids = load_split(resolve_path(args.split)) if args.split else None
    summary = preshard(
        h5_path=args.h5,
        out_dir=args.out,
        pdb_ids=pdb_ids,
        edge_cutoff=args.cutoff,
        keep_trajectory=not args.no_traj,
        strip_hydrogens=not args.keep_h,
        strip_solvent=not args.keep_solvent,
        workers=args.workers,
        overwrite=args.overwrite,
        device=args.device,
        affinity=affinity,
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:10s} {v}")


if __name__ == "__main__":
    _cli()
