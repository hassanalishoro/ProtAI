"""Model + dataset service layer.

Separated from Flask routes so it's testable and reusable (e.g. from a CLI,
notebook, or a different web framework). The routes module imports from here.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch

from ..config import REPO_ROOT, resolve_path
from ..data.graph import build_radius_graph, one_hot_elements
from ..training.lit_module import ProtAILitModule


# Run-dir name prefixes that indicate junk/sanity training and should never
# back the live demo. Edit here if you adopt new naming conventions.
_SKIP_PREFIXES = ("tiny_", "sanity_", "debug_", "test_", "smoke_")

# Preferred run names in priority order — first match wins. Lets the demo
# auto-pick the headline checkpoint without needing PROTAI_MODEL_PATH set.
_PREFERRED_RUNS = (
    "random_frame_random",   # headline: trajectory-aware on random split
    "random_frame",          # legacy alias
    "headline",              # explicit headline alias if present
    "schnet_aff_random",     # next-best architecture cell
)


def find_latest_best_checkpoint(logs_dir: Path) -> Optional[Path]:
    """Pick the right `best.ckpt` for the demo.

    Priority order:
        1. Any run whose name appears in `_PREFERRED_RUNS` (in that order).
        2. Otherwise, the most recently modified run dir, after filtering
           out junk names listed in `_SKIP_PREFIXES`.

    Falls back to legacy `best_model.pt` when `best.ckpt` is missing.
    Set the env var `PROTAI_MODEL_PATH` to override entirely.
    """
    if not logs_dir.exists():
        return None

    candidates: list[Path] = []
    for run in logs_dir.iterdir():
        if not run.is_dir():
            continue
        if any(run.name.startswith(p) for p in _SKIP_PREFIXES):
            continue
        for name in ("best.ckpt", "best_model.pt"):
            ckpt = run / name
            if ckpt.exists():
                candidates.append(ckpt)
                break

    if not candidates:
        return None

    by_name = {c.parent.name: c for c in candidates}
    for preferred in _PREFERRED_RUNS:
        if preferred in by_name:
            return by_name[preferred]

    # Fall back to most recently modified checkpoint.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


class ProtAIService:
    """Encapsulates the trained model + the H5 dataset for inference.

    Construct once at server startup; reuse for every request.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        # Resolve model checkpoint.
        env_model = os.environ.get("PROTAI_MODEL_PATH")
        if env_model:
            self.model_path: Optional[Path] = Path(env_model)
        elif model_path is not None:
            self.model_path = Path(model_path)
        else:
            self.model_path = find_latest_best_checkpoint(REPO_ROOT / "runs")

        # Resolve dataset path.
        env_data = os.environ.get("PROTAI_DATA_PATH")
        self.data_path = Path(env_data) if env_data else (
            data_path if data_path else REPO_ROOT / "data" / "MD" / "h5_files" / "MD.hdf5"
        )

        # Pick inference device (env var > arg > auto).
        env_dev = os.environ.get("PROTAI_DEVICE")
        spec = env_dev or device or "auto"
        if spec == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(spec)

        self.module: Optional[ProtAILitModule] = None
        self._h5: Optional[h5py.File] = None
        self._load_model()
        self._load_data()

    # ------------------------------------------------------------------ load

    def _load_model(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            print(f"[service] No checkpoint found (set PROTAI_MODEL_PATH or train first).")
            return
        try:
            if self.model_path.suffix == ".ckpt":
                self.module = ProtAILitModule.load_from_checkpoint(
                    str(self.model_path), map_location=str(self.device),
                )
            else:
                print(f"[service] Legacy .pt checkpoint detected — limited functionality.")
                self.module = None
                return
            self.module.to(self.device).eval()
            n_params = sum(p.numel() for p in self.module.parameters())
            print(f"[service] Loaded {self.model_path.relative_to(REPO_ROOT)} on {self.device} ({n_params:,} params)")
        except Exception as e:
            print(f"[service] Failed to load checkpoint: {e}")
            self.module = None

    def _load_data(self) -> None:
        if not self.data_path.exists():
            print(f"[service] Dataset not found at {self.data_path}")
            return
        self._h5 = h5py.File(self.data_path, "r")
        print(f"[service] Loaded {self.data_path.relative_to(REPO_ROOT)} ({len(self._h5)} structures)")

    # ------------------------------------------------------------------ status

    @property
    def model_loaded(self) -> bool:
        return self.module is not None

    @property
    def data_loaded(self) -> bool:
        return self._h5 is not None

    def list_structures(self) -> list[str]:
        if not self._h5:
            return []
        return sorted(self._h5.keys())

    def has_structure(self, pdb_id: str) -> bool:
        return bool(self._h5 and pdb_id in self._h5)

    # ------------------------------------------------------------------ public API

    def get_structure_info(self, pdb_id: str) -> Dict[str, Any]:
        """Return atom coordinates, element labels, true energy, adaptability."""
        if not self.has_structure(pdb_id):
            raise KeyError(pdb_id)
        grp = self._h5[pdb_id]

        # Use frame 0 for visualization. Strip H + solvent for sane atom count.
        atoms_element = grp["atoms_element"][:]
        atoms_number = grp["atoms_number"][:]
        mol_idx = grp["molecules_begin_atom_index"][:]
        coords = grp["trajectory_coordinates"][0]
        energy = float(grp["frames_interaction_energy"][:].mean())

        keep = np.ones(len(atoms_element), dtype=bool)
        if len(mol_idx) >= 1:
            keep[int(mol_idx[-1]):] = False
        keep &= atoms_number != 1

        # Adaptability from full trajectory variance.
        traj = grp["trajectory_coordinates"][:][:, keep, :]
        adapt = self._compute_adaptability(traj)

        protein_end = int(mol_idx[1]) if len(mol_idx) > 1 else len(coords)
        n_protein = int((keep[:protein_end]).sum())
        n_total = int(keep.sum())

        return {
            "pdb_id": pdb_id,
            "num_atoms": n_total,
            "num_protein_atoms": n_protein,
            "num_ligand_atoms": n_total - n_protein,
            "coordinates": coords[keep].tolist(),
            "elements": atoms_element[keep].tolist(),
            "atomic_numbers": atoms_number[keep].tolist(),
            "adaptability": adapt.tolist(),
            "true_affinity": energy,
        }

    def predict_affinity(self, pdb_id: str) -> Dict[str, Any]:
        """Run model inference for one structure. Returns predicted + true affinity."""
        if not self.has_structure(pdb_id):
            raise KeyError(pdb_id)
        grp = self._h5[pdb_id]

        atoms_element = grp["atoms_element"][:]
        atoms_number = grp["atoms_number"][:]
        mol_idx = grp["molecules_begin_atom_index"][:]
        coords = grp["trajectory_coordinates"][0]
        true_energy = float(grp["frames_interaction_energy"][:].mean())

        # Strip H + solvent.
        keep = np.ones(len(atoms_element), dtype=bool)
        if len(mol_idx) >= 1:
            keep[int(mol_idx[-1]):] = False
        keep &= atoms_number != 1

        if not self.model_loaded:
            return {"pdb_id": pdb_id, "predicted_affinity": None,
                    "true_affinity": round(true_energy, 4),
                    "model_type": "no model loaded"}

        # Build inference batch directly on the inference device (GPU when available).
        from torch_geometric.data import Batch, Data
        dev = self.device
        pos = torch.tensor(coords[keep], dtype=torch.float32, device=dev)
        z = torch.tensor(atoms_number[keep], dtype=torch.long, device=dev)
        elem = torch.tensor(atoms_element[keep], dtype=torch.long, device=dev)
        edge_index, edge_attr = build_radius_graph(pos, cutoff=self.module.cfg.data.edge_cutoff)

        if self.module.cfg.model.name == "schnet":
            x = z.view(-1, 1).float()
        else:
            x = one_hot_elements(elem)

        data = Data(x=x, pos=pos, z=z, edge_index=edge_index, edge_attr=edge_attr)
        batch = Batch.from_data_list([data])

        with torch.no_grad():
            pred = self.module(batch)

        # IMPORTANT: model output is in NORMALIZED z-score space for any
        # target trained with target normalization. Denormalize before
        # exposing through the API, otherwise the demo shows z-scores while
        # the "true" reference is in kcal/mol — a single 1A1B request would
        # display "predicted -0.23" against "true -115" and look broken.
        target = self.module.cfg.model.target
        if isinstance(pred, dict):
            # Multitask: pick the head whose label we're showing as the
            # primary prediction (energy for legacy multitask, logk for new).
            head_name = "logk" if target == "multitask_logk_energy" else "energy"
            raw = pred[head_name]
            pred_val = float(self.module._denormalize(raw).item())
        elif target == "adaptability":
            # Per-atom raw target; no normalization applied during training.
            pred_val = float(pred.flatten()[0].item())
        else:
            # Single-scalar normalized targets: binding_affinity, log_k.
            pred_val = float(self.module._denormalize(pred).flatten()[0].item())

        return {
            "pdb_id": pdb_id,
            "predicted_affinity": round(pred_val, 4),
            "true_affinity": round(true_energy, 4),
            "error": round(abs(pred_val - true_energy), 4),
            "model_type": f"{self.module.cfg.model.name} ({self.module.cfg.model.target})",
            "num_atoms": int(keep.sum()),
        }

    def get_frame(self, pdb_id: str, frame: int) -> Dict[str, Any]:
        """Return atomic coordinates at frame `frame` of the MD trajectory.

        Used by the frontend's frame scrubber to animate trajectories
        without re-sending the whole structure record.
        """
        if not self.has_structure(pdb_id):
            raise KeyError(pdb_id)
        grp = self._h5[pdb_id]

        traj = grp["trajectory_coordinates"]
        total = int(traj.shape[0])
        if frame < 0 or frame >= total:
            raise ValueError(f"frame {frame} out of range [0, {total})")

        atoms_number = grp["atoms_number"][:]
        mol_idx = grp["molecules_begin_atom_index"][:]
        keep = np.ones(len(atoms_number), dtype=bool)
        if len(mol_idx) >= 1:
            keep[int(mol_idx[-1]):] = False
        keep &= atoms_number != 1

        coords = traj[frame][:][keep]
        energy = float(grp["frames_interaction_energy"][frame])
        return {
            "pdb_id": pdb_id,
            "frame": frame,
            "total_frames": total,
            "coordinates": coords.tolist(),
            "energy": energy,
        }

    def analyze_pocket(self, pdb_id: str, cutoff: float = 4.5) -> Dict[str, Any]:
        """Identify protein atoms within `cutoff` Å of any ligand atom."""
        if not self.has_structure(pdb_id):
            raise KeyError(pdb_id)
        grp = self._h5[pdb_id]
        coords = grp["trajectory_coordinates"][0]
        mol_idx = grp["molecules_begin_atom_index"][:]
        if len(mol_idx) < 2:
            raise ValueError(f"{pdb_id}: no ligand boundary in molecules_begin_atom_index")
        protein_end = int(mol_idx[1])
        protein = coords[:protein_end]
        ligand = coords[protein_end:int(mol_idx[-1])] if len(mol_idx) > 2 else coords[protein_end:]
        if len(ligand) == 0:
            raise ValueError(f"{pdb_id}: empty ligand")
        from scipy.spatial.distance import cdist
        d = cdist(protein, ligand)
        protein_pocket = np.where(d.min(axis=1) < cutoff)[0].tolist()
        ligand_contacts = (np.where(d.min(axis=0) < cutoff)[0] + protein_end).tolist()
        return {
            "pdb_id": pdb_id, "cutoff_angs": cutoff,
            "pocket_size": len(protein_pocket),
            "protein_pocket_indices": protein_pocket,
            "ligand_indices": ligand_contacts,
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compute_adaptability(traj: np.ndarray) -> np.ndarray:
        """Per-atom mean pairwise inter-frame distance, upper-triangle only.

        Matches the formula used by `protai/data/preshard.py` (fixed
        2026-05-08): for each atom, average over the unique frame-pair
        distances (T*(T-1)/2 pairs) rather than the full T×T matrix. The
        full-matrix version double-counts each pair and dilutes the value
        with the T zero-distance diagonal entries.
        """
        # traj: (T, N, 3). Transpose to (N, T, 3) so each atom owns a (T, 3) block.
        coords = np.transpose(traj, (1, 0, 2)).astype(np.float32)
        n_atoms, t = coords.shape[0], coords.shape[1]
        if t < 2:
            return np.zeros(n_atoms, dtype=np.float32)
        # Upper-triangular pair indices (i < j) — same convention as preshard.
        iu, ju = np.triu_indices(t, k=1)
        # (N, P, 3) where P = T*(T-1)/2; squared distance then sqrt + mean.
        diffs = coords[:, iu, :] - coords[:, ju, :]
        dist = np.sqrt((diffs ** 2).sum(axis=-1))
        return dist.mean(axis=1)


# Singleton accessor — lets routes.py grab the same instance on every request.
@lru_cache(maxsize=1)
def get_service() -> ProtAIService:
    return ProtAIService()
