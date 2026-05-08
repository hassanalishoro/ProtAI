"""Fork-safe dataset loading pre-sharded .pt files with multi-frame support.

Why a new dataset class:
  * The old ProtDataset opens the H5 in __init__ → unsafe with DataLoader workers.
  * It used pandas as an intermediate → 5-10x slower than direct tensors.
  * The legacy code used a `Data(pos_arg, ids=...)` kwarg that PyG silently dropped.

This dataset reads the per-complex .pt files produced by preshard.py and
attaches the PDB id correctly via `Data(pdb_id=...)` (PyG attaches arbitrary
keys when passed as kwargs after the standard ones).

Multi-frame strategy options:
  * frame_zero  : always use the reference frame stored in `pos`
  * random      : sample one trajectory frame uniformly per __getitem__
  * mean        : average over all frames (deterministic, single-graph output)
  * attention_pool: store all frames as a stacked tensor; the model is responsible
                    for pooling (handled in the LightningModule via batched forward)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph import build_radius_graph
from .splits import load_split


SUPPORTED_TARGETS = {"binding_affinity", "adaptability", "multitask"}
SUPPORTED_FRAME_STRATEGIES = {"frame_zero", "random", "mean", "attention_pool"}


class PreShardedDataset(Dataset):
    """Loads per-complex .pt files emitted by `protai.data.preshard`.

    Each __getitem__ returns a torch_geometric.data.Data with:
        x          : node features (one-hot element by default; configurable)
        pos        : (N, 3) coords for the chosen frame
        edge_index : graph edges (recomputed if frame != reference, else cached)
        edge_attr  : inverse distances aligned with edge_index
        z          : atomic numbers (for SchNet-style models)
        pdb_id     : str
        y          : prediction target (shape depends on `target`)

    Args:
        processed_dir: directory of {pdb_id}.pt files.
        split_file:    text file with one pdb_id per line. Filters which files load.
        target:        "binding_affinity" (graph-level scalar, mean energy),
                       "adaptability" (per-atom flexibility),
                       "multitask" (both, model must handle a tuple).
        frame_strategy: see module docstring.
        edge_cutoff:   only used when re-computing graph for non-reference frames.
        node_feature:  "one_hot_element" (uses element_idx, default) or "atomic_number"
                       (uses raw z; recommended for SchNet).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_file: Optional[str | Path] = None,
        target: str = "binding_affinity",
        frame_strategy: str = "frame_zero",
        edge_cutoff: float = 4.5,
        node_feature: str = "one_hot_element",
    ):
        if target not in SUPPORTED_TARGETS:
            raise ValueError(f"target must be one of {SUPPORTED_TARGETS}, got {target!r}")
        if frame_strategy not in SUPPORTED_FRAME_STRATEGIES:
            raise ValueError(f"frame_strategy must be one of {SUPPORTED_FRAME_STRATEGIES}, got {frame_strategy!r}")

        self.processed_dir = Path(processed_dir)
        self.target = target
        self.frame_strategy = frame_strategy
        self.edge_cutoff = edge_cutoff
        self.node_feature = node_feature

        # Build the file index.
        if split_file is not None:
            ids = load_split(split_file)
            files = [self.processed_dir / f"{i}.pt" for i in ids]
            self.files = [f for f in files if f.exists()]
            missing = len(files) - len(self.files)
            if missing:
                print(f"[PreShardedDataset] {missing} ids in split missing from {self.processed_dir}")
        else:
            self.files = sorted(self.processed_dir.glob("*.pt"))

        if not self.files:
            raise FileNotFoundError(
                f"No .pt files found in {self.processed_dir} matching split. "
                "Did you run `python -m protai.data.preshard` first?"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        record = torch.load(self.files[idx], weights_only=False)
        return self._record_to_data(record)

    # ------------------------------------------------------------------ helpers

    def _record_to_data(self, rec: dict) -> Data:
        # Pick coordinates based on frame strategy.
        if self.frame_strategy == "frame_zero" or "pos_traj" not in rec:
            pos = rec["pos"]
            edge_index = rec["edge_index"]
            edge_attr = rec["edge_attr"]
        elif self.frame_strategy == "random":
            traj = rec["pos_traj"]  # (T, N, 3)
            t = random.randint(0, traj.shape[0] - 1)
            pos = traj[t]
            edge_index, edge_attr = build_radius_graph(pos, cutoff=self.edge_cutoff)
        elif self.frame_strategy == "mean":
            pos = rec["pos_traj"].mean(dim=0)
            edge_index, edge_attr = build_radius_graph(pos, cutoff=self.edge_cutoff)
        elif self.frame_strategy == "attention_pool":
            # Use reference frame for graph topology; model receives full traj.
            pos = rec["pos"]
            edge_index = rec["edge_index"]
            edge_attr = rec["edge_attr"]
        else:  # pragma: no cover
            raise RuntimeError(f"unhandled frame_strategy {self.frame_strategy!r}")

        # Node features.
        if self.node_feature == "one_hot_element":
            from .graph import one_hot_elements
            x = one_hot_elements(rec["element_idx"])
        elif self.node_feature == "atomic_number":
            x = rec["z"].view(-1, 1).float()
        else:
            raise ValueError(f"Unknown node_feature {self.node_feature!r}")

        # Target.
        if self.target == "binding_affinity":
            y = rec["y_energy_mean"].view(1)
        elif self.target == "adaptability":
            y = rec["adaptability"]
        elif self.target == "multitask":
            y = {"energy": rec["y_energy_mean"].view(1), "adaptability": rec["adaptability"]}
        else:  # pragma: no cover
            raise RuntimeError(f"unhandled target {self.target!r}")

        kwargs = dict(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            z=rec["z"],
            pdb_id=rec["pdb_id"],
        )
        # Multitask is a dict; PyG Data only handles tensor / list-of-tensor / scalar.
        # Use separate keys for clarity.
        if self.target == "multitask":
            kwargs["y_energy"] = y["energy"]
            kwargs["y_adapt"] = y["adaptability"]
        else:
            kwargs["y"] = y

        if self.frame_strategy == "attention_pool":
            kwargs["pos_traj"] = rec["pos_traj"]  # (T, N, 3)
            kwargs["y_energy_per_frame"] = rec["y_energy_per_frame"]

        return Data(**kwargs)
