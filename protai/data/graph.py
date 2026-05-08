"""Vectorized graph construction for protein-ligand complexes.

Replaces the old `src/data/components/graph.py` which used pandas, Python list
comprehensions, and a bare `except:`. This version:
  * Builds the radius graph directly from a (N, 3) tensor — no pandas
  * Vectorizes edge-distance computation with `torch.norm`
  * Vectorizes one-hot encoding via `F.one_hot`
  * Surfaces errors instead of swallowing them

Implementation strategy: prefers `torch_cluster.radius_graph` (native CUDA,
2-4x faster for sparse graphs) when available, falls back to torch.cdist on
GPU or cKDTree on CPU otherwise. The fallback exists because torch_cluster
ships C++ extensions that fail to load on Windows when there's any
torch/CUDA version skew — extremely common in practice. On Linux pods with
correctly-installed PyG wheels the fast path activates automatically.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from ..config import N_ELEMENT_CLASSES


# Detect torch_cluster at import time. ImportError covers "not installed";
# OSError covers "installed but DLL/so failed to load" (the WinError 127
# case we hit constantly on Windows due to torch/CUDA version mismatches).
try:
    from torch_cluster import radius_graph as _tc_radius_graph  # type: ignore
    _HAS_TORCH_CLUSTER = True
except (ImportError, OSError):
    _tc_radius_graph = None  # type: ignore
    _HAS_TORCH_CLUSTER = False


def build_radius_graph(
    pos: torch.Tensor,
    cutoff: float = 4.5,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an undirected radius graph with inverse-distance edge weights.

    Device-aware: if `pos` is on CUDA, runs all-pairs distance on GPU (no
    cKDTree round-trip to CPU). For typical N<10k complexes this is faster
    than cKDTree on CPU.

    Args:
        pos: (N, 3) atomic coordinates as a float tensor.
        cutoff: Maximum Euclidean distance (in Å) for edge inclusion.
        eps: Numerical floor for edge-weight denominator (avoid div-by-zero).

    Returns:
        edge_index: (2, E) int64 on the same device as `pos`.
        edge_attr:  (E,) float32 on the same device as `pos`.
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`pos` must be (N, 3); got {tuple(pos.shape)}")

    device = pos.device
    n = pos.shape[0]
    if n < 2:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )

    # Fast path: native torch_cluster (CUDA + CPU). Activates on Linux pods
    # with the PyG extension wheels installed; gracefully unavailable on
    # Windows where the C++ extension fails to load.
    if _HAS_TORCH_CLUSTER:
        return _build_torch_cluster(pos, cutoff, eps)

    # Fallback paths (kept working for Windows / no-extensions environments).
    if pos.is_cuda:
        return _build_gpu(pos, cutoff, eps)
    return _build_cpu(pos, cutoff, eps)


def _build_torch_cluster(
    pos: torch.Tensor, cutoff: float, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Native CUDA / CPU radius search via torch_cluster.

    `radius_graph(loop=False)` returns a directed edge tensor (2, E) that
    already includes both directions for each radius pair (the radius
    relation is symmetric), matching the convention of our cdist/cKDTree
    paths exactly. `max_num_neighbors=128` is safely above what protein
    complexes produce at typical 4.5 Å cutoffs (median ~10-12 neighbors).
    """
    edge_index = _tc_radius_graph(  # type: ignore[misc]
        pos, r=cutoff, loop=False, max_num_neighbors=128
    )
    if edge_index.numel() == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=pos.device),
            torch.zeros((0,), dtype=torch.float32, device=pos.device),
        )
    src, dst = edge_index[0], edge_index[1]
    edge_dists = (pos[src] - pos[dst]).norm(dim=1)
    edge_attr = (1.0 / (edge_dists + eps)).to(torch.float32)
    return edge_index, edge_attr


def _build_gpu(
    pos: torch.Tensor, cutoff: float, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """All-pairs on GPU. O(N^2) memory but fine for N < ~15k."""
    n = pos.shape[0]
    # cdist is autograd-friendly and uses cuBLAS.
    dists = torch.cdist(pos, pos)  # (N, N)
    # Upper triangle, within cutoff, and not self.
    mask = (dists < cutoff) & (dists > 0)
    mask = torch.triu(mask, diagonal=1)
    pairs = mask.nonzero(as_tuple=False)  # (P, 2) on device

    if pairs.numel() == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=pos.device),
            torch.zeros((0,), dtype=torch.float32, device=pos.device),
        )

    edge_index = torch.cat([pairs.T, pairs.flip(1).T], dim=1).contiguous()
    src, dst = edge_index[0], edge_index[1]
    edge_dists = (pos[src] - pos[dst]).norm(dim=1)
    edge_attr = (1.0 / (edge_dists + eps)).to(torch.float32)
    return edge_index, edge_attr


def _build_cpu(
    pos: torch.Tensor, cutoff: float, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """cKDTree on CPU — fast for very large graphs (N > 15k) and no CUDA needed."""
    tree = cKDTree(pos.detach().numpy())
    pairs = np.fromiter(
        (idx for pair in tree.query_pairs(r=cutoff) for idx in pair),
        dtype=np.int64,
    ).reshape(-1, 2)

    if pairs.size == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float32),
        )

    pairs_t = torch.from_numpy(pairs)
    edge_index = torch.cat([pairs_t.T, pairs_t.flip(1).T], dim=1).contiguous()
    src, dst = edge_index[0], edge_index[1]
    dists = torch.norm(pos[src] - pos[dst], dim=1)
    edge_attr = (1.0 / (dists + eps)).to(torch.float32)
    return edge_index, edge_attr


def one_hot_elements(element_idx: torch.Tensor, n_classes: int = N_ELEMENT_CLASSES) -> torch.Tensor:
    """Vectorized one-hot encoding for MISATO element indices (1-based).

    MISATO encodes elements as 1..10 (H..I) in `atoms_element`. Unknown values
    fall into the last bin. This replaces the old per-atom Python loop.

    Args:
        element_idx: (N,) integer tensor with values typically in [1, 10].
        n_classes: total number of bins (default 11 = 10 known + 1 unknown).

    Returns:
        (N, n_classes) float tensor.
    """
    idx = element_idx.long() - 1  # 1-based -> 0-based
    # Clamp out-of-range indices to the last bin (UNK).
    idx = torch.where(
        (idx >= 0) & (idx < n_classes - 1),
        idx,
        torch.full_like(idx, n_classes - 1),
    )
    return F.one_hot(idx, num_classes=n_classes).to(torch.float32)
