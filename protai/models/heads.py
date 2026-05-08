"""Pluggable prediction heads.

Decouples "produce per-atom embeddings" from "predict the final scalar/vector":
this lets GNN_MD and SchNet share heads, and lets us swap targets (per-atom
adaptability vs graph-level binding affinity) by changing the head, not the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, GlobalAttention


class PerAtomHead(nn.Module):
    """Maps node embeddings (N, H) → per-atom scalar (N,).

    Used for per-atom regression targets (e.g. flexibility / adaptability).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # (N,)


class GraphLevelHead(nn.Module):
    """Pools node embeddings to a graph-level vector, then maps → scalar.

    Used for graph-level regression (e.g. binding affinity per complex).

    Args:
        hidden_dim: input feature dim.
        pool: "mean" | "sum" | "attention".
        dropout: applied between the pooled vector and the output projection.
    """

    def __init__(self, hidden_dim: int, pool: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.pool_kind = pool
        if pool == "attention":
            gate_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.SiLU(),
                                    nn.Linear(hidden_dim // 2, 1))
            self.attention_pool = GlobalAttention(gate_nn=gate_nn)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool_kind == "mean":
            g = global_mean_pool(x, batch)
        elif self.pool_kind == "sum":
            g = global_add_pool(x, batch)
        elif self.pool_kind == "attention":
            g = self.attention_pool(x, batch)
        else:
            raise ValueError(f"Unknown pool {self.pool_kind!r}")
        return self.mlp(g).squeeze(-1)  # (B,)


class MultiTaskHead(nn.Module):
    """Two heads sharing the trunk: graph-level energy + per-atom adaptability."""

    def __init__(self, hidden_dim: int, pool: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.energy_head = GraphLevelHead(hidden_dim, pool=pool, dropout=dropout)
        self.adaptability_head = PerAtomHead(hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> dict:
        return {
            "energy": self.energy_head(x, batch),         # (B,)
            "adaptability": self.adaptability_head(x),    # (N,)
        }


def build_head(target: str, hidden_dim: int, pool: str = "mean", dropout: float = 0.1) -> nn.Module:
    """Factory used by models to attach the correct head from config."""
    if target == "binding_affinity":
        return GraphLevelHead(hidden_dim, pool=pool, dropout=dropout)
    if target == "adaptability":
        return PerAtomHead(hidden_dim, dropout=dropout)
    if target == "multitask":
        return MultiTaskHead(hidden_dim, pool=pool, dropout=dropout)
    raise ValueError(f"Unknown target {target!r}")
