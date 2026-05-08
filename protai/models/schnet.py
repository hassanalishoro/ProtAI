"""SchNet wrapper — E(3)-equivariant baseline using PyG's built-in SchNet.

Why SchNet:
  * Continuous-filter convolutions on atomic distances → rotation/translation
    invariant by construction.
  * Standard, well-cited reference equivariant architecture.
  * Available as `torch_geometric.nn.models.SchNet` — minimal code surface here.

Note: PyG's SchNet returns a pre-pooled scalar per graph by default. We expose
`return_node_embeddings=True` to grab per-atom features so we can plug in the
same heads used by GNN_MD (per-atom, graph-level, multitask).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SchNet

from ..data.graph import build_radius_graph
from .heads import build_head


class SchNetModel(nn.Module):
    """Per-atom SchNet trunk + a configurable head on top.

    Args:
        hidden_dim: width of SchNet feature channels.
        num_layers: number of interaction blocks.
        num_filters: number of continuous filters.
        num_gaussians: RBF basis size for distance expansion.
        cutoff: radius (Å) for the dynamic neighbor graph.
        target: head type — "binding_affinity" | "adaptability" | "multitask".
        graph_pool: pooling for graph-level head.
        dropout: head dropout.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_filters: int = 128,
        num_gaussians: int = 50,
        cutoff: float = 4.5,
        target: str = "binding_affinity",
        graph_pool: str = "mean",
        dropout: float = 0.1,
        **_unused,  # tolerate extra config keys
    ):
        super().__init__()
        # Use PyG SchNet for the message-passing trunk only.
        self.trunk = SchNet(
            hidden_channels=hidden_dim,
            num_filters=num_filters,
            num_interactions=num_layers,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        # Replace the trunk's atom-property head with identity so we get
        # per-atom embeddings directly.
        self.trunk.lin1 = nn.Identity()
        self.trunk.act = nn.Identity()
        self.trunk.lin2 = nn.Identity()

        self.cutoff = cutoff
        self.head = build_head(target=target, hidden_dim=hidden_dim, pool=graph_pool, dropout=dropout)
        self.target = target

    def forward(self, data):
        """SchNet wants z (atomic numbers), pos, batch — builds its own neighbors.

        We bypass `SchNet.forward` (which pools and produces a scalar) and walk
        the trunk manually so we expose node embeddings to our pluggable head.

        Uses `data.edge_index` if present (built by `protai.data.graph` via cKDTree
        on CPU or torch.cdist on GPU, avoiding the torch-cluster dependency that
        breaks on Windows due to PyTorch/CUDA version mismatches). Falls back to
        building edges per-graph in the batch otherwise.
        """
        z = data.z.long()
        pos = data.pos
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(z.size(0), dtype=torch.long, device=z.device)

        if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
            edge_index = data.edge_index
        else:
            # Per-graph fallback (rare path; preshard always populates edges).
            chunks = []
            for g in range(int(batch.max().item()) + 1):
                mask = batch == g
                idx = mask.nonzero(as_tuple=False).flatten()
                ei, _ = build_radius_graph(pos[mask], cutoff=self.cutoff)
                chunks.append(idx[ei])  # remap local -> global
            edge_index = torch.cat(chunks, dim=1) if chunks else \
                         torch.zeros((2, 0), dtype=torch.long, device=pos.device)

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.trunk.distance_expansion(edge_weight)

        h = self.trunk.embedding(z)
        for interaction in self.trunk.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # h is (N, hidden_dim) — pass through the configurable head.
        return self.head(h, batch)
