"""Cleaned GNN_MD — the message-passing + attention baseline from the FYP1 report.

Differences from the legacy `examples/MDmodel.py`:
  * BatchNorm → LayerNorm (BatchNorm is theoretically wrong on variable-size
    graphs and silently desynchronizes across DDP ranks without SyncBatchNorm).
  * Configurable depth / hidden_dim instead of hardcoded values.
  * Outputs node embeddings; the prediction head is attached separately
    (so the same trunk drives binding-affinity, adaptability, or multitask).
  * SiLU activations (smoother gradients than ReLU; standard in modern GNNs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, NNConv

from .heads import build_head


class GNN_MD(nn.Module):
    """Hybrid GNN: NNConv → GATv2 ×2 → GCN ×(num_layers-3) → head.

    Edge attribute is a 1-D scalar (inverse distance from `build_radius_graph`).

    Args:
        num_features: input node-feature dim (e.g. 11 for one-hot element).
        hidden_dim: width of all internal layers.
        num_layers: total message-passing layers (>= 3).
        attention_heads: heads in GATv2 layers.
        target: which head to attach ("binding_affinity" | "adaptability" | "multitask").
        graph_pool: "mean" | "sum" | "attention" — only used when target is graph-level.
        dropout: applied inside the head.
    """

    def __init__(
        self,
        num_features: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 5,
        attention_heads: int = 4,
        target: str = "binding_affinity",
        graph_pool: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 3:
            raise ValueError("num_layers must be >= 3 (one NNConv + two GATv2 minimum)")

        # 1. NNConv — uses edge feature to modulate the message linearly.
        edge_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, num_features * hidden_dim)
        )
        self.conv1 = NNConv(num_features, hidden_dim, nn=edge_mlp, aggr="mean")
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 2-3. GATv2 attention layers.
        h2 = hidden_dim * 2
        self.conv2 = GATv2Conv(hidden_dim, h2 // attention_heads, heads=attention_heads, edge_dim=1)
        self.norm2 = nn.LayerNorm(h2)

        h3 = hidden_dim * 4
        self.conv3 = GATv2Conv(h2, h3 // attention_heads, heads=attention_heads, edge_dim=1)
        self.norm3 = nn.LayerNorm(h3)

        # 4..num_layers. GCN layers (use edge weight scalar).
        gcn_layers = []
        gcn_norms = []
        cur = h3
        for _ in range(num_layers - 3):
            gcn_layers.append(GCNConv(cur, cur))
            gcn_norms.append(nn.LayerNorm(cur))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        self.gcn_norms = nn.ModuleList(gcn_norms)

        self.out_dim = cur
        self.head = build_head(target=target, hidden_dim=cur, pool=graph_pool, dropout=dropout)
        self.target = target

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr.view(-1, 1)  # NNConv / GATv2 want (E, 1)
        edge_weight = data.edge_attr.view(-1)   # GCNConv wants (E,)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.silu(self.norm1(self.conv1(x, edge_index, edge_attr)))
        x = F.silu(self.norm2(self.conv2(x, edge_index, edge_attr)))
        x = F.silu(self.norm3(self.conv3(x, edge_index, edge_attr)))
        for conv, norm in zip(self.gcn_layers, self.gcn_norms):
            x = F.silu(norm(conv(x, edge_index, edge_weight)))

        # Head returns either per-node (N,), per-graph (B,), or dict.
        return self.head(x, batch)
