"""Models: GNN_MD baseline, SchNet (equivariant), pluggable heads."""

from .gnn_md import GNN_MD
from .schnet import SchNetModel
from .heads import PerAtomHead, GraphLevelHead, MultiTaskHead

__all__ = ["GNN_MD", "SchNetModel", "PerAtomHead", "GraphLevelHead", "MultiTaskHead"]


def build_model(name: str, **kwargs):
    """Factory for selecting model by name from config."""
    name = name.lower()
    if name == "gnn_md":
        return GNN_MD(**kwargs)
    if name == "schnet":
        return SchNetModel(**kwargs)
    raise ValueError(f"Unknown model name: {name!r}. Options: gnn_md, schnet")
