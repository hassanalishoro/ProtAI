"""Single source of truth for ProtAI hyperparameters.

Loads/dumps YAML configs into typed dataclasses so every module reads the same
values. No more `EDGE_DIST_CUTOFF = 4.5` scattered across five files.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Project-wide constants
# ---------------------------------------------------------------------------

# MISATO atom-element internal mapping (1-indexed in the H5).
#   1=H, 2=C, 3=N, 4=O, 5=F, 6=P, 7=S, 8=Cl, 9=Br, 10=I, plus an "unknown" bin.
ELEMENT_NAMES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "UNK"]
N_ELEMENT_CLASSES = len(ELEMENT_NAMES)

# Hydrogen atomic number — used to strip H atoms during preshard.
H_ATOMIC_NUMBER = 1

# Project root (resolved at import time, works regardless of CWD).
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Where data lives + graph construction params."""
    raw_h5: str = "data/MD/h5_files/MD.hdf5"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/MD/splits"
    train_split: str = "train_MD.txt"
    val_split: str = "val_MD.txt"
    test_split: str = "test_MD.txt"

    # Graph construction
    edge_cutoff: float = 4.5  # Angstroms

    # Multi-frame trajectory sampling
    frame_strategy: str = "frame_zero"  # frame_zero | random | mean | attention_pool
    n_frames: int = 8  # used by attention_pool / mean strategies

    # Preshard options
    keep_trajectory: bool = True  # if False, only frame 0 stored (smaller files)
    strip_hydrogens: bool = True
    strip_solvent: bool = True


@dataclass
class ModelConfig:
    """Architecture selection + hyperparameters."""
    name: str = "schnet"  # gnn_md | schnet
    target: str = "binding_affinity"  # binding_affinity | adaptability | multitask

    # Shared
    hidden_dim: int = 128
    num_layers: int = 4

    # GNN_MD specifics
    gnn_md_attention_heads: int = 4

    # SchNet specifics
    schnet_num_filters: int = 128
    schnet_num_gaussians: int = 50

    # Pooling for graph-level prediction
    graph_pool: str = "mean"  # mean | sum | attention


@dataclass
class TrainConfig:
    """Training-loop knobs."""
    batch_size: int = 16
    num_workers: int = 4
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    grad_clip: float = 1.0
    loss: str = "huber"  # mse | mae | huber

    # LR schedule
    lr_schedule: str = "cosine"  # none | cosine | reduce_on_plateau
    warmup_epochs: int = 2

    # Regularization
    dropout: float = 0.1

    # Reproducibility
    seed: int = 42

    # Lightning
    precision: str = "bf16-mixed"  # 32 | 16-mixed | bf16-mixed
    accelerator: str = "auto"  # auto | gpu | cpu
    devices: str = "auto"

    # Early stopping
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    # Output
    log_dir: str = "runs"
    run_name: Optional[str] = None  # auto-generated if None


@dataclass
class Config:
    """Top-level config combining all sub-sections."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load a YAML file. Missing keys fall back to defaults."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)


def resolve_path(p: str | Path) -> Path:
    """Resolve a path against REPO_ROOT if it's relative."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)
