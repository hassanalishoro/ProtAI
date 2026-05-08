"""ProtAI: trajectory-aware protein-ligand binding affinity prediction."""

# CUDA memory allocator config — must be set BEFORE torch initializes CUDA.
#
# Linux: `expandable_segments:True` is purpose-built for variable-shape
#        GNN/transformer workloads (our protein graphs vary in N and E
#        every batch). PyTorch team recommends it for this exact use case.
#
# Windows: we deliberately do NOT set PYTORCH_CUDA_ALLOC_CONF here.
#        We tested `max_split_size_mb:128` + `garbage_collection_threshold:0.8`
#        and it made throughput WORSE (0.18 → 0.07 it/s on RTX 4070 Laptop,
#        VRAM spillover from 12 GB → 15 GB). The aggressive split-size cap
#        prevents large edge tensors from getting contiguous blocks and the
#        GC threshold triggers constant sweeps. Default allocator wins.
#
# Auto-detected so cloud (Linux) deployments automatically get the right
# config without anyone needing to remember to flip it manually.
import os as _os
import sys as _sys
if _sys.platform.startswith("linux"):
    _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Silence known-harmless Windows warnings emitted on every PyG import.
# pyg-lib, torch-scatter, torch-cluster, torch-spline-conv, torch-sparse are
# C++ extensions whose compiled DLLs fail to load on Windows when the wheel was
# built against a different PyTorch/CUDA combo than what's installed. PyG falls
# back to pure-Python implementations which work fine for our use case.
# Also silences torch.cuda's pynvml deprecation warning.
#
# This filter is set BEFORE any submodule import, so it's in place by the time
# torch_geometric is imported anywhere in the package.
import warnings as _warnings

for _pattern in (
    ".*pyg-lib.*",
    ".*torch-scatter.*",
    ".*torch-cluster.*",
    ".*torch-spline-conv.*",
    ".*torch-sparse.*",
    ".*pynvml.*",
):
    _warnings.filterwarnings("ignore", message=_pattern)

__version__ = "0.1.0"
