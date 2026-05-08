"""ProtAI: trajectory-aware protein-ligand binding affinity prediction."""

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
