#!/usr/bin/env bash
# ProtAI cloud pod bring-up.
#
# Idempotent: safe to re-run after pod restarts or if a step fails.
# Reads CLOUD_DEPLOYMENT.txt for the why behind every step.
#
# Usage on a fresh Vast.ai / Lambda / Runpod Linux pod:
#     git clone https://github.com/LastPredator/ProtAI.git
#     cd ProtAI
#     bash scripts/cloud_setup.sh
#
# After this finishes successfully:
#   1. Download MD.hdf5 into data/MD/h5_files/  (124 GB)
#   2. Run:  protai-preshard --h5 data/MD/h5_files/MD.hdf5 --out data/processed --device auto
#   3. Run:  protai-train --config configs/headline.yaml --override train.num_workers=12 train.batch_size=32

set -euo pipefail

# Pretty-print helpers.
say()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '    \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[1;33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# Verify Python 3.11 is on PATH. The `python3.11` binary is present on
# every Vast.ai pytorch image we've used; if you picked a 3.12-only
# image the wheel install in step 3 will fail (PyG extensions ship
# fewer 3.12 wheels).
say "Checking Python interpreter"
command -v python3.11 >/dev/null || die "python3.11 not on PATH. Pick a pod image with 3.11 installed (most pytorch images include it)."
PY_VER=$(python3.11 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
ok "python3.11 → ${PY_VER}"

# Detect installed torch + cuda versions. The PyG wheel index URL
# embeds these exact strings; any mismatch and the extensions fail
# to load with the same WinError-127-style symptom we hit on Windows.
say "Detecting torch + cuda versions"
python3.11 -c 'import torch' 2>/dev/null || {
    warn "torch not installed yet; installing the protai package will pull it"
}
TORCH_VER=$(python3.11 -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || echo "")
CUDA_VER=$(python3.11 -c 'import torch; v=torch.version.cuda; print("" if v is None else v.replace(".",""))' 2>/dev/null || echo "")

# Install the protai package itself (editable). After this, torch is
# guaranteed present even if the pod image was minimal.
say "Installing protai package"
python3.11 -m pip install --upgrade pip wheel >/dev/null
python3.11 -m pip install -e ".[api,dev]"
ok "protai installed"

# Re-detect versions in case torch was just installed.
TORCH_VER=$(python3.11 -c 'import torch; print(torch.__version__.split("+")[0])')
CUDA_VER=$(python3.11 -c 'import torch; v=torch.version.cuda; print("cpu" if v is None else v.replace(".",""))')
ok "torch ${TORCH_VER}  cuda ${CUDA_VER}"

if [ "${CUDA_VER}" = "cpu" ]; then
    die "torch was installed without CUDA support. This pod won't train at usable speed; pick a CUDA-enabled image."
fi

# Install PyG extension wheels matching the exact torch+cuda combo.
# This is the single most common cloud bring-up failure: the default
# PyPI wheels don't have CUDA-built versions; you must use the PyG
# wheel index for prebuilt linux-x86_64 wheels.
WHEEL_INDEX="https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html"
say "Installing PyG extension wheels"
echo "    wheel index: ${WHEEL_INDEX}"
python3.11 -m pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f "${WHEEL_INDEX}"

# Verify each extension actually loads — installation success doesn't
# guarantee runtime success when there's any version skew.
say "Verifying PyG extensions load and execute on GPU"
python3.11 - <<'PY'
import torch
mods = ["torch_scatter", "torch_cluster", "torch_sparse", "torch_spline_conv"]
for name in mods:
    __import__(name)
    print(f"    {name}: import ok")

# Hardware sanity: actually run a CUDA kernel through scatter and cluster
from torch_scatter import scatter_add
from torch_cluster import radius_graph
x   = torch.randn(10000, 64, device='cuda')
idx = torch.randint(0, 1000, (10000,), device='cuda')
out = scatter_add(x, idx, dim=0)
assert out.device.type == 'cuda', "scatter_add did not run on GPU"
print(f"    scatter_add GPU: {tuple(out.shape)} on {out.device}")

pos = torch.randn(2000, 3, device='cuda')
ei  = radius_graph(pos, r=0.3, loop=False, max_num_neighbors=64)
assert ei.device.type == 'cuda', "radius_graph did not run on GPU"
print(f"    radius_graph GPU: {ei.shape[1]} edges on {ei.device}")
PY
ok "all extensions exercise GPU paths cleanly"

# Verify allocator config — should be expandable_segments on Linux.
# This is the cheap, silent-failure check that bites people in cloud
# deployment. If protai/__init__.py auto-detection ever breaks, this
# catches it before training.
say "Verifying CUDA allocator config"
python3.11 - <<'PY'
import os, protai  # importing protai triggers its env-var setup
conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')
print(f"    PYTORCH_CUDA_ALLOC_CONF = {conf}")
assert 'expandable_segments' in conf, f"Expected expandable_segments on Linux, got: {conf!r}"
print("    Linux fast-path allocator active")
PY
ok "allocator configured correctly"

# Final nudge.
cat <<'NEXT'

╭──────────────────────────────────────────────────────────────────────╮
│  Cloud setup complete.                                               │
│                                                                      │
│  Next steps:                                                         │
│    1. Get MD.hdf5 into data/MD/h5_files/ (124 GB).                   │
│    2. protai-preshard --h5 data/MD/h5_files/MD.hdf5 \                │
│                       --out data/processed --device auto             │
│    3. protai-train --config configs/headline.yaml \                  │
│                    --override train.num_workers=12 \                 │
│                              train.batch_size=32                     │
│                                                                      │
│  See CLOUD_DEPLOYMENT.txt for the full procedure and gotchas.        │
╰──────────────────────────────────────────────────────────────────────╯
NEXT
