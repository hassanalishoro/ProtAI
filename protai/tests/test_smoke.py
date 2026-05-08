"""End-to-end smoke test on the 20-complex tiny_md.hdf5.

Verifies the full pipeline without ever touching MD.hdf5:
  1. Pre-shard the tiny H5 → .pt files
  2. Load with PreShardedDataset
  3. Build graphs match the expected shape
  4. Forward pass through both models works for all targets
  5. Each head produces a tensor of the expected shape

Run from anywhere:
    py -3.11 -m protai.tests.test_smoke
    py -3.11 protai/tests/test_smoke.py
    py -3.11 protai\\tests\\test_smoke.py    (Windows shell)
"""
from __future__ import annotations

# ---- Environment setup (must happen before importing torch / torch_geometric) ----
import os
import sys
import warnings
from pathlib import Path

# Silence known-harmless Windows warnings:
#   * PyG optional C++ extensions (pyg-lib, torch-scatter, etc.) — PyG falls back to pure-Python
#   * pynvml deprecation warning from torch.cuda
warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*")
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore", message=".*torch-sparse.*")
warnings.filterwarnings("ignore", message=".*pynvml.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Bootstrap sys.path so this works as both `python -m` and `python file.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---- Now safe to import the package ----
import shutil
import tempfile

import torch
from torch_geometric.loader import DataLoader

from protai.config import REPO_ROOT
from protai.data.graph import build_radius_graph, one_hot_elements
from protai.data.preshard import preshard
from protai.data.dataset import PreShardedDataset
from protai.models import GNN_MD, SchNetModel


TINY_H5 = REPO_ROOT / "data" / "MD" / "h5_files" / "tiny_md.hdf5"


def _scaffold_tmpdir(device: str = "auto") -> Path:
    """Pre-shard tiny_md.hdf5 into a tmpdir; return the path."""
    tmp = Path(tempfile.mkdtemp(prefix="protai_smoke_"))
    summary = preshard(
        h5_path=TINY_H5,
        out_dir=tmp,
        edge_cutoff=4.5,
        keep_trajectory=True,
        workers=1,
        device=device,
    )
    assert summary.get("ok", 0) > 0, f"Preshard produced 0 ok files. Summary: {summary}"
    return tmp


def test_graph_builder():
    # CPU path
    pos = torch.randn(50, 3) * 5.0
    ei, ea = build_radius_graph(pos, cutoff=4.5)
    assert ei.shape[0] == 2 and ei.shape[1] == ea.shape[0]
    if ei.numel() > 0:
        assert ei.shape[1] % 2 == 0
    print(f"[graph cpu]  {pos.shape[0]} nodes -> {ei.shape[1]} edges  OK")

    # GPU path (uses identical inputs by moving the same tensor to GPU)
    if torch.cuda.is_available():
        torch.manual_seed(0)
        cpu_pos = torch.randn(50, 3) * 5.0
        gpu_pos = cpu_pos.to("cuda")
        ei_cpu, _ = build_radius_graph(cpu_pos, cutoff=4.5)
        ei_gpu, _ = build_radius_graph(gpu_pos, cutoff=4.5)
        # cKDTree uses float64 internally; torch.cdist uses float32. A handful of
        # boundary edges right at the cutoff may flip. Allow ±5% tolerance.
        diff = abs(ei_cpu.shape[1] - ei_gpu.shape[1])
        rel = diff / max(ei_cpu.shape[1], 1)
        assert rel < 0.05, \
            f"GPU/CPU graph differ too much: cpu={ei_cpu.shape[1]} gpu={ei_gpu.shape[1]} ({rel:.1%})"
        print(f"[graph gpu]  cpu={ei_cpu.shape[1]} gpu={ei_gpu.shape[1]} (diff={diff})  OK")
    else:
        print(f"[graph gpu]  skipped (no CUDA)")


def test_one_hot():
    idx = torch.tensor([1, 2, 3, 7, 99])  # 99 is out of range -> UNK bin
    oh = one_hot_elements(idx, n_classes=11)
    assert oh.shape == (5, 11)
    assert oh.sum() == 5
    assert oh[-1, -1] == 1.0
    print("[one_hot]    vectorized one-hot encoding  OK")


def test_dataset_and_models():
    if not TINY_H5.exists():
        print(f"[!] {TINY_H5} not found -- skipping dataset/model smoke test.")
        return

    tmp = _scaffold_tmpdir()
    try:
        # ---- target = binding_affinity, model = GNN_MD ----
        ds = PreShardedDataset(tmp, target="binding_affinity",
                               frame_strategy="frame_zero", node_feature="one_hot_element")
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        gnn = GNN_MD(num_features=11, hidden_dim=32, num_layers=4,
                     attention_heads=2, target="binding_affinity").eval()
        with torch.no_grad():
            out = gnn(batch)
        assert out.shape == (batch.num_graphs,), f"GNN_MD/affinity bad shape {out.shape}"
        print(f"[gnn_md]     binding_affinity batch={batch.num_graphs} -> {tuple(out.shape)}  OK")

        # ---- target = adaptability, model = GNN_MD ----
        ds_a = PreShardedDataset(tmp, target="adaptability",
                                 frame_strategy="frame_zero", node_feature="one_hot_element")
        batch_a = next(iter(DataLoader(ds_a, batch_size=2, shuffle=False)))
        gnn_a = GNN_MD(num_features=11, hidden_dim=32, num_layers=4,
                       attention_heads=2, target="adaptability").eval()
        with torch.no_grad():
            out_a = gnn_a(batch_a)
        assert out_a.shape == (batch_a.num_nodes,), f"GNN_MD/adapt bad shape {out_a.shape}"
        print(f"[gnn_md]     adaptability nodes={batch_a.num_nodes} -> {tuple(out_a.shape)}  OK")

        # ---- target = binding_affinity, model = SchNet ----
        ds_s = PreShardedDataset(tmp, target="binding_affinity",
                                 frame_strategy="frame_zero", node_feature="atomic_number")
        batch_s = next(iter(DataLoader(ds_s, batch_size=2, shuffle=False)))
        sch = SchNetModel(hidden_dim=32, num_layers=2, num_filters=32, num_gaussians=10,
                          cutoff=4.5, target="binding_affinity").eval()
        with torch.no_grad():
            out_s = sch(batch_s)
        assert out_s.shape == (batch_s.num_graphs,), f"SchNet bad shape {out_s.shape}"
        print(f"[schnet]     binding_affinity batch={batch_s.num_graphs} -> {tuple(out_s.shape)}  OK")

        # ---- multitask ----
        ds_m = PreShardedDataset(tmp, target="multitask",
                                 frame_strategy="frame_zero", node_feature="one_hot_element")
        batch_m = next(iter(DataLoader(ds_m, batch_size=2, shuffle=False)))
        gnn_m = GNN_MD(num_features=11, hidden_dim=32, num_layers=4,
                       attention_heads=2, target="multitask").eval()
        with torch.no_grad():
            out_m = gnn_m(batch_m)
        assert isinstance(out_m, dict) and "energy" in out_m and "adaptability" in out_m
        assert out_m["energy"].shape == (batch_m.num_graphs,)
        assert out_m["adaptability"].shape == (batch_m.num_nodes,)
        print(f"[gnn_md]     multitask energy={tuple(out_m['energy'].shape)} adapt={tuple(out_m['adaptability'].shape)}  OK")

        # ---- random frame strategy ----
        ds_r = PreShardedDataset(tmp, target="binding_affinity",
                                 frame_strategy="random", node_feature="one_hot_element")
        b_r = next(iter(DataLoader(ds_r, batch_size=2, shuffle=False)))
        with torch.no_grad():
            _ = gnn(b_r)
        print(f"[frame]      random-frame strategy works  OK")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    import time
    print("=" * 70)
    print("ProtAI smoke test")
    cuda = torch.cuda.is_available()
    print(f"  CUDA available: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    print("=" * 70)
    t0 = time.time()
    test_graph_builder()
    test_one_hot()
    test_dataset_and_models()
    print("=" * 70)
    print(f"ALL OK  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
