# ProtAI

**Trajectory-aware protein–ligand binding affinity prediction with graph neural networks on the MISATO dataset.**

Most published binding-affinity models train on a single static crystal snapshot per complex. ProtAI trains on the whole MD trajectory — 100 frames per complex × 16,972 complexes — so the model sees how the binding pocket actually moves during binding.

> **Status:** code complete, production preshard verified at 16,972 / 16,972 complexes. Training runs in progress.

| | |
|---|---|
| **Team** | Ibaad Ahmed Chaudhry (22I-0585) · Abdullah Kaif Sheikh (22I-2142) · Hassan Ali Shoro (22I-0561) |
| **Supervisor** | Mr. Shoaib Saleem Khattak |
| **Institution** | Department of Computer Science, FAST NUCES Islamabad |
| **Session** | 2022 — 2026 |

---

## What's in this repo

```
ProtAI/
├── protai/             Python package — data, models, training, API
├── configs/            YAML configs for every ablation cell
├── scripts/            Wrapper scripts (preshard, train, eval)
├── backend/            Flask app — thin wrapper over protai.api
├── frontend-new/       Astro + React + Tailwind UI (gitignored, see its own README)
├── data/               MISATO splits + H5 files (large files gitignored)
├── runs/               Training outputs (gitignored)
├── docker/             Original MISATO Docker image (kept for reproducibility)
├── pyproject.toml      Package metadata + dependencies
└── *.pdf               Project proposal, thesis, reference papers
```

---

## Quick start

### Requirements

- Python **3.11+** (PyTorch 2 + PyG 2.5 baseline)
- ~150 GB free disk (for the presharded dataset)
- A GPU is strongly recommended for training (works on RTX 3060+ / 4070 / 4090 / cloud A100)
- For the frontend: Node 20+ and `npm`

### Install

```bash
git clone <this-repo>
cd ProtAI

# Editable install with all optional extras (training + Flask backend + dev tools)
py -3.11 -m pip install -e ".[api,dev]"
```

### Get the MISATO dataset

The 124 GB `MD.hdf5` is hosted on Zenodo and is **not** committed to git. Download it from MISATO's Zenodo release into `data/MD/h5_files/MD.hdf5`. The split files (`data/MD/splits/`) are already in the repo.

### Pre-shard the data (one-time, ~40 min on RTX 4070)

```bash
protai-preshard --h5 data/MD/h5_files/MD.hdf5 --out data/processed --device auto
# or:
py -3.11 scripts/preshard.py --h5 data/MD/h5_files/MD.hdf5 --out data/processed --device auto
```

This walks the H5, strips hydrogens + solvent, builds the radius graph, computes per-atom adaptability, and emits one `.pt` file per complex (~9 MB each, ~150 GB total).

### Verify the preshard

```bash
py -3.11 scripts/verify_preshard.py
```

You should see all 16,972 files present and splits aligned 100%.

### Train a model

```bash
# Quick sanity check on the tiny dataset (~3 minutes on RTX 4070)
protai-train --config configs/tiny.yaml

# A single ablation cell on full data
protai-train --config configs/arch_target_grid/schnet_binding_affinity.yaml

# The headline configuration (with all best choices)
protai-train --config configs/headline.yaml
```

Outputs land in `runs/<run_name>/`:

| File | Purpose |
|---|---|
| `best.ckpt` | Lightning checkpoint of the best validation epoch |
| `last.ckpt` | Resume point |
| `resolved_config.yaml` | The actual config used (post-overrides) |
| `events.out.tfevents.*` | TensorBoard logs |
| `hparams.yaml` | Hyperparameters frozen into the checkpoint |

### Evaluate on a different split (e.g. cross-dataset)

```bash
protai-eval --ckpt runs/headline/best.ckpt --split data/MD/splits/test_MD.txt
```

### Run the backend + frontend

Two terminals:

```bash
# Terminal 1 — Flask API on port 5000
py -3.11 backend/app.py

# Terminal 2 — Astro dev server on port 4321
cd frontend-new
npm install      # one-time
npm run dev
```

Open http://localhost:4321. The dev server proxies `/api/*` to Flask automatically.

For production deployment, build the frontend first (`npm run build`) — Flask's `static_folder` auto-detects `frontend-new/dist/` and serves the built site directly.

---

## What makes this different

Three things, in priority order:

1. **Trajectory-aware training.** Every other PDBbind-trained model uses one crystal snapshot per complex. We use 100 MD frames per complex with random-frame sampling — essentially free 100× data augmentation that exposes the model to pocket flexibility.

2. **Equivariant baseline.** The headline configuration uses SchNet (E(3)-equivariant by construction), with a hybrid GNN_MD baseline for ablation.

3. **Open + reproducible.** Every ablation in the paper is one config file + one command. Pretrained weights ship with each release.

---

## Configuration system

All hyperparameters live in `configs/*.yaml`, parsed into typed dataclasses by `protai/config.py`. Override anything from the CLI:

```bash
protai-train --config configs/headline.yaml \
  --override train.max_epochs=20 model.hidden_dim=64 train.seed=1337
```

See `configs/base.yaml` for the full list of available knobs.

---

## Repository hygiene

| Path | Status |
|---|---|
| `data/processed/` | gitignored — 150 GB, regenerable |
| `data/MD/h5_files/MD.hdf5` | gitignored — 124 GB, download from Zenodo |
| `runs/` | gitignored — checkpoints + tensorboard events |
| `frontend-new/` | gitignored — kept local until model is trained, then merged |
| `__pycache__/` | gitignored |

Anything outside that list is committed.

---

## Acknowledgements

This work builds on the [MISATO dataset](https://github.com/t7morgen/misato-dataset) (Siebenmorgen et al., 2024) and uses [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural network primitives.

The original MISATO codebase is LGPL-2.1; we've inherited that license.

---

## Citation

If you use ProtAI in your work, please cite:

```bibtex
@thesis{protai2026,
  title  = {ProtAI: Trajectory-Aware Protein-Ligand
            Binding Affinity Prediction},
  author = {Chaudhry, I.A. and Sheikh, A.K. and Shoro, H.A.},
  school = {National University of Computer and Emerging Sciences,
            Islamabad},
  year   = {2026},
  type   = {Final Year Project},
  supervisor = {Khattak, S.S.}
}
```
