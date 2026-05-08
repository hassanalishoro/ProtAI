"""Quick test: load latest model and run inference on one structure.

Now uses the `protai.api.service` layer so it shares all model-loading and
graph-building logic with the backend.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from protai.api.service import ProtAIService


def main():
    svc = ProtAIService()
    if not svc.data_loaded:
        raise SystemExit(f"Dataset not found at {svc.data_path}")
    if not svc.model_loaded:
        raise SystemExit(
            f"No checkpoint found. Train a model first or set PROTAI_MODEL_PATH."
        )

    pdb_id = os.environ.get("PROTAI_TEST_PDB", "1A1B")
    if not svc.has_structure(pdb_id):
        pdb_id = svc.list_structures()[0]
        print(f"  (env PDB not in dataset, falling back to {pdb_id})")

    print(f"Predicting for {pdb_id}...")
    result = svc.predict_affinity(pdb_id)
    for k, v in result.items():
        print(f"  {k:25s} {v}")
    print("\n[OK]")


if __name__ == "__main__":
    main()
