"""Standalone evaluation: load a checkpoint, run inference on a split, dump metrics.

Useful for cross-dataset evaluation (train on MISATO, test on PDBbind core, etc.)
without retraining.

Usage:
    python -m protai.training.eval \\
        --ckpt runs/headline/best.ckpt \\
        --split data/MD/splits/test_MD.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr

from ..config import Config, resolve_path
from ..data import PreShardedDataset
from .lit_module import ProtAILitModule


def _metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    if y.std() > 1e-6 and yhat.std() > 1e-6:
        pearson = float(np.corrcoef(y, yhat)[0, 1])
        spearman = float(spearmanr(y, yhat).statistic)
    else:
        pearson = float("nan"); spearman = float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman, "r2": r2,
            "n_samples": int(y.shape[0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .ckpt file")
    p.add_argument("--split", required=True, help="Path to split .txt file")
    p.add_argument("--processed", default=None, help="Processed dir (default: from ckpt config)")
    p.add_argument("--out", default=None, help="Output JSON path")
    args = p.parse_args()

    # Load module and config from checkpoint.
    module = ProtAILitModule.load_from_checkpoint(args.ckpt, map_location="cpu")
    cfg: Config = module.cfg

    processed = Path(args.processed) if args.processed else resolve_path(cfg.data.processed_dir)
    ds = PreShardedDataset(
        processed,
        Path(args.split),
        target=cfg.model.target,
        frame_strategy="frame_zero",
        edge_cutoff=cfg.data.edge_cutoff,
        node_feature="atomic_number" if cfg.model.name == "schnet" else "one_hot_element",
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        logger=False,
    )
    trainer.test(module, loader)

    # Re-collect predictions for richer metrics + dump.
    # IMPORTANT: the model's forward returns predictions in NORMALIZED z-score
    # space for any target trained with target normalization (binding_affinity,
    # log_k, and the energy head of multitask_*). Denormalize before computing
    # RMSE/MAE — otherwise the JSON contains z-score units while batch.y is in
    # the target's original units, and RMSE/MAE come out roughly equal to the
    # target standard deviation regardless of model quality.
    module.eval()
    ys, yhats = [], []
    import torch
    device = next(module.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = module(batch)
            t = cfg.model.target
            if t == "multitask":
                # Legacy energy+adaptability multitask
                ys.append(batch.y_energy.cpu().numpy().flatten())
                yhats.append(module._denormalize(pred["energy"]).cpu().numpy().flatten())
            elif t == "multitask_logk_energy":
                # New log_K + energy multitask — headline is the log_K head.
                ys.append(batch.y_logk.cpu().numpy().flatten())
                yhats.append(module._denormalize(pred["logk"]).cpu().numpy().flatten())
            elif t == "adaptability":
                ys.append(batch.y.cpu().numpy().flatten())
                yhats.append(pred.cpu().numpy().flatten())
            else:  # binding_affinity, log_k — single-scalar normalized targets
                ys.append(batch.y.cpu().numpy().flatten())
                yhats.append(module._denormalize(pred).cpu().numpy().flatten())

    y = np.concatenate(ys); yhat = np.concatenate(yhats)
    m = _metrics(y, yhat)
    m["checkpoint"] = args.ckpt
    m["split"] = args.split
    print(json.dumps(m, indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)
        print(f"[done] metrics → {args.out}")


if __name__ == "__main__":
    main()
