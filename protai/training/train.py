"""CLI entry point: train a model from a YAML config.

Usage:
    python -m protai.training.train --config configs/headline.yaml
    python -m protai.training.train --config configs/base.yaml --override model.name=schnet train.max_epochs=20
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set in
# protai/__init__.py to guarantee it lands before torch CUDA init.

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from ..config import Config, resolve_path
from ..data import PreShardedDataset
from .lit_module import ProtAILitModule


# ---------------------------------------------------------------------------
# `set_float32_matmul_precision("high")`:
#     Lets cuBLAS use TF32 on Ampere+ (RTX 30xx / 40xx / A100 / H100). 10–15%
#     throughput win for fp32 matmuls (optimizer step, reductions); bf16
#     mixed-precision paths unaffected.
#
# Note: we deliberately do NOT enable `cudnn.benchmark`. It autotunes conv
# kernels and keeps speculative workspace allocated — pure waste for SchNet
# (no conv layers) and actively harmful with our variable-shape graph batches
# where there's no stable shape for it to cache against.
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")


def _apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Parse `key.subkey=value` overrides, mutate cfg in place, return it.

    Examples:
        train.max_epochs=10
        model.name=schnet
        data.frame_strategy=random
    """
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Override must be key=value, got {o!r}")
        key, raw = o.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(f"Override key must be section.field, got {key!r}")
        section, field = parts
        target = getattr(cfg, section)
        # Try to coerce to the existing field's type.
        existing = getattr(target, field)
        if isinstance(existing, bool):
            value = raw.lower() in ("1", "true", "yes")
        elif isinstance(existing, int):
            value = int(raw)
        elif isinstance(existing, float):
            value = float(raw)
        else:
            value = raw
        setattr(target, field, value)
    return cfg


def _make_run_dir(cfg: Config) -> Path:
    log_dir = resolve_path(cfg.train.log_dir)
    name = cfg.train.run_name or f"{cfg.model.name}_{cfg.model.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = log_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--override", nargs="*", default=[], help="key.subkey=value overrides")
    p.add_argument("--ckpt", default=None, help="Resume from checkpoint")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg = _apply_overrides(cfg, args.override)

    pl.seed_everything(cfg.train.seed, workers=True)

    run_dir = _make_run_dir(cfg)
    cfg.to_yaml(run_dir / "resolved_config.yaml")

    # Datasets / loaders.
    processed = resolve_path(cfg.data.processed_dir)
    splits = resolve_path(cfg.data.splits_dir)
    common = dict(
        target=cfg.model.target,
        frame_strategy=cfg.data.frame_strategy,
        edge_cutoff=cfg.data.edge_cutoff,
        node_feature="atomic_number" if cfg.model.name == "schnet" else "one_hot_element",
    )
    train_ds = PreShardedDataset(processed, splits / cfg.data.train_split, **common)
    # For val/test, force frame_zero so metrics are deterministic (no random sampling).
    eval_common = {**common, "frame_strategy": "frame_zero"}
    val_ds = PreShardedDataset(processed, splits / cfg.data.val_split, **eval_common)
    test_ds = PreShardedDataset(processed, splits / cfg.data.test_split, **eval_common)

    print(f"[train] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # DataLoader perf knobs:
    #   pin_memory=True            → faster H2D copies (page-locked staging buffer)
    #   persistent_workers=True    → don't tear down/respawn workers between epochs
    #                                (especially expensive on Windows where
    #                                multiprocessing uses spawn, not fork)
    #   prefetch_factor=4          → each worker queues up 4 batches ahead of the
    #                                GPU so disk I/O overlaps with compute
    nw = max(0, int(cfg.train.num_workers))
    nw_eval = max(0, nw // 2) if nw > 0 else 0
    loader_kwargs = dict(pin_memory=True)
    if nw > 0:
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=nw, persistent_workers=nw > 0,
        **loader_kwargs,
    )
    eval_loader_kwargs = dict(pin_memory=True)
    if nw_eval > 0:
        eval_loader_kwargs["prefetch_factor"] = 4
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=nw_eval, persistent_workers=nw_eval > 0,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=nw_eval, persistent_workers=nw_eval > 0,
        **eval_loader_kwargs,
    )

    # Module + callbacks + trainer.
    module = ProtAILitModule(cfg)

    # Monitor val/pearson (mode=max) for both checkpointing and early stopping.
    # Why pearson:
    #   * It's the metric that matters for binding-affinity ranking tasks
    #   * RMSE/MAE are dominated by a few outlier complexes (e.g., 4CP5 at
    #     -675 kcal/mol) and barely move during training, causing premature
    #     early stopping and poor checkpoint selection
    #   * Pearson rank-correlates predictions with truth and improves smoothly
    #     epoch-by-epoch as the model learns
    callbacks = [
        ModelCheckpoint(
            dirpath=str(run_dir), filename="best",
            monitor="val/pearson", mode="max", save_top_k=1, save_last=True,
        ),
        EarlyStopping(
            monitor="val/pearson", patience=cfg.train.early_stop_patience,
            min_delta=cfg.train.early_stop_min_delta, mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(save_dir=str(run_dir.parent), name=run_dir.name, version="")

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
        deterministic=False,  # GNN ops aren't fully deterministic; seed handles repro
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.ckpt)
    print("\n=== Test on best checkpoint ===")
    trainer.test(module, test_loader, ckpt_path="best")
    print(f"\n[done] outputs in {run_dir}")


if __name__ == "__main__":
    main()
