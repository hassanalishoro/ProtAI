"""ProtAI training CLI — clean rewrite.

Orchestrates one training run: config → datasets → model → trainer → fit → test.

Design principles:
  1. Single responsibility. This file orchestrates training. Data, models,
     metrics, and the Lightning module live in their own files.
  2. CLI-first. Single config file + key.subkey=value overrides + optional
     resume-from-checkpoint flag. No environment-variable magic.
  3. Reproducibility. Seed everything, dump the resolved config next to
     each checkpoint, log full hyperparameters into the checkpoint itself.
  4. Performance defaults that suit our workload.
       * TF32 matmul (Ampere+ tensor cores for fp32 ops)
       * pin_memory + prefetch_factor + persistent_workers in DataLoader
       * NO cudnn.benchmark (we have no conv layers; it just hoards VRAM)
  5. Best-checkpoint selection on val/pearson (mode=max). RMSE/MAE on this
     dataset are dominated by outliers and barely move; pearson is the
     signal that actually tracks generalization.

Usage:
    protai-train --config configs/headline.yaml
    protai-train --config configs/base.yaml --override train.max_epochs=20 train.seed=7
    protai-train --config configs/headline.yaml --ckpt runs/headline/last.ckpt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from ..config import Config, resolve_path
from ..data import PreShardedDataset
from .lit_module import ProtAILitModule


class EpochSummaryCallback(Callback):
    """Pipe-friendly one-line-per-epoch metric summary.

    Replaces tqdm's interactive bar when stdout isn't a TTY (i.e., when
    `tee` or a redirect has captured stdout). Avoids the carriage-return
    spam that turns one progress line into hundreds in the log file.

    Output format (one line per epoch end):
        [epoch  3]  train_loss=0.578 val_loss=0.989 val_pearson=0.122
                    val_spearman=0.136 val_rmse=1.85 val_mae=1.50 val_r2=0.0009

    The single \r-free line plays nicely with grep / less / tail.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = {k: float(v) for k, v in trainer.callback_metrics.items()
             if isinstance(v, (int, float, torch.Tensor))}
        # Pull the keys we know about; quietly skip any that aren't there yet.
        def g(name):
            v = m.get(name)
            return f"{v:+.4g}" if v is not None else "—"
        line = (
            f"[epoch {trainer.current_epoch:3d}] "
            f"train_loss={g('train/loss')} "
            f"val_loss={g('val/loss')} "
            f"val_pearson={g('val/pearson')} "
            f"val_spearman={g('val/spearman')} "
            f"val_rmse={g('val/rmse')} "
            f"val_mae={g('val/mae')} "
            f"val_r2={g('val/r2')}"
        )
        print(line, flush=True)


# ---------------------------------------------------------------------------
# Global PyTorch performance knobs (set once at module import)
# ---------------------------------------------------------------------------
#
# `set_float32_matmul_precision("high")`:
#     Enables TF32 in cuBLAS for fp32 matmuls on Ampere+ GPUs (RTX 30xx,
#     40xx, A100, H100). Roughly 10-15% throughput improvement on the
#     optimizer step and any non-bf16 fp32 ops. The bf16-mixed forward/
#     backward path is unaffected. Free win, always-on.
#
# Note: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set in
# protai/__init__.py so it lands before torch initializes CUDA. That's
# the right place for env vars; this is for runtime knobs.
torch.set_float32_matmul_precision("high")


# Monitor metric used by both ModelCheckpoint (which epoch's weights to save)
# and EarlyStopping (when to bail out). Pearson with mode=max is the right
# choice for binding-affinity ranking tasks where RMSE is outlier-bound and
# pearson is the actual scientific objective.
MONITOR_METRIC = "val/pearson"
MONITOR_MODE = "max"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="protai-train",
        description="Train a ProtAI model from a YAML config.",
    )
    p.add_argument(
        "--config", required=True,
        help="Path to YAML config (e.g., configs/headline.yaml).",
    )
    p.add_argument(
        "--override", nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Config overrides as section.field=value (e.g., train.max_epochs=20).",
    )
    p.add_argument(
        "--ckpt", default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return p


def _apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply key.subkey=value strings to cfg in place. Returns cfg for chaining.

    Supports two-level keys (section.field). Coerces value to the existing
    field's type (bool / int / float / str). Reasonable but not exhaustive —
    nested dicts inside a section field aren't supported. Add a second dot
    or extend the parser if you need that.
    """
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"Override must look like key=value, got {o!r}")
        key, raw = o.split("=", 1)
        parts = key.split(".")
        if len(parts) != 2:
            raise ValueError(
                f"Override key must be section.field, got {key!r}"
            )
        section_name, field_name = parts
        if not hasattr(cfg, section_name):
            raise ValueError(f"Unknown config section: {section_name!r}")
        section = getattr(cfg, section_name)
        if not hasattr(section, field_name):
            raise ValueError(
                f"Unknown field {field_name!r} in section {section_name!r}"
            )
        existing = getattr(section, field_name)
        # Type-coerce based on the existing field's type. Bool first because
        # bool is a subclass of int in Python.
        if isinstance(existing, bool):
            value: Any = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(existing, int):
            value = int(raw)
        elif isinstance(existing, float):
            value = float(raw)
        elif existing is None:
            # None can become a string (most common) or be left None.
            value = raw
        else:
            value = raw
        setattr(section, field_name, value)
    return cfg


# ---------------------------------------------------------------------------
# Run directory + reproducibility
# ---------------------------------------------------------------------------

def _make_run_dir(cfg: Config) -> Path:
    """Compute and create the run output directory.

    If `train.run_name` is null, generates a timestamped name from the
    model/target combo so concurrent unnamed runs don't collide.
    """
    log_dir = resolve_path(cfg.train.log_dir)
    if cfg.train.run_name:
        name = cfg.train.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{cfg.model.name}_{cfg.model.target}_{ts}"
    run_dir = log_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _make_loaders(cfg: Config):
    """Build train / val / test DataLoaders.

    Validation and test deliberately use frame_strategy=frame_zero regardless
    of training's setting, so eval is deterministic and comparable across
    runs (no random frame sampling polluting the metric).
    """
    processed = resolve_path(cfg.data.processed_dir)
    splits = resolve_path(cfg.data.splits_dir)

    common = dict(
        target=cfg.model.target,
        edge_cutoff=cfg.data.edge_cutoff,
        node_feature=("atomic_number"
                      if cfg.model.name == "schnet"
                      else "one_hot_element"),
    )

    train_ds = PreShardedDataset(
        processed_dir=processed,
        split_file=splits / cfg.data.train_split,
        frame_strategy=cfg.data.frame_strategy,
        **common,
    )
    val_ds = PreShardedDataset(
        processed_dir=processed,
        split_file=splits / cfg.data.val_split,
        frame_strategy="frame_zero",
        **common,
    )
    test_ds = PreShardedDataset(
        processed_dir=processed,
        split_file=splits / cfg.data.test_split,
        frame_strategy="frame_zero",
        **common,
    )

    # DataLoader knobs:
    #   pin_memory:           page-locked host buffer → faster H2D transfer
    #   persistent_workers:   reuse worker processes across epochs (huge on
    #                         Windows where mp uses spawn; harmless on Linux)
    #   prefetch_factor=4:    each worker queues 4 batches ahead of the GPU
    nw_train = max(0, int(cfg.train.num_workers))
    nw_eval = max(0, nw_train // 2) if nw_train > 0 else 0

    def _common_kwargs(nw: int) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(pin_memory=True, persistent_workers=nw > 0)
        if nw > 0:
            kw["prefetch_factor"] = 4
        return kw

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=nw_train,
        **_common_kwargs(nw_train),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=nw_eval,
        **_common_kwargs(nw_eval),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=nw_eval,
        **_common_kwargs(nw_eval),
    )
    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------

def _make_callbacks(cfg: Config, run_dir: Path) -> List[pl.Callback]:
    """Build the standard set of callbacks.

    ModelCheckpoint and EarlyStopping both monitor val/pearson (mode=max).
    See module docstring for why. LearningRateMonitor is a free observability
    win — its log appears in the tensorboard scalars panel.
    """
    return [
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename="best",
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            save_top_k=1,
            save_last=True,  # `last.ckpt` lets you resume mid-run
        ),
        EarlyStopping(
            monitor=MONITOR_METRIC,
            mode=MONITOR_MODE,
            patience=cfg.train.early_stop_patience,
            min_delta=cfg.train.early_stop_min_delta,
            # Don't kill training on NaN — the lit_module already handles
            # transient NaN by returning 0.0, but this is defensive insurance.
            check_finite=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def _make_trainer(cfg: Config, run_dir: Path) -> pl.Trainer:
    callbacks = _make_callbacks(cfg, run_dir)

    # Pipe-friendly logging: when stdout isn't a TTY (piped through tee, nohup,
    # tmux capture, etc.), tqdm's carriage-return progress bar becomes one
    # full line per percentage tick — thousands of redundant log lines per
    # epoch. Detect that and:
    #   1. Disable the interactive progress bar entirely.
    #   2. Add EpochSummaryCallback so we still get one clean line per epoch.
    is_tty = sys.stdout.isatty()
    enable_progress_bar = is_tty
    if not is_tty:
        callbacks = list(callbacks) + [EpochSummaryCallback()]

    logger = TensorBoardLogger(
        save_dir=str(run_dir.parent),
        name=run_dir.name,
        version="",  # don't add a "version_0" subdir
    )
    return pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
        log_every_n_steps=10,
        enable_progress_bar=enable_progress_bar,
        # GNN scatter ops have non-deterministic kernels; the seed handles
        # what determinism we get. Setting deterministic=True would slow us
        # down significantly with no real gain.
        deterministic=False,
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def _print_run_banner(cfg: Config, run_dir: Path,
                      n_train: int, n_val: int, n_test: int) -> None:
    """One-glance summary printed before training starts."""
    print()
    print("=" * 72)
    print(f"  ProtAI training run: {run_dir.name}")
    print("=" * 72)
    print(f"  model:           {cfg.model.name} (target={cfg.model.target})")
    print(f"  hidden_dim:      {cfg.model.hidden_dim}")
    print(f"  num_layers:      {cfg.model.num_layers}")
    print(f"  graph_pool:      {cfg.model.graph_pool}")
    print()
    print(f"  frame_strategy:  {cfg.data.frame_strategy}")
    print(f"  edge_cutoff:     {cfg.data.edge_cutoff} Å")
    print()
    print(f"  loss:            {cfg.train.loss}")
    print(f"  optimizer:       {cfg.train.optimizer} (lr={cfg.train.learning_rate}, wd={cfg.train.weight_decay})")
    print(f"  lr_schedule:     {cfg.train.lr_schedule} (warmup={cfg.train.warmup_epochs} epochs)")
    print(f"  grad_clip:       {cfg.train.grad_clip}")
    print(f"  precision:       {cfg.train.precision}")
    print(f"  batch_size:      {cfg.train.batch_size}")
    print(f"  num_workers:     {cfg.train.num_workers}")
    print(f"  max_epochs:      {cfg.train.max_epochs}")
    print(f"  early stopping:  patience={cfg.train.early_stop_patience} on {MONITOR_METRIC} ({MONITOR_MODE})")
    print(f"  seed:            {cfg.train.seed}")
    print()
    print(f"  dataset sizes:   train={n_train}  val={n_val}  test={n_test}")
    print(f"  output dir:      {run_dir}")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_argparser().parse_args()

    # ---- 1. Load + override config, seed everything ----
    cfg = Config.from_yaml(args.config)
    cfg = _apply_overrides(cfg, args.override)
    pl.seed_everything(cfg.train.seed, workers=True)

    # ---- 2. Set up the output directory; freeze the resolved config ----
    run_dir = _make_run_dir(cfg)
    cfg.to_yaml(run_dir / "resolved_config.yaml")

    # ---- 3. Build data, model, trainer ----
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = _make_loaders(cfg)
    module = ProtAILitModule(cfg)
    trainer = _make_trainer(cfg, run_dir)

    _print_run_banner(cfg, run_dir, len(train_ds), len(val_ds), len(test_ds))

    # ---- 4. Train ----
    trainer.fit(module, train_loader, val_loader, ckpt_path=args.ckpt)

    # ---- 5. Test on the best checkpoint, not the final epoch ----
    print("\n" + "=" * 72)
    print("  Loading best checkpoint and evaluating on test set")
    print("=" * 72)
    trainer.test(module, test_loader, ckpt_path="best")

    print(f"\n[done] outputs in {run_dir}")


if __name__ == "__main__":
    main()
