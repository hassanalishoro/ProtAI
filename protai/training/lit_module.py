"""PyTorch Lightning module wrapping any ProtAI model.

Why Lightning: free DDP, free mixed precision, free checkpointing, free logging.
The 350-line custom training loop in `examples/train_with_metrics.py` is replaced
by ~150 lines here that handle more cases correctly:
  * Configurable loss (MSE / MAE / Huber)
  * Gradient clipping (final report mentioned training instability)
  * AdamW + cosine LR schedule with warmup
  * Per-atom RMSE for adaptability target (legacy code averaged per-graph — wrong
    when graphs have different atom counts)
  * Pearson / Spearman tracked every val epoch
  * Multitask handling (joint loss with configurable weighting)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.stats import spearmanr

from ..config import Config, DataConfig, ModelConfig, TrainConfig
from ..models import build_model


def _coerce_config(cfg) -> Config:
    """Accept Config / dict / None, return a Config. Used so Lightning's
    `load_from_checkpoint` can re-instantiate from saved hparams (which come
    back as a plain dict)."""
    if cfg is None:
        raise TypeError("ProtAILitModule requires `cfg` (Config object or dict).")
    if isinstance(cfg, Config):
        return cfg
    if isinstance(cfg, dict):
        return Config(
            data=DataConfig(**cfg.get("data", {})),
            model=ModelConfig(**cfg.get("model", {})),
            train=TrainConfig(**cfg.get("train", {})),
        )
    raise TypeError(f"`cfg` must be Config or dict, got {type(cfg).__name__}")


def _build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unknown loss {name!r}")


class ProtAILitModule(pl.LightningModule):
    """Lightning wrapper around any ProtAI model.

    The model's `forward(data)` produces:
      * (N,) tensor for `target == "adaptability"`
      * (B,) tensor for `target == "binding_affinity"`
      * dict {"energy": (B,), "adaptability": (N,)} for `target == "multitask"`
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        # Lightning's load_from_checkpoint passes saved hparams as kwargs.
        # Accept the modern key ("cfg"), the legacy key ("config") from older
        # checkpoints, and the case where `cfg` is a dict (round-trip from yaml).
        if cfg is None:
            cfg = kwargs.pop("config", None)
        cfg = _coerce_config(cfg)

        # Save under "cfg" so future `load_from_checkpoint` finds the right key.
        self.save_hyperparameters({"cfg": cfg.to_dict()})
        self.cfg = cfg

        # Instantiate model.
        m = cfg.model
        model_kwargs: Dict[str, Any] = dict(
            hidden_dim=m.hidden_dim,
            num_layers=m.num_layers,
            target=m.target,
            graph_pool=m.graph_pool,
            dropout=cfg.train.dropout,
        )
        if m.name == "gnn_md":
            model_kwargs.update(
                num_features=11,  # one-hot element classes (see protai.config)
                attention_heads=m.gnn_md_attention_heads,
            )
        elif m.name == "schnet":
            model_kwargs.update(
                num_filters=m.schnet_num_filters,
                num_gaussians=m.schnet_num_gaussians,
                cutoff=cfg.data.edge_cutoff,
            )
        self.model = build_model(m.name, **model_kwargs)

        self.loss_fn = _build_loss(cfg.train.loss)
        self.target = m.target

        # Validation buffers — keep tensors ON DEVICE per batch and only sync
        # to CPU once at epoch end. The previous version called .cpu().tolist()
        # in every validation_step, forcing a CUDA sync ~100 times per epoch.
        self._val_y: List[torch.Tensor] = []
        self._val_yhat: List[torch.Tensor] = []

    # ------------------------------------------------------------- forward

    def forward(self, data):
        return self.model(data)

    # ------------------------------------------------------------- losses

    def _compute_loss(self, pred, data):
        """Handle single-target and multitask cases uniformly."""
        if self.target == "multitask":
            energy_loss = self.loss_fn(pred["energy"], data.y_energy)
            adapt_loss = self.loss_fn(pred["adaptability"], data.y_adapt)
            # 50/50 weighting; can be made configurable later.
            return 0.5 * energy_loss + 0.5 * adapt_loss, {
                "loss/energy": energy_loss.detach(),
                "loss/adaptability": adapt_loss.detach(),
            }
        return self.loss_fn(pred, data.y), {}

    # ------------------------------------------------------------- steps

    def training_step(self, data, _idx):
        pred = self(data)
        loss, parts = self._compute_loss(pred, data)
        self.log("train/loss", loss, batch_size=data.num_graphs, prog_bar=True)
        for k, v in parts.items():
            self.log(f"train/{k}", v, batch_size=data.num_graphs)
        return loss

    def validation_step(self, data, _idx):
        pred = self(data)
        loss, parts = self._compute_loss(pred, data)
        self.log("val/loss", loss, batch_size=data.num_graphs, prog_bar=True)
        for k, v in parts.items():
            self.log(f"val/{k}", v, batch_size=data.num_graphs)

        # Stash y / yhat for end-of-epoch correlation metrics.
        # Keep tensors on-device; we'll concat + transfer once in epoch_end.
        if self.target == "multitask":
            y, yhat = data.y_energy, pred["energy"]
        else:
            y, yhat = data.y, pred
        self._val_y.append(y.detach().flatten())
        self._val_yhat.append(yhat.detach().flatten())

    def on_validation_epoch_end(self):
        if not self._val_y:
            return
        # Single GPU→CPU sync at the very end (was: once per val batch).
        y = torch.cat(self._val_y).float().cpu().numpy()
        yhat = torch.cat(self._val_yhat).float().cpu().numpy()
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae = float(np.mean(np.abs(y - yhat)))
        # Correlations require >= 2 distinct values to be defined.
        if y.std() > 1e-6 and yhat.std() > 1e-6:
            pearson = float(np.corrcoef(y, yhat)[0, 1])
            spearman = float(spearmanr(y, yhat).statistic)
        else:
            pearson = float("nan")
            spearman = float("nan")
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot

        self.log_dict({
            "val/rmse": rmse,
            "val/mae": mae,
            "val/pearson": pearson,
            "val/spearman": spearman,
            "val/r2": r2,
        }, prog_bar=True)
        self._val_y.clear()
        self._val_yhat.clear()

    def test_step(self, data, _idx):
        # Reuse validation logic — Lightning aggregates with the test/ prefix instead.
        return self.validation_step(data, _idx)

    def on_test_epoch_end(self):
        # Same metric computation; just rename the prefix.
        if not self._val_y:
            return
        y = torch.cat(self._val_y).float().cpu().numpy()
        yhat = torch.cat(self._val_yhat).float().cpu().numpy()
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae = float(np.mean(np.abs(y - yhat)))
        if y.std() > 1e-6 and yhat.std() > 1e-6:
            pearson = float(np.corrcoef(y, yhat)[0, 1])
            spearman = float(spearmanr(y, yhat).statistic)
        else:
            pearson = float("nan")
            spearman = float("nan")
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        self.log_dict({
            "test/rmse": rmse, "test/mae": mae,
            "test/pearson": pearson, "test/spearman": spearman, "test/r2": r2,
        })
        self._val_y.clear()
        self._val_yhat.clear()

    # ------------------------------------------------------------- optim

    def configure_optimizers(self):
        t = self.cfg.train
        if t.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        elif t.optimizer.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer {t.optimizer!r}")

        if t.lr_schedule == "none":
            return opt

        if t.lr_schedule == "reduce_on_plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=3
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val/loss"}}

        # Cosine with warmup (default).
        warm = max(1, t.warmup_epochs)
        total = max(t.max_epochs, warm + 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warm:
                return (epoch + 1) / warm
            progress = (epoch - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": sched}
