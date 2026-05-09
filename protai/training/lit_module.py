"""ProtAI Lightning module — clean rewrite.

Design principles, after learning the hard way what didn't work:

1.  Target normalization is mandatory.
    Energy targets range roughly -180 to +25 kcal/mol with mean ~-28, std ~30.
    Without normalization, MSE loss values are O(1000), gradients are huge,
    and optimization is poorly conditioned. Huber loss caps gradients at ±1
    per sample which prevents scale learning entirely. Both fail in practice.

    Solution: at fit start, compute (mean, std) of training targets ONCE and
    store as buffers. Train in normalized (z-score) space — the loss surface
    is well-conditioned, gradients are O(1), and standard MSE works correctly.
    Predictions are denormalized back to kcal/mol for metric computation, so
    val/test RMSE/MAE/Pearson are all reported in the original units that
    reviewers and the paper care about.

2.  MSE on normalized data, not Huber.
    Huber's appeal is robustness to outliers, but at delta=1 on unit-variance
    data it behaves identically to MSE for typical errors and only kicks in
    for severe outliers. Normalization already mitigates outliers; standard
    MSE is the right tool.

3.  Track val/pearson — that's what matters for the task.
    Binding affinity prediction is fundamentally a ranking problem. RMSE is
    dominated by a handful of outlier complexes (e.g., 4CP5 at -675 kcal/mol)
    and barely moves during training. Pearson rank-correlates predictions
    with truth and improves smoothly as the model learns. Use it for both
    early stopping (mode=max) and best-checkpoint selection.

4.  Per-target normalization policy.
    * binding_affinity: graph-level scalar in kcal/mol — normalize.
    * adaptability:     per-atom in Å, already O(1-10) — leave raw.
    * multitask:        normalize the energy head, leave adaptability raw.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import spearmanr

from ..config import Config, DataConfig, ModelConfig, TrainConfig, resolve_path
from ..data.splits import load_split
from ..models import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_config(cfg) -> Config:
    """Accept Config / dict / None, return a Config.

    Lightning's `load_from_checkpoint` re-instantiates the module by passing
    saved hparams (a plain dict) as kwargs. This helper makes both paths
    work transparently.
    """
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
        # Note: Huber on z-score-normalized data is fine (delta=1 is meaningful
        # in unit-variance space). On unnormalized kcal/mol it does not work.
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unknown loss {name!r}")


def _compute_target_stats(processed_dir: Path, train_split: Path) -> Tuple[float, float, int, int]:
    """Read training-set y_energy_mean values and return (mean, std, n_used, n_skipped).

    Cached: writes a small JSON file to processed_dir keyed by a fingerprint
    of (split file content + processed_dir path). First run takes ~30-60 sec
    reading 13K+ small files; subsequent runs are instant. Cache invalidates
    automatically if you re-preshard or change the split list.

    Robust to a few missing or corrupt files — those are skipped and counted.
    """
    import hashlib
    import json

    # Cache key: hash of (processed_dir absolute path + split file content)
    split_text = Path(train_split).read_text()
    fingerprint = hashlib.md5(
        f"{Path(processed_dir).resolve()}|{split_text}".encode()
    ).hexdigest()[:12]
    cache_file = Path(processed_dir) / f".target_stats_{fingerprint}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return (
                float(data["mean"]),
                float(data["std"]),
                int(data["n_used"]),
                int(data["n_skipped"]),
            )
        except Exception:
            pass  # fall through and recompute

    # Cache miss: compute fresh, with progress bar so the user sees activity.
    print(f"[ProtAI] Computing target normalization stats from training set "
          f"(cache miss; this is a one-time ~1 min cost)...")
    try:
        from tqdm import tqdm
        iterator = tqdm(load_split(train_split), desc="reading targets", unit="cplx")
    except ImportError:
        iterator = load_split(train_split)

    targets: List[float] = []
    skipped = 0
    for pid in iterator:
        path = Path(processed_dir) / f"{pid}.pt"
        if not path.exists():
            skipped += 1
            continue
        try:
            rec = torch.load(path, weights_only=False, mmap=True)
            targets.append(float(rec["y_energy_mean"]))
        except Exception:
            skipped += 1
            continue
    if not targets:
        raise RuntimeError(f"No training targets readable in {processed_dir}")

    t = torch.tensor(targets, dtype=torch.float32)
    mean = t.mean().item()
    std = max(t.std().item(), 1e-3)  # floor so we never divide by ~0
    n_used = len(targets)

    # Save cache for next time. Don't fail training if cache write fails
    # (read-only volume, permissions, etc) — just skip.
    try:
        cache_file.write_text(json.dumps({
            "mean": mean, "std": std,
            "n_used": n_used, "n_skipped": skipped,
        }))
    except Exception:
        pass

    return mean, std, n_used, skipped


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class ProtAILitModule(pl.LightningModule):
    """Lightning wrapper around any ProtAI model with target normalization.

    Forward contract for each `target` mode:
      * "binding_affinity" : model returns (B,) tensor — graph-level scalar.
                             Trained in normalized space, metrics in kcal/mol.
      * "adaptability"     : model returns (N,) tensor — per-atom scalar.
                             Trained in raw Å (already O(1)).
      * "multitask"        : model returns dict {"energy": (B,), "adaptability": (N,)}.
                             Energy head normalized, adaptability raw.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = kwargs.pop("config", None)
        cfg = _coerce_config(cfg)
        self.save_hyperparameters({"cfg": cfg.to_dict()})
        self.cfg = cfg

        # -------- Build model from config --------
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
                num_features=11,
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

        # -------- Target normalization buffers --------
        # Initialized to identity (mean=0, std=1) so the model behaves
        # correctly even before stats are computed (e.g., when loading a
        # checkpoint for inference). Real values populated in setup() at
        # the start of fit. Buffers travel with .to(device) automatically.
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))
        # If a checkpoint already contains these buffers, _stats_computed
        # stays True so we don't recompute and overwrite.
        self._stats_computed = False

        # -------- Validation accumulators (on-device) --------
        # Append per-batch tensors during validation_step, concat + sync
        # to CPU once at epoch end. Saves ~100 GPU stalls per val pass.
        self._val_y: List[torch.Tensor] = []
        self._val_yhat: List[torch.Tensor] = []

    # ----------------------------------------------------------------
    # Setup: compute target normalization stats once before training
    # ----------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Compute target normalization stats from the training split.

        Runs once at the start of `fit`. Skipped for adaptability/multitask
        targets that don't need scalar normalization. Skipped if stats
        already loaded from a checkpoint.
        """
        if self._stats_computed:
            return
        if stage != "fit":
            return
        if self.target == "adaptability":
            # Per-atom adaptability is already O(1-10), no normalization needed.
            self._stats_computed = True
            return

        processed_dir = resolve_path(self.cfg.data.processed_dir)
        train_split = resolve_path(self.cfg.data.splits_dir) / self.cfg.data.train_split

        mean, std, n_used, n_skipped = _compute_target_stats(processed_dir, train_split)
        # `.fill_` mutates the buffer in place, preserving its registered status.
        self.target_mean.fill_(mean)
        self.target_std.fill_(std)
        self._stats_computed = True

        # Print only on rank 0 (avoid duplicate logs in multi-GPU runs).
        if self.trainer is not None and self.trainer.is_global_zero:
            print(f"[ProtAI] Target normalization stats:")
            print(f"  source: {n_used} train samples (skipped {n_skipped})")
            print(f"  mean:   {mean:+.3f} kcal/mol")
            print(f"  std:    {std:.3f} kcal/mol")
            print(f"  → loss computed in normalized z-score space")
            print(f"  → metrics reported in original kcal/mol")

    # ----------------------------------------------------------------
    # Normalization helpers
    # ----------------------------------------------------------------

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def _denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.target_std + self.target_mean

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(self, data):
        """The model produces predictions in normalized space (for binding_affinity)
        or raw space (for adaptability). The choice is encoded by the target."""
        return self.model(data)

    # ----------------------------------------------------------------
    # Loss computation (in normalized space where applicable)
    # ----------------------------------------------------------------

    def _compute_loss(self, pred, data):
        """Returns (total_loss, dict_of_logging_components)."""
        if self.target == "binding_affinity":
            y_norm = self._normalize(data.y)
            return self.loss_fn(pred, y_norm), {}

        if self.target == "adaptability":
            # Per-atom values, no normalization.
            return self.loss_fn(pred, data.y), {}

        if self.target == "multitask":
            # Energy head normalized, adaptability raw.
            y_e_norm = self._normalize(data.y_energy)
            energy_loss = self.loss_fn(pred["energy"], y_e_norm)
            adapt_loss = self.loss_fn(pred["adaptability"], data.y_adapt)
            total = 0.5 * energy_loss + 0.5 * adapt_loss
            return total, {
                "loss/energy": energy_loss.detach(),
                "loss/adaptability": adapt_loss.detach(),
            }

        raise RuntimeError(f"unhandled target {self.target!r}")

    # ----------------------------------------------------------------
    # Training / validation steps
    # ----------------------------------------------------------------

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
        # Important: metrics are computed in ORIGINAL UNITS (kcal/mol for
        # energy, Å for adaptability). Predictions are denormalized accordingly.
        if self.target == "binding_affinity":
            y = data.y
            yhat = self._denormalize(pred)
        elif self.target == "adaptability":
            y = data.y
            yhat = pred
        elif self.target == "multitask":
            y = data.y_energy
            yhat = self._denormalize(pred["energy"])
        else:  # pragma: no cover
            raise RuntimeError(f"unhandled target {self.target!r}")

        self._val_y.append(y.detach().flatten())
        self._val_yhat.append(yhat.detach().flatten())

    def on_validation_epoch_end(self):
        if not self._val_y:
            return
        # Single GPU→CPU sync at epoch end (vs ~100 stalls per batch).
        y = torch.cat(self._val_y).float().cpu().numpy()
        yhat = torch.cat(self._val_yhat).float().cpu().numpy()
        metrics = self._compute_metrics(y, yhat, prefix="val")
        self.log_dict(metrics, prog_bar=True)
        self._val_y.clear()
        self._val_yhat.clear()

    def test_step(self, data, _idx):
        return self.validation_step(data, _idx)

    def on_test_epoch_end(self):
        if not self._val_y:
            return
        y = torch.cat(self._val_y).float().cpu().numpy()
        yhat = torch.cat(self._val_yhat).float().cpu().numpy()
        metrics = self._compute_metrics(y, yhat, prefix="test")
        self.log_dict(metrics)
        self._val_y.clear()
        self._val_yhat.clear()

    @staticmethod
    def _compute_metrics(y: np.ndarray, yhat: np.ndarray, prefix: str) -> Dict[str, float]:
        """Standard regression metrics. RMSE/MAE in target units, R²/Pearson/Spearman dimensionless.

        Defensively filters NaN/Inf predictions before computing — bf16-mixed
        precision occasionally produces a NaN for a single complex (e.g.,
        atypical structure triggering an overflow in the message-passing
        steps), and a single NaN would otherwise corrupt every aggregate.

        Returns 0.0 (not NaN) when variance is too small for correlation —
        Lightning's EarlyStopping treats NaN as "stop now" and would kill
        epoch 0 before the model has differentiated its outputs enough.
        """
        # Drop any NaN/Inf in either array before computing aggregates.
        finite_mask = np.isfinite(y) & np.isfinite(yhat)
        n_dropped = int((~finite_mask).sum())
        if n_dropped > 0:
            print(f"[lit_module] WARN: dropped {n_dropped}/{len(y)} non-finite "
                  f"values from {prefix} metrics (likely bf16 numerical artifact)")
        y = y[finite_mask]
        yhat = yhat[finite_mask]

        if len(y) == 0:
            # Pathological case: all predictions were NaN. Return zeros so
            # EarlyStopping doesn't bail.
            return {f"{prefix}/{k}": 0.0
                    for k in ("rmse", "mae", "pearson", "spearman", "r2")}

        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        mae = float(np.mean(np.abs(y - yhat)))
        # Correlation requires nonzero variance on both sides. At epoch 0
        # the model hasn't differentiated outputs yet (variance ~0), so
        # pearson is undefined. Return 0.0 instead of NaN so EarlyStopping
        # treats it as "no improvement yet" rather than "stop immediately."
        if y.std() > 1e-6 and yhat.std() > 1e-6:
            pearson = float(np.corrcoef(y, yhat)[0, 1])
            spearman = float(spearmanr(y, yhat).statistic)
            # corrcoef can still return NaN if inputs are pathological.
            if not np.isfinite(pearson):
                pearson = 0.0
            if not np.isfinite(spearman):
                spearman = 0.0
        else:
            pearson = 0.0
            spearman = 0.0
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        return {
            f"{prefix}/rmse": rmse,
            f"{prefix}/mae": mae,
            f"{prefix}/pearson": pearson,
            f"{prefix}/spearman": spearman,
            f"{prefix}/r2": r2,
        }

    # ----------------------------------------------------------------
    # Optimizer + LR schedule
    # ----------------------------------------------------------------

    def configure_optimizers(self):
        t = self.cfg.train
        if t.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay,
            )
        elif t.optimizer.lower() == "adam":
            opt = torch.optim.Adam(
                self.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer {t.optimizer!r}")

        if t.lr_schedule == "none":
            return opt

        if t.lr_schedule == "reduce_on_plateau":
            # Reduce when val/pearson stops improving (mode=max).
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=3,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val/pearson"},
            }

        # Default: cosine with warmup.
        warm = max(1, t.warmup_epochs)
        total = max(t.max_epochs, warm + 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < warm:
                return (epoch + 1) / warm
            progress = (epoch - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": sched}
