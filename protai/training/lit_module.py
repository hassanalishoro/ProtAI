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
from typing import Any, Dict, List, Optional, Tuple

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


def _compute_target_stats(
    processed_dir: Path,
    train_split: Path,
    target_field: str = "y_energy_mean",
    winsorize: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Compute (mean, std) of `target_field` over the training-set .pt files.

    Cached on disk: the cache key includes:
      * `processed_dir` absolute path
      * sha256(split file contents)
      * the target field name (so log_k vs y_energy_mean don't collide)
      * the most-recent .pt mtime referenced in the split (so a re-preshard
        invalidates the cache automatically — fixes review item A1).
      * winsorize percentiles (so toggling outlier handling re-derives stats)

    Returns a dict:
        {"mean": float, "std": float, "n_used": int, "n_skipped": int,
         "winsorize_bounds": (low, high) | None}
    """
    import hashlib
    import json

    pdb_ids = load_split(train_split)
    proc = Path(processed_dir).resolve()

    # mtime probe: take the latest mtime of up to 64 referenced .pt files.
    # 64 is enough to detect a re-preshard (which always overwrites in
    # bulk), without paying full I/O for the cache-key alone.
    sample_paths = [proc / f"{p}.pt" for p in pdb_ids[:64] if (proc / f"{p}.pt").exists()]
    if not sample_paths:
        sample_paths = [p for p in proc.glob("*.pt")][:64]
    latest_mtime = max((p.stat().st_mtime for p in sample_paths), default=0.0)

    split_text = Path(train_split).read_text()
    key_blob = (
        f"{proc}|{target_field}|{winsorize}|"
        f"{int(latest_mtime)}|{hashlib.sha256(split_text.encode()).hexdigest()}"
    )
    fingerprint = hashlib.md5(key_blob.encode()).hexdigest()[:16]
    cache_file = proc / f".target_stats_{target_field}_{fingerprint}.json"

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            wb = data.get("winsorize_bounds")
            return {
                "mean": float(data["mean"]),
                "std": float(data["std"]),
                "n_used": int(data["n_used"]),
                "n_skipped": int(data["n_skipped"]),
                "winsorize_bounds": tuple(wb) if wb is not None else None,
                "from_cache": True,
            }
        except Exception:
            pass

    print(
        f"[ProtAI] Computing target stats: field={target_field!r} "
        f"on {len(pdb_ids):,} train complexes (cache miss)..."
    )
    try:
        from tqdm import tqdm
        iterator = tqdm(pdb_ids, desc=f"reading {target_field}", unit="cplx")
    except ImportError:
        iterator = pdb_ids

    targets: List[float] = []
    skipped = 0
    for pid in iterator:
        path = proc / f"{pid}.pt"
        if not path.exists():
            skipped += 1
            continue
        try:
            rec = torch.load(path, weights_only=False, mmap=True)
            v = rec.get(target_field)
            if v is None:
                skipped += 1
                continue
            v = float(v) if not isinstance(v, torch.Tensor) else float(v.item())
            if not np.isfinite(v):
                skipped += 1
                continue
            targets.append(v)
        except Exception:
            skipped += 1
            continue

    if not targets:
        raise RuntimeError(
            f"No training targets readable for field {target_field!r} in {processed_dir}"
        )

    t = np.asarray(targets, dtype=np.float64)
    bounds: Optional[Tuple[float, float]] = None
    if winsorize is not None:
        lo, hi = float(np.percentile(t, winsorize[0])), float(np.percentile(t, winsorize[1]))
        t = np.clip(t, lo, hi)
        bounds = (lo, hi)

    mean = float(t.mean())
    std = float(max(t.std(), 1e-3))
    n_used = len(targets)

    try:
        cache_file.write_text(json.dumps({
            "mean": mean, "std": std,
            "n_used": n_used, "n_skipped": skipped,
            "winsorize_bounds": list(bounds) if bounds is not None else None,
            "target_field": target_field,
        }))
    except Exception:
        pass

    return {
        "mean": mean, "std": std,
        "n_used": n_used, "n_skipped": skipped,
        "winsorize_bounds": bounds,
        "from_cache": False,
    }


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

        # -------- Target normalization buffers (primary head) --------
        # Initialized to identity (mean=0, std=1) so the model behaves
        # correctly even before stats are computed (e.g., when loading a
        # checkpoint for inference). Real values populated in setup() at
        # the start of fit. Buffers travel with .to(device) automatically.
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

        # -------- Auxiliary normalization buffers --------
        # Used only by `multitask_logk_energy`: the primary head normalizes
        # log_k (mean ~6.5, std ~1.9), the auxiliary head normalizes the
        # MD energy (mean ~ -28, std ~51). Always allocated so checkpoints
        # round-trip cleanly across target choices.
        self.register_buffer("aux_target_mean", torch.tensor(0.0))
        self.register_buffer("aux_target_std", torch.tensor(1.0))

        # -------- Winsorization bounds (auxiliary head only) --------
        # When `cfg.train.winsorize_aux_pct` is set (e.g. (1.0, 99.0)), the
        # auxiliary energy targets are clipped at the training-set
        # percentiles before normalization. Bounds are computed once at
        # fit setup; defaults to ±inf (no-op) so checkpoints without
        # winsorization round-trip correctly.
        self.register_buffer("aux_winsor_lo", torch.tensor(float("-inf")))
        self.register_buffer("aux_winsor_hi", torch.tensor(float("+inf")))

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

        Runs once at the start of `fit`. Skipped for adaptability (per-atom,
        already O(1)) and skipped if stats already loaded from a checkpoint.

        Per-target field selection:
          * binding_affinity         → primary buffers from `y_energy_mean`
          * log_k                    → primary buffers from `y_logk`
          * multitask                → primary from `y_energy_mean`,
                                       adaptability head untouched (raw)
          * multitask_logk_energy    → primary from `y_logk`,
                                       auxiliary from `y_energy_mean`
                                       (with optional Winsorization)

        Winsorization is applied only to the AUXILIARY MD-energy head, where
        the long-tailed distribution and outliers like 4CP5 (-675 kcal/mol)
        actually motivate it. Log K targets are well-behaved (range 0-14)
        and don't need clipping.
        """
        if self._stats_computed:
            return
        if stage != "fit":
            return
        if self.target == "adaptability":
            self._stats_computed = True
            return

        processed_dir = resolve_path(self.cfg.data.processed_dir)
        train_split = resolve_path(self.cfg.data.splits_dir) / self.cfg.data.train_split

        # Pick the primary field by target.
        if self.target in ("binding_affinity", "multitask"):
            primary_field = "y_energy_mean"
        elif self.target in ("log_k", "multitask_logk_energy"):
            primary_field = "y_logk"
        else:
            primary_field = "y_energy_mean"

        primary = _compute_target_stats(processed_dir, train_split, target_field=primary_field)
        self.target_mean.fill_(primary["mean"])
        self.target_std.fill_(primary["std"])

        if self.trainer is not None and self.trainer.is_global_zero:
            unit = "kcal/mol" if primary_field == "y_energy_mean" else "(-log10 K)"
            print(f"[ProtAI] Primary normalization ({primary_field}):")
            print(f"  source : {primary['n_used']:,} train samples (skipped {primary['n_skipped']})")
            print(f"  mean   : {primary['mean']:+.3f} {unit}")
            print(f"  std    : {primary['std']:.3f} {unit}")
            print(f"  cached : {primary['from_cache']}")

        # Auxiliary stats only for the new multitask target.
        if self.target == "multitask_logk_energy":
            winsor = getattr(self.cfg.train, "winsorize_aux_pct", None)
            if winsor is not None:
                winsor = (float(winsor[0]), float(winsor[1]))
            aux = _compute_target_stats(
                processed_dir, train_split,
                target_field="y_energy_mean",
                winsorize=winsor,
            )
            self.aux_target_mean.fill_(aux["mean"])
            self.aux_target_std.fill_(aux["std"])
            if aux["winsorize_bounds"] is not None:
                self.aux_winsor_lo.fill_(aux["winsorize_bounds"][0])
                self.aux_winsor_hi.fill_(aux["winsorize_bounds"][1])
            if self.trainer is not None and self.trainer.is_global_zero:
                print(f"[ProtAI] Auxiliary normalization (y_energy_mean):")
                print(f"  source : {aux['n_used']:,} train samples (skipped {aux['n_skipped']})")
                print(f"  mean   : {aux['mean']:+.3f} kcal/mol")
                print(f"  std    : {aux['std']:.3f} kcal/mol")
                if aux["winsorize_bounds"] is not None:
                    print(f"  winsor : [{aux['winsorize_bounds'][0]:.2f}, {aux['winsorize_bounds'][1]:.2f}] kcal/mol "
                          f"(percentiles {winsor})")

        self._stats_computed = True

    # ----------------------------------------------------------------
    # Normalization helpers
    # ----------------------------------------------------------------

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize against the PRIMARY head's stats."""
        return (y - self.target_mean) / self.target_std

    def _denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize against the PRIMARY head's stats."""
        return y * self.target_std + self.target_mean

    def _normalize_aux(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize against the AUXILIARY head's stats. Applies Winsorization
        first when bounds are set (no-op if bounds are ±inf)."""
        y_clipped = torch.clamp(y, min=self.aux_winsor_lo, max=self.aux_winsor_hi)
        return (y_clipped - self.aux_target_mean) / self.aux_target_std

    def _denormalize_aux(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.aux_target_std + self.aux_target_mean

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

        if self.target == "log_k":
            y_norm = self._normalize(data.y)
            return self.loss_fn(pred, y_norm), {}

        if self.target == "adaptability":
            # Per-atom values, no normalization.
            return self.loss_fn(pred, data.y), {}

        if self.target == "multitask":
            y_e_norm = self._normalize(data.y_energy)
            energy_loss = self.loss_fn(pred["energy"], y_e_norm)
            adapt_loss = self.loss_fn(pred["adaptability"], data.y_adapt)
            total = 0.5 * energy_loss + 0.5 * adapt_loss
            return total, {
                "loss/energy": energy_loss.detach(),
                "loss/adaptability": adapt_loss.detach(),
            }

        if self.target == "multitask_logk_energy":
            # Headline: PDBbind log_k (primary head). Auxiliary: MD energy
            # with optional Winsorization. Loss weights configurable; default
            # is 0.9 logk + 0.1 energy so the headline metric dominates the
            # gradient signal but the energy head still gets enough learning
            # signal to act as a useful trunk regularizer.
            #
            # Gradient asymmetry caveat: the energy head's parameters only
            # see 0.1 * energy_loss in their gradient (the shared trunk sees
            # both contributions). So val/aux_energy_mae will be worse than
            # a standalone binding_affinity-target run reaches — that's the
            # expected behavior of an under-weighted auxiliary, not a bug.
            w_logk = float(getattr(self.cfg.train, "multitask_logk_weight", 0.9))
            w_energy = float(getattr(self.cfg.train, "multitask_energy_weight", 0.1))

            y_l_norm = self._normalize(data.y_logk)
            y_e_norm = self._normalize_aux(data.y_energy)

            logk_loss = self.loss_fn(pred["logk"], y_l_norm)
            energy_loss = self.loss_fn(pred["energy"], y_e_norm)
            total = w_logk * logk_loss + w_energy * energy_loss
            return total, {
                "loss/logk": logk_loss.detach(),
                "loss/energy_aux": energy_loss.detach(),
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
        # IMPORTANT: metrics reported in the target's ORIGINAL units. The
        # primary head's denormalize lifts back to kcal/mol or -log10(K)
        # depending on the target. The early-stopping signal (val/pearson)
        # is computed against this denormalized prediction.
        if self.target == "binding_affinity":
            y = data.y
            yhat = self._denormalize(pred)
        elif self.target == "log_k":
            y = data.y
            yhat = self._denormalize(pred)
        elif self.target == "adaptability":
            y = data.y
            yhat = pred
        elif self.target == "multitask":
            y = data.y_energy
            yhat = self._denormalize(pred["energy"])
        elif self.target == "multitask_logk_energy":
            # Headline metric is log_k; the energy head's metric is logged
            # separately below for diagnostics but doesn't drive early stop.
            y = data.y_logk
            yhat = self._denormalize(pred["logk"])
            # Auxiliary metric on energy (per-batch, eyeballed via TB).
            with torch.no_grad():
                e_pred = self._denormalize_aux(pred["energy"])
                e_err = (e_pred - data.y_energy).abs().mean()
                self.log("val/aux_energy_mae", e_err, batch_size=data.num_graphs)
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
